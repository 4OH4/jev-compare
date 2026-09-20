"""HTTP client for Jev (``typesafe-ai/jev``) via Vercel AI Gateway.

Contract (docs: https://vercel.com/docs/ai-gateway/modalities/evaluation, and confirmed against a
live call on 2026-09-20): ``POST /v1/evaluate`` with ``model``, ``state`` and ``questions``,
authenticated by ``Authorization: Bearer $AI_GATEWAY_API_KEY``.

Question types on this route are ``boolean`` (TypeSafe's "noul"), ``choice`` and ``score``.
Observed response shape::

    {"model": "typesafe-ai/jev",
     "answers": {
        "<boolean q>": {"type": "boolean", "probability": 0.98},
        "<choice q>":  {"type": "choice", "choice": "billing",
                        "probabilities": {"billing": 1, ...}, "confidence": 1},
        "<score q>":   {"type": "score", "score": 1.02,
                        "probabilities": {"0": 0.03, "1": 0.92, "2": 0.05}, "confidence": 0.89}},
     "usage": {"inputTokens": 413, "outputTokens": 68},
     "providerMetadata": {"typesafe": {"confidence": {...}}, "gateway": {"cost": ..., ...}}}

Notes from that call: boolean answers carry no ``confidence`` (the probability is the signal);
score probabilities are keyed by the level number (0, 1, 2 ...) as a string; there is no model
version string in the body, headers, or the models listing (only a release date), and
``gateway.cost`` was "0" while ``gateway.marketCost`` held the real price.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/evaluate"
MODELS_URL = "https://ai-gateway.vercel.sh/typesafe/v1/models"
DEFAULT_MODEL = "typesafe-ai/jev"


class JevError(RuntimeError):
    """A non-2xx response or a body that does not have the expected shape."""

    def __init__(self, message: str, raw: RawResponse):
        super().__init__(message)
        self.raw = raw


@dataclass
class RawResponse:
    """One HTTP round trip, kept verbatim. Failures are recorded, never swallowed."""

    request: dict[str, Any]
    status_code: int
    latency_s: float
    body: Any  # parsed JSON if the body was JSON, else the raw text
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


@dataclass
class JevResult:
    """A parsed successful response. ``raw`` keeps the full round trip."""

    model: str
    answers: dict[str, dict[str, Any]]
    input_tokens: int
    output_tokens: int
    market_cost: float  # what the call would cost at list price (gateway.cost was 0 in testing)
    latency_s: float
    raw: RawResponse


def boolean(instructions: str, criteria: dict[str, str] | None = None) -> dict[str, Any]:
    q: dict[str, Any] = {"type": "boolean", "instructions": instructions}
    if criteria:
        q["criteria"] = criteria  # {"true": ..., "false": ...}
    return q


def choice(instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
    """``criteria`` is ordered lowest to highest; answers key each level by its 0-based index."""
    return {"type": "score", "instructions": instructions, "criteria": criteria}


def _api_key() -> str:
    load_dotenv()
    key = os.environ.get("AI_GATEWAY_API_KEY")
    if not key:
        raise RuntimeError("AI_GATEWAY_API_KEY is not set (put it in .env; see .env.example)")
    return key


def evaluate_raw(
    state: str | dict | list,
    questions: dict[str, dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> RawResponse:
    """POST one evaluation request and return the untouched response.

    Non-2xx statuses are returned, not raised, so callers can log them as failures. Transport
    errors (timeouts, connection resets) still raise ``httpx.HTTPError``.
    """
    payload = {"model": model, "state": state, "questions": questions}
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}

    start = time.perf_counter()
    if client is None:
        with httpx.Client(timeout=timeout) as c:
            resp = c.post(GATEWAY_URL, json=payload, headers=headers)
    else:
        resp = client.post(GATEWAY_URL, json=payload, headers=headers)
    latency = time.perf_counter() - start

    try:
        body: Any = resp.json()
    except ValueError:
        body = resp.text
    return RawResponse(payload, resp.status_code, latency, body, dict(resp.headers))


def parse(raw: RawResponse) -> JevResult:
    """Parse a response into a ``JevResult``; raise ``JevError`` on failure or unexpected shape."""
    if not raw.ok:
        raise JevError(f"HTTP {raw.status_code}: {raw.body}", raw)
    try:
        body = raw.body
        gateway = body.get("providerMetadata", {}).get("gateway", {})
        return JevResult(
            model=body["model"],
            answers=body["answers"],
            input_tokens=body["usage"]["inputTokens"],
            output_tokens=body["usage"]["outputTokens"],
            market_cost=float(gateway.get("marketCost", 0)),
            latency_s=raw.latency_s,
            raw=raw,
        )
    except (KeyError, TypeError, AttributeError) as e:
        raise JevError(f"unexpected response shape ({e!r}): {raw.body}", raw) from e


def evaluate(
    state: str | dict | list, questions: dict[str, dict[str, Any]], **kwargs: Any
) -> JevResult:
    return parse(evaluate_raw(state, questions, **kwargs))


def list_models(timeout: float = 30.0) -> Any:
    """GET the TypeSafe-compatible model listing (name, description, release_date)."""
    headers = {"Authorization": f"Bearer {_api_key()}"}
    resp = httpx.get(MODELS_URL, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
