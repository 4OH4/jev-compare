"""Providers for the benchmark: Jev and three frontier LLMs behind one ``classify`` interface.

Every provider takes the shared label set and one ``state`` string (the text to classify, plus the
few-shot examples in the few-shot condition) and returns an ``Attempt``. Anything that goes wrong
is raised as ``CallFailure`` with a ``kind``, so the runner can retry transient failures and
record the rest explicitly.

The LLMs are asked for structured output that names one label and reports a ``confidence`` (their
own estimate, from 0 to 1, that the label is right). The schema forces the label to be one of the
valid labels. There is no chain of thought. Jev is asked one ``choice`` question whose criteria are
the same label descriptions.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

from jev_compare import jev
from jev_compare.data import LabelSet

load_dotenv()

# Failure kinds. Transient kinds are retried; the rest are recorded as they are.
TRANSIENT = {"rate_limit", "server", "timeout", "connection"}
# Kinds that describe what the model did with the request, not the infrastructure.
MODEL_BEHAVIOUR = {"schema_violation", "incomplete", "refusal", "context_length"}

MAX_OUTPUT_TOKENS = 1000  # room for Gemini's thinking tokens, which count against the cap

# Wording of a "request too long" error. It must not match errors about the output limit
# (``max_tokens``, ``max_output_tokens``), which are configuration mistakes, not model behaviour.
CONTEXT_LENGTH_ERROR = re.compile(
    r"context[ _](length|window|limit)|(prompt|input|request) is too long|too many tokens"
    r"|(prompt|input|request)\b.{0,60}\bexceeds?\b.{0,40}\b(maximum|limit)"
    r"|exceeds? the (maximum|limit)",
    re.I,
)


class CallFailure(Exception):
    """One failed attempt. ``raw`` holds whatever the provider returned, if anything.

    ``retry_after`` is the number of seconds the server asked us to wait, if it said.
    """

    def __init__(self, kind: str, message: str, raw: Any = None, retry_after: float | None = None):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message
        self.raw = raw
        self.retry_after = retry_after


class Pacer:
    """Spaces the start of requests at least ``60 / per_minute`` seconds apart, across threads.

    Used where the API has a fixed request quota. Sending at the quota's rate wastes no requests on
    rejections, whereas sending faster only produces 429s and retries. ``per_minute=None`` turns it
    off. ``hold`` pushes the next slot back, for when the server has said how long to wait.
    """

    def __init__(self, per_minute: float | None):
        self.interval = 60.0 / per_minute if per_minute else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        time.sleep(max(0.0, start - now))

    def hold(self, seconds: float) -> None:
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


def retry_after_seconds(headers: Any) -> float | None:
    """The ``Retry-After`` header as a number of seconds, or None if absent or not a number."""
    try:
        value = float(headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


@dataclass
class Attempt:
    """One successful call. ``probs`` is set for Jev only; the LLMs report ``confidence``."""

    label: str
    confidence: float | None
    probs: dict[str, float] | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # only where the provider reports it (Jev, via the gateway)
    latency_s: float
    model_returned: str | None
    raw: Any
    # ``input_tokens`` is the total input, including the two counts below.
    cached_input_tokens: int = 0  # read from the provider's prompt cache (billed at a discount)
    cache_write_tokens: int = 0  # written to the cache (billed at a premium by Anthropic only)


# --- prompts and schema --------------------------------------------------------------------------


def system_prompt(labels: LabelSet) -> str:
    listing = "\n".join(f"- {name}: {desc}" for name, desc in labels.labels.items())
    return (
        f"{labels.task}\n\nLabels:\n{listing}\n\n"
        "Reply with JSON containing `label` (exactly one of the labels above) and `confidence` "
        "(a number from 0 to 1: your estimate of the probability that the label is correct)."
    )


def response_schema(labels: LabelSet) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(labels.labels)},
            "confidence": {"type": "number"},
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }


def parse_llm_json(text: str, labels: LabelSet, raw: Any) -> tuple[str, float]:
    try:
        data = json.loads(text)
        label, conf = data["label"], float(data["confidence"])
    except (ValueError, KeyError, TypeError) as e:
        raise CallFailure("schema_violation", f"unparseable output {text!r}: {e!r}", raw) from e
    if label not in labels.labels:
        raise CallFailure("schema_violation", f"label {label!r} is not a valid label", raw)
    return label, min(1.0, max(0.0, conf))


def classify_status(status: Any, name: str, text: str) -> CallFailure:
    if "Timeout" in name:
        kind = "timeout"
    elif "Connection" in name:
        kind = "connection"
    elif status == 429:
        kind = "rate_limit"
    elif isinstance(status, int) and status >= 500:
        kind = "server"
    elif status in (401, 403):
        kind = "auth"
    elif status == 400 and CONTEXT_LENGTH_ERROR.search(text) and "max_" not in text.lower():
        kind = "context_length"
    elif status == 400:
        kind = "bad_request"
    elif status == 404:
        kind = "not_found"
    else:
        kind = "other"
    return CallFailure(kind, f"{name}: {text[:500]}")


def classify_exception(e: Exception) -> CallFailure:
    """Map an SDK or transport exception to a failure kind, without importing every SDK's types."""
    name = type(e).__name__
    if isinstance(e, httpx.TimeoutException):
        name = "Timeout" + name
    elif isinstance(e, httpx.TransportError):
        name = "Connection" + name
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    failure = classify_status(status, name, str(e))
    failure.retry_after = retry_after_seconds(
        getattr(getattr(e, "response", None), "headers", None)
    )
    return failure


# --- providers -----------------------------------------------------------------------------------


class JevProvider:
    QUESTION = "label"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.client = httpx.Client(timeout=90.0)
        # The gateway allows a fixed number of requests a minute, so requests are paced to stay
        # under it (``requests_per_minute`` in configs/models.json).
        self.pacer = Pacer(cfg.get("requests_per_minute"))

    def classify(self, labels: LabelSet, state: str, shared_prefix: str = "") -> Attempt:
        questions = {self.QUESTION: jev.choice(labels.task, labels.labels)}
        self.pacer.wait()  # before the request is timed, so the wait is not counted as latency
        try:
            raw = jev.evaluate_raw(state, questions, model=self.cfg["model_id"], client=self.client)
        except httpx.HTTPError as e:
            raise classify_exception(e) from e
        if not raw.ok:
            failure = classify_status(raw.status_code, "HTTPStatus", str(raw.body))
            failure.raw = raw.body
            failure.retry_after = retry_after_seconds(raw.headers)
            if failure.kind == "rate_limit" and failure.retry_after:
                self.pacer.hold(failure.retry_after)  # every worker waits, not just this one
            raise failure
        try:
            result = jev.parse(raw)
            answer = result.answers[self.QUESTION]
            label, probs = answer["choice"], answer["probabilities"]
        except (jev.JevError, KeyError, TypeError) as e:
            raise CallFailure("schema_violation", repr(e), raw.body) from e
        if label not in labels.labels:
            raise CallFailure("schema_violation", f"label {label!r} is not a valid label", raw.body)
        return Attempt(
            label=label,
            confidence=answer.get("confidence"),
            probs=probs,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.market_cost,
            latency_s=raw.latency_s,
            model_returned=result.model,
            raw=raw.body,
        )


class AnthropicProvider:
    def __init__(self, cfg: dict[str, Any]):
        import anthropic

        self.cfg = cfg
        self.client = anthropic.Anthropic(max_retries=0, timeout=90.0)

    def classify(self, labels: LabelSet, state: str, shared_prefix: str = "") -> Attempt:
        # Prompt caching is explicit on Anthropic: the cache breakpoint goes at the end of the part
        # that is the same for every row. With few-shot examples that is the system prompt plus the
        # examples (one breakpoint covers both). Without them it is the system prompt alone.
        # Prefixes shorter than the model's minimum (4096 tokens for Haiku 4.5) are not cached.
        cache = {"type": "ephemeral"}
        if shared_prefix:
            system: Any = system_prompt(labels)
            content: Any = [
                {"type": "text", "text": shared_prefix, "cache_control": cache},
                {"type": "text", "text": state[len(shared_prefix) :]},
            ]
        else:
            system = [{"type": "text", "text": system_prompt(labels), "cache_control": cache}]
            content = state
        start = time.perf_counter()
        try:
            resp = self.client.messages.create(
                model=self.cfg["model_id"],
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config={
                    "format": {"type": "json_schema", "schema": response_schema(labels)}
                },
                **self.cfg.get("params", {}),
            )
        except Exception as e:
            raise classify_exception(e) from e
        latency = time.perf_counter() - start
        raw = resp.to_dict()
        if resp.stop_reason != "end_turn":
            kind = "refusal" if resp.stop_reason == "refusal" else "incomplete"
            raise CallFailure(kind, f"stop_reason={resp.stop_reason}", raw)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        label, conf = parse_llm_json(text, labels, raw)
        u = resp.usage
        read, write = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
        # Anthropic's input_tokens counts only the tokens after the last cache breakpoint.
        return Attempt(
            label, conf, None, u.input_tokens + read + write, u.output_tokens, None, latency,
            resp.model, raw, read, write,
        )  # fmt: skip


class OpenAIProvider:
    def __init__(self, cfg: dict[str, Any]):
        import openai

        self.cfg = cfg
        self.client = openai.OpenAI(max_retries=0, timeout=90.0)

    def classify(self, labels: LabelSet, state: str, shared_prefix: str = "") -> Attempt:
        start = time.perf_counter()
        try:
            resp = self.client.responses.create(
                model=self.cfg["model_id"],
                instructions=system_prompt(labels),
                input=state,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "classification",
                        "schema": response_schema(labels),
                        "strict": True,
                    }
                },
                max_output_tokens=MAX_OUTPUT_TOKENS,
                # Caching is automatic for prompts of 1,024 tokens or more, matched on the prefix.
                # The key sends requests that share a prefix to the same cache.
                prompt_cache_key=f"jev-compare-{labels.dataset}-{bool(shared_prefix)}",
                **self.cfg.get("params", {}),
            )
        except Exception as e:
            raise classify_exception(e) from e
        latency = time.perf_counter() - start
        raw = resp.model_dump(mode="json")
        if resp.status != "completed":
            raise CallFailure("incomplete", f"status={resp.status} {resp.incomplete_details}", raw)
        label, conf = parse_llm_json(resp.output_text, labels, raw)
        cached = resp.usage.input_tokens_details.cached_tokens or 0
        return Attempt(
            label, conf, None, resp.usage.input_tokens, resp.usage.output_tokens, None, latency,
            resp.model, raw, cached,
        )  # fmt: skip


class GeminiProvider:
    """Gemini, with explicit prompt caching of the part of the request shared by every row.

    Implicit caching was tried first and reported no cached tokens, so the system prompt and the
    few-shot examples are cached explicitly (``client.caches``). The cache is created on first use
    and deleted by ``cleanup``. The row's text is then sent as a second user turn. A prompt too
    short to cache (the API rejects it) is sent uncached.
    """

    CACHE_TTL = "7200s"  # longer than the slowest condition, so a cache cannot expire mid-run

    def __init__(self, cfg: dict[str, Any]):
        from google import genai
        from google.genai import types

        self.cfg = cfg
        self.types = types
        self.client = genai.Client(
            api_key=os.environ[cfg["env_key"]],
            http_options=types.HttpOptions(timeout=90_000),
        )
        self._caches: dict[tuple[str, str], tuple[str, int] | None] = {}
        self._lock = threading.Lock()

    def _cache_for(self, labels: LabelSet, shared_prefix: str) -> tuple[str | None, int]:
        """Return (cache name, tokens written by this call). The name is None if not cacheable."""
        key = (labels.dataset, shared_prefix)
        with self._lock:
            if key in self._caches:
                entry = self._caches[key]
                return (entry[0], 0) if entry else (None, 0)
            types = self.types
            contents = (
                [types.Content(role="user", parts=[types.Part(text=shared_prefix)])]
                if shared_prefix
                else None
            )
            try:
                cache = self.client.caches.create(
                    model=self.cfg["model_id"],
                    config=types.CreateCachedContentConfig(
                        system_instruction=system_prompt(labels),
                        contents=contents,
                        ttl=self.CACHE_TTL,
                    ),
                )
            except Exception:
                self._caches[key] = None  # too short to cache, or caching refused
                return None, 0
            tokens = cache.usage_metadata.total_token_count or 0
            self._caches[key] = (cache.name, tokens)
            return cache.name, tokens

    def cleanup(self) -> None:
        for entry in self._caches.values():
            if entry:
                with contextlib.suppress(Exception):  # the cache expires by itself anyway
                    self.client.caches.delete(name=entry[0])
        self._caches.clear()

    def classify(self, labels: LabelSet, state: str, shared_prefix: str = "") -> Attempt:
        cache_name, created = self._cache_for(labels, shared_prefix)
        settings: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_json_schema": response_schema(labels),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            **self.cfg.get("params", {}),
        }
        if cache_name:
            settings["cached_content"] = cache_name  # holds the system prompt and the examples
            contents = state[len(shared_prefix) :]
        else:
            settings["system_instruction"] = system_prompt(labels)
            contents = state
        config = self.types.GenerateContentConfig(**settings)
        start = time.perf_counter()
        try:
            resp = self.client.models.generate_content(
                model=self.cfg["model_id"], contents=contents, config=config
            )
        except Exception as e:
            raise classify_exception(e) from e
        latency = time.perf_counter() - start
        raw = resp.model_dump(mode="json")
        candidate = resp.candidates[0] if resp.candidates else None
        finish = getattr(getattr(candidate, "finish_reason", None), "name", None)
        if candidate is None or finish != "STOP" or not resp.text:
            raise CallFailure("incomplete", f"finish_reason={finish}", raw)
        label, conf = parse_llm_json(resp.text, labels, raw)
        usage = resp.usage_metadata
        # Thinking tokens are billed as output, so they are counted with the visible output.
        out_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        cached = usage.cached_content_token_count or 0
        # The call that created the cache also paid for writing it: those tokens are counted as
        # extra input, written once, and billed at the normal input price. Storage (about $0.002
        # per hour for a few thousand tokens) is not counted.
        return Attempt(
            label, conf, None, (usage.prompt_token_count or 0) + created, out_tokens, None, latency,
            resp.model_version, raw, cached, created,
        )  # fmt: skip


PROVIDERS = {
    "jev": JevProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "google": GeminiProvider,
}


def has_key(cfg: dict[str, Any]) -> bool:
    return bool(os.environ.get(cfg.get("env_key", "AI_GATEWAY_API_KEY")))


def make_provider(cfg: dict[str, Any]):
    return PROVIDERS[cfg["provider"]](cfg)


def check_model_id(cfg: dict[str, Any]) -> tuple[bool, str]:
    """Ask the provider whether the configured model ID exists. Jev is checked by the first call."""
    if cfg["provider"] == "jev":
        return True, "checked by the first call"
    provider = make_provider(cfg)
    try:
        if cfg["provider"] == "google":
            provider.client.models.get(model=cfg["model_id"])
        else:
            provider.client.models.retrieve(cfg["model_id"])
    except Exception as e:
        return False, classify_exception(e).message
    return True, "found in the provider's model list"
