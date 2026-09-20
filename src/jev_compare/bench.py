"""Benchmark runner and scoring, shared by the Phase 2 notebook (and reusable by Phase 3).

Failure policy:

- Transient failures (rate limit, 5xx, timeout, connection) are retried, up to ``MAX_ATTEMPTS``
  attempts per row, after the wait the server asked for (``Retry-After``) or an exponential
  backoff. Retries are logged on the row.
- Any failure that remains is recorded on the row with its kind. It is never dropped and never
  silently scored.
- Balanced accuracy is reported two ways: on the rows that got an answer, and with every failed
  row counted as wrong. Failure counts are reported next to both.
- An exception that is not an API failure (a bug, or an unexpected response) is recorded on the row
  as ``unexpected`` and counted with the infrastructure failures.
- A condition stops early, with ``RunAborted``, if its first rows show a systematic problem (see
  ``run_condition``), so that a bad key or setting does not spend the whole run.
- No response is cached. Every run makes fresh calls and overwrites its result file.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import subprocess
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from jev_compare.data import LABELS_DIR, ROOT, LabelSet
from jev_compare.llm import MODEL_BEHAVIOUR, TRANSIENT, CallFailure, system_prompt

RESULTS = ROOT / "results"
MAX_ATTEMPTS = 8
MAX_RETRY_WAIT = 120  # seconds: the longest wait before a retry, whatever the server asks for
FAILED = "__failed__"
# Early stop: within the first GATE_ROWS rows of a condition, more than GATE_MAX_FAILED_SHARE of
# them failing (after retries), or a failure that no other row can fix (a bad key or model ID).
GATE_ROWS = 20
GATE_MAX_FAILED_SHARE = 0.25
GATE_FATAL_KINDS = {"auth", "not_found"}
N_BOOT = 2000  # bootstrap resamples for confidence intervals
BOOT_SEED = 0
PACKAGES = (
    "anthropic", "openai", "google-genai", "httpx", "scikit-learn", "pandas", "numpy", "datasets",
)  # fmt: skip
ROW_COLUMNS = (
    "pred", "confidence", "probs", "input_tokens", "output_tokens", "cost_usd", "latency_s",
    "model_returned", "cached_input_tokens", "cache_write_tokens", "error_kind", "error",
)  # fmt: skip


# --- prompts -------------------------------------------------------------------------------------


def shared_prefix(examples: list[tuple[str, str]] | None) -> str:
    """The part of the state that is the same for every row: the few-shot examples, or nothing."""
    if not examples:
        return ""
    shots = "\n\n".join(f"Text: {t}\nLabel: {label}" for t, label in examples)
    return f"Labelled examples:\n\n{shots}\n\nText to classify:\n"


def build_state(text: str, examples: list[tuple[str, str]] | None) -> str:
    """The one string every model sees for a row.

    Zero-shot: the text alone. Few-shot: the labelled examples first, then the text. Jev receives
    this as its ``state`` and the LLMs as the user message, so the formatting is identical. The
    row's own text comes last so that everything before it can be served from a prompt cache.
    """
    return shared_prefix(examples) + text


# --- running -------------------------------------------------------------------------------------


def _classify_with_retry(
    provider, labels: LabelSet, state: str, prefix: str = ""
) -> dict[str, Any]:
    """Classify one row. ``wall_s`` is the time from the first attempt to the outcome, including
    the failed attempts and the waits between them; ``latency_s`` is the successful call only."""
    retries: list[dict[str, Any]] = []
    start = time.perf_counter()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            a = provider.classify(labels, state, prefix)
        except CallFailure as f:
            if f.kind in TRANSIENT and attempt < MAX_ATTEMPTS:
                retries.append(
                    {"kind": f.kind, "message": f.message[:300], "retry_after": f.retry_after}
                )
                # Wait as long as the server asked. Without a request, back off exponentially.
                wait = f.retry_after if f.retry_after is not None else min(60, 2**attempt)
                time.sleep(min(wait, MAX_RETRY_WAIT) + random.random())
                continue
            return {
                "status": "failed",
                "error_kind": f.kind,
                "error": f.message,
                "raw": f.raw,
                "attempts": attempt,
                "retries": retries,
                "wall_s": time.perf_counter() - start,
            }
        except Exception as e:  # a bug or an unexpected response shape: record it, do not abort
            return {
                "status": "failed",
                "error_kind": "unexpected",
                "error": f"{type(e).__name__}: {e}"[:500],
                "raw": {"traceback": traceback.format_exc()},
                "attempts": attempt,
                "retries": retries,
                "wall_s": time.perf_counter() - start,
            }
        return {
            "status": "ok",
            "pred": a.label,
            "confidence": a.confidence,
            "probs": a.probs,
            "input_tokens": a.input_tokens,
            "output_tokens": a.output_tokens,
            "cost_usd": a.cost_usd,
            "latency_s": a.latency_s,
            "model_returned": a.model_returned,
            "cached_input_tokens": a.cached_input_tokens,
            "cache_write_tokens": a.cache_write_tokens,
            "raw": a.raw,
            "attempts": attempt,
            "retries": retries,
            "wall_s": time.perf_counter() - start,
        }
    raise AssertionError("unreachable")


class RunAborted(RuntimeError):
    """A condition was stopped early because its first rows failed in a systematic way."""


def provenance(dataset: str) -> dict[str, Any]:
    """What produced a results file, so a number can be traced to the code and settings behind it.

    ``code_sha256`` covers every source file in ``src/jev_compare``. The git commit alone would not
    identify the code, because work that is not committed yet is not in it.
    """

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    def sha(*paths: Path) -> str:
        h = hashlib.sha256()
        for p in paths:
            h.update(p.name.encode() + p.read_bytes())
        return h.hexdigest()

    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    changed = git("status", "--porcelain", "--", ".", ":(exclude)results", ":(exclude)data")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if changed is None else bool(changed),
        "code_sha256": sha(*sorted((ROOT / "src" / "jev_compare").glob("*.py"))),
        "labels_sha256": sha(LABELS_DIR / f"{dataset}.json"),
        "models_sha256": sha(ROOT / "configs" / "models.json"),
        "packages": versions,
    }


def run_condition(
    *,
    model_key: str,
    cfg: dict[str, Any],
    provider,
    labels: LabelSet,
    rows: pd.DataFrame,
    shot: str,
    examples: list[tuple[str, str]] | None,
    split: str,
    path: Path,
    workers: int = 4,
) -> pd.DataFrame:
    """Classify every row and write ``path`` (JSON lines: one header, then one record per row).

    The header holds everything shared by the rows (prompt, label descriptions, few-shot examples,
    model ID, parameters, provenance) so it is stored once. Each row record keeps the full raw
    response.

    Early stop: among the first ``GATE_ROWS`` rows to finish, a failure of a kind in
    ``GATE_FATAL_KINDS`` or more than ``GATE_MAX_FAILED_SHARE`` of the rows failing stops the
    condition. Rows not yet started are skipped and ``RunAborted`` is raised, with the distinct
    error messages. The file is left as far as it got, so it has fewer rows than ``n_rows``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "record": "header",
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_key": model_key,
        "provider": cfg["provider"],
        "model_id_requested": cfg["model_id"],
        "params": cfg.get("params", {}),
        "dataset": labels.dataset,
        "split": split,
        "shot": shot,
        "n_rows": len(rows),
        "workers": workers,
        "task": labels.task,
        "labels": labels.labels,
        "system_prompt": system_prompt(labels),
        "few_shot_examples": [{"text": t, "label": lab} for t, lab in examples or []],
        "provenance": provenance(labels.dataset),
    }
    lock = threading.Lock()
    records: list[dict[str, Any]] = []
    stop: list[str] = []  # the reason, once the gate has tripped

    def check_gate() -> None:
        """Call with the lock held, after a record was added."""
        if stop or len(records) > GATE_ROWS:
            return
        failed = [r for r in records if r["status"] != "ok"]
        fatal = [r for r in failed if r["error_kind"] in GATE_FATAL_KINDS]
        if fatal or len(failed) > GATE_MAX_FAILED_SHARE * GATE_ROWS:
            errors = sorted({(r["error_kind"], str(r["error"])[:200]) for r in failed})
            listing = "; ".join(f"{kind}: {msg}" for kind, msg in errors[:5])
            stop.append(
                f"{model_key} {labels.dataset} {shot} {split} stopped after {len(records)} rows, "
                f"{len(failed)} failed. {listing}"
            )

    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")

        def work(row) -> None:
            if stop:
                return
            state = build_state(row.text, examples)
            out = _classify_with_retry(provider, labels, state, shared_prefix(examples))
            rec = {
                "record": "row",
                "row_id": int(row.row_id),
                "dataset": labels.dataset,
                "split": split,
                "model": model_key,
                "shot": shot,
                "true_label": row.label,
                "text": row.text,
                **out,
            }
            with lock:
                f.write(json.dumps(rec, default=str) + "\n")
                f.flush()
                records.append(rec)
                check_gate()

        # The first row runs alone so that it can fill the providers' prompt caches. Rows sent in
        # parallel from the start would all miss the cache.
        first, rest = rows.iloc[:1], rows.iloc[1:]
        for row in first.itertuples(index=False):
            work(row)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, rest.itertuples(index=False)))

    if stop:
        raise RunAborted(stop[0])
    return load_results(path)


def load_results(path: Path) -> pd.DataFrame:
    """Row records of a results file as a frame, in the order of ``row_id``."""
    recs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("record") == "row":
                recs.append(rec)
    df = pd.DataFrame(recs).sort_values("row_id").reset_index(drop=True)
    for col in ROW_COLUMNS:  # an all-failed run has no answer columns
        if col not in df:
            df[col] = None
    return df


def read_header(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.loads(f.readline())


# --- scoring -------------------------------------------------------------------------------------


def score_frame(df: pd.DataFrame, pred_col: str = "pred") -> dict[str, Any]:
    """Accuracy measures and failure counts for one condition.

    ``bal_acc`` uses the rows that got an answer. ``bal_acc_fail_wrong`` counts every failed row as
    wrong. Failures split into model behaviour (bad output, refusal, context overflow) and
    infrastructure (rate limits, network) so the two can be read separately.
    """
    ok = df[df.status == "ok"]
    failed = df[df.status != "ok"]
    n_model = int(failed.error_kind.isin(MODEL_BEHAVIOUR).sum()) if len(failed) else 0

    def bal(true, pred):
        if not len(true):
            return float("nan")
        with warnings.catch_warnings():
            # Predicting a class that is absent from a small sample is expected here.
            warnings.simplefilter("ignore", UserWarning)
            return balanced_accuracy_score(true, pred)

    filled = df[pred_col].where(df.status == "ok", FAILED)
    lo, hi = bal_acc_interval(ok.true_label, ok[pred_col])
    return {
        "n": len(df),
        "n_failed": len(failed),
        "n_failed_model": n_model,
        "n_failed_infra": len(failed) - n_model,
        "bal_acc": bal(ok.true_label, ok[pred_col]),
        "bal_acc_lo": lo,
        "bal_acc_hi": hi,
        "bal_acc_fail_wrong": bal(df.true_label, filled),
        "accuracy": accuracy_score(ok.true_label, ok[pred_col]) if len(ok) else float("nan"),
    }


# --- uncertainty ---------------------------------------------------------------------------------


def _boot_bal_acc(
    true: np.ndarray, correct: list[np.ndarray], n_boot: int, seed: int
) -> np.ndarray:
    """Bootstrap balanced accuracy for one or more predictors on the same rows.

    Rows are resampled with replacement from the whole sample, so the mix of classes varies between
    resamples. Resampling within each class would give no spread for a class with a single row,
    which is the case for Banking77 in a small sample. Balanced accuracy is the mean per-class
    recall over the classes present in a resample. The same resampled rows are used for every
    predictor, which is what makes differences between predictors paired. Returns an array
    (predictors, n_boot).
    """
    rng = np.random.default_rng(seed)
    classes, codes = np.unique(true, return_inverse=True)
    pick = rng.integers(0, len(true), size=(n_boot, len(true)), dtype=np.int32)
    picked_codes = codes.astype(np.int32)[pick]
    picked_correct = [c[pick] for c in correct]
    recalls = np.full((len(correct), n_boot, len(classes)), np.nan)
    for k in range(len(classes)):
        in_class = picked_codes == k
        seen = in_class.sum(axis=1)
        for m, c in enumerate(picked_correct):
            hits = (in_class & c).sum(axis=1)
            recalls[m, :, k] = np.where(seen > 0, hits / np.maximum(seen, 1), np.nan)
    return np.nanmean(recalls, axis=2)


def bal_acc_interval(
    true, pred, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[float, float]:
    """95% bootstrap interval for balanced accuracy over the rows (see ``_boot_bal_acc``)."""
    true, pred = np.asarray(true), np.asarray(pred)
    if not len(true):
        return float("nan"), float("nan")
    boot = _boot_bal_acc(true, [true == pred], n_boot, seed)[0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def paired_difference(
    true, pred_a, pred_b, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> dict[str, float]:
    """Balanced accuracy of A minus B on the same rows, with a 95% paired bootstrap interval."""
    true, pred_a, pred_b = np.asarray(true), np.asarray(pred_a), np.asarray(pred_b)
    correct = [true == pred_a, true == pred_b]
    classes = np.unique(true)
    recall = [np.mean([c[true == cls].mean() for cls in classes]) for c in correct]
    boot = _boot_bal_acc(true, correct, n_boot, seed)
    lo, hi = np.percentile(boot[0] - boot[1], [2.5, 97.5])
    return {
        "n_rows": len(true),
        "bal_acc_a": float(recall[0]),
        "bal_acc_b": float(recall[1]),
        "diff": float(recall[0] - recall[1]),
        "diff_lo": float(lo),
        "diff_hi": float(hi),
    }


def paired_comparisons(
    runs: dict[tuple, pd.DataFrame],
    svm: dict[str, pd.DataFrame],
    datasets: tuple[str, ...],
    shots: tuple[str, ...] = ("zeroshot", "fewshot"),
) -> pd.DataFrame:
    """The fixed list of comparisons, each on the rows both sides answered.

    Comparisons: Jev against each LLM in the same shot condition, Jev against the SVM in each shot
    condition, and few-shot against zero-shot for every API model. ``runs`` is keyed by
    (dataset, model, shot, split) and ``svm`` by dataset (frames with ``row_id``, ``true_label``,
    ``pred`` and ``status``). A comparison is skipped if either side did not run.
    """

    def get(ds, model, shot):
        df = svm.get(ds) if model == "svm" else runs.get((ds, model, shot, "test"))
        if df is None:
            return None
        return df[df.status == "ok"].set_index("row_id")

    models = sorted({m for (_, m, _, sp) in runs if sp == "test"})
    llms = [m for m in models if m != "jev"]
    plan = []
    for ds in datasets:
        for shot in shots:
            plan += [(ds, ("jev", shot), (m, shot)) for m in llms]
            plan.append((ds, ("jev", shot), ("svm", "trained")))
        plan += [(ds, (m, shots[1]), (m, shots[0])) for m in models]

    out = []
    for ds, (ma, sa), (mb, sb) in plan:
        a, b = get(ds, ma, sa), get(ds, mb, sb)
        if a is None or b is None:
            continue
        common = a.index.intersection(b.index)
        if not len(common):
            continue
        d = paired_difference(
            a.loc[common, "true_label"], a.loc[common, "pred"], b.loc[common, "pred"]
        )
        out.append({"dataset": ds, "a": f"{ma} {sa}", "b": f"{mb} {sb}", **d})
    res = pd.DataFrame(out)
    if len(res):
        res["interval_excludes_zero"] = (res.diff_lo > 0) | (res.diff_hi < 0)
    return res


def p_positive(df: pd.DataFrame, positive: str = "positive") -> pd.Series:
    """Probability of the positive class per row, for the binary threshold fit.

    Jev reports a probability for each label. The LLMs report a confidence in the label they
    chose, so a negative answer with confidence c counts as 1 - c.
    """

    def one(r):
        if r.status != "ok":
            return np.nan
        if isinstance(r.probs, dict):
            return float(r.probs.get(positive, 0.0))
        return r.confidence if r.pred == positive else 1 - r.confidence

    return df.apply(one, axis=1)


def fit_threshold(df: pd.DataFrame, positive: str = "positive") -> float:
    """The threshold on P(positive) that maximises balanced accuracy; ties go closest to 0.5."""
    ok = df[df.status == "ok"]
    p = p_positive(ok, positive).to_numpy()
    y = (ok.true_label == positive).to_numpy()
    values = np.unique(p)
    candidates = np.unique(
        np.concatenate([[0.5], (values[:-1] + values[1:]) / 2, [values[0] - 1e-9]])
    )
    best = max(
        candidates,
        key=lambda t: (balanced_accuracy_score(y, p >= t), -abs(t - 0.5)),
    )
    return float(best)


def apply_threshold(
    df: pd.DataFrame, threshold: float, positive: str = "positive", negative: str = "negative"
) -> pd.DataFrame:
    out = df.copy()
    p = p_positive(out, positive)
    out["pred_adjusted"] = np.where(p >= threshold, positive, negative)
    out.loc[out.status != "ok", "pred_adjusted"] = None
    return out


# --- cost and latency ----------------------------------------------------------------------------


def cost_latency(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Tokens, cost and latency for one condition, from the successful rows.

    Cost uses the provider's cache prices for tokens read from or written to the prompt cache
    (``cached_input_per_mtok``, ``cache_write_per_mtok``; each defaults to the normal input price).
    ``cost_usd_per_1k_rows_no_cache`` prices the same tokens with no caching, to show the saving.
    Cost counts the successful attempt of each row only. Tokens of failed attempts are not known.
    """
    ok = df[df.status == "ok"]
    if ok.empty:
        return {}
    inp, out = ok.input_tokens.sum(), ok.output_tokens.sum()
    cached, written = ok.cached_input_tokens.sum(), ok.cache_write_tokens.sum()
    if ok.cost_usd.notna().all():
        cost = no_cache = float(ok.cost_usd.sum())  # Jev: the gateway's list-price cost
    else:
        p_in, p_out = cfg["input_per_mtok"], cfg["output_per_mtok"]
        p_read = cfg.get("cached_input_per_mtok", p_in)
        p_write = cfg.get("cache_write_per_mtok", p_in)
        fresh = inp - cached - written
        cost = (fresh * p_in + cached * p_read + written * p_write + out * p_out) / 1e6
        no_cache = (inp * p_in + out * p_out) / 1e6
    lat = ok.latency_s.to_numpy()
    n = len(ok)
    return {
        "input_tokens_per_row": float(inp / n),
        "output_tokens_per_row": float(out / n),
        "cached_share": float(cached / inp) if inp else 0.0,
        "cost_usd_per_1k_rows": cost / n * 1000,
        "cost_usd_per_1k_rows_no_cache": no_cache / n * 1000,
        "latency_median_s": float(statistics.median(lat)),
        "latency_p95_s": float(np.percentile(lat, 95)),
        # Latency is the successful call alone. These include the failed attempts and the waits
        # between them, so they show what a rate-limited model costs in time.
        "wall_median_s": float(ok.wall_s.median()),
        "wall_p95_s": float(np.percentile(ok.wall_s, 95)),
        "retry_rate": float((df.attempts > 1).mean()),
    }


# --- SVM -----------------------------------------------------------------------------------------


def run_svm(
    train: pd.DataFrame,
    test_sample: pd.DataFrame,
    test_full: pd.DataFrame,
    path: Path,
    seed: int,
    c_grid: tuple[float, ...] = (0.1, 0.3, 1, 3, 10),
) -> dict[str, Any]:
    """TF-IDF plus a linear SVM, fitted on the official train split only.

    ``C`` is chosen by 5-fold cross-validation inside train (the TF-IDF vocabulary is refitted in
    each fold, so no fold leaks into another). Predictions on the sample are written to ``path``.
    Inference time is also measured on the full official test split, since ten rows is too few to
    time.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LinearSVC())
    search = GridSearchCV(
        pipe,
        {"linearsvc__C": list(c_grid)},
        scoring="balanced_accuracy",
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    search.fit(train.text, train.label)  # grid search, then a refit on all of train
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = search.predict(test_sample.text)
    sample_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_full = search.predict(test_full.text)
    full_s = time.perf_counter() - t0

    out = test_sample[["row_id", "label"]].rename(columns={"label": "true_label"}).copy()
    out["pred"] = pred
    out["status"] = "ok"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    lo, hi = bal_acc_interval(test_full.label, pred_full, n_boot=500)  # keeps memory modest
    return {
        "best_C": search.best_params_["linearsvc__C"],
        "cv_bal_acc": float(search.best_score_),
        "fit_time_s": fit_s,
        "predict_ms_per_row": full_s / len(test_full) * 1000,
        "predict_sample_s": sample_s,
        "n_train": len(train),
        # The whole official test split, for reference. The sample is what the other models see.
        "n_full_test": len(test_full),
        "bal_acc_full_test": float(balanced_accuracy_score(test_full.label, pred_full)),
        "bal_acc_full_test_lo": lo,
        "bal_acc_full_test_hi": hi,
        **score_frame(out.assign(status="ok")),
    }
