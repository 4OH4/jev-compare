"""Calibration measures for the calibration notebook (03).

All functions take one confidence per row (a number from 0 to 1) and whether the row's label was
right. The confidence is that of the label the model chose, so the measures are top-label
calibration measures.

- Buckets are open on the right ([lo, hi)), except the last, which also includes 1.0. Every
  confidence therefore falls in exactly one bucket, so no row is dropped from a table or an ECE.
- ECE is the weighted mean, over buckets, of the gap between the bucket's accuracy and its mean
  confidence. The weight of a bucket is its share of the rows.
- Brier score is the mean squared difference between the confidence and the outcome (1 if right,
  0 if wrong). For Jev it is a top-label Brier score, not the multiclass score over all 77 labels,
  so that it can be compared with a model that reports one confidence only.
- Intervals come from resampling rows with replacement. The same resampled rows are used for every
  model, so differences between models are paired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Buckets of 0.1 from 0.5 up, and one below 0.5 so that low-confidence answers are not dropped.
BIN_EDGES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
Z95 = 1.959964
N_BOOT = 2000
BOOT_SEED = 0
TOL = 1e-9


def assign_bins(conf, edges=BIN_EDGES) -> np.ndarray:
    """Index of the bucket for each confidence. 1.0 goes in the last bucket."""
    idx = np.searchsorted(np.asarray(edges), np.asarray(conf, dtype=float), side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def wilson_interval(k, n, z: float = Z95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a proportion k/n. Empty groups give NaN."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def reliability_table(conf, correct, edges=BIN_EDGES) -> pd.DataFrame:
    """One row per bucket: count, mean confidence, accuracy (with a Wilson interval) and gap."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=bool)
    b = assign_bins(conf, edges)
    rows = []
    for i in range(len(edges) - 1):
        m = b == i
        n = int(m.sum())
        hits = int(correct[m].sum())
        lo, hi = wilson_interval(hits, n)
        close = "]" if i == len(edges) - 2 else ")"
        rows.append(
            {
                "bucket": f"[{edges[i]:.1f}, {edges[i + 1]:.1f}{close}",
                "lo": edges[i],
                "hi": edges[i + 1],
                "n": n,
                "n_correct": hits,
                "mean_confidence": float(conf[m].mean()) if n else np.nan,
                "accuracy": hits / n if n else np.nan,
                "accuracy_lo": float(lo) if n else np.nan,
                "accuracy_hi": float(hi) if n else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["gap"] = out.accuracy - out.mean_confidence  # positive: under-confident
    return out


def ece(conf, correct, edges=BIN_EDGES) -> float:
    """Expected calibration error: sum over buckets of (n_b / N) * |accuracy_b - confidence_b|."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    b = assign_bins(conf, edges)
    total = 0.0
    for i in range(len(edges) - 1):
        m = b == i
        total += abs(correct[m].sum() - conf[m].sum())  # n_b * |acc - conf| = |sum(y) - sum(c)|
    return float(total / len(conf))


def brier(conf, correct) -> float:
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    return float(np.mean((conf - correct) ** 2))


def gated_curve(conf, correct, thresholds) -> pd.DataFrame:
    """Accuracy on the rows with confidence at or above each threshold, and the share kept.

    ``accuracy`` is NaN where no row clears the threshold. The interval is a Wilson interval, which
    is wide when few rows are left; read it, not the line alone.
    """
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=bool)
    rows = []
    for x in thresholds:
        keep = conf >= x - TOL  # tolerance: 0.95 from a float range must still keep a stated 0.95
        n = int(keep.sum())
        hits = int(correct[keep].sum())
        lo, hi = wilson_interval(hits, n)
        rows.append(
            {
                "threshold": float(x),
                "n_kept": n,
                "coverage": n / len(conf),
                "accuracy": hits / n if n else np.nan,
                "accuracy_lo": float(lo) if n else np.nan,
                "accuracy_hi": float(hi) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap(
    conf: dict[str, np.ndarray],
    correct: dict[str, np.ndarray],
    edges=BIN_EDGES,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, dict[str, np.ndarray]]:
    """Bootstrap ECE, Brier score and accuracy for each predictor, on shared resampled rows.

    ``conf`` and ``correct`` are keyed by predictor name and must cover the same rows in the same
    order. Returns ``{predictor: {"ece": array, "brier": array, "accuracy": array}}``, each array
    of length ``n_boot``.
    """
    names = list(conf)
    n = len(conf[names[0]])
    if any(len(conf[k]) != n or len(correct[k]) != n for k in names):
        raise ValueError("every predictor must have one confidence and outcome per row")
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n, size=(n_boot, n), dtype=np.int32)
    out = {}
    for k in names:
        c = np.asarray(conf[k], dtype=float)[pick]
        y = np.asarray(correct[k], dtype=float)[pick]
        b = assign_bins(c, edges)
        e = np.zeros(n_boot)
        for i in range(len(edges) - 1):
            m = b == i
            e += np.abs((y * m).sum(axis=1) - (c * m).sum(axis=1))
        out[k] = {
            "ece": e / n,
            "brier": ((c - y) ** 2).mean(axis=1),
            "accuracy": y.mean(axis=1),
        }
    return out


def summarise(
    conf: dict[str, np.ndarray],
    correct: dict[str, np.ndarray],
    edges=BIN_EDGES,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    reference: str | None = None,
) -> pd.DataFrame:
    """Accuracy, ECE and Brier score per predictor, each with a 95% bootstrap interval.

    If ``reference`` names a predictor, every other predictor also gets a row per measure for the
    difference (that predictor minus the reference), with a paired interval. Differences are
    labelled ``"<name> minus <reference>"``.
    """
    boot = bootstrap(conf, correct, edges, n_boot, seed)
    points = {
        k: {
            "accuracy": float(np.mean(correct[k])),
            "ece": ece(conf[k], correct[k], edges),
            "brier": brier(conf[k], correct[k]),
        }
        for k in conf
    }
    rows = []

    def add(name, measure, value, draws):
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append({"predictor": name, "measure": measure, "value": value, "lo": lo, "hi": hi})

    for k in conf:
        for measure in ("accuracy", "ece", "brier"):
            add(k, measure, points[k][measure], boot[k][measure])
    if reference is not None:
        for k in conf:
            if k == reference:
                continue
            for measure in ("accuracy", "ece", "brier"):
                add(
                    f"{k} minus {reference}",
                    measure,
                    points[k][measure] - points[reference][measure],
                    boot[k][measure] - boot[reference][measure],
                )
    out = pd.DataFrame(rows)
    out["interval_excludes_zero"] = np.where(
        out.predictor.str.contains(" minus "), (out.lo > 0) | (out.hi < 0), np.nan
    )
    return out
