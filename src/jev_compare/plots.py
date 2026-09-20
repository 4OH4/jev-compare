"""Charts for the benchmark notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

# Light-mode ink and surface, and the first five categorical slots of the default palette.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SLOTS = {
    "jev": "#2a78d6",
    "haiku": "#eb6834",
    "openai": "#1baf7a",
    "gemini": "#eda100",
    "svm": "#e87ba4",
}
NAMES = {
    "jev": "Jev",
    "haiku": "Claude Haiku 4.5",
    "openai": "OpenAI small",
    "gemini": "Gemini Flash",
    "svm": "TF-IDF + SVM",
}
ORDER = ["jev", "haiku", "openai", "gemini", "svm"]
DATASET_NAMES = {"banking77": "Banking77", "ag_news": "AG News", "imdb": "IMDb"}


def _tint(colour: str, amount: float) -> tuple[float, float, float]:
    """Blend a colour towards the surface. ``amount`` is the share of the colour kept."""
    c, s = np.array(to_rgb(colour)), np.array(to_rgb(SURFACE))
    return tuple(amount * c + (1 - amount) * s)


def plot_balanced_accuracy(
    summary: pd.DataFrame,
    datasets: tuple[str, ...],
    n_classes: dict[str, int],
    path: Path | None = None,
    shots: tuple[str, str] = ("zeroshot", "fewshot"),
):
    """Vertical bars of balanced accuracy, one group per dataset, one bar per model and condition.

    Colour identifies the model. Within a model, few-shot is the full colour and zero-shot is a
    lighter tint. The SVM is trained once, so it has a single bar. A model that was not run leaves
    its slot empty, so positions do not move between runs.
    """
    raw = summary[summary.panel == "raw"]
    slots = [(m, s) for m in ORDER for s in (("trained",) if m == "svm" else shots)]
    n_test = int(raw.n.max())

    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    width, group_gap = 0.8, 1.6
    pitch = len(slots) + group_gap
    centres = []
    for g, ds in enumerate(datasets):
        left = g * pitch
        centres.append(left + (len(slots) - 1) / 2)
        for i, (model, shot) in enumerate(slots):
            hit = raw[(raw.dataset == ds) & (raw.model == model) & (raw.shot == shot)]
            if hit.empty:
                continue
            colour = SLOTS[model] if shot in ("fewshot", "trained") else _tint(SLOTS[model], 0.5)
            height = hit.bal_acc.iloc[0]
            ax.bar(left + i, height, width, color=colour, linewidth=0, zorder=3)
            has_interval = {"bal_acc_lo", "bal_acc_hi"} <= set(hit.columns)
            lo = hit.bal_acc_lo.iloc[0] if has_interval else np.nan
            hi = hit.bal_acc_hi.iloc[0] if has_interval else np.nan
            if pd.notna(lo) and pd.notna(hi):
                ax.errorbar(
                    left + i,
                    height,
                    yerr=[[height - lo], [hi - height]],
                    fmt="none",
                    ecolor=INK_2,
                    elinewidth=1,
                    capsize=2,
                    zorder=4,
                )

    ax.set_xticks(centres)
    ax.set_xticklabels(
        [f"{DATASET_NAMES.get(d, d)}\n{n_classes[d]} classes" for d in datasets],
        color=INK,
        fontsize=11,
    )
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)], color=INK_MUTED, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1, (len(datasets) - 1) * pitch + len(slots))
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_ylabel("Balanced accuracy", color=INK_2, fontsize=10)

    # Legend: model identity (colour), then the shot condition (full colour or tint).
    model_handles = [Patch(facecolor=SLOTS[m], label=NAMES[m]) for m in ORDER]
    shot_handles = [
        Patch(facecolor=INK_MUTED, label="Few-shot (or trained)"),
        Patch(facecolor=_tint(INK_MUTED, 0.5), label="Zero-shot"),
    ]
    fig.legend(
        handles=model_handles + shot_handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        handlelength=1.1,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.text(
        0.075,
        0.955,
        "Balanced accuracy by model and dataset",
        color=INK,
        fontsize=14,
        weight="bold",
        ha="left",
    )
    subtitle = (
        f"Official test split, {n_test} sampled rows per dataset, one run per condition. "
        "Bars: answered rows. Whiskers: 95% bootstrap interval."
    )
    fig.text(0.075, 0.915, subtitle, color=INK_2, fontsize=9, ha="left")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.2)

    if path is not None:
        fig.savefig(path, facecolor=SURFACE)
    return fig


def _style_axes(ax, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(axis="both", length=0, labelcolor=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def plot_reliability(
    tables: dict[str, pd.DataFrame],
    edges: tuple[float, ...],
    subtitle: str,
    path: Path | None = None,
):
    """Reliability diagram with the bucket counts beneath it.

    ``tables`` maps a model key to the output of ``calibration.reliability_table``. Each model is a
    line through its non-empty buckets, at the bucket's mean confidence, with the marker area in
    proportion to the number of rows. Whiskers are 95% Wilson intervals. A point on the dotted
    diagonal is perfectly calibrated. The lower panel gives the counts, because a bucket with few
    rows says little.
    """
    fig, (ax, cnt) = plt.subplots(
        2, 1, figsize=(7.4, 7.2), dpi=150, gridspec_kw={"height_ratios": [3.4, 1]}, sharex=True
    )
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax, "Accuracy in the bucket")
    _style_axes(cnt, "Rows")
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle=(0, (2, 3)), zorder=2)
    ax.text(0.62, 0.52, "Perfect calibration", color=INK_MUTED, fontsize=8)

    n_max = max(int(t.n.max()) for t in tables.values())
    keys = list(tables)
    for j, key in enumerate(keys):
        t = tables[key]
        t = t[t.n > 0]
        colour = SLOTS[key]
        ax.plot(t.mean_confidence, t.accuracy, color=colour, linewidth=2, zorder=3)
        ax.errorbar(
            t.mean_confidence,
            t.accuracy,
            yerr=[t.accuracy - t.accuracy_lo, t.accuracy_hi - t.accuracy],
            fmt="none",
            ecolor=colour,
            elinewidth=1,
            capsize=2,
            alpha=0.6,
            zorder=3,
        )
        ax.scatter(
            t.mean_confidence,
            t.accuracy,
            s=30 + 220 * t.n / n_max,
            color=colour,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=4,
            label=NAMES[key],
        )
        full = tables[key]
        centre = (full.lo + full.hi) / 2
        width = (full.hi - full.lo) / (len(keys) + 1)
        offset = (j - (len(keys) - 1) / 2) * width
        cnt.bar(centre + offset, full.n, width * 0.92, color=colour, linewidth=0, zorder=3)
    ax.set_xlim(-0.02, 1.02)  # room for a marker at exactly 1.0
    ax.set_ylim(0, 1.05)
    cnt.set_xticks(list(edges))
    cnt.set_xticklabels([f"{e:.1f}" for e in edges])
    cnt.set_xlabel("Stated confidence", color=INK_2, fontsize=10)
    cnt.set_ylim(0, None)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.text(0.1, 0.965, "Reliability: stated confidence against accuracy", color=INK,
             fontsize=14, weight="bold", ha="left")  # fmt: skip
    fig.text(0.1, 0.935, subtitle, color=INK_2, fontsize=9, ha="left")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.08, hspace=0.12)
    if path is not None:
        fig.savefig(path, facecolor=SURFACE)
    return fig


def plot_gated(
    curves: dict[str, pd.DataFrame],
    subtitle: str,
    path: Path | None = None,
):
    """Accuracy and coverage as the confidence threshold rises, one panel each.

    ``curves`` maps a model key to the output of ``calibration.gated_curve``. Accuracy is on the
    rows at or above the threshold, with a shaded 95% Wilson interval. Coverage is the share of all
    rows that are at or above it. The two are separate panels because they have different scales
    and meanings.
    """
    fig, (acc, cov) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(acc, "Accuracy of the answers kept")
    _style_axes(cov, "Share of all rows kept")
    for key, c in curves.items():
        colour = SLOTS[key]
        shown = c[c.n_kept > 0]
        acc.fill_between(shown.threshold, shown.accuracy_lo, shown.accuracy_hi, color=colour,
                         alpha=0.15, linewidth=0, zorder=2)  # fmt: skip
        acc.plot(shown.threshold, shown.accuracy, color=colour, linewidth=2, zorder=3,
                 label=NAMES[key])  # fmt: skip
        cov.plot(c.threshold, c.coverage, color=colour, linewidth=2, zorder=3, label=NAMES[key])
    acc.set_ylim(None, 1.01)
    cov.set_ylim(0, 1.03)
    for ax in (acc, cov):
        ax.set_xlabel("Confidence threshold", color=INK_2, fontsize=10)
    cov.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.text(0.06, 0.955, "What a confidence threshold does", color=INK, fontsize=14,
             weight="bold", ha="left")  # fmt: skip
    fig.text(0.06, 0.915, subtitle, color=INK_2, fontsize=9, ha="left")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.14, wspace=0.16)
    if path is not None:
        fig.savefig(path, facecolor=SURFACE)
    return fig
