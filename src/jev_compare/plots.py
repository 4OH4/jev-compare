"""Charts for the benchmark notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


SHOT_NAMES = {"zeroshot": "Zero-shot", "fewshot": "Few-shot", "trained": "Trained"}


def plot_balanced_accuracy(
    summary: pd.DataFrame,
    datasets: tuple[str, ...],
    n_classes: dict[str, int],
    path: Path | None = None,
    shots: tuple[str, str] = ("zeroshot", "fewshot"),
):
    """Vertical bars of balanced accuracy, one group per dataset, one bar per model and condition.

    Within each dataset the bars are split by condition: zero-shot, then few-shot, then the trained
    SVM. The condition is named on the axis under its bars. Colour identifies the model only. A
    model that was not run leaves its slot empty, so positions do not move between runs.
    """
    raw = summary[summary.panel == "raw"]
    prompted = [m for m in ORDER if m != "svm"]
    subgroups = [(shot, prompted) for shot in shots] + [("trained", ["svm"])]
    n_test = int(raw.n.max())

    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Bars sit one unit apart. Condition sub-groups are separated by a small gap, datasets by a
    # larger one, so the spacing alone shows which bars belong together.
    width, sub_gap, group_gap = 0.8, 0.7, 2.4
    x = 0.0
    sub_ticks, sub_labels, group_ticks = [], [], []
    for g, ds in enumerate(datasets):
        if g:
            x += group_gap - sub_gap
        group_left = x
        for shot, models in subgroups:
            sub_left = x
            for model in models:
                hit = raw[(raw.dataset == ds) & (raw.model == model) & (raw.shot == shot)]
                if not hit.empty:
                    height = hit.bal_acc.iloc[0]
                    ax.bar(x, height, width, color=SLOTS[model], linewidth=0, zorder=3)
                    has_interval = {"bal_acc_lo", "bal_acc_hi"} <= set(hit.columns)
                    lo = hit.bal_acc_lo.iloc[0] if has_interval else np.nan
                    hi = hit.bal_acc_hi.iloc[0] if has_interval else np.nan
                    if pd.notna(lo) and pd.notna(hi):
                        ax.errorbar(
                            x,
                            height,
                            yerr=[[height - lo], [hi - height]],
                            fmt="none",
                            ecolor=INK_2,
                            elinewidth=1,
                            capsize=2,
                            zorder=4,
                        )
                x += 1
            sub_ticks.append((sub_left + x - 1) / 2)
            sub_labels.append(SHOT_NAMES[shot])
            x += sub_gap
        group_ticks.append((group_left + x - sub_gap - 1) / 2)
    right = x - sub_gap - 1

    # Two rows of labels under the bars: the condition, then the dataset.
    ax.set_xticks(sub_ticks)
    ax.set_xticklabels(sub_labels, color=INK_2, fontsize=9)
    ax.tick_params(axis="x", length=0, pad=6)
    datasets_axis = ax.secondary_xaxis("bottom")
    datasets_axis.set_xticks(group_ticks)
    datasets_axis.set_xticklabels(
        [f"{DATASET_NAMES.get(d, d)}\n{n_classes[d]} classes" for d in datasets],
        color=INK,
        fontsize=11,
    )
    datasets_axis.tick_params(axis="x", length=0, pad=24)
    datasets_axis.spines["bottom"].set_visible(False)

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)], color=INK_MUTED, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1, right + 1)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_ylabel("Balanced accuracy", color=INK_2, fontsize=10)

    fig.legend(
        handles=[Patch(facecolor=SLOTS[m], label=NAMES[m]) for m in ORDER],
        loc="lower center",
        ncol=len(ORDER),
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
        "Comparing Jev, frontier LLMs and TF-IDF/SVM for text classification tasks",
        color=INK,
        fontsize=14,
        weight="bold",
        ha="left",
    )
    subtitle = (
        f"Balanced accuracy by model and dataset, {n_test} sampled rows per dataset, "
        "one run per condition. Whiskers: 95% bootstrap interval."
    )
    fig.text(0.075, 0.915, subtitle, color=INK_2, fontsize=9, ha="left")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.22)

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
    title: str = "Reliability: stated confidence against accuracy",
    whiskers: bool | None = None,
    min_rows: int = 10,
):
    """Reliability diagram with the bucket counts beneath it.

    ``tables`` maps a model key to the output of ``calibration.reliability_table``. Each model is a
    line through its non-empty buckets, at the bucket's mean confidence, with the marker area in
    proportion to the number of rows. Whiskers are 95% Wilson intervals, drawn by default for two
    models or fewer (more would overlap; the intervals are in the tables). A point on the dotted
    diagonal is perfectly calibrated. A bucket with fewer than ``min_rows`` rows says little, so
    it is drawn as a hollow point and left out of the line. The lower panel gives the counts.
    """
    if whiskers is None:
        whiskers = len(tables) <= 2
    fig, (ax, cnt) = plt.subplots(
        2, 1, figsize=(7.4, 7.2), dpi=150, gridspec_kw={"height_ratios": [3.4, 1]}, sharex=True
    )
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax, "Accuracy in the bucket")
    _style_axes(cnt, "Rows")
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle=(0, (2, 3)), zorder=2,
            label="Perfect calibration")  # fmt: skip

    n_max = max(int(t.n.max()) for t in tables.values())
    keys = list(tables)
    for j, key in enumerate(keys):
        t = tables[key]
        t = t[t.n > 0]
        colour = SLOTS[key]
        solid, sparse = t[t.n >= min_rows], t[t.n < min_rows]
        ax.plot(solid.mean_confidence, solid.accuracy, color=colour, linewidth=2, zorder=3)
        if whiskers:
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
            solid.mean_confidence,
            solid.accuracy,
            s=30 + 220 * solid.n / n_max,
            color=colour,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=4,
        )
        ax.scatter(
            sparse.mean_confidence,
            sparse.accuracy,
            s=30,
            facecolor=SURFACE,
            edgecolor=colour,
            linewidth=1.3,
            zorder=4,
        )
        ax.plot([], [], color=colour, linewidth=2, marker="o", markersize=7, label=NAMES[key])
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
    fig.text(0.1, 0.965, title, color=INK, fontsize=14, weight="bold", ha="left")
    fig.text(0.1, 0.935, subtitle, color=INK_2, fontsize=9, ha="left")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.08, hspace=0.12)
    if path is not None:
        fig.savefig(path, facecolor=SURFACE)
    return fig


def plot_gated(
    curves: dict[str, pd.DataFrame],
    subtitle: str,
    path: Path | None = None,
    title: str = "What a confidence threshold does",
    bands: bool | None = None,
):
    """Accuracy and coverage as the confidence threshold rises, one panel each.

    ``curves`` maps a model key to the output of ``calibration.gated_curve``. Accuracy is on the
    rows at or above the threshold, with a shaded 95% Wilson interval. Coverage is the share of all
    rows that are at or above it. The two are separate panels because they have different scales
    and meanings. The bands are drawn by default for two models or fewer (more would overlap).
    """
    if bands is None:
        bands = len(curves) <= 2
    fig, (acc, cov) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(acc, "Accuracy of the answers kept")
    _style_axes(cov, "Share of all rows kept")
    for key, c in curves.items():
        colour = SLOTS[key]
        shown = c[c.n_kept > 0]
        if bands:
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
    fig.text(0.06, 0.955, title, color=INK, fontsize=14, weight="bold", ha="left")
    fig.text(0.06, 0.915, subtitle, color=INK_2, fontsize=9, ha="left")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.14, wspace=0.16)
    if path is not None:
        fig.savefig(path, facecolor=SURFACE)
    return fig
