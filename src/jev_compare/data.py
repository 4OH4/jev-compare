"""Dataset loading, seeded stratified sampling, few-shot pools and label descriptions.

Every split returned here is an *official* published split. Rows are identified by ``row_id``, the
row's position in its official split, so a sample can always be traced back to the source.

Sources:

- Banking77: the official CSVs published by PolyAI (10,003 train / 3,080 test). The Hub copy
  (``PolyAI/banking77``) is a loading script, which ``datasets`` 5 no longer runs.
- AG News: ``fancyzhx/ag_news`` on the Hub (120,000 train / 7,600 test).
- IMDb: ``stanfordnlp/imdb`` on the Hub (25,000 train / 25,000 test).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABELS_DIR = ROOT / "configs" / "labels"
CACHE_DIR = ROOT / "data" / "raw"  # gitignored download cache

DATASETS = ("banking77", "ag_news", "imdb")
BANKING77_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
)
IMDB_LABELS = {0: "negative", 1: "positive"}


@dataclass(frozen=True)
class LabelSet:
    """The task instruction and one description per label, shared by every model."""

    dataset: str
    task: str
    labels: dict[str, str]


def load_labels(dataset: str) -> LabelSet:
    spec = json.loads((LABELS_DIR / f"{dataset}.json").read_text(encoding="utf-8"))
    return LabelSet(dataset=spec["dataset"], task=spec["task"], labels=spec["labels"])


def _from_hub(repo: str, label_names: dict[int, str] | None = None) -> dict[str, pd.DataFrame]:
    from datasets import load_dataset

    ds = load_dataset(repo)
    out = {}
    for split in ("train", "test"):
        df = ds[split].to_pandas()[["text", "label"]]
        names = label_names or dict(enumerate(ds[split].features["label"].names))
        df["label"] = df["label"].map(names)
        out[split] = df
    return out


def load_splits(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the official (train, test) splits as frames with ``row_id``, ``text``, ``label``."""
    if dataset == "banking77":
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frames = {}
        for split in ("train", "test"):
            path = CACHE_DIR / f"banking77_{split}.csv"
            if not path.exists():
                pd.read_csv(BANKING77_URL + f"{split}.csv").to_csv(path, index=False)
            frames[split] = pd.read_csv(path).rename(columns={"category": "label"})
    elif dataset == "ag_news":
        frames = _from_hub("fancyzhx/ag_news")
    elif dataset == "imdb":
        frames = _from_hub("stanfordnlp/imdb", IMDB_LABELS)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")

    out = []
    for split in ("train", "test"):
        df = frames[split][["text", "label"]].copy()
        df.insert(0, "row_id", np.arange(len(df)))
        out.append(df)
    train, test = out
    labels = load_labels(dataset).labels
    unknown = (set(train.label) | set(test.label)) - set(labels)
    if unknown:
        raise ValueError(f"{dataset}: labels missing from the label file: {sorted(unknown)}")
    return train, test


def fingerprint(df: pd.DataFrame) -> str:
    """A hash of the texts and labels of a split, in order, to show two runs used the same data."""
    rows = pd.util.hash_pandas_object(df[["text", "label"]], index=False)
    return hashlib.sha256(rows.to_numpy().tobytes()).hexdigest()


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` rows, spread as evenly as possible over the classes.

    Each class gets ``n // k`` rows and the remainder goes to classes picked at random, so with
    more classes than rows (Banking77 at n=10) each row comes from a different class. The result
    is deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    classes = sorted(df.label.unique())
    k = len(classes)
    if n > len(df):
        raise ValueError(f"cannot sample {n} rows from {len(df)}")
    quota = dict.fromkeys(classes, n // k)
    extra = rng.choice(k, size=n - (n // k) * k, replace=False) if n % k else []
    for i in extra:
        quota[classes[i]] += 1

    parts = []
    for cls in classes:
        rows = df[df.label == cls]
        take = min(quota[cls], len(rows))
        parts.append(rows.sample(n=take, random_state=int(rng.integers(2**31))))
    out = pd.concat(parts)
    # Order is shuffled so that any prefix of the sample is not sorted by class.
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def few_shot_pool(train: pd.DataFrame, seed: int, max_chars: int = 1200) -> pd.DataFrame:
    """One labelled example per class from the official train split, in a shuffled order.

    Examples longer than ``max_chars`` are skipped so that a long IMDb review does not dominate
    the prompt. Every class in the frame must have at least one row that short. The examples are
    shuffled (seeded) so that the example next to the text to classify is not always the same
    class. The choice of rows does not depend on the shuffle.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for cls in sorted(train.label.unique()):
        rows = train[(train.label == cls) & (train.text.str.len() <= max_chars)]
        if rows.empty:
            raise ValueError(f"no train example for {cls!r} within {max_chars} characters")
        parts.append(rows.sample(n=1, random_state=int(rng.integers(2**31))))
    pool = pd.concat(parts)
    return pool.sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)


def threshold_carveout(
    train: pd.DataFrame, n: int, seed: int, exclude_row_ids: set[int]
) -> pd.DataFrame:
    """A stratified chunk of train, disjoint from the few-shot examples, for fitting thresholds.

    This is our own carve-out from the official train split, not a published validation split.
    """
    pool = train[~train.row_id.isin(exclude_row_ids)]
    return stratified_sample(pool, n, seed)
