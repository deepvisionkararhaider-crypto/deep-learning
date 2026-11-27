"""UCI Adult loading and preprocessing for the MLP.

Source: https://archive.ics.uci.edu/dataset/2/adult  (CC BY 4.0)
Files: adult.data (32,561 rows, training) and adult.test (16,281 rows, held out).
The preprocessing statistics are fitted ONLY on the training portion to prevent leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]
NUMERIC = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "education", "marital_status", "occupation",
               "relationship", "race", "sex", "native_country"]
LABELS = ["<=50K", ">50K"]


def _read(path: Path, skip_header: bool) -> pd.DataFrame:
    df = pd.read_csv(
        path, names=COLUMNS, skipinitialspace=True, na_values="?",
        skiprows=1 if skip_header else 0, comment=None,
    )
    # adult.test labels carry a trailing period, e.g. "<=50K."
    df["income"] = df["income"].astype(str).str.rstrip(".")
    return df.dropna().reset_index(drop=True)


@dataclass
class AdultPreprocessor:
    """Fits numeric standardisation and one-hot vocabularies on training data only."""
    means: dict = field(default_factory=dict)
    stds: dict = field(default_factory=dict)
    vocab: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "AdultPreprocessor":
        for c in NUMERIC:
            self.means[c] = float(df[c].mean())
            self.stds[c] = float(df[c].std() or 1.0)
        for c in CATEGORICAL:
            self.vocab[c] = sorted(df[c].unique().tolist())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for c in NUMERIC:
            parts.append(((df[c].astype(float) - self.means[c]) / self.stds[c]).to_numpy()[:, None])
        for c in CATEGORICAL:
            cats = self.vocab[c]
            idx = {v: i for i, v in enumerate(cats)}
            onehot = np.zeros((len(df), len(cats) + 1), dtype=np.float32)  # last slot = unseen
            for row, v in enumerate(df[c].tolist()):
                onehot[row, idx.get(v, len(cats))] = 1.0
            parts.append(onehot)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def feature_names(self) -> list[str]:
        names = list(NUMERIC)
        for c in CATEGORICAL:
            names += [f"{c}={v}" for v in self.vocab[c]] + [f"{c}=<unseen>"]
        return names

    def to_dict(self) -> dict:
        return {"means": self.means, "stds": self.stds, "vocab": self.vocab}

    @classmethod
    def from_dict(cls, d: dict) -> "AdultPreprocessor":
        return cls(means=d["means"], stds=d["stds"], vocab=d["vocab"])


def load_splits(raw_dir: str | Path, val_fraction: float = 0.1, seed: int = 42):
    """Return (train_df, val_df, test_df). Validation is a stratified split of adult.data."""
    raw_dir = Path(raw_dir)
    full_train = _read(raw_dir / "adult.data", skip_header=False)
    test = _read(raw_dir / "adult.test", skip_header=True)
    rng = np.random.default_rng(seed)
    val_idx = []
    for label in LABELS:
        idx = full_train.index[full_train["income"] == label].to_numpy()
        rng.shuffle(idx)
        val_idx.extend(idx[: int(round(len(idx) * val_fraction))].tolist())
    val = full_train.loc[sorted(val_idx)].reset_index(drop=True)
    train = full_train.drop(index=val_idx).reset_index(drop=True)
    return train, val, test


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    return (df["income"] == LABELS[1]).astype(np.float32).to_numpy()
