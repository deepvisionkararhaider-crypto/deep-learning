"""Architecture shape contract and preprocessing leakage checks for the MLP."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
data = importlib.import_module("models.01_mlp.data")
model_mod = importlib.import_module("models.01_mlp.model")

RAW = ROOT / "datasets" / "raw" / "uci_adult"
needs_data = pytest.mark.skipif(not (RAW / "adult.data").exists(), reason="UCI Adult not downloaded")


def test_forward_shape_contract():
    m = model_mod.MLP(in_features=50, hidden=(16, 8), dropout=0.0).eval()
    out = m(torch.randn(7, 50))
    assert out.shape == (7,)


@needs_data
def test_split_sizes_and_validation_disjointness():
    tr, va, te = data.load_splits(RAW, seed=42)
    full = data._read(RAW / "adult.data", skip_header=False)
    # validation is carved out of adult.data without overlap with training rows
    assert len(tr) + len(va) == len(full)
    assert len(te) == len(data._read(RAW / "adult.test", skip_header=True))
    cols = data.COLUMNS[:-1]
    tr_keys = set(map(tuple, tr[cols].astype(str).values.tolist()))
    va_keys = set(map(tuple, va[cols].astype(str).values.tolist()))
    assert len(tr_keys & va_keys) <= 5  # only UCI's own duplicate rows may collide


@needs_data
def test_known_dataset_duplicate_overlap_is_small():
    """UCI Adult contains identical feature rows across files (a dataset property, not
    a split bug). We assert the measured overlap stays small and documented (<0.5% of test)."""
    tr, _, te = data.load_splits(RAW, seed=42)
    cols = data.COLUMNS[:-1]
    train_keys = set(map(tuple, tr[cols].astype(str).values.tolist()))  # built once
    test_keys = list(map(tuple, te[cols].astype(str).values.tolist()))
    overlap = sum(1 for k in test_keys if k in train_keys)
    assert overlap / len(te) < 0.005


@needs_data
def test_preprocessor_fitted_on_train_only():
    tr, _, te = data.load_splits(RAW, seed=42)
    p = data.AdultPreprocessor().fit(tr)
    assert p.means["age"] == pytest.approx(tr["age"].mean())
    x = p.transform(te)
    assert x.shape[1] == len(p.feature_names())
    assert np.isfinite(x).all()
