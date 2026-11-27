"""CPU inference for models whose checkpoints are verified and registered as available."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.checkpoint_manager import load_checkpoint  # noqa: E402

_adult_data = importlib.import_module("models.01_mlp.data")
_adult_model = importlib.import_module("models.01_mlp.model")


class AdultIncomeModel:
    """Input contract: one record with the 14 UCI Adult raw columns.
    Output contract: {'probability_gt_50k': float, 'label': '<=50K'|'>50K', 'threshold': float}.
    """

    model_id = "01_mlp"

    def __init__(self) -> None:
        ckpt = load_checkpoint(self.model_id)
        arch = ckpt["architecture"]
        self.prep = _adult_data.AdultPreprocessor.from_dict(ckpt["preprocessing"])
        self.threshold = float(ckpt["threshold"])
        self.labels = ckpt["labels"]
        self.model = _adult_model.MLP(arch["in_features"], tuple(arch["hidden"]), arch["dropout"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        torch.set_num_threads(1)

    @property
    def vocab(self) -> dict[str, list[str]]:
        return self.prep.vocab

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        expected = _adult_data.NUMERIC + _adult_data.CATEGORICAL
        missing = [c for c in expected if c not in record]
        if missing:
            raise ValueError(f"Missing fields: {missing}")
        df = pd.DataFrame([{c: record[c] for c in expected}])
        for c in _adult_data.NUMERIC:
            try:
                df[c] = float(df[c].iloc[0])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Field '{c}' must be numeric") from exc
        x = torch.from_numpy(self.prep.transform(df))
        with torch.no_grad():
            p = float(torch.sigmoid(self.model(x)).item())
        return {
            "probability_gt_50k": p,
            "label": self.labels["1"] if p >= self.threshold else self.labels["0"],
            "threshold": self.threshold,
        }


_REGISTRY_BUILDERS = {"01_mlp": AdultIncomeModel}


def get_model(model_id: str):
    """Lazily build and cache one model. Only registered, verified models can be built."""
    return _cached(model_id)


from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=4)
def _cached(model_id: str):
    if model_id not in _REGISTRY_BUILDERS:
        raise KeyError(f"No inference implementation registered for {model_id}")
    return _REGISTRY_BUILDERS[model_id]()
