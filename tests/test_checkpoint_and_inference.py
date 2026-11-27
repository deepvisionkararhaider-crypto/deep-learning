"""Tests for checkpoint safety, registry gating, the MLP contract, and CPU inference."""
import shutil
from pathlib import Path

import pytest
import torch

from core import checkpoint_manager as cm
from core.checksum import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "mlp_adult.pt"


def test_registry_loads_all_entries():
    entries = cm.load_registry()
    assert "01_mlp" in entries and entries["01_mlp"].status == "available"
    assert entries["02_cnn"].status == "not_trained"


def test_unregistered_or_untrained_model_is_refused():
    with pytest.raises(cm.CheckpointError, match="not currently available"):
        cm.verify_checkpoint("02_cnn")


def test_registry_checksum_matches_file():
    entry = cm.get_entry("01_mlp")
    assert sha256_file(CKPT) == entry.sha256


def test_tampered_checkpoint_is_rejected(tmp_path):
    copy = tmp_path / "mlp_adult.pt"
    shutil.copy(CKPT, copy)
    with open(copy, "ab") as f:
        f.write(b"tamper")
    with pytest.raises(cm.CheckpointError, match="SHA-256 mismatch"):
        cm.verify_checkpoint("01_mlp", ckpt_dir=tmp_path)


def test_missing_checkpoint_file_is_reported(tmp_path):
    with pytest.raises(cm.CheckpointError, match="missing"):
        cm.verify_checkpoint("01_mlp", ckpt_dir=tmp_path)


def test_pickle_payload_is_refused_by_weights_only(tmp_path):
    # A malicious pickle must not execute; weights_only=True rejects arbitrary objects.
    class Evil:
        def __reduce__(self):
            return (print, ("EXECUTED",))
    bad = tmp_path / "evil.pt"
    torch.save({"x": Evil()}, bad)
    with pytest.raises(Exception):
        torch.load(bad, map_location="cpu", weights_only=True)


def test_mlp_inference_contract_and_output():
    from core.inference_engine import get_model
    model = get_model("01_mlp")
    record = {
        "age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors",
        "education_num": 13, "marital_status": "Never-married", "occupation": "Adm-clerical",
        "relationship": "Not-in-family", "race": "White", "sex": "Male",
        "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40, "native_country": "United-States",
    }
    out = model.predict(record)
    assert set(out) == {"probability_gt_50k", "label", "threshold"}
    assert 0.0 <= out["probability_gt_50k"] <= 1.0
    assert out["label"] in {"<=50K", ">50K"}


def test_mlp_rejects_missing_or_non_numeric_fields():
    from core.inference_engine import get_model
    model = get_model("01_mlp")
    with pytest.raises(ValueError, match="Missing"):
        model.predict({"age": 30})
    base = {c: model.vocab[c][0] for c in model.vocab}
    base.update({"age": 30, "fnlwgt": 1, "education_num": 9, "capital_gain": 0,
                 "capital_loss": 0, "hours_per_week": 40})
    base["age"] = "abc"
    with pytest.raises(ValueError, match="numeric"):
        model.predict(base)


def test_unseen_category_is_handled_without_crashing():
    from core.inference_engine import get_model
    model = get_model("01_mlp")
    record = {c: model.vocab[c][0] for c in model.vocab}
    record.update({"age": 30, "fnlwgt": 1, "education_num": 9, "capital_gain": 0,
                   "capital_loss": 0, "hours_per_week": 40, "native_country": "Atlantis"})
    assert 0.0 <= model.predict(record)["probability_gt_50k"] <= 1.0
