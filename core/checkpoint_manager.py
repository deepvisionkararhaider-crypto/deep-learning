"""Safe checkpoint loading for the public inference app.

Rules enforced here:
* Only checkpoints listed in config/checkpoints.yaml with status "available" can load.
* SHA-256 of the file must match the registry before it is deserialised.
* torch.load(weights_only=True) is always used: no arbitrary pickled code runs.
* Nothing is downloaded or trained here for models absent from the registry.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from core.checksum import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config" / "checkpoints.yaml"
CKPT_DIR = ROOT / "checkpoints"

NOT_AVAILABLE_MSG = (
    "This model is not currently available for inference. Please check back after "
    "its training and checkpoint publication are completed."
)


class CheckpointError(RuntimeError):
    """Raised for any checkpoint that cannot be safely loaded."""


@dataclass(frozen=True)
class CheckpointEntry:
    model_id: str
    status: str
    file: str | None
    sha256: str | None
    notes: str


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, CheckpointEntry]:
    data = yaml.safe_load(path.read_text()) or {}
    entries = {}
    for model_id, item in data.get("models", {}).items():
        entries[model_id] = CheckpointEntry(
            model_id=model_id,
            status=item.get("status", "not_trained"),
            file=item.get("file"),
            sha256=item.get("sha256"),
            notes=item.get("notes", ""),
        )
    return entries


def get_entry(model_id: str) -> CheckpointEntry:
    entries = load_registry()
    if model_id not in entries:
        raise CheckpointError(f"Unknown model '{model_id}'. " + NOT_AVAILABLE_MSG)
    return entries[model_id]


def verify_checkpoint(model_id: str, ckpt_dir: Path = CKPT_DIR) -> Path:
    """Return the verified checkpoint path or raise CheckpointError."""
    entry = get_entry(model_id)
    if entry.status != "available" or not entry.file or not entry.sha256:
        raise CheckpointError(NOT_AVAILABLE_MSG)
    path = ckpt_dir / entry.file
    if not path.is_file():
        raise CheckpointError(f"Checkpoint file missing: {entry.file}. " + NOT_AVAILABLE_MSG)
    actual = sha256_file(path)
    if actual != entry.sha256:
        raise CheckpointError(
            f"SHA-256 mismatch for {entry.file}: expected {entry.sha256}, got {actual}. "
            "The file is corrupted or was replaced; it will not be loaded."
        )
    return path


@functools.lru_cache(maxsize=8)
def load_checkpoint(model_id: str, ckpt_dir: str = str(CKPT_DIR)) -> dict[str, Any]:
    """Verify then safely deserialise a checkpoint (cached per process)."""
    path = verify_checkpoint(model_id, Path(ckpt_dir))
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pickle/unsafe content, truncated file, etc.
        raise CheckpointError(f"Could not safely deserialise {path.name}: {exc}") from exc
