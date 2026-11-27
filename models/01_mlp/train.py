"""Reproducible training + evaluation for the MLP on UCI Adult.

Usage:  python -m models.01_mlp.train --raw datasets/raw/uci_adult --out checkpoints
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.checksum import sha256_file

# Package names start with a digit ("01_mlp"), so they cannot be imported with
# dotted "from x import" syntax; importlib handles them.
_data = importlib.import_module("models.01_mlp.data")
_model = importlib.import_module("models.01_mlp.model")
AdultPreprocessor, encode_labels, load_splits = _data.AdultPreprocessor, _data.encode_labels, _data.load_splits
MLP = _model.MLP

SEED = 42
CONFIG = {"hidden": [128, 64], "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4,
          "batch_size": 256, "max_epochs": 30, "patience": 5}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(x)).numpy()


def metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y)),
    }


def train(raw_dir: str, out_dir: str) -> dict:
    set_seed(SEED)
    torch.set_num_threads(2)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_splits(raw_dir, seed=SEED)
    prep = AdultPreprocessor().fit(train_df)  # fitted on train only
    x_tr, y_tr = torch.from_numpy(prep.transform(train_df)), torch.from_numpy(encode_labels(train_df))
    x_va, y_va = prep.transform(val_df), encode_labels(val_df)
    x_te, y_te = prep.transform(test_df), encode_labels(test_df)

    model = MLP(x_tr.shape[1], tuple(CONFIG["hidden"]), CONFIG["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(x_tr, y_tr.float()), batch_size=CONFIG["batch_size"], shuffle=True)

    history, best_val, best_state, bad = [], float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(x_tr)
        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(torch.from_numpy(x_va)), torch.from_numpy(y_va)).item()
        sched.step(va_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": va_loss})
        print(f"epoch {epoch:02d} train_loss={train_loss:.4f} val_loss={va_loss:.4f}")
        if va_loss < best_val - 1e-4:
            best_val, bad = va_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CONFIG["patience"]:
                print(f"early stopping at epoch {epoch}")
                break
    train_seconds = time.time() - t0
    model.load_state_dict(best_state)

    # Threshold chosen on validation F1 only, then frozen for test.
    p_va = predict_proba(model, torch.from_numpy(x_va))
    thresholds = np.linspace(0.2, 0.8, 61)
    best_t = max(thresholds, key=lambda t: f1_score(y_va, (p_va >= t).astype(int), zero_division=0))
    p_te = predict_proba(model, torch.from_numpy(x_te))
    test_metrics = metrics(y_te, p_te, threshold=float(best_t))
    val_metrics = metrics(y_va, p_va, threshold=float(best_t))

    ckpt_path = out / "mlp_adult.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "architecture": {"name": "MLP", "in_features": int(x_tr.shape[1]),
                         "hidden": CONFIG["hidden"], "dropout": CONFIG["dropout"]},
        "preprocessing": prep.to_dict(),
        "threshold": float(best_t),
        "labels": {"0": "<=50K", "1": ">50K"},
        "seed": SEED,
        "config": CONFIG,
    }, ckpt_path)

    report = {
        "model": "01_mlp", "dataset": "UCI Adult (adult.data train/val split, adult.test held out)",
        "status": "trained_cpu_metrics_recorded", "config": CONFIG, "threshold": float(best_t),
        "epochs_run": len(history), "train_seconds": round(train_seconds, 1),
        "train_samples": int(len(train_df)), "val_samples": int(len(val_df)),
        "history": history, "val_metrics": val_metrics, "test_metrics": test_metrics,
        "checkpoint": {"file": ckpt_path.name, "bytes": ckpt_path.stat().st_size,
                       "sha256": sha256_file(ckpt_path)},
    }
    (out / "mlp_adult_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="datasets/raw/uci_adult")
    ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()
    r = train(a.raw, a.out)
    print(json.dumps({k: r[k] for k in ["epochs_run", "train_seconds", "threshold", "test_metrics"]}, indent=2))
