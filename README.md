# Deep Learning Laboratory

An educational and research laboratory for deep-learning architectures, with a public
CPU-only Streamlit inference app. **This is an in-progress project. Most of the planned
scope is not yet implemented or trained.** The status below is the source of truth.

> Educational and research prototype. Not a medical device. Must not be used for diagnosis,
> treatment decisions, or clinical decision-making.

## Current status (verified)

| Component | Status |
|---|---|
| 01 MLP on UCI Adult (income >50K) | ✅ Trained on CPU, checkpoint SHA-256 verified, served in app |
| Safe checkpoint manager (SHA-256 + `weights_only=True`) | ✅ Implemented and tested |
| Streamlit app shell + MLP inference page | ✅ Smoke-tested with Streamlit AppTest |
| Models 02–20 | ⬜ Not implemented, not trained |
| Medical AI (classification, U-Net, YOLO, Mask R-CNN) | ⬜ Not implemented (needs GPU + licensed datasets) |
| Smart Agriculture (25-class, segmentation, cotton detection) | ⬜ Not implemented (needs verified 25-class dataset) |
| Notebook archive (ZIP) | ⬜ Not produced |
| Public deployment | ⬜ Not deployed (awaiting choice of deploy path, see below) |

## 01 MLP — Results (held-out UCI Adult test set, n = 15,060)

Threshold chosen on validation F1 only, then frozen for test. Run: `models/01_mlp/train.py`
(seed 42, CPU, 12 epochs with early stopping, 16 s training time).

| Metric | Value |
|---|---|
| Accuracy | 0.836 |
| Precision (>50K) | 0.640 |
| Recall (>50K) | 0.760 |
| F1 (>50K) | 0.695 |
| ROC-AUC | 0.908 |

Full history and metrics: `checkpoints/mlp_adult_report.json`.

**Dataset:** UCI Adult, https://archive.ics.uci.edu/dataset/2/adult (CC BY 4.0).
`adult.data` is split into train and validation (10% stratified, seed 42). `adult.test` is
held out. Preprocessing statistics are fitted on training data only.
Known property: 19 of 15,060 test rows (0.13%) exactly match training feature rows, because
the UCI data contains duplicate records. This is documented and tested (< 0.5%).

## Repository layout

```
app.py                      Streamlit public app (CPU inference only, never trains/downloads)
core/checkpoint_manager.py  Registry gating, SHA-256 verification, safe torch.load
core/inference_engine.py    Per-model inference wrappers with explicit input/output contracts
core/checksum.py            SHA-256 helper
config/checkpoints.yaml     Checkpoint registry (only status: available is served)
models/01_mlp/              Data loader, preprocessing, architecture, training script
checkpoints/                mlp_adult.pt + training report (small; committed)
datasets/raw/               Downloaded datasets (git-ignored; fetch instructions below)
tests/                      pytest suite (13 tests)
```

## Inference safety rules (enforced in code)

- A model is served only if its registry status is `available`, its file exists, and its
  SHA-256 matches. Any other case returns the "not currently available" message.
- `torch.load(..., weights_only=True)` is always used. A test confirms a malicious pickle is refused.
- The public app never trains models and never downloads datasets.

## Reproduce

```bash
pip install -r requirements-dev.txt
# Download UCI Adult into datasets/raw/uci_adult (adult.zip from the UCI page, then unzip)
python3 models/01_mlp/train.py --raw datasets/raw/uci_adult --out checkpoints
python3 -m pytest -q tests
streamlit run app.py
```

## Test results

`python3 -m pytest -q tests` → 13 passed (checkpoint safety, tamper rejection, registry gating,
MLP contract, input validation, unseen-category handling, split integrity).

## Limitations

- Only 1 of 20 core architectures is implemented and trained.
- The MLP is trained on 1994 US census data. It is a demonstration, not a tool for decisions about individuals.
- Medical AI and agriculture modules are not started. No clinical validity or diagnostic accuracy is claimed.

## Next steps

1. Implement CNN (CIFAR-10) and ResNet/DenseNet (CIFAR-100) on CPU, with real metrics.
2. Add LSTM/GRU on a verified time-series dataset.
3. Obtain GPU compute for medical and agriculture models with verified dataset licences.
4. Publish checkpoints to Hugging Face and record their URLs in the registry.
