# Project Status — chexlocalize-medical-imaging

_Last updated: 2026-09-14_

This file tracks the implementation status of each component. It is the single
source of truth for "what actually exists" versus "what is planned." No results,
metrics, or Dice scores are recorded until they are genuinely produced.

## Status overview

| Component        | Status                |
| ---------------- | --------------------- |
| Architecture     | ✅ Initialized         |
| Dataset          | 🟡 TOOLING READY — awaiting gated access (SAS URL) |
| Preprocessing    | ⛔ NOT YET IMPLEMENTED |
| nnU-Net          | ⛔ NOT YET IMPLEMENTED |
| SAM 2            | ⛔ NOT YET IMPLEMENTED |
| MedSAM           | ⛔ NOT YET IMPLEMENTED |
| NVIDIA method    | ⛔ NOT YET IMPLEMENTED |
| Evaluation       | ⛔ NOT YET IMPLEMENTED |
| Frontend         | ⛔ NOT YET IMPLEMENTED |

## Notes

- **Architecture: initialized** — Directory structure, central configuration
  system (`configs/project.yaml` + `configs/loader.py`), `.gitignore`,
  `requirements.txt` (infrastructure only), and a project-validation script are
  in place. No models, datasets, or results yet.
- **Dataset: TOOLING READY, awaiting access** — CheXlocalize is GATED. The
  reproducible downloader (`scripts/download_chexlocalize.sh`), read-only
  inspector (`scripts/inspect_dataset.py`), and docs (`docs/dataset.md`) are in
  place. Download is blocked only on the user completing the Stanford AIMI
  registration/agreement and providing the private SAS URL via `.env`
  (`CHEXLOCALIZE_SAS_URL`). Nothing is downloaded yet. Its real structure must
  be inspected before any split/train/eval strategy is chosen. The official
  validation/test images + annotations are a localization **benchmark** and must
  not be treated as ordinary training data. **Note: CheXlocalize contains NO
  dedicated segmentation training set** — see `docs/dataset.md`.
- **DEMO vs FULL** — Development modes are supported via `PROJECT_MODE`. DEMO
  and FULL outputs are written to separate subdirectories and must never be
  mixed.

## Definition of "fine-tuned"

A model is labeled **fine-tuned only if its weights are actually updated using
training data.** Pretrained/zero-shot inference is **never** called fine-tuning.

## What is intentionally NOT done yet

- ⛔ Download CheXlocalize
- ⛔ Install SAM 2 / MedSAM / nnU-Net / NVIDIA method
- ⛔ Any training, inference, or predictions
- ⛔ Any placeholder/example Dice scores or fabricated results
