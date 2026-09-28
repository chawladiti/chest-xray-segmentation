# Project Status — chexlocalize-medical-imaging

_Last updated: 2026-09-14_

This file tracks the implementation status of each component. It is the single
source of truth for "what actually exists" versus "what is planned." No results,
metrics, or Dice scores are recorded until they are genuinely produced.

## Status overview

| Component        | Status                |
| ---------------- | --------------------- |
| Architecture     | ✅ Initialized         |
| Dataset          | ✅ DOWNLOADED & INSPECTED (5.3 GB; val 234 / test 668) |
| Preprocessing    | ✅ IMPLEMENTED (GT masks decoded; manifest + validation PASS) |
| Experimental split | ✅ CREATED (patient-level dev split; official test untouched; checks PASS) |
| Training targets | ✅ BUILT (VALID_POSITIVE 643 / VALID_NEGATIVE 1696 / IGNORE 1; multi-label; checks PASS) |
| nnU-Net          | ⛔ NOT YET IMPLEMENTED |
| SAM 2            | ⛔ NOT YET IMPLEMENTED |
| MedSAM           | ⛔ NOT YET IMPLEMENTED |
| NVIDIA method    | ⛔ NOT YET IMPLEMENTED |
| Evaluation       | ✅ FRAMEWORK IMPLEMENTED (Dice/IoU, per-pathology, comparison; 63 tests PASS) |
| Frontend         | ⛔ NOT YET IMPLEMENTED |

## Notes

- **Architecture: initialized** — Directory structure, central configuration
  system (`configs/project.yaml` + `configs/loader.py`), `.gitignore`,
  `requirements.txt` (infrastructure only), and a project-validation script are
  in place. No models, datasets, or results yet.
- **Dataset: DOWNLOADED & INSPECTED (2026-09-28)** — CheXlocalize pulled into
  `data/raw/chexlocalize/` (5.3 GB, 3,256 files, 0 failed) via
  `scripts/download_chexlocalize.sh`, and verified read-only with
  `scripts/inspect_dataset.py`. Confirmed: val = 234 images / 200 patients,
  test = 668 images / 500 patients. GT masks cover 187/234 val and 499/668 test
  studies (not every image has a mask). Naming caveat: CSV "Lung Opacity" ==
  JSON "Airspace Opacity". Full details + nuances in `docs/dataset.md`.
  **Still NO training split created** — the official test set is the localization
  **benchmark** and must not be trained on. **CheXlocalize contains NO dedicated
  segmentation training set**; train/eval strategy is the next decision.
  (Dataset files are git-ignored and never committed.)
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
