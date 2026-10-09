# RunPod dataset (pipeline subset)

This pod copy contains **only files required** for nnU-Net, SAM 2, MedSAM,
segmentation training, and Dice/IoU evaluation. Grad-CAM pickle maps are
**intentionally excluded** and are not an error.

Official CheXlocalize **test images are stored separately** under
`CheXpert/test/` and are **not used for training or model selection**.

## Sizes
- raw (`data/raw/chexlocalize`): 1.6 GB (1,765,054,485 bytes)
- processed (`data/processed`): 48.7 MB (51,053,651 bytes)
- splits (`data/splits`): 2.0 MB

## Required raw files
| Path | Present |
| --- | --- |
| `CheXpert/val/` | yes |
| `CheXpert/test/` | yes |
| `CheXpert/val_labels.csv` | yes |
| `CheXpert/test_labels.csv` | yes |
| `CheXlocalize/gt_segmentations_val.json` | yes |
| `CheXlocalize/gt_segmentations_test.json` | yes |
| `CheXlocalize/gt_annotations_val.json` | yes |
| `CheXlocalize/gt_annotations_test.json` | yes |
| `CheXlocalize/chexlocalize_tasks.json` | yes |

## Image / mask counts
- CheXpert/val JPGs: **234** (expected 234)
- CheXpert/test JPGs: **277** (668 expected for later benchmark; **not required to start training**)
- processed masks: **2177**
- processed training-target PNGs: **2339**

## Split pair counts
- train rows: 1870; VALID_POSITIVE: **506** (expected 506)
- internal_val rows: 470; VALID_POSITIVE: **137** (expected 137)
- test rows: 6680 (668 images × 10 pathologies; not used for training)
- unique train image_ids: 187

## Patient leakage
- train ∩ internal_val patients: 0 (must be 0)
- train ∩ test patients: 0 (must be 0)
- internal_val ∩ test patients: 0 (must be 0)

## Path resolution
- missing train image/mask files after remap: 0
- missing internal_val image/mask files after remap: 0
- missing train/internal_val training-target files after remap: 0
- missing test image files after remap: 383 (incomplete test copy is **not** a training blocker)

## DEMO subset check
- train image_ids beyond the 3-CXR DEMO set: 184
- not limited to DEMO: yes

## Grad-CAM (intentionally excluded)
- `gradcam_maps_val/` exists: True
- `.pkl` files under raw: **383** (expected 0 on this pod)
- `gradcam_segmentations_val.json` present: True (expected false)
- Missing Grad-CAM files are **not** an error for this project.

## PASS/FAIL (development / training readiness)
- **Overall: PASS**
- Train + internal_val data are complete. Training can begin **without** official test images.
- Official test JPGs complete: no (not a blocker)

