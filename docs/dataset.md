# CheXlocalize dataset

Radiologist-annotated chest X-ray **localization/segmentation** dataset from the
Stanford AIMI group. Annotations cover **10 pathologies**: Atelectasis,
Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Lung Lesion,
Lung Opacity, Pleural Effusion, Pneumothorax, Support Devices.

- Landing page: https://aimi.stanford.edu/datasets/chexlocalize
- Code / structure reference: https://github.com/rajpurkarlab/cheXlocalize
- Cite (DOI): https://doi.org/10.71718/hap9-kn94
- Paper: "Benchmarking saliency methods for chest X-ray interpretation",
  Nature Machine Intelligence, 2022 — https://doi.org/10.1038/s42256-022-00536-x

## Access (GATED — manual, one-time)

CheXlocalize requires accepting a data-use agreement. **These steps must be done
by you; they cannot be automated.**

1. Log in / create an account at Stanford AIMI:
   https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c
   (Use Google Chrome; the download button may not render in Firefox.)
2. Click the download link, fill out the registration form, and **accept the
   terms of agreement**.
3. You will be given a private **Azure Blob SAS URL**.
4. Store it locally as a secret (NEVER commit it):
   ```bash
   echo 'CHEXLOCALIZE_SAS_URL="https://<account>.blob.core.windows.net/<container>?<sas-token>"' >> .env
   ```
5. Download into `data/raw/`:
   ```bash
   bash scripts/download_chexlocalize.sh --dry-run   # sanity check
   bash scripts/download_chexlocalize.sh             # real download
   ```
   The script bootstraps a local `azcopy` if needed and pulls both folders.
6. Inspect what actually landed (no assumptions):
   ```bash
   python scripts/inspect_dataset.py
   ```

## Actual structure (VERIFIED on 2026-09-28 via `scripts/inspect_dataset.py`)

Downloaded size: **5.3 GB**, 3,256 files, 0 failed. Note the archive unpacks
under an extra lowercase `chexlocalize/` top-level folder:

```
data/raw/chexlocalize/
├── CheXpert/
│   ├── val/                 # 234 CXR images, 200 patients  (patient*/study*/view*.jpg)
│   ├── val_labels.csv       # 234 rows; image-level labels (14 CheXpert classes)
│   ├── test/                # 668 CXR images, 500 patients
│   └── test_labels.csv      # 668 rows; image-level labels
└── CheXlocalize/
    ├── chexlocalize_tasks.json         # task list (key: 'chexplanation_tasks')
    ├── gradcam_maps_val/               # 2,340 .pkl DenseNet121+Grad-CAM heatmaps (val)
    ├── gradcam_segmentations_val.json  # 234 studies — Grad-CAM pixel segmentations (val)
    ├── gt_annotations_val.json         # 187 studies — RAW radiologist annotations (val)
    ├── gt_annotations_test.json        # 668 studies — RAW radiologist annotations (test)
    ├── gt_segmentations_val.json       # 187 studies — GT PIXEL segmentations (val)
    ├── gt_segmentations_test.json      # 499 studies — GT PIXEL segmentations (test)
    ├── hb_annotations_test.json        # 502 studies — HUMAN-BENCHMARK raw annotations
    ├── hb_salient_pt_test.json         # 503 studies — human-benchmark salient points
    └── hb_segmentations_test.json      # 502 studies — human-benchmark pixel segmentations
```

### Verified facts & non-obvious nuances (these matter downstream)

- **Counts confirmed:** val = 234 images / 200 patients; test = 668 images /
  500 patients. Images are JPG under `patient*/study*/view*_frontal|lateral.jpg`.
- **Not every image has a mask.** GT segmentations exist for **187 / 234** val
  studies and **499 / 668** test studies — images with no positive localizable
  pathology simply have no segmentation entry. Do **not** assume 1 mask per image.
- **Pathology-name mismatch to reconcile in preprocessing:** the CSV labels use
  **"Lung Opacity"**, but the segmentation/annotation JSONs use
  **"Airspace Opacity"** for the same class. The 10 localized pathologies are:
  Enlarged Cardiomediastinum, Cardiomegaly, Lung Lesion, Airspace Opacity
  (= Lung Opacity), Edema, Consolidation, Atelectasis, Pneumothorax,
  Pleural Effusion, Support Devices. (CheXpert also has Pneumonia / No Finding,
  which are **not** localized.)
- **Two JSON representations per split:** `gt_annotations_*` = raw radiologist
  polygons plus `img_size`; `gt_segmentations_*` = encoded pixel-level masks.
  The exact mask encoding will be handled in preprocessing (not yet touched).
- **Grad-CAM** heatmaps (`.pkl`) and Grad-CAM segmentations are provided for the
  **val** split only — these are the saliency-method outputs from the paper.
- **Human benchmark (`hb_*`)** exists for **test only** — the radiologist
  reference used to compare *models* against; never a training target.

## ⚠️ Critical methodological / data-leakage notes

These constraints shape every downstream decision:

1. **There is NO segmentation training set.** CheXlocalize provides pixel-level
   ground-truth segmentations only for the CheXpert **val** (234) and **test**
   (668) sets. It is a *benchmark*, not a standard train/val/test corpus.
2. **The `test` split is the official localization benchmark.** Do **not** train
   on it, do **not** leak its annotations into training, and do **not**
   arbitrarily re-split it into train/test.
3. **Human-benchmark (`hb_*`) segmentations exist only for `test`.** They are the
   radiologist reference to compare *models* against — not training targets.
4. **Supervised models (e.g., nnU-Net) need training masks that this dataset
   does not directly provide.** Strategy options to evaluate later (decision
   deferred, nothing chosen yet):
   - Treat the `val` GT segmentations as a small supervised train set and report
     on the `test` benchmark (small-data regime; document limitations).
   - Use CheXpert *training* images (no seg masks) only for pretraining/other
     tasks — not as segmentation supervision.
   - Foundation-model **inference** (SAM 2 / MedSAM zero-shot) needs no training
     masks; fine-tuning would require a legitimately labeled train split.
5. **DEMO vs FULL isolation** still applies: any small debugging sample drawn
   from the data must live under the demo output tree and never mix with FULL.
6. **Terminology:** a model is "fine-tuned" only if its weights are actually
   updated on training data; zero-shot/pretrained inference is never called
   fine-tuning.

The concrete train/eval strategy will be chosen next, now that
`inspect_dataset.py` has confirmed the real on-disk structure (see verified
facts above). Nothing has been split, moved, or trained.
