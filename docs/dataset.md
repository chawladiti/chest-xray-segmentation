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

## Expected structure (per the official repo)

> Confirm against `scripts/inspect_dataset.py` after download — do not assume.

```
data/raw/
├── CheXpert/
│   ├── val/                 # CheXpert VALIDATION CXR images (234 imgs, 200 patients)
│   ├── val_labels.csv       # validation ground-truth image-level labels
│   ├── test/                # CheXpert TEST CXR images (668 imgs, 500 patients)
│   └── test_labels.csv      # test ground-truth image-level labels
└── CheXlocalize/
    ├── gradcam_maps_val/               # val DenseNet121 + Grad-CAM heatmaps
    ├── gradcam_segmentations_val.json  # val Grad-CAM pixel-level segmentations
    ├── gt_annotations_val.json         # val ground-truth RAW radiologist annotations
    ├── gt_annotations_test.json        # test ground-truth RAW radiologist annotations
    ├── gt_segmentations_val.json       # val ground-truth PIXEL segmentations
    ├── gt_segmentations_test.json      # test ground-truth PIXEL segmentations
    ├── hb_annotations_test.json        # test HUMAN-BENCHMARK raw annotations
    ├── hb_salient_pt_test.json         # test human-benchmark most-salient points
    └── hb_segmentations_test.json      # test human-benchmark pixel segmentations
```

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

The concrete train/eval strategy will be chosen **after** `inspect_dataset.py`
confirms the real on-disk structure.
