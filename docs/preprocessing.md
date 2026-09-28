# CheXlocalize preprocessing

Converts the official CheXlocalize **ground-truth segmentation** JSON files
(COCO-RLE) into usable per-pathology binary masks, plus a manifest and QC
visualizations. The original images and JSON files are treated as **read-only**
and are never modified. No models are trained and no metrics are computed here.

## One command (reproducible)

```bash
python scripts/preprocess.py            # decode -> manifest -> validate -> visualize
python scripts/preprocess.py --no-viz   # skip visualizations
```

Individual stages (each runnable standalone):

```bash
python preprocessing/decode_masks.py
python preprocessing/build_manifest.py
python preprocessing/validate_masks.py
python preprocessing/visualize_samples.py
```

Configuration: [`configs/preprocessing.yaml`](../configs/preprocessing.yaml)
(base data paths come from [`configs/project.yaml`](../configs/project.yaml)).

## Source files (inputs, read-only)

Under `data/raw/chexlocalize/`:

- `CheXlocalize/gt_segmentations_val.json` — 187 CXR IDs (val)
- `CheXlocalize/gt_segmentations_test.json` — 499 CXR IDs (test)
- `CheXpert/val/`, `CheXpert/test/` — the corresponding CXR JPGs
- `CheXpert/val_labels.csv`, `CheXpert/test_labels.csv` — image-level labels

## RLE decoding

The GT segmentation JSON stores each mask as a **COCO-RLE** object
`{"size": [h, w], "counts": "<compressed-string>"}`. We decode with
`pycocotools.mask.decode` (the `counts` string is ASCII-encoded to bytes first),
yielding an `H×W` array of `{0,1}`. See `preprocessing/decode_masks.py::decode_rle`.

## Mask format & output structure

Masks are written as 8-bit grayscale **PNG** (`0` = background, `255` =
foreground; configurable to `.npy` with `{0,1}`). Layout under
`data/processed/` (the `processed/` tree is git-ignored):

```
data/processed/
├── images/                     # (empty by default; copy_images=false — originals referenced)
├── masks/
│   ├── val/<image_id>/<Pathology>.png
│   └── test/<image_id>/<Pathology>.png
├── manifests/
│   ├── manifest.csv            # one row per image_id x pathology
│   ├── dataset_summary.json    # counts per split / pathology
│   ├── validation_report.json  # integrity check results
│   └── _mask_index.csv         # intermediate (decode stage output)
└── visualizations/             # QC overlays (>=12 samples + grid)
```

Only **non-empty** masks are written to disk; empty (all-zero) masks are recorded
in the manifest with `has_mask=false` and an empty `mask_path`
(`save_empty_masks: false`). Pathology names with spaces are made filesystem-safe
(`Pleural Effusion` → `Pleural_Effusion.png`) while the manifest keeps the exact
original name.

### Manifest columns

Required: `image_id`, `split`, `pathology`, `image_path`, `mask_path`,
`image_width`, `image_height`, `has_mask`.
Extras (for validation/provenance): `mask_width`, `mask_height`, `mask_area`,
`rle_status`, `csv_alias`.

## Pathology mapping (naming discrepancy — nothing silently renamed)

The 10 localized pathologies are, **exactly as in the segmentation source of
truth** (`chexlocalize_tasks.json` and `gt_segmentations_*.json`):

Enlarged Cardiomediastinum · Cardiomegaly · Lung Lesion · **Airspace Opacity** ·
Edema · Consolidation · Atelectasis · Pneumothorax · Pleural Effusion ·
Support Devices

There is one naming discrepancy, verified against the downloaded files:

| Segmentation JSON / tasks.json | CheXpert CSV label files |
| --- | --- |
| **Airspace Opacity** | **Lung Opacity** |

These are the **same clinical class** (CheXpert originally used "Airspace
Opacity", later renamed to "Lung Opacity" in the public labeler). We keep
**"Airspace Opacity"** as the canonical mask/manifest name (matching the
segmentation data) and record the CSV alias in
`configs/preprocessing.yaml::pathology_csv_aliases` and in the manifest's
`csv_alias` column. No labels are renamed in place.

## Image / mask alignment

The RLE `size` is `[height, width]`. For every non-empty mask we read the true
JPG pixel size (via Pillow) and assert `image_width×image_height ==
mask_width×mask_height`. Verified result: **0 dimension mismatches** across all
2,177 non-empty masks. QC overlays (see `data/processed/visualizations/`)
visually confirm anatomically correct placement.

## Validation checks (`validate_masks.py`)

Reported to `validation_report.json`; a run passes iff all **critical** counts
are zero. Latest run:

| Check | Result |
| --- | --- |
| Duplicate (image_id, pathology) rows | 0 |
| Missing image files | 0 |
| Missing mask files (has_mask but no PNG) | 0 |
| Invalid RLE | 0 |
| Dimension mismatches | 0 |
| Naming issues | none |
| Empty masks (informational) | 4,683 |
| Images without a seg entry / negatives (informational) | val 47, test 169 |

## Notes / non-results

This stage produces **data artifacts only** — masks, a manifest, a summary, a
validation report, and QC figures. It performs no training, no inference, and no
metric computation. Dice and model comparisons are explicitly out of scope here.
