# CheXlocalize dataset verification

_Read-only verification. No files were downloaded, moved, or modified. No secrets/SAS URLs are read or displayed._

## 1. Location & size
- **Dataset root:** `/Users/aditi/Projects/chexlocalize-medical-imaging/data/raw/chexlocalize`
- **Total disk usage:** 5.3 GB (5,728,033,748 bytes)

## 2. Required directories / files

| Path | Exists |
| --- | --- |
| `CheXpert/` | ✅ |
| `CheXpert/val/` | ✅ |
| `CheXpert/test/` | ✅ |
| `CheXpert/val_labels.csv` | ✅ |
| `CheXpert/test_labels.csv` | ✅ |
| `CheXlocalize/` | ✅ |
| `CheXlocalize/gt_segmentations_val.json` | ✅ |
| `CheXlocalize/gt_segmentations_test.json` | ✅ |
| `CheXlocalize/gt_annotations_val.json` | ✅ |
| `CheXlocalize/gt_annotations_test.json` | ✅ |

## 3. Image counts
- **CheXpert/val/**: 234 images (expected 234)
- **CheXpert/test/**: 668 images (expected 668)

## 4. Annotation JSON file sizes

| File | Size |
| --- | --- |
| `gt_segmentations_val.json` | 1.7 MB |
| `gt_segmentations_test.json` | 4.3 MB |
| `gt_annotations_val.json` | 2.8 MB |
| `gt_annotations_test.json` | 7.6 MB |

## 5. JSON contents

### gt_segmentations_val.json
- CXR IDs: **187**
- Pathologies (10): ['Airspace Opacity', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Lung Lesion', 'Pleural Effusion', 'Pneumothorax', 'Support Devices']
- Example CXR ID: `patient64622_study1_view1_frontal`
- Example stored mask size [h, w]: [2320, 2828]
- Non-empty (id, pathology) mask pairs: **643**
- CXR IDs with ≥1 non-empty mask: **187**

### gt_segmentations_test.json
- CXR IDs: **499**
- Pathologies (10): ['Airspace Opacity', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Lung Lesion', 'Pleural Effusion', 'Pneumothorax', 'Support Devices']
- Example CXR ID: `patient64744_study1_view1_frontal`
- Example stored mask size [h, w]: [2021, 2021]
- Non-empty (id, pathology) mask pairs: **1534**
- CXR IDs with ≥1 non-empty mask: **499**

### gt_annotations_val.json
- CXR IDs: **187**
- Pathologies (10): ['Airspace Opacity', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Lung Lesion', 'Pleural Effusion', 'Pneumothorax', 'Support Devices']
- Example CXR ID: `patient64622_study1_view1_frontal`
- Example stored img_size: [2320, 2828]
- CXR IDs with ≥1 non-empty polygon: **187**

### gt_annotations_test.json
- CXR IDs: **668**
- Pathologies (10): ['Airspace Opacity', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Lung Lesion', 'Pleural Effusion', 'Pneumothorax', 'Support Devices']
- Example CXR ID: `patient64744_study1_view1_frontal`
- Example stored img_size: [2021, 2021]
- CXR IDs with ≥1 non-empty polygon: **499**

## 6. CXR-ID ↔ image-file matching

### gt_segmentations_val vs val images
- Referenced CXR IDs: 187
- Matched to an image file: 187
- Orphaned (no matching image): 0

### gt_segmentations_test vs test images
- Referenced CXR IDs: 499
- Matched to an image file: 499
- Orphaned (no matching image): 0

## 7. Coverage (images without a GT segmentation entry)
- val images with no seg entry: 47 of 234 (expected: negatives have no mask)
- test images with no seg entry: 169 of 668

## 8. Asset inventory
- Raw CXR images: ✅ present (902 total)
- Ground-truth segmentations (val+test): ✅ present
- Grad-CAM maps (val, .pkl): ✅ present (2340 files)
- Other benchmark files:
  - `hb_segmentations_test.json`: ✅
  - `hb_annotations_test.json`: ✅
  - `hb_salient_pt_test.json`: ✅
  - `gradcam_segmentations_val.json`: ✅
  - `chexlocalize_tasks.json`: ✅

## 9. Warnings
- None. All checks within expected ranges.

## 10. PASS/FAIL summary

| Check | Result |
| --- | --- |
| A. Images downloaded | PASS |
| B. Ground-truth masks downloaded | PASS |
| C. Validation data usable | PASS |
| D. Test data usable | PASS |
| E. Image/annotation alignment | PASS |
| F. Dataset ready for preprocessing | PASS |
