# Mask overlap analysis (TASK 1)

Spatial-overlap study of CheXlocalize ground-truth pathology masks, used to decide the segmentation target representation. Read-only; no training.

- **Scope:** `all_positive` (images with ≥1 non-empty mask across val + test)
- **Images with ≥1 mask:** 686 ({'test': 499, 'val': 187})
- **Images with multiple pathology masks:** 526
- **Images with ≥1 overlapping pair:** 508
- **Overlapping pathology-mask pairs:** 3030 of 3481 pairs checked
- **% of multi-mask images with overlap:** 96.58%
- **% of all masked images with overlap:** 74.05%

## Most frequent overlapping class pairs

| Pathology A | Pathology B | # images |
| --- | --- | --- |
| Airspace Opacity | Enlarged Cardiomediastinum | 270 |
| Airspace Opacity | Atelectasis | 248 |
| Cardiomegaly | Enlarged Cardiomediastinum | 243 |
| Enlarged Cardiomediastinum | Support Devices | 216 |
| Airspace Opacity | Cardiomegaly | 174 |
| Atelectasis | Enlarged Cardiomediastinum | 162 |
| Airspace Opacity | Pleural Effusion | 151 |
| Airspace Opacity | Support Devices | 137 |
| Airspace Opacity | Edema | 125 |
| Cardiomegaly | Support Devices | 124 |
| Atelectasis | Pleural Effusion | 115 |
| Atelectasis | Cardiomegaly | 109 |
| Edema | Enlarged Cardiomediastinum | 108 |
| Enlarged Cardiomediastinum | Pleural Effusion | 103 |
| Atelectasis | Support Devices | 89 |

## Example overlaps

| Split | Image ID | Classes | Intersection px | IoU |
| --- | --- | --- | --- | --- |
| test | `patient64741_study1_view1_frontal` | Airspace Opacity + Cardiomegaly | 508151 | 0.5685 |
| test | `patient64741_study1_view1_frontal` | Airspace Opacity + Enlarged Cardiomediastinum | 510263 | 0.4025 |
| test | `patient64741_study1_view1_frontal` | Airspace Opacity + Support Devices | 146184 | 0.1091 |
| test | `patient64741_study1_view1_frontal` | Cardiomegaly + Enlarged Cardiomediastinum | 863872 | 0.6968 |
| test | `patient64741_study1_view1_frontal` | Cardiomegaly + Support Devices | 323891 | 0.2177 |
| test | `patient64741_study1_view1_frontal` | Enlarged Cardiomediastinum + Support Devices | 523559 | 0.3147 |
| test | `patient64742_study1_view1_frontal` | Airspace Opacity + Atelectasis | 119214 | 0.7763 |
| test | `patient64743_study1_view1_frontal` | Airspace Opacity + Atelectasis | 459279 | 0.7358 |

## Implication

Pathology masks **overlap spatially** (a single pixel can belong to more than one finding, e.g. Airspace Opacity co-located with Consolidation/Atelectasis/Edema). A single mutually-exclusive multi-class label map is therefore **invalid** — see `docs/segmentation_target_strategy.md`.

_Machine-readable report: `mask_overlap.json`._