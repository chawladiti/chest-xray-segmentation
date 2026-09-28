# Model-independent evaluation

Every model (nnU-Net, SAM 2, MedSAM, NVIDIA method) is scored with this **same**
pipeline on the **same** official CheXlocalize test set. This is a **10-pathology
multi-label** problem: each pathology is an independent **binary** channel. There
is **no** mutually-exclusive multiclass label map.

Code: `evaluation/{dice,iou,metrics,evaluate,aggregate_results,compare_models}.py`.
Config: [`configs/evaluation.yaml`](../configs/evaluation.yaml).

## Pathology set (canonical — confirmed)

The 10 evaluated tasks are the official CheXlocalize localization/segmentation
tasks, in this canonical order (identical in `configs/preprocessing.yaml` and
`configs/evaluation.yaml`; enforced by `tests/test_pathology_consistency.py`):

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Enlarged Cardiomediastinum
6. Lung Lesion
7. **Airspace Opacity** (canonical segmentation name; CheXpert CSV alias = "Lung Opacity")
8. Pleural Effusion
9. Pneumothorax
10. **Support Devices**

> **Confirmed decision:** **Support Devices** is one of the 10 tasks and is kept.
> **Pneumonia is NOT** a CheXlocalize localization task (it has no segmentation
> masks) and is intentionally excluded. "Airspace Opacity" remains the canonical
> name; "Lung Opacity" is used **only** as the CheXpert CSV alias. No files,
> masks, or manifest values are renamed.

## Dice

    Dice = 2 * |P ∩ G| / (|P| + |G|)

Inputs are binarized first (probability maps thresholded at
`metric.threshold`, default 0.5; 0/255 masks scaled). `dice_percentage = Dice × 100`.

## IoU

    IoU = |P ∩ G| / |P ∪ G|

## Empty-mask convention (explicit)

We adopt, and document, the following (configurable) convention:

| Prediction | Ground truth | Dice / IoU |
| --- | --- | --- |
| empty | empty | **1.0** (perfect agreement / correct true-negative) |
| non-empty | empty | 0.0 |
| empty | non-empty | 0.0 |

The both-empty value is `metric.both_empty_dice` / `both_empty_iou` (default 1.0).
This matters for `VALID_NEGATIVE` pairs: a model that correctly predicts nothing
scores 1.0; a false positive scores 0.0.

## Target-status (IGNORE) handling

Per (image, pathology) `target_status` drives evaluation:

- **VALID_POSITIVE** — score prediction against the decoded GT mask.
- **VALID_NEGATIVE** — score prediction against an empty GT (all-zero).
- **IGNORE** — positive-but-unlocalized (or uncertain/blank). **Excluded** from
  supervised aggregation by default (`evaluate_pair` returns `None`). It can be
  included only for explicit analyses via `include_ignore=True`.

## Robustness (handled cases)

`evaluation.metrics.evaluate_pair` handles: empty prediction, empty ground truth,
both empty, **missing prediction** (treated as empty, flagged
`missing_prediction`), **missing target** (`evaluated=False`, NaN, flagged),
**dimension mismatch** (raises `ValueError`, or nearest-neighbour aligns the
prediction when `align_prediction=True`), thresholded probability maps, binary
masks, and **multi-channel** stacks (`evaluate_multichannel`, one binary channel
per pathology).

## Per-pathology aggregation

`aggregate_results.py` groups per-image rows by
`(model, training_type, split, pathology)` and reports `num_evaluated,
mean_dice, median_dice, std_dice, mean_iou, dice_percentage`. An overall
**macro-average across pathologies** is added per `(model, training_type, split)`
as `pathology == "__overall__"`, while per-pathology rows are retained.

## Model comparison & percentage improvement

`compare_models.py` compares two `(model, training_type)` identities:

    absolute_dice_improvement   = candidate_mean_dice - baseline_mean_dice
    percentage_dice_improvement = ((candidate - baseline) / baseline) × 100

- **Zero-baseline guard:** if `baseline_mean_dice == 0`, percentage improvement
  is reported as `null` with an explicit note (never divide by zero).
- **Same-subset rule:** the evaluated `(image_id, pathology)` sets of the two
  models are compared; any difference is reported (`same_evaluation_subset`,
  `subset_difference`) and never compared silently.
- **Terminology:** a model is compared as `fine_tuned` only when its weights were
  actually updated (`weights_updated = true`). Pretrained/zero-shot runs use
  `training_type = pretrained_inference` and are **never** called fine-tuning.

## Test-set policy

- All final comparisons use **only** the official CheXlocalize **test** split
  (`test_policy.final_comparison_split`).
- The official test annotations are **never** used for training or model
  selection (enforced upstream by the patient-level split; test has no training
  targets).

## Figures

`compare_models.generate_figures()` produces (only when results exist — never
fabricated): mean Dice by pathology, mean Dice by model, pretrained vs
fine-tuned, Dice distribution by model, and an optional per-pathology heatmap,
saved under `results/<mode>/figures/`.

## Outputs

Written under the mode-isolated `results_root` (`results/<demo|full>/`):
`per_image_metrics.csv`, `per_pathology_metrics.csv`, `model_comparison.csv`,
`model_comparison.json`. No results are fabricated; with no predictions present,
these are header-only schema files.
