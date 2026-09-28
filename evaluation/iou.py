"""Intersection-over-Union (Jaccard) for binary segmentation masks.

    IoU = |P ∩ G| / |P ∪ G|

Conventions (see docs/evaluation.md):
  * Inputs are binarized (value > 0 is foreground).
  * BOTH-EMPTY: if the union is empty, IoU is defined as ``both_empty_value``
    (default 1.0 — perfect agreement).
  * Dimension mismatch raises ValueError.
"""

from __future__ import annotations

import numpy as np


def iou_score(prediction, ground_truth, *, both_empty_value: float = 1.0) -> float:
    pred = np.asarray(prediction)
    gt = np.asarray(ground_truth)
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: prediction {pred.shape} vs ground_truth {gt.shape}"
        )
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    union = int(np.logical_or(pred_b, gt_b).sum())
    if union == 0:
        return float(both_empty_value)
    inter = int(np.logical_and(pred_b, gt_b).sum())
    return float(inter / union)


__all__ = ["iou_score"]
