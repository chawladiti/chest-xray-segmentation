"""Dice coefficient for binary segmentation masks.

    Dice = 2 * |P ∩ G| / (|P| + |G|)

Conventions (see docs/evaluation.md):
  * Inputs are binarized: any value > 0 is foreground (threshold probability maps
    BEFORE calling, e.g. via evaluation.metrics.binarize).
  * BOTH-EMPTY: if |P| = |G| = 0, Dice is defined as ``both_empty_value``
    (default 1.0 — perfect agreement / correct true-negative).
  * One-empty: if exactly one of P, G is empty, Dice = 0.0.
  * Dimension mismatch raises ValueError (strict; align upstream if intended).
"""

from __future__ import annotations

import numpy as np


def dice_coefficient(prediction, ground_truth, *, both_empty_value: float = 1.0) -> float:
    pred = np.asarray(prediction)
    gt = np.asarray(ground_truth)
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: prediction {pred.shape} vs ground_truth {gt.shape}"
        )
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    p = int(pred_b.sum())
    g = int(gt_b.sum())
    denom = p + g
    if denom == 0:
        return float(both_empty_value)
    inter = int(np.logical_and(pred_b, gt_b).sum())
    return float(2.0 * inter / denom)


def dice_percentage(dice: float) -> float:
    """Dice x 100."""
    return float(dice) * 100.0


__all__ = ["dice_coefficient", "dice_percentage"]
