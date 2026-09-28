"""Model-independent per-pair and multi-channel metric computation.

Wraps dice/iou with:
  * probability-map thresholding and binary-mask handling,
  * target_status logic (VALID_POSITIVE / VALID_NEGATIVE / IGNORE),
  * robust handling of empty / missing / mismatched inputs,
  * multi-channel (multi-label) evaluation.

This 10-pathology problem is MULTI-LABEL: each pathology is an independent binary
channel. Nothing is turned into a mutually-exclusive multiclass map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from evaluation.dice import dice_coefficient, dice_percentage
from evaluation.iou import iou_score

# Target-status constants (kept in sync with configs/targets.yaml).
VALID_POSITIVE = "VALID_POSITIVE"
VALID_NEGATIVE = "VALID_NEGATIVE"
IGNORE = "IGNORE"


def binarize(mask, threshold: float = 0.5) -> np.ndarray:
    """Binarize a prediction/target.

    * float / probability maps (max > 1 handled too): foreground = value >= threshold
      for values in [0, 1]; for 0-255 style, scale by 255 first if max > 1.
    * boolean / integer masks: foreground = value > 0.
    """
    arr = np.asarray(mask)
    if arr.dtype == bool:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        a = arr
        if a.size and float(np.nanmax(a)) > 1.0:  # 0-255 style probabilities
            a = a / 255.0
        return a >= threshold
    # integer masks (0/1 or 0/255)
    return arr > 0


@dataclass
class PairResult:
    dice: float
    dice_percentage: float
    iou: float
    target_status: str
    evaluated: bool                      # False => excluded from supervised aggregation
    flags: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dice": self.dice,
            "dice_percentage": self.dice_percentage,
            "iou": self.iou,
            "target_status": self.target_status,
            "evaluated": self.evaluated,
            "flags": ";".join(self.flags),
        }


def evaluate_pair(
    prediction,
    target,
    target_status: str,
    *,
    threshold: float = 0.5,
    both_empty_dice: float = 1.0,
    both_empty_iou: float = 1.0,
    align_prediction: bool = False,
    include_ignore: bool = False,
) -> Optional[PairResult]:
    """Evaluate one (image, pathology) pair, respecting target_status.

    Returns a PairResult, or None if the pair is IGNORE and include_ignore=False.

    Edge cases:
      * IGNORE               -> excluded (None) unless include_ignore.
      * missing target       -> cannot score; evaluated=False, dice/iou = NaN.
      * missing prediction   -> treated as an all-zero prediction; flagged.
      * empty pred / empty gt / both empty -> handled by dice/iou conventions.
      * dimension mismatch   -> raises ValueError unless align_prediction=True
                                (nearest-neighbour resize of prediction to target).
    """
    flags: List[str] = []

    if target_status == IGNORE and not include_ignore:
        return None

    # Missing target: nothing to score against.
    if target is None:
        flags.append("missing_target")
        return PairResult(float("nan"), float("nan"), float("nan"),
                          target_status, evaluated=False, flags=flags)

    gt = binarize(target, threshold)

    # Missing prediction -> treat as empty prediction (model produced nothing).
    if prediction is None:
        flags.append("missing_prediction")
        pred = np.zeros_like(gt, dtype=bool)
    else:
        pred = binarize(prediction, threshold)

    # Dimension handling.
    if pred.shape != gt.shape:
        if align_prediction:
            pred = _resize_nn(pred, gt.shape)
            flags.append("prediction_resized")
        else:
            raise ValueError(
                f"Dimension mismatch: prediction {pred.shape} vs target {gt.shape}"
            )

    if not pred.any():
        flags.append("empty_prediction")
    if not gt.any():
        flags.append("empty_ground_truth")

    d = dice_coefficient(pred, gt, both_empty_value=both_empty_dice)
    j = iou_score(pred, gt, both_empty_value=both_empty_iou)
    evaluated = not (target_status == IGNORE)  # IGNORE only reaches here if include_ignore
    return PairResult(d, dice_percentage(d), j, target_status, evaluated=evaluated, flags=flags)


def _resize_nn(mask_bool: np.ndarray, shape) -> np.ndarray:
    """Nearest-neighbour resize a boolean mask to `shape` without extra deps."""
    from PIL import Image
    h, w = shape
    im = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
    im = im.resize((w, h), Image.NEAREST)
    return np.asarray(im) > 127


def evaluate_multichannel(
    predictions: Sequence,
    targets: Sequence,
    pathologies: Sequence[str],
    target_statuses: Sequence[str],
    **kwargs,
) -> Dict[str, Optional[PairResult]]:
    """Evaluate a multi-label stack: one independent binary channel per pathology.

    `predictions`/`targets` are sequences (or arrays with leading channel axis) of
    length len(pathologies). Returns {pathology: PairResult|None}.
    """
    n = len(pathologies)
    if not (len(predictions) == len(targets) == len(target_statuses) == n):
        raise ValueError("predictions, targets, statuses, pathologies must align in length")
    out: Dict[str, Optional[PairResult]] = {}
    for i, pathology in enumerate(pathologies):
        out[pathology] = evaluate_pair(
            predictions[i], targets[i], target_statuses[i], **kwargs
        )
    return out


__all__ = [
    "VALID_POSITIVE", "VALID_NEGATIVE", "IGNORE",
    "binarize", "PairResult", "evaluate_pair", "evaluate_multichannel",
]
