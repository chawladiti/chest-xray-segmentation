"""Unit tests for Dice and IoU (evaluation.dice / evaluation.iou)."""

import numpy as np
import pytest

from evaluation.dice import dice_coefficient, dice_percentage
from evaluation.iou import iou_score


# --- perfect overlap ---------------------------------------------------------
def test_dice_perfect_overlap():
    a = np.array([[1, 1], [0, 0]])
    assert dice_coefficient(a, a) == 1.0
    assert iou_score(a, a) == 1.0


# --- no overlap --------------------------------------------------------------
def test_dice_no_overlap():
    pred = np.array([[1, 1], [0, 0]])
    gt = np.array([[0, 0], [1, 1]])
    assert dice_coefficient(pred, gt) == 0.0
    assert iou_score(pred, gt) == 0.0


# --- partial overlap ---------------------------------------------------------
def test_dice_partial_overlap():
    pred = np.array([1, 1, 0, 0])
    gt = np.array([1, 0, 0, 0])
    # inter=1, |P|=2, |G|=1 -> 2*1/3
    assert dice_coefficient(pred, gt) == pytest.approx(2 / 3)
    # IoU: inter=1, union=2 -> 0.5
    assert iou_score(pred, gt) == pytest.approx(0.5)


# --- empty prediction (gt non-empty) ----------------------------------------
def test_dice_empty_prediction():
    pred = np.zeros((3, 3))
    gt = np.ones((3, 3))
    assert dice_coefficient(pred, gt) == 0.0
    assert iou_score(pred, gt) == 0.0


# --- empty target (pred non-empty) ------------------------------------------
def test_dice_empty_target():
    pred = np.ones((3, 3))
    gt = np.zeros((3, 3))
    assert dice_coefficient(pred, gt) == 0.0
    assert iou_score(pred, gt) == 0.0


# --- both empty (documented convention = 1.0) --------------------------------
def test_dice_both_empty_default_convention():
    z = np.zeros((4, 4))
    assert dice_coefficient(z, z) == 1.0
    assert iou_score(z, z) == 1.0


def test_dice_both_empty_configurable():
    z = np.zeros((4, 4))
    assert dice_coefficient(z, z, both_empty_value=0.0) == 0.0
    assert iou_score(z, z, both_empty_value=0.0) == 0.0


# --- dimension mismatch raises ----------------------------------------------
def test_dice_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        dice_coefficient(np.zeros((2, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        iou_score(np.zeros((2, 2)), np.zeros((3, 3)))


# --- dice percentage ---------------------------------------------------------
def test_dice_percentage():
    assert dice_percentage(0.5) == 50.0
    assert dice_percentage(1.0) == 100.0
    assert dice_percentage(0.0) == 0.0


# --- binary vs 0/255 masks give same result ---------------------------------
def test_dice_handles_0_255_masks():
    pred = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    gt = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    # inter=1, |P|=2, |G|=1 -> 2/3
    assert dice_coefficient(pred, gt) == pytest.approx(2 / 3)
