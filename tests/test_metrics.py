"""Unit tests for evaluation.metrics (status handling, thresholding, multi-channel)."""

import numpy as np
import pytest

from evaluation.metrics import (
    VALID_POSITIVE,
    VALID_NEGATIVE,
    IGNORE,
    binarize,
    evaluate_pair,
    evaluate_multichannel,
)


# --- thresholding probability maps ------------------------------------------
def test_binarize_probability_map():
    prob = np.array([[0.2, 0.6], [0.9, 0.49]])
    b = binarize(prob, threshold=0.5)
    assert b.tolist() == [[False, True], [True, False]]


def test_binarize_0_255():
    arr = np.array([0, 128, 255], dtype=np.uint8)
    # scaled by 255 -> [0, .502, 1] >= 0.5 -> [F, T, T]
    assert binarize(arr, threshold=0.5).tolist() == [False, True, True]


def test_evaluate_pair_thresholds_prediction():
    prob = np.array([[0.9, 0.1], [0.1, 0.1]])
    gt = np.array([[1, 0], [0, 0]])
    res = evaluate_pair(prob, gt, VALID_POSITIVE, threshold=0.5)
    assert res.dice == 1.0
    assert res.dice_percentage == 100.0


# --- perfect / partial on VALID_POSITIVE ------------------------------------
def test_evaluate_pair_partial():
    pred = np.array([1, 1, 0, 0])
    gt = np.array([1, 0, 0, 0])
    res = evaluate_pair(pred, gt, VALID_POSITIVE)
    assert res.dice == pytest.approx(2 / 3)
    assert res.iou == pytest.approx(0.5)
    assert res.evaluated is True


# --- empty prediction / empty target / both empty ----------------------------
def test_evaluate_pair_empty_prediction_flagged():
    res = evaluate_pair(np.zeros((3, 3)), np.ones((3, 3)), VALID_POSITIVE)
    assert res.dice == 0.0
    assert "empty_prediction" in res.flags


def test_evaluate_pair_empty_target_valid_negative():
    # VALID_NEGATIVE with empty prediction and empty GT -> perfect (true negative)
    res = evaluate_pair(np.zeros((3, 3)), np.zeros((3, 3)), VALID_NEGATIVE)
    assert res.dice == 1.0
    assert "empty_ground_truth" in res.flags


def test_evaluate_pair_valid_negative_false_positive():
    # VALID_NEGATIVE but model predicts something -> Dice 0
    res = evaluate_pair(np.ones((3, 3)), np.zeros((3, 3)), VALID_NEGATIVE)
    assert res.dice == 0.0


# --- IGNORE target -----------------------------------------------------------
def test_evaluate_pair_ignore_excluded_by_default():
    res = evaluate_pair(np.ones((2, 2)), np.zeros((2, 2)), IGNORE)
    assert res is None


def test_evaluate_pair_ignore_included_when_requested():
    res = evaluate_pair(np.ones((2, 2)), np.ones((2, 2)), IGNORE, include_ignore=True)
    assert res is not None
    assert res.target_status == IGNORE


# --- missing prediction / missing target ------------------------------------
def test_evaluate_pair_missing_prediction():
    res = evaluate_pair(None, np.ones((3, 3)), VALID_POSITIVE)
    assert res.dice == 0.0
    assert "missing_prediction" in res.flags


def test_evaluate_pair_missing_target():
    res = evaluate_pair(np.ones((3, 3)), None, VALID_POSITIVE)
    assert res.evaluated is False
    assert "missing_target" in res.flags
    assert np.isnan(res.dice)


# --- dimension mismatch ------------------------------------------------------
def test_evaluate_pair_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_pair(np.zeros((2, 2)), np.zeros((3, 3)), VALID_POSITIVE)


def test_evaluate_pair_dimension_mismatch_align():
    pred = np.ones((2, 2))
    gt = np.ones((4, 4))
    res = evaluate_pair(pred, gt, VALID_POSITIVE, align_prediction=True)
    assert res.dice == 1.0
    assert "prediction_resized" in res.flags


# --- multi-channel evaluation ------------------------------------------------
def test_evaluate_multichannel():
    pathologies = ["A", "B", "C"]
    preds = [np.ones((2, 2)), np.zeros((2, 2)), np.ones((2, 2))]
    targets = [np.ones((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))]
    statuses = [VALID_POSITIVE, VALID_NEGATIVE, IGNORE]
    out = evaluate_multichannel(preds, targets, pathologies, statuses)
    assert out["A"].dice == 1.0            # perfect positive
    assert out["B"].dice == 1.0            # both empty -> true negative
    assert out["C"] is None                # IGNORE excluded


def test_evaluate_multichannel_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_multichannel([np.ones((2, 2))], [np.ones((2, 2))], ["A", "B"],
                              [VALID_POSITIVE, VALID_POSITIVE])
