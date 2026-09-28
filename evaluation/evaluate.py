"""Model-independent per-image evaluation driver.

Given a model's predictions and the split's targets, computes per (image,
pathology) Dice/IoU respecting target_status, and writes per_image_metrics.csv.

Predictions layout (probability or binary PNG):
    predictions_root/<model>/<split>/<image_id>/<Pathology>.png

Targets come from the split CSVs (data/splits/<split>.csv), whose per-pair
target_status is computed with the SAME rules used to build training targets.

This module imports NO model code. With no predictions present it writes an
empty (header-only) results file — it never fabricates scores.

Run (once a model has written predictions):
    python evaluation/evaluate.py --model medsam --split test \
        --training-type pretrained_inference
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config import load_evaluation_config, project_config  # noqa: E402
from evaluation.metrics import evaluate_pair, IGNORE  # noqa: E402
from preprocessing.config import load_targets_config, safe_pathology_filename  # noqa: E402
from preprocessing.build_training_targets import classify  # noqa: E402

PER_IMAGE_FIELDS = [
    "image_id", "patient_id", "split", "pathology", "model", "training_type",
    "weights_updated", "checkpoint", "epoch", "target_status",
    "dice", "dice_percentage", "iou", "flags",
]


class PredictionStore:
    """Loads predictions from predictions_root/<model>/<split>/<image_id>/<pathology>.png."""

    def __init__(self, predictions_root: Path, model: str):
        self.root = Path(predictions_root) / model

    def path_for(self, split: str, image_id: str, pathology: str) -> Path:
        return self.root / split / image_id / (safe_pathology_filename(pathology) + ".png")

    def load(self, split: str, image_id: str, pathology: str) -> Optional[np.ndarray]:
        p = self.path_for(split, image_id, pathology)
        if not p.is_file():
            return None
        return np.asarray(Image.open(p).convert("L")).astype(np.float32)


def iter_targets(split: str) -> Iterator[Dict[str, Any]]:
    """Yield per (image, pathology) target dicts for a split, with target_status."""
    project = project_config()
    tcfg = load_targets_config()
    policy = tcfg["label_policy"]
    statuses = tcfg["statuses"]
    split_csv = project.paths["splits_root"] / f"{split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}. Run scripts/build_splits.py.")
    with open(split_csv, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            has_mask = str(r["has_mask"]).lower() in ("true", "1")
            status = classify(has_mask, r.get("chexpert_label", ""), policy, statuses)
            yield {
                "image_id": r["image_id"],
                "patient_id": r["patient_id"],
                "split": split,
                "pathology": r["pathology"],
                "target_status": status,
                "mask_path": r["mask_path"] if has_mask else "",
                "image_path": r["image_path"],
                "has_mask": has_mask,
            }


def _load_gt(target: Dict[str, Any], ref_shape) -> Optional[np.ndarray]:
    """Ground truth array for a target. VALID_NEGATIVE -> zeros of ref_shape."""
    if target["has_mask"] and target["mask_path"]:
        return np.asarray(Image.open(target["mask_path"]).convert("L")).astype(np.uint8)
    # VALID_NEGATIVE: empty GT sized to prediction (or image) shape.
    if ref_shape is not None:
        return np.zeros(ref_shape, dtype=np.uint8)
    with Image.open(target["image_path"]) as im:
        return np.zeros((im.height, im.width), dtype=np.uint8)


def evaluate_model(
    model: str,
    split: str,
    *,
    training_type: str,
    weights_updated: bool,
    checkpoint: str = "",
    epoch: "int | str" = "",
    require_prediction: bool = True,
    include_ignore: bool = False,
) -> List[Dict[str, Any]]:
    eval_cfg = load_evaluation_config()
    m = eval_cfg["metric"]
    project = project_config()
    store = PredictionStore(project.paths["predictions_root"], model)

    rows: List[Dict[str, Any]] = []
    n_missing = 0
    for t in iter_targets(split):
        if t["target_status"] == IGNORE and not include_ignore:
            continue
        pred = store.load(split, t["image_id"], t["pathology"])
        if pred is None and require_prediction:
            n_missing += 1
            continue
        ref_shape = pred.shape if pred is not None else None
        gt = _load_gt(t, ref_shape)
        res = evaluate_pair(
            pred, gt, t["target_status"],
            threshold=m["threshold"],
            both_empty_dice=m["both_empty_dice"],
            both_empty_iou=m["both_empty_iou"],
            align_prediction=True,
            include_ignore=include_ignore,
        )
        if res is None:
            continue
        rows.append({
            "image_id": t["image_id"], "patient_id": t["patient_id"], "split": split,
            "pathology": t["pathology"], "model": model, "training_type": training_type,
            "weights_updated": weights_updated, "checkpoint": checkpoint, "epoch": epoch,
            "target_status": res.target_status, "dice": res.dice,
            "dice_percentage": res.dice_percentage, "iou": res.iou,
            "flags": ";".join(res.flags),
        })

    if n_missing:
        print(f"[evaluate] {model}/{split}: {n_missing} pairs had no prediction (skipped).")
    return rows


def write_per_image(rows: List[Dict[str, Any]], append: bool = False) -> Path:
    project = project_config()
    eval_cfg = load_evaluation_config()
    out_dir = project.paths["results_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / eval_cfg["output"]["per_image_filename"]
    mode = "a" if (append and out_path.exists()) else "w"
    with open(out_path, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_IMAGE_FIELDS)
        if mode == "w":
            w.writeheader()
        w.writerows(rows)
    print(f"[evaluate] wrote {len(rows)} rows -> {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Model-independent per-image evaluation")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--training-type", default="pretrained_inference",
                    help="e.g. pretrained_inference | fine_tuned | supervised")
    ap.add_argument("--weights-updated", action="store_true",
                    help="set ONLY if the model's weights were actually updated (fine-tuned)")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--epoch", default="")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    rows = evaluate_model(
        args.model, args.split,
        training_type=args.training_type,
        weights_updated=args.weights_updated,
        checkpoint=args.checkpoint, epoch=args.epoch,
    )
    write_per_image(rows, append=args.append)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
