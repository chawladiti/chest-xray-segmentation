#!/usr/bin/env python3
"""Evaluate nnU-Net predictions with the project's EXISTING Dice/IoU evaluator.

Compares nnU-Net predicted label maps (PNG, values {0,1}) against the held-out
internal_val ground-truth masks produced by convert_to_nnunet.py. Uses the
validated evaluation.dice / evaluation.iou functions unchanged — no new metric
logic is introduced.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# make the project importable (evaluation package lives at repo root)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.dice import dice_coefficient, dice_percentage  # noqa: E402
from evaluation.iou import iou_score  # noqa: E402


def _load_binary(path: Path) -> np.ndarray:
    return (np.array(Image.open(path).convert("L")) > 0).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True, help="nnU-Net prediction dir (PNG per case)")
    ap.add_argument("--gt-dir", required=True, help="internal_val ground-truth dir (PNG per case)")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--csv", default=None, help="optional per-case CSV output path")
    args = ap.parse_args()

    gt_dir, pred_dir = Path(args.gt_dir), Path(args.pred_dir)
    rows, dices, ious = [], [], []
    for gt_path in sorted(gt_dir.glob("*.png")):
        case = gt_path.stem
        pred_path = pred_dir / f"{case}.png"
        if not pred_path.exists():
            rows.append({"case": case, "status": "MISSING_PREDICTION"})
            continue
        gt = _load_binary(gt_path)
        pred = _load_binary(pred_path)
        d = dice_coefficient(pred, gt)
        i = iou_score(pred, gt)
        dices.append(d); ious.append(i)
        rows.append({"case": case, "dice": round(d, 6), "iou": round(i, 6),
                     "gt_pos": int(gt.sum()), "pred_pos": int(pred.sum())})

    summary = {
        "num_cases": len([r for r in rows if "dice" in r]),
        "num_missing": len([r for r in rows if r.get("status") == "MISSING_PREDICTION"]),
        "mean_dice": round(float(np.mean(dices)), 6) if dices else None,
        "mean_dice_pct": round(dice_percentage(float(np.mean(dices))), 4) if dices else None,
        "mean_iou": round(float(np.mean(ious)), 6) if ious else None,
    }
    print(json.dumps({"summary": summary, "per_case": rows}, indent=2))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "dice", "iou", "gt_pos", "pred_pos", "status"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_case": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
