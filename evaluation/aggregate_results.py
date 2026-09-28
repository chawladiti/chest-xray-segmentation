"""Aggregate per-image metrics into per-pathology and overall summaries.

Reads results_root/per_image_metrics.csv and writes
results_root/per_pathology_metrics.csv with, per (model, training_type, split,
pathology):
    num_evaluated, mean_dice, median_dice, std_dice, mean_iou, dice_percentage

Plus an overall macro-average row per (model, training_type, split) using
pathology == "__overall__" (mean of the per-pathology mean_dice), while the
per-pathology rows are retained.

Empty input -> header-only output (never fabricates).
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config import load_evaluation_config, project_config  # noqa: E402

PER_PATHOLOGY_FIELDS = [
    "model", "training_type", "split", "pathology",
    "num_evaluated", "mean_dice", "median_dice", "std_dice",
    "mean_iou", "dice_percentage",
]


def _read_per_image(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def aggregate(rows: List[Dict[str, str]], overall_key: str) -> List[Dict[str, Any]]:
    groups: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: {"dice": [], "iou": []})
    for r in rows:
        # Only evaluated (non-IGNORE) rows contribute; IGNORE never appears here.
        key = (r["model"], r["training_type"], r["split"], r["pathology"])
        try:
            groups[key]["dice"].append(float(r["dice"]))
            groups[key]["iou"].append(float(r["iou"]))
        except (ValueError, KeyError):
            continue

    out: List[Dict[str, Any]] = []
    per_model_split_pathology_mean: Dict[tuple, List[float]] = defaultdict(list)

    for (model, ttype, split, pathology), vals in sorted(groups.items()):
        dices, ious = vals["dice"], vals["iou"]
        if not dices:
            continue
        mean_dice = statistics.fmean(dices)
        row = {
            "model": model, "training_type": ttype, "split": split, "pathology": pathology,
            "num_evaluated": len(dices),
            "mean_dice": round(mean_dice, 6),
            "median_dice": round(statistics.median(dices), 6),
            "std_dice": round(statistics.pstdev(dices), 6) if len(dices) > 1 else 0.0,
            "mean_iou": round(statistics.fmean(ious), 6),
            "dice_percentage": round(mean_dice * 100.0, 4),
        }
        out.append(row)
        per_model_split_pathology_mean[(model, ttype, split)].append(mean_dice)

    # Overall macro-average across pathologies.
    for (model, ttype, split), means in sorted(per_model_split_pathology_mean.items()):
        macro = statistics.fmean(means)
        out.append({
            "model": model, "training_type": ttype, "split": split, "pathology": overall_key,
            "num_evaluated": len(means),  # number of pathologies aggregated
            "mean_dice": round(macro, 6),
            "median_dice": round(statistics.median(means), 6),
            "std_dice": round(statistics.pstdev(means), 6) if len(means) > 1 else 0.0,
            "mean_iou": "",
            "dice_percentage": round(macro * 100.0, 4),
        })
    return out


def main() -> int:
    project = project_config()
    eval_cfg = load_evaluation_config()
    results_dir = project.paths["results_root"]
    results_dir.mkdir(parents=True, exist_ok=True)

    per_image = results_dir / eval_cfg["output"]["per_image_filename"]
    rows = _read_per_image(per_image)
    agg = aggregate(rows, eval_cfg["aggregation"]["overall_key"])

    out_path = results_dir / eval_cfg["output"]["per_pathology_filename"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_PATHOLOGY_FIELDS)
        w.writeheader()
        w.writerows(agg)
    print(f"[aggregate] read {len(rows)} per-image rows; wrote {len(agg)} aggregate rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
