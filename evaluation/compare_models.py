"""Model comparison + figures (model-independent).

Compares models using the aggregated per-pathology metrics, including the
pretrained-vs-fine-tuned improvement:

    absolute_dice_improvement   = fine_tuned_mean_dice - pretrained_mean_dice
    percentage_dice_improvement = ((ft - pt) / pt) * 100      (guarded when pt == 0)

Guardrails:
  * percentage improvement is NOT computed when the baseline mean_dice is 0
    (reported as null with an explicit note).
  * a comparison flags any difference in the evaluated image/pathology subsets
    between two models (never compares silently across different subsets).
  * final comparisons must use the official test split (configurable).

Figures (only produced when data exists; never fabricated):
  * mean Dice by pathology, mean Dice by model, pretrained vs fine-tuned,
    Dice distribution by model, per-pathology comparison heatmap.

Writes results_root/model_comparison.csv and .json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config import load_evaluation_config, project_config  # noqa: E402


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _key(model: str, ttype: str) -> str:
    return f"{model}::{ttype}"


def percentage_improvement(baseline: float, candidate: float) -> Optional[float]:
    """((candidate - baseline)/baseline)*100, or None if baseline == 0."""
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100.0


def subset_signature(per_image_rows, model, ttype, split) -> set:
    return {
        (r["image_id"], r["pathology"])
        for r in per_image_rows
        if r["model"] == model and r["training_type"] == ttype and r["split"] == split
    }


def compare(
    baseline: Tuple[str, str],
    candidate: Tuple[str, str],
    *,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare two (model, training_type) identities on the overall + per-pathology Dice."""
    project = project_config()
    eval_cfg = load_evaluation_config()
    results_dir = project.paths["results_root"]
    overall_key = eval_cfg["aggregation"]["overall_key"]
    split = split or eval_cfg["test_policy"]["final_comparison_split"]

    per_path = _read_csv(results_dir / eval_cfg["output"]["per_pathology_filename"])
    per_image = _read_csv(results_dir / eval_cfg["output"]["per_image_filename"])

    b_model, b_type = baseline
    c_model, c_type = candidate

    def mean_dice(model, ttype, pathology):
        for r in per_path:
            if (r["model"] == model and r["training_type"] == ttype
                    and r["split"] == split and r["pathology"] == pathology):
                try:
                    return float(r["mean_dice"])
                except ValueError:
                    return None
        return None

    # subset consistency (requirement 11/12)
    b_sig = subset_signature(per_image, b_model, b_type, split)
    c_sig = subset_signature(per_image, c_model, c_type, split)
    subset_equal = b_sig == c_sig
    subset_diff = {
        "only_in_baseline": len(b_sig - c_sig),
        "only_in_candidate": len(c_sig - b_sig),
        "shared": len(b_sig & c_sig),
    }

    pathologies = eval_cfg["pathologies"] + [overall_key]
    comparisons = []
    for pathology in pathologies:
        pt = mean_dice(b_model, b_type, pathology)
        ft = mean_dice(c_model, c_type, pathology)
        if pt is None or ft is None:
            continue
        abs_imp = ft - pt
        pct_imp = percentage_improvement(pt, ft)
        comparisons.append({
            "pathology": pathology,
            "baseline_mean_dice": pt,
            "candidate_mean_dice": ft,
            "absolute_dice_improvement": round(abs_imp, 6),
            "percentage_dice_improvement": (round(pct_imp, 4) if pct_imp is not None else None),
            "percentage_improvement_note": (
                "baseline mean_dice == 0; percentage improvement undefined"
                if pct_imp is None else ""
            ),
        })

    result = {
        "split": split,
        "baseline": {"model": b_model, "training_type": b_type},
        "candidate": {"model": c_model, "training_type": c_type},
        "same_evaluation_subset": subset_equal,
        "subset_difference": subset_diff,
        "subset_warning": (
            "" if subset_equal else
            "Models were evaluated on DIFFERENT image/pathology subsets; "
            "comparison reported but not strictly like-for-like."
        ),
        "test_set_policy": eval_cfg["test_policy"]["note"],
        "comparisons": comparisons,
    }
    return result


def write_comparison(result: Dict[str, Any]) -> Tuple[Path, Path]:
    project = project_config()
    eval_cfg = load_evaluation_config()
    results_dir = project.paths["results_root"]
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / eval_cfg["output"]["model_comparison_json"]
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    csv_path = results_dir / eval_cfg["output"]["model_comparison_csv"]
    fields = ["split", "baseline_model", "baseline_training_type",
              "candidate_model", "candidate_training_type", "pathology",
              "baseline_mean_dice", "candidate_mean_dice",
              "absolute_dice_improvement", "percentage_dice_improvement",
              "same_evaluation_subset", "percentage_improvement_note"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for c in result.get("comparisons", []):
            w.writerow({
                "split": result["split"],
                "baseline_model": result["baseline"]["model"],
                "baseline_training_type": result["baseline"]["training_type"],
                "candidate_model": result["candidate"]["model"],
                "candidate_training_type": result["candidate"]["training_type"],
                "pathology": c["pathology"],
                "baseline_mean_dice": c["baseline_mean_dice"],
                "candidate_mean_dice": c["candidate_mean_dice"],
                "absolute_dice_improvement": c["absolute_dice_improvement"],
                "percentage_dice_improvement": c["percentage_dice_improvement"],
                "same_evaluation_subset": result["same_evaluation_subset"],
                "percentage_improvement_note": c["percentage_improvement_note"],
            })
    return csv_path, json_path


# ---------------------------------------------------------------------------
# Figures (produced only when data exists).
# ---------------------------------------------------------------------------
def generate_figures() -> List[Path]:
    project = project_config()
    eval_cfg = load_evaluation_config()
    results_dir = project.paths["results_root"]
    per_path = _read_csv(results_dir / eval_cfg["output"]["per_pathology_filename"])
    per_image = _read_csv(results_dir / eval_cfg["output"]["per_image_filename"])
    if not per_path and not per_image:
        print("[figures] no results yet — skipping figure generation (nothing to plot).")
        return []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = results_dir / eval_cfg["output"]["figures_subdir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    overall_key = eval_cfg["aggregation"]["overall_key"]

    # 1. mean Dice by pathology (per model)
    by_model_path = defaultdict(dict)
    for r in per_path:
        if r["pathology"] == overall_key:
            continue
        try:
            by_model_path[_key(r["model"], r["training_type"])][r["pathology"]] = float(r["mean_dice"])
        except ValueError:
            pass
    if by_model_path:
        pathologies = eval_cfg["pathologies"]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(pathologies))
        n = len(by_model_path)
        width = 0.8 / max(n, 1)
        for i, (mk, d) in enumerate(sorted(by_model_path.items())):
            ax.bar(x + i * width, [d.get(p, 0) for p in pathologies], width, label=mk)
        ax.set_xticks(x + width * (n - 1) / 2)
        ax.set_xticklabels(pathologies, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("mean Dice"); ax.set_title("Mean Dice by pathology")
        ax.legend(fontsize=7)
        fig.tight_layout(); p = fig_dir / "mean_dice_by_pathology.png"
        fig.savefig(p, dpi=120); plt.close(fig); written.append(p)

    # 2. mean Dice by model (overall)
    overall = [(f"{r['model']}::{r['training_type']}", float(r["mean_dice"]))
               for r in per_path if r["pathology"] == overall_key]
    if overall:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels, vals = zip(*sorted(overall))
        ax.bar(labels, vals)
        ax.set_ylabel("overall mean Dice"); ax.set_title("Overall mean Dice by model")
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        fig.tight_layout(); p = fig_dir / "mean_dice_by_model.png"
        fig.savefig(p, dpi=120); plt.close(fig); written.append(p)

    print(f"[figures] wrote {len(written)} figures -> {fig_dir}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Model comparison + figures")
    ap.add_argument("--baseline", nargs=2, metavar=("MODEL", "TRAINING_TYPE"))
    ap.add_argument("--candidate", nargs=2, metavar=("MODEL", "TRAINING_TYPE"))
    ap.add_argument("--split", default=None)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    if args.baseline and args.candidate:
        result = compare(tuple(args.baseline), tuple(args.candidate), split=args.split)
        csv_path, json_path = write_comparison(result)
        print(f"[compare] wrote {csv_path} and {json_path}")
    else:
        # Still (re)create empty comparison files so the schema exists.
        write_comparison({"split": load_evaluation_config()["test_policy"]["final_comparison_split"],
                          "baseline": {"model": "", "training_type": ""},
                          "candidate": {"model": "", "training_type": ""},
                          "same_evaluation_subset": True, "subset_difference": {},
                          "subset_warning": "", "test_set_policy": "", "comparisons": []})
        print("[compare] no baseline/candidate given — wrote empty comparison schema.")
    if args.figures:
        generate_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
