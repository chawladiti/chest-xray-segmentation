"""TASK 3 & 4 — build model-ready segmentation training targets.

Reads the dev-pool split CSVs (train.csv, internal_val.csv) and assigns each
(image, pathology) pair a target_status:

  VALID_POSITIVE  — a non-empty CheXlocalize pixel mask exists  -> save that mask
  VALID_NEGATIVE  — CheXpert label is a confirmed negative      -> save all-zero mask
  IGNORE          — positive label but no localization (or       -> NO mask file
                     uncertain/blank); excluded from the loss

Never converts a positive-but-unlocalized pair into an all-zero (negative) mask.
The official TEST set is excluded (no training targets). Read-only inputs.

Outputs:
  processed/targets/<split>/<image_id>/<Pathology>.png   (POSITIVE + NEGATIVE)
  processed/manifests/training_targets.csv

Run:
    python preprocessing/build_training_targets.py
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.config import (  # noqa: E402
    PreprocessingContext,
    load_preprocessing_context,
    load_targets_config,
    safe_pathology_filename,
)

MANIFEST_FIELDS = [
    "image_id", "patient_id", "split", "pathology", "image_path",
    "mask_path", "target_status", "has_mask", "csv_alias", "chexpert_label",
]


def classify(has_mask: bool, label: str, policy: Dict[str, Any], statuses: Dict[str, str]) -> str:
    if has_mask:
        return statuses["valid_positive"]
    v = str(label).strip()
    if v in policy["negative_values"]:
        return statuses["valid_negative"]
    if v in policy["positive_values"]:
        return statuses["ignore"]
    if v in policy["uncertain_values"]:
        return policy["uncertain_status"]
    # blank / anything else
    return policy["blank_status"] if v == "" else statuses["ignore"]


def _read_split_rows(splits_dir: Path, filename: str) -> List[Dict[str, str]]:
    path = splits_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}. Run scripts/build_splits.py first.")
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_zero_mask(target_path: Path, size_wh, values) -> None:
    w, h = size_wh
    lo, _hi = values
    arr = np.full((h, w), lo, dtype=np.uint8)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(target_path)


def build(ctx: PreprocessingContext) -> List[Dict[str, Any]]:
    tcfg = load_targets_config()
    policy = tcfg["label_policy"]
    statuses = tcfg["statuses"]
    out = tcfg["output"]
    values = out["binary_values"]
    ext = ".png"

    targets_root = ctx.processed_root / out["targets_subdir"]
    targets_root.mkdir(parents=True, exist_ok=True)
    splits_dir = ctx.project.paths["splits_root"]

    rows_out: List[Dict[str, Any]] = []
    counts = {statuses["valid_positive"]: 0, statuses["valid_negative"]: 0, statuses["ignore"]: 0}
    dim_cache: Dict[str, tuple] = {}

    for split, filename in tcfg["input"]["split_csvs"].items():
        for r in _read_split_rows(splits_dir, filename):
            image_id = r["image_id"]
            pathology = r["pathology"]
            has_mask = str(r["has_mask"]).lower() in ("true", "1")
            label = r.get("chexpert_label", "")
            status = classify(has_mask, label, policy, statuses)
            counts[status] = counts.get(status, 0) + 1

            rel = Path(split) / image_id / (safe_pathology_filename(pathology) + ext)
            target_mask_path = targets_root / rel
            mask_path_str = ""

            if status == statuses["valid_positive"]:
                # Reuse the already-decoded ground-truth mask (copy into targets/).
                src = Path(r["mask_path"])
                target_mask_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target_mask_path)
                mask_path_str = str(target_mask_path)
            elif status == statuses["valid_negative"] and out["save_negative_zero_masks"]:
                # All-zero mask sized to the ORIGINAL image (dimensional consistency).
                if image_id not in dim_cache:
                    with Image.open(r["image_path"]) as im:
                        dim_cache[image_id] = (im.width, im.height)
                _write_zero_mask(target_mask_path, dim_cache[image_id], values)
                mask_path_str = str(target_mask_path)
            # IGNORE -> deliberately no mask file, mask_path stays ""

            rows_out.append({
                "image_id": image_id,
                "patient_id": r["patient_id"],
                "split": split,
                "pathology": pathology,
                "image_path": r["image_path"],
                "mask_path": mask_path_str,
                "target_status": status,
                "has_mask": has_mask,
                "csv_alias": r.get("csv_alias", ""),
                "chexpert_label": label,
            })

    manifest_path = ctx.manifests_dir / out["manifest_filename"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows_out)

    print("=" * 66)
    print("Build training targets")
    print("=" * 66)
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  total pairs: {len(rows_out)}")
    print(f"  targets dir: {targets_root}")
    print(f"  manifest   : {manifest_path}")
    return rows_out


def main() -> int:
    ctx = load_preprocessing_context()
    build(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
