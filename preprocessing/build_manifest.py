"""Build the preprocessing manifest CSV and dataset summary.

Reads the intermediate mask index produced by decode_masks.py, resolves each
CXR's original image path and true pixel dimensions, and writes:

  * processed/manifests/manifest.csv   (one row per image_id x pathology)
  * processed/manifests/dataset_summary.json

Manifest columns (required + a few extras for validation):
  image_id, split, pathology, image_path, mask_path,
  image_width, image_height, has_mask,
  mask_width, mask_height, mask_area, rle_status, csv_alias

Run standalone (auto-runs decode if the index is missing):
    python preprocessing/build_manifest.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from preprocessing.config import (
    PreprocessingContext,
    image_path_for,
    load_preprocessing_context,
)
from preprocessing import decode_masks

MANIFEST_FIELDS = [
    "image_id", "split", "pathology", "image_path", "mask_path",
    "image_width", "image_height", "has_mask",
    "mask_width", "mask_height", "mask_area", "rle_status", "csv_alias",
]


def _load_index(ctx: PreprocessingContext) -> List[Dict[str, Any]]:
    index_path = ctx.manifests_dir / ctx.prep["output"]["mask_index_filename"]
    if not index_path.exists():
        print("[manifest] mask index not found; running decode_masks first...")
        return decode_masks.decode_all(ctx)
    with open(index_path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _coerce(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize CSV-string values back to typed values."""
    def to_int(v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return ""
    has_mask = rec.get("has_mask")
    rec["has_mask"] = str(has_mask).lower() in ("true", "1")
    rec["mask_area"] = to_int(rec.get("mask_area"))
    rec["mask_width"] = to_int(rec.get("mask_width"))
    rec["mask_height"] = to_int(rec.get("mask_height"))
    return rec


def build_manifest(ctx: PreprocessingContext) -> List[Dict[str, Any]]:
    records = [_coerce(dict(r)) for r in _load_index(ctx)]
    aliases = ctx.prep.get("pathology_csv_aliases", {})

    # Cache image dimensions per (split, image_id) so we read each JPG once.
    dim_cache: Dict[tuple, tuple] = {}

    def image_dims(split: str, image_id: str, path: Path):
        key = (split, image_id)
        if key in dim_cache:
            return dim_cache[key]
        wh = ("", "")
        if path.is_file():
            try:
                with Image.open(path) as im:
                    wh = (im.width, im.height)
            except Exception:
                wh = ("", "")
        dim_cache[key] = wh
        return wh

    rows: List[Dict[str, Any]] = []
    for rec in records:
        split = rec["split"]
        image_id = rec["image_id"]
        img_path = image_path_for(ctx, split, image_id)
        iw, ih = image_dims(split, image_id, img_path)
        rows.append({
            "image_id": image_id,
            "split": split,
            "pathology": rec["pathology"],
            "image_path": str(img_path),
            "mask_path": rec.get("mask_path", ""),
            "image_width": iw,
            "image_height": ih,
            "has_mask": rec["has_mask"],
            "mask_width": rec.get("mask_width", ""),
            "mask_height": rec.get("mask_height", ""),
            "mask_area": rec.get("mask_area", 0),
            "rle_status": rec.get("rle_status", ""),
            "csv_alias": aliases.get(rec["pathology"], ""),
        })

    manifest_path = ctx.manifests_dir / ctx.prep["output"]["manifest_filename"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[manifest] wrote {manifest_path} ({len(rows)} rows)")

    _write_summary(ctx, rows)
    return rows


def _write_summary(ctx: PreprocessingContext, rows: List[Dict[str, Any]]) -> None:
    per_split = defaultdict(lambda: {"images": set(), "masks": 0})
    per_pathology = defaultdict(lambda: {"masks": 0})
    per_split_pathology: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in rows:
        per_split[r["split"]]["images"].add(r["image_id"])
        if r["has_mask"]:
            per_split[r["split"]]["masks"] += 1
            per_pathology[r["pathology"]]["masks"] += 1
            per_split_pathology[r["split"]][r["pathology"]] += 1

    summary = {
        "n_manifest_rows": len(rows),
        "n_pathologies": len(ctx.pathologies),
        "pathologies": ctx.pathologies,
        "pathology_csv_aliases": ctx.prep.get("pathology_csv_aliases", {}),
        "splits": {
            split: {
                "n_images_with_seg_entry": len(v["images"]),
                "n_non_empty_masks": v["masks"],
                "masks_per_pathology": dict(per_split_pathology[split]),
            }
            for split, v in per_split.items()
        },
        "total_non_empty_masks": sum(v["masks"] for v in per_split.values()),
        "non_empty_masks_per_pathology": {k: v["masks"] for k, v in per_pathology.items()},
    }
    summary_path = ctx.manifests_dir / ctx.prep["output"]["summary_filename"]
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[manifest] wrote summary: {summary_path}")


def main() -> int:
    ctx = load_preprocessing_context()
    build_manifest(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
