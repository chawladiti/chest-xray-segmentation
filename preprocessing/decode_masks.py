"""Decode CheXlocalize GT segmentation RLE into usable binary masks.

Reads the official gt_segmentations_{val,test}.json (COCO-RLE, pycocotools
compatible), decodes each (image_id, pathology) mask, and writes it as a PNG
(or .npy) under processed/masks/<split>/<image_id>/<pathology>.png.

* The original JSON files are opened READ-ONLY and never modified.
* Empty (all-zero) masks are recorded but, by default, not written to disk.
* An intermediate index (_mask_index.csv) is written for build_manifest.py.

Run standalone:
    python preprocessing/decode_masks.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from preprocessing.config import (
    PreprocessingContext,
    load_preprocessing_context,
    safe_pathology_filename,
)


def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    """Decode one COCO-RLE dict {'size':[h,w],'counts':str|bytes} -> HxW uint8 {0,1}."""
    counts = rle["counts"]
    obj = {
        "size": list(rle["size"]),
        "counts": counts.encode("ascii") if isinstance(counts, str) else counts,
    }
    m = mask_utils.decode(obj)
    if m.ndim == 3:  # (h, w, 1) -> (h, w)
        m = m[:, :, 0]
    return m.astype(np.uint8)


def _write_mask(path: Path, mask01: np.ndarray, fmt: str, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(path, mask01)
    else:  # png
        lo, hi = values
        img = np.where(mask01 > 0, hi, lo).astype(np.uint8)
        Image.fromarray(img, mode="L").save(path)


def decode_all(ctx: PreprocessingContext) -> List[Dict[str, Any]]:
    """Decode every (image_id, pathology) mask across all splits.

    Returns a list of record dicts (one per image_id x pathology).
    """
    ctx.ensure_output_dirs()
    out = ctx.prep["output"]
    fmt = out["mask_format"]
    save_empty = bool(out["save_empty_masks"])
    values = out["binary_mask_values"]
    ext = ".npy" if fmt == "npy" else ".png"

    records: List[Dict[str, Any]] = []

    for split, split_cfg in ctx.splits.items():
        seg_path = ctx.dataset_root / split_cfg["gt_segmentations"]
        with open(seg_path, "r", encoding="utf-8") as fh:  # READ-ONLY
            seg_data = json.load(fh)

        print(f"[decode] {split}: {len(seg_data)} CXR IDs from {seg_path.name}")
        n_written = 0
        for image_id, entry in seg_data.items():
            for pathology in ctx.pathologies:
                rec: Dict[str, Any] = {
                    "image_id": image_id,
                    "split": split,
                    "pathology": pathology,
                    "mask_path": "",
                    "mask_width": "",
                    "mask_height": "",
                    "mask_area": 0,
                    "has_mask": False,
                    "rle_status": "absent",
                }
                seg = entry.get(pathology)
                if isinstance(seg, dict) and "counts" in seg and "size" in seg:
                    try:
                        m = decode_rle(seg)
                        h, w = int(m.shape[0]), int(m.shape[1])
                        area = int(m.sum())
                        rec.update(
                            mask_width=w, mask_height=h, mask_area=area,
                            has_mask=area > 0, rle_status="ok",
                        )
                        if area > 0 or save_empty:
                            rel = Path(split) / image_id / (safe_pathology_filename(pathology) + ext)
                            mask_path = ctx.masks_dir / rel
                            _write_mask(mask_path, m, fmt, values)
                            rec["mask_path"] = str(mask_path)
                            n_written += 1
                    except Exception as exc:  # invalid RLE
                        rec["rle_status"] = f"invalid: {exc}"
                records.append(rec)
        print(f"[decode] {split}: wrote {n_written} non-empty masks")

    # Write intermediate index for build_manifest.py
    index_path = ctx.manifests_dir / out["mask_index_filename"]
    fieldnames = ["image_id", "split", "pathology", "mask_path",
                  "mask_width", "mask_height", "mask_area", "has_mask", "rle_status"]
    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[decode] wrote mask index: {index_path} ({len(records)} rows)")
    return records


def main() -> int:
    ctx = load_preprocessing_context()
    decode_all(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
