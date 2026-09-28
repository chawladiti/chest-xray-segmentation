"""Validate the decoded masks and manifest against the requested checks.

Reads processed/manifests/manifest.csv and reports:
  * missing image files
  * missing annotations (image files with no GT segmentation entry)
  * invalid RLE
  * empty masks
  * duplicate (image_id, pathology) rows
  * dimension mismatches (mask h/w vs image h/w)
  * naming issues (pathology set vs expected; CSV alias reconciliation)

Writes processed/manifests/validation_report.json and prints a summary.
Exit code 0 iff no CRITICAL problems (missing images, invalid RLE, duplicates,
dimension mismatches). Empty masks and negatives-without-masks are informational.

Run standalone:
    python preprocessing/validate_masks.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from preprocessing.config import (
    PreprocessingContext,
    image_path_for,
    load_preprocessing_context,
)


def _read_manifest(ctx: PreprocessingContext) -> List[Dict[str, Any]]:
    path = ctx.manifests_dir / ctx.prep["output"]["manifest_filename"]
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run build_manifest.py first."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def validate(ctx: PreprocessingContext) -> Dict[str, Any]:
    rows = _read_manifest(ctx)
    expected_pathologies = set(ctx.pathologies)

    # --- duplicates ---
    pair_counts = Counter((r["image_id"], r["pathology"]) for r in rows)
    duplicates = [f"{a} | {b}" for (a, b), c in pair_counts.items() if c > 1]

    # --- unique images / image existence ---
    images = {}
    for r in rows:
        images.setdefault((r["split"], r["image_id"]), r["image_path"])
    missing_images = sorted(
        f"{split}:{img}" for (split, img), p in images.items() if not Path(p).is_file()
    )

    # --- invalid RLE / empty masks / dimension mismatches ---
    invalid_rle = []
    empty_masks = 0
    dim_mismatches = []
    missing_mask_files = []
    for r in rows:
        status = r.get("rle_status", "")
        if status.startswith("invalid"):
            invalid_rle.append(f"{r['image_id']} | {r['pathology']} | {status}")
        has_mask = str(r.get("has_mask", "")).lower() in ("true", "1")
        if status == "ok" and not has_mask:
            empty_masks += 1
        # dimension check only where a mask exists
        if has_mask:
            mp = r.get("mask_path", "")
            if mp and not Path(mp).is_file():
                missing_mask_files.append(f"{r['image_id']} | {r['pathology']}")
            try:
                iw, ih = int(r["image_width"]), int(r["image_height"])
                mw, mh = int(r["mask_width"]), int(r["mask_height"])
                if (iw, ih) != (mw, mh):
                    dim_mismatches.append(
                        f"{r['image_id']} | {r['pathology']} | img={iw}x{ih} mask={mw}x{mh}"
                    )
            except (ValueError, KeyError):
                dim_mismatches.append(f"{r['image_id']} | {r['pathology']} | non-numeric dims")

    # --- missing annotations: image files present on disk with NO seg entry ---
    seg_image_ids = {(r["split"], r["image_id"]) for r in rows}
    images_without_seg = {}
    for split, split_cfg in ctx.splits.items():
        split_dir = ctx.dataset_root / split_cfg["images_subdir"]
        count = 0
        if split_dir.exists():
            for img in split_dir.rglob("*.jpg"):
                rel = img.relative_to(split_dir).parts
                if len(rel) >= 3:
                    cxr_id = f"{rel[0]}_{rel[1]}_{img.stem}"
                    if (split, cxr_id) not in seg_image_ids:
                        count += 1
        images_without_seg[split] = count

    # --- naming issues ---
    seen_pathologies = {r["pathology"] for r in rows}
    naming_issues = []
    unexpected = seen_pathologies - expected_pathologies
    absent = expected_pathologies - seen_pathologies
    if unexpected:
        naming_issues.append(f"unexpected pathology names in manifest: {sorted(unexpected)}")
    if absent:
        naming_issues.append(f"expected pathologies missing from manifest: {sorted(absent)}")
    aliases = ctx.prep.get("pathology_csv_aliases", {})

    report = {
        "n_manifest_rows": len(rows),
        "n_unique_images": len(images),
        "duplicates": duplicates,
        "missing_image_files": missing_images,
        "missing_mask_files": missing_mask_files,
        "invalid_rle": invalid_rle,
        "empty_masks_count": empty_masks,
        "dimension_mismatches": dim_mismatches,
        "images_without_seg_entry_per_split": images_without_seg,
        "pathologies_seen": sorted(seen_pathologies),
        "naming_issues": naming_issues,
        "pathology_csv_aliases": aliases,
    }

    # Critical = anything that would break preprocessing correctness.
    critical = {
        "duplicates": len(duplicates),
        "missing_image_files": len(missing_images),
        "missing_mask_files": len(missing_mask_files),
        "invalid_rle": len(invalid_rle),
        "dimension_mismatches": len(dim_mismatches),
        "naming_issues": len(naming_issues),
    }
    report["critical_issue_counts"] = critical
    report["passed"] = all(v == 0 for v in critical.values())

    report_path = ctx.manifests_dir / ctx.prep["output"]["validation_report_filename"]
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # --- print summary ---
    print("=" * 68)
    print("Mask / manifest validation")
    print("=" * 68)
    print(f"manifest rows: {report['n_manifest_rows']}")
    print(f"unique images: {report['n_unique_images']}")
    print(f"duplicates: {len(duplicates)}")
    print(f"missing image files: {len(missing_images)}")
    print(f"missing mask files: {len(missing_mask_files)}")
    print(f"invalid RLE: {len(invalid_rle)}")
    print(f"empty masks (recorded, not written): {empty_masks}")
    print(f"dimension mismatches: {len(dim_mismatches)}")
    print(f"images without seg entry (negatives): {images_without_seg}")
    print(f"naming issues: {naming_issues if naming_issues else 'none'}")
    print(f"report: {report_path}")
    print(f"RESULT: {'PASS' if report['passed'] else 'FAIL'}")
    return report


def main() -> int:
    ctx = load_preprocessing_context()
    report = validate(ctx)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
