#!/usr/bin/env python3
"""READ-ONLY verification of the downloaded CheXlocalize dataset.

Performs the full integrity check requested for the project:
  * locate dataset root, report path + total disk usage
  * verify required directories/files exist
  * count val/test images
  * report JSON file sizes
  * parse JSONs: #CXR IDs, pathologies, example ID, stored image size,
    #non-empty annotations/segmentations (RLE decoded WITHOUT pycocotools)
  * match CXR IDs referenced by JSON to image files
  * detect missing images and orphaned annotations
  * report which asset types are present (images / GT masks / Grad-CAM / benchmark)

It DOES NOT download, move, modify, or delete anything. It NEVER prints secrets
(no .env values, SAS URLs, or tokens are read or displayed).

Writes a human-readable report to docs/dataset_verification.md and prints a
PASS/FAIL summary. Exit code 0 iff all critical checks pass.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import load_config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
REPORT_PATH = PROJECT_ROOT / "docs" / "dataset_verification.md"

# ---------------------------------------------------------------------------
# COCO RLE decode (compressed string -> run lengths), no external deps.
# Mirrors pycocotools' rleFrString. Foreground area = sum of odd-index runs.
# ---------------------------------------------------------------------------
def rle_counts_to_runs(s: str) -> list[int]:
    cnts: list[int] = []
    p = 0
    data = [ord(c) for c in s]
    n = len(data)
    while p < n:
        x = 0
        k = 0
        more = 1
        while more:
            c = data[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= (-1 << (5 * k))
        if len(cnts) > 2:
            x += cnts[len(cnts) - 2]
        cnts.append(x)
    return cnts


def rle_foreground_area(seg: dict) -> int:
    """Return number of foreground pixels for a COCO-RLE segmentation dict."""
    counts = seg.get("counts")
    if counts is None:
        return 0
    if isinstance(counts, list):  # uncompressed RLE
        runs = counts
    else:
        runs = rle_counts_to_runs(counts)
    # RLE starts with a background run; foreground runs are the odd indices.
    return sum(runs[1::2])


# ---------------------------------------------------------------------------
def find_dataset_root(raw_root: Path) -> Path | None:
    """Find the directory that directly contains BOTH CheXpert/ and CheXlocalize/."""
    if not raw_root.exists():
        return None
    candidates = [raw_root, *[p for p in raw_root.rglob("*") if p.is_dir()]]
    for d in candidates:
        if (d / "CheXpert").is_dir() and (d / "CheXlocalize").is_dir():
            return d
    return None


def du_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024:
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} PB"


def list_image_ids(split_dir: Path) -> set[str]:
    """Build CXR IDs from image files: patient/study/view*.jpg -> patient_study_view*."""
    ids: set[str] = set()
    if not split_dir.exists():
        return ids
    for img in split_dir.rglob("*"):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
            rel = img.relative_to(split_dir)  # patientX/studyY/view1_frontal.jpg
            parts = rel.parts
            if len(parts) >= 3:
                ids.add(f"{parts[0]}_{parts[1]}_{img.stem}")
            else:
                ids.add(img.stem)
    return ids


def count_images(split_dir: Path) -> int:
    if not split_dir.exists():
        return 0
    return sum(1 for p in split_dir.rglob("*")
               if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def analyze_annotation_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    ids = list(data.keys())
    pathologies: set[str] = set()
    non_empty_ids = 0
    example_img_size = None
    for cid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        has_any = False
        for k, v in entry.items():
            if k == "img_size":
                if example_img_size is None:
                    example_img_size = v
                continue
            pathologies.add(k)
            if isinstance(v, list) and len(v) > 0:
                has_any = True
        if has_any:
            non_empty_ids += 1
    return {
        "n_ids": len(ids),
        "pathologies": sorted(pathologies),
        "example_id": ids[0] if ids else None,
        "example_img_size": example_img_size,
        "non_empty_ids": non_empty_ids,
    }


def analyze_segmentation_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    ids = list(data.keys())
    pathologies: set[str] = set()
    non_empty_pairs = 0
    non_empty_ids = 0
    example_size = None
    for cid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        id_has = False
        for pth, seg in entry.items():
            pathologies.add(pth)
            if isinstance(seg, dict) and "size" in seg:
                if example_size is None:
                    example_size = seg["size"]
                if rle_foreground_area(seg) > 0:
                    non_empty_pairs += 1
                    id_has = True
        if id_has:
            non_empty_ids += 1
    return {
        "n_ids": len(ids),
        "pathologies": sorted(pathologies),
        "example_id": ids[0] if ids else None,
        "example_size": example_size,
        "non_empty_pairs": non_empty_pairs,
        "non_empty_ids": non_empty_ids,
        "raw": data,
    }


def main() -> int:
    cfg = load_config()
    raw_root = cfg.paths["raw_data_root"]
    root = find_dataset_root(raw_root)

    lines: list[str] = []
    warnings: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("# CheXlocalize dataset verification")
    out("")
    out("_Read-only verification. No files were downloaded, moved, or modified. "
        "No secrets/SAS URLs are read or displayed._")
    out("")

    if root is None:
        out("## RESULT: FAIL — dataset root not found")
        out(f"Searched under: `{raw_root}` for a folder containing both "
            "`CheXpert/` and `CheXlocalize/`.")
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        print("\nDataset root not found.")
        return 1

    # 1-3. Location + size
    total = du_bytes(root)
    out("## 1. Location & size")
    out(f"- **Dataset root:** `{root}`")
    out(f"- **Total disk usage:** {human(total)} ({total:,} bytes)")
    out("")

    chexpert = root / "CheXpert"
    chexlocalize = root / "CheXlocalize"
    val_dir = chexpert / "val"
    test_dir = chexpert / "test"

    # 4. Required paths
    required = {
        "CheXpert/": chexpert.is_dir(),
        "CheXpert/val/": val_dir.is_dir(),
        "CheXpert/test/": test_dir.is_dir(),
        "CheXpert/val_labels.csv": (chexpert / "val_labels.csv").is_file(),
        "CheXpert/test_labels.csv": (chexpert / "test_labels.csv").is_file(),
        "CheXlocalize/": chexlocalize.is_dir(),
        "CheXlocalize/gt_segmentations_val.json": (chexlocalize / "gt_segmentations_val.json").is_file(),
        "CheXlocalize/gt_segmentations_test.json": (chexlocalize / "gt_segmentations_test.json").is_file(),
        "CheXlocalize/gt_annotations_val.json": (chexlocalize / "gt_annotations_val.json").is_file(),
        "CheXlocalize/gt_annotations_test.json": (chexlocalize / "gt_annotations_test.json").is_file(),
    }
    out("## 2. Required directories / files")
    out("")
    out("| Path | Exists |")
    out("| --- | --- |")
    missing_required = []
    for name, ok in required.items():
        out(f"| `{name}` | {'✅' if ok else '❌ MISSING'} |")
        if not ok:
            missing_required.append(name)
    out("")
    if missing_required:
        warnings.append(f"Missing required paths: {missing_required}")

    # 5. Image counts
    n_val = count_images(val_dir)
    n_test = count_images(test_dir)
    out("## 3. Image counts")
    out(f"- **CheXpert/val/**: {n_val} images (expected 234)")
    out(f"- **CheXpert/test/**: {n_test} images (expected 668)")
    out("")
    if n_val != 234:
        warnings.append(f"val image count {n_val} != expected 234")
    if n_test != 668:
        warnings.append(f"test image count {n_test} != expected 668")

    # 6. JSON file sizes
    json_files = {
        "gt_segmentations_val.json": chexlocalize / "gt_segmentations_val.json",
        "gt_segmentations_test.json": chexlocalize / "gt_segmentations_test.json",
        "gt_annotations_val.json": chexlocalize / "gt_annotations_val.json",
        "gt_annotations_test.json": chexlocalize / "gt_annotations_test.json",
    }
    out("## 4. Annotation JSON file sizes")
    out("")
    out("| File | Size |")
    out("| --- | --- |")
    for name, p in json_files.items():
        sz = human(p.stat().st_size) if p.is_file() else "MISSING"
        out(f"| `{name}` | {sz} |")
    out("")

    # 7. Parse JSONs
    out("## 5. JSON contents")
    out("")
    seg_val = analyze_segmentation_json(json_files["gt_segmentations_val.json"])
    seg_test = analyze_segmentation_json(json_files["gt_segmentations_test.json"])
    ann_val = analyze_annotation_json(json_files["gt_annotations_val.json"])
    ann_test = analyze_annotation_json(json_files["gt_annotations_test.json"])

    out("### gt_segmentations_val.json")
    out(f"- CXR IDs: **{seg_val['n_ids']}**")
    out(f"- Pathologies ({len(seg_val['pathologies'])}): {seg_val['pathologies']}")
    out(f"- Example CXR ID: `{seg_val['example_id']}`")
    out(f"- Example stored mask size [h, w]: {seg_val['example_size']}")
    out(f"- Non-empty (id, pathology) mask pairs: **{seg_val['non_empty_pairs']}**")
    out(f"- CXR IDs with ≥1 non-empty mask: **{seg_val['non_empty_ids']}**")
    out("")
    out("### gt_segmentations_test.json")
    out(f"- CXR IDs: **{seg_test['n_ids']}**")
    out(f"- Pathologies ({len(seg_test['pathologies'])}): {seg_test['pathologies']}")
    out(f"- Example CXR ID: `{seg_test['example_id']}`")
    out(f"- Example stored mask size [h, w]: {seg_test['example_size']}")
    out(f"- Non-empty (id, pathology) mask pairs: **{seg_test['non_empty_pairs']}**")
    out(f"- CXR IDs with ≥1 non-empty mask: **{seg_test['non_empty_ids']}**")
    out("")
    out("### gt_annotations_val.json")
    out(f"- CXR IDs: **{ann_val['n_ids']}**")
    out(f"- Pathologies ({len(ann_val['pathologies'])}): {ann_val['pathologies']}")
    out(f"- Example CXR ID: `{ann_val['example_id']}`")
    out(f"- Example stored img_size: {ann_val['example_img_size']}")
    out(f"- CXR IDs with ≥1 non-empty polygon: **{ann_val['non_empty_ids']}**")
    out("")
    out("### gt_annotations_test.json")
    out(f"- CXR IDs: **{ann_test['n_ids']}**")
    out(f"- Pathologies ({len(ann_test['pathologies'])}): {ann_test['pathologies']}")
    out(f"- Example CXR ID: `{ann_test['example_id']}`")
    out(f"- Example stored img_size: {ann_test['example_img_size']}")
    out(f"- CXR IDs with ≥1 non-empty polygon: **{ann_test['non_empty_ids']}**")
    out("")

    # 8-9. ID matching + orphans
    val_img_ids = list_image_ids(val_dir)
    test_img_ids = list_image_ids(test_dir)

    def matchup(label: str, ref_ids, img_ids):
        ref = set(ref_ids)
        orphans = sorted(ref - img_ids)      # referenced by JSON, no image file
        matched = ref & img_ids
        out(f"### {label}")
        out(f"- Referenced CXR IDs: {len(ref)}")
        out(f"- Matched to an image file: {len(matched)}")
        out(f"- Orphaned (no matching image): {len(orphans)}")
        if orphans:
            out(f"  - examples: {orphans[:5]}")
            warnings.append(f"{label}: {len(orphans)} orphaned annotations")
        out("")
        return orphans

    out("## 6. CXR-ID ↔ image-file matching")
    out("")
    matchup("gt_segmentations_val vs val images", seg_val["raw"].keys(), val_img_ids)
    matchup("gt_segmentations_test vs test images", seg_test["raw"].keys(), test_img_ids)

    # Images without any GT segmentation entry (expected: negatives)
    val_imgs_wo_seg = len(val_img_ids - set(seg_val["raw"].keys()))
    test_imgs_wo_seg = len(test_img_ids - set(seg_test["raw"].keys()))
    out("## 7. Coverage (images without a GT segmentation entry)")
    out(f"- val images with no seg entry: {val_imgs_wo_seg} of {len(val_img_ids)} "
        "(expected: negatives have no mask)")
    out(f"- test images with no seg entry: {test_imgs_wo_seg} of {len(test_img_ids)}")
    out("")

    # 10. Asset inventory
    gradcam_dir = chexlocalize / "gradcam_maps_val"
    n_gradcam = sum(1 for p in gradcam_dir.rglob("*.pkl")) if gradcam_dir.exists() else 0
    benchmark_files = {
        "hb_segmentations_test.json": (chexlocalize / "hb_segmentations_test.json").is_file(),
        "hb_annotations_test.json": (chexlocalize / "hb_annotations_test.json").is_file(),
        "hb_salient_pt_test.json": (chexlocalize / "hb_salient_pt_test.json").is_file(),
        "gradcam_segmentations_val.json": (chexlocalize / "gradcam_segmentations_val.json").is_file(),
        "chexlocalize_tasks.json": (chexlocalize / "chexlocalize_tasks.json").is_file(),
    }
    out("## 8. Asset inventory")
    out(f"- Raw CXR images: {'✅ present' if (n_val + n_test) > 0 else '❌ absent'} "
        f"({n_val + n_test} total)")
    out(f"- Ground-truth segmentations (val+test): "
        f"{'✅ present' if json_files['gt_segmentations_val.json'].is_file() and json_files['gt_segmentations_test.json'].is_file() else '❌ absent'}")
    out(f"- Grad-CAM maps (val, .pkl): {'✅ present' if n_gradcam > 0 else '❌ absent'} "
        f"({n_gradcam} files)")
    out("- Other benchmark files:")
    for name, ok in benchmark_files.items():
        out(f"  - `{name}`: {'✅' if ok else '❌ missing'}")
    out("")

    # Warnings
    out("## 9. Warnings")
    if warnings:
        for w in warnings:
            out(f"- ⚠️ {w}")
    else:
        out("- None. All checks within expected ranges.")
    out("")

    # PASS/FAIL summary
    A = (n_val + n_test) > 0
    B = seg_val["non_empty_pairs"] > 0 and seg_test["non_empty_pairs"] > 0
    C = required["CheXpert/val/"] and json_files["gt_segmentations_val.json"].is_file() and n_val == 234
    D = required["CheXpert/test/"] and json_files["gt_segmentations_test.json"].is_file() and n_test == 668
    # alignment: no orphaned annotations for either split
    seg_val_orphans = set(seg_val["raw"].keys()) - val_img_ids
    seg_test_orphans = set(seg_test["raw"].keys()) - test_img_ids
    E = len(seg_val_orphans) == 0 and len(seg_test_orphans) == 0
    F = A and B and C and D and E and not missing_required

    out("## 10. PASS/FAIL summary")
    out("")
    out("| Check | Result |")
    out("| --- | --- |")
    out(f"| A. Images downloaded | {'PASS' if A else 'FAIL'} |")
    out(f"| B. Ground-truth masks downloaded | {'PASS' if B else 'FAIL'} |")
    out(f"| C. Validation data usable | {'PASS' if C else 'FAIL'} |")
    out(f"| D. Test data usable | {'PASS' if D else 'FAIL'} |")
    out(f"| E. Image/annotation alignment | {'PASS' if E else 'FAIL'} |")
    out(f"| F. Dataset ready for preprocessing | {'PASS' if F else 'FAIL'} |")
    out("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {REPORT_PATH}")
    return 0 if F else 1


if __name__ == "__main__":
    raise SystemExit(main())
