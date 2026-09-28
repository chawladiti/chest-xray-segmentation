"""TASK 1 — spatial overlap analysis of CheXlocalize pathology masks.

For every image with >= 2 non-empty pathology masks, checks whether any pair of
masks overlaps spatially (shared foreground pixels). This informs the target
representation decision (multi-class vs multi-label). Read-only; no training.

Scope (configs/targets.yaml -> overlap_analysis.scope):
  "all_positive" — every image with >=1 non-empty mask across val + test.

Writes processed/manifests/mask_overlap.json and docs/mask_overlap_analysis.md,
and prints a summary.

Run:
    python preprocessing/analyze_mask_overlap.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.config import (
    PreprocessingContext,
    load_preprocessing_context,
    load_targets_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = PROJECT_ROOT / "docs" / "mask_overlap_analysis.md"


def _load_mask_bool(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def _iter_image_dirs(masks_dir: Path):
    """Yield (split, image_id, {pathology: mask_png_path}) for each image with masks."""
    for split_dir in sorted(p for p in masks_dir.iterdir() if p.is_dir()):
        split = split_dir.name
        for image_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            masks = {}
            for png in sorted(image_dir.glob("*.png")):
                pathology = png.stem.replace("_", " ")
                masks[pathology] = png
            if masks:
                yield split, image_dir.name, masks


def analyze(ctx: PreprocessingContext) -> Dict[str, Any]:
    tcfg = load_targets_config()
    scope = tcfg["overlap_analysis"]["scope"]
    example_cap = int(tcfg["overlap_analysis"]["iou_example_count"])

    total_images_with_mask = 0
    multi_mask_images = 0
    images_with_overlap = 0
    overlapping_pairs = 0
    total_pairs_checked = 0
    class_pair_counter: Counter = Counter()
    examples: List[Dict[str, Any]] = []
    per_split = Counter()

    for split, image_id, masks in _iter_image_dirs(ctx.masks_dir):
        total_images_with_mask += 1
        per_split[split] += 1
        if len(masks) < 2:
            continue
        multi_mask_images += 1

        # Load masks once per image.
        loaded = {p: _load_mask_bool(path) for p, path in masks.items()}
        image_has_overlap = False
        for a, b in combinations(sorted(loaded), 2):
            total_pairs_checked += 1
            ma, mb = loaded[a], loaded[b]
            if ma.shape != mb.shape:  # should not happen within an image
                continue
            inter = int(np.logical_and(ma, mb).sum())
            if inter > 0:
                overlapping_pairs += 1
                image_has_overlap = True
                class_pair_counter[(a, b)] += 1
                if len(examples) < example_cap:
                    union = int(np.logical_or(ma, mb).sum())
                    examples.append({
                        "split": split,
                        "image_id": image_id,
                        "pair": [a, b],
                        "intersection_px": inter,
                        "iou": round(inter / union, 4) if union else 0.0,
                    })
        if image_has_overlap:
            images_with_overlap += 1

    pct_multi_with_overlap = (
        round(100 * images_with_overlap / multi_mask_images, 2) if multi_mask_images else 0.0
    )
    pct_all_with_overlap = (
        round(100 * images_with_overlap / total_images_with_mask, 2) if total_images_with_mask else 0.0
    )

    top_pairs = [
        {"classes": [a, b], "count": c}
        for (a, b), c in class_pair_counter.most_common(15)
    ]

    report = {
        "scope": scope,
        "images_with_any_mask": total_images_with_mask,
        "images_per_split": dict(per_split),
        "images_with_multiple_pathology_masks": multi_mask_images,
        "images_with_overlap": images_with_overlap,
        "overlapping_pathology_mask_pairs": overlapping_pairs,
        "total_pathology_mask_pairs_checked": total_pairs_checked,
        "percent_of_multimask_images_with_overlap": pct_multi_with_overlap,
        "percent_of_all_masked_images_with_overlap": pct_all_with_overlap,
        "top_overlapping_class_pairs": top_pairs,
        "examples": examples,
    }

    report_path = ctx.manifests_dir / load_targets_config()["output"]["overlap_report_filename"]
    ctx.manifests_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    _write_doc(report, report_path)

    print("=" * 66)
    print("Mask overlap analysis")
    print("=" * 66)
    print(f"scope: {scope}")
    print(f"images with >=1 mask: {total_images_with_mask}")
    print(f"images with multiple pathology masks: {multi_mask_images}")
    print(f"images with >=1 overlapping pair: {images_with_overlap} "
          f"({pct_multi_with_overlap}% of multi-mask, {pct_all_with_overlap}% of all)")
    print(f"overlapping pathology-mask pairs: {overlapping_pairs} "
          f"of {total_pairs_checked} checked")
    print("top overlapping class pairs:")
    for tp in top_pairs[:8]:
        print(f"  {tp['classes'][0]} + {tp['classes'][1]}: {tp['count']}")
    print(f"report: {report_path}")
    return report


def _write_doc(report: Dict[str, Any], report_path: Path) -> None:
    lines = []
    a = lines.append
    a("# Mask overlap analysis (TASK 1)\n")
    a("Spatial-overlap study of CheXlocalize ground-truth pathology masks, used to "
      "decide the segmentation target representation. Read-only; no training.\n")
    a(f"- **Scope:** `{report['scope']}` (images with ≥1 non-empty mask across "
      "val + test)")
    a(f"- **Images with ≥1 mask:** {report['images_with_any_mask']} "
      f"({report['images_per_split']})")
    a(f"- **Images with multiple pathology masks:** "
      f"{report['images_with_multiple_pathology_masks']}")
    a(f"- **Images with ≥1 overlapping pair:** {report['images_with_overlap']}")
    a(f"- **Overlapping pathology-mask pairs:** "
      f"{report['overlapping_pathology_mask_pairs']} of "
      f"{report['total_pathology_mask_pairs_checked']} pairs checked")
    a(f"- **% of multi-mask images with overlap:** "
      f"{report['percent_of_multimask_images_with_overlap']}%")
    a(f"- **% of all masked images with overlap:** "
      f"{report['percent_of_all_masked_images_with_overlap']}%\n")
    a("## Most frequent overlapping class pairs\n")
    a("| Pathology A | Pathology B | # images |")
    a("| --- | --- | --- |")
    for tp in report["top_overlapping_class_pairs"]:
        a(f"| {tp['classes'][0]} | {tp['classes'][1]} | {tp['count']} |")
    a("\n## Example overlaps\n")
    a("| Split | Image ID | Classes | Intersection px | IoU |")
    a("| --- | --- | --- | --- | --- |")
    for ex in report["examples"]:
        a(f"| {ex['split']} | `{ex['image_id']}` | {ex['pair'][0]} + {ex['pair'][1]} "
          f"| {ex['intersection_px']} | {ex['iou']} |")
    a("\n## Implication\n")
    a("Pathology masks **overlap spatially** (a single pixel can belong to more "
      "than one finding, e.g. Airspace Opacity co-located with Consolidation/"
      "Atelectasis/Edema). A single mutually-exclusive multi-class label map is "
      "therefore **invalid** — see `docs/segmentation_target_strategy.md`.\n")
    a(f"_Machine-readable report: `{report_path.name}`._")
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ctx = load_preprocessing_context()
    analyze(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
