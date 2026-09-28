#!/usr/bin/env python3
"""One-command CheXlocalize preprocessing pipeline.

Runs, in order:
  1. decode_masks   — decode GT RLE segmentations -> PNG masks
  2. build_manifest — write manifest.csv + dataset_summary.json
  3. validate_masks — integrity checks -> validation_report.json
  4. visualize_samples — QC overlays (>=10 samples)

Reproducible entry point:
    python scripts/preprocess.py
    python scripts/preprocess.py --no-viz     # skip visualizations

Does NOT modify original data, train models, or compute performance metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.config import load_preprocessing_context  # noqa: E402
from preprocessing import decode_masks, build_manifest, validate_masks, visualize_samples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="CheXlocalize preprocessing pipeline")
    parser.add_argument("--no-viz", action="store_true", help="skip visualizations")
    args = parser.parse_args()

    ctx = load_preprocessing_context()
    print("#" * 72)
    print("CheXlocalize preprocessing pipeline")
    print(f"  dataset_root   : {ctx.dataset_root}")
    print(f"  processed_root : {ctx.processed_root}")
    print("#" * 72)

    print("\n=== [1/4] decode masks ===")
    decode_masks.decode_all(ctx)

    print("\n=== [2/4] build manifest ===")
    build_manifest.build_manifest(ctx)

    print("\n=== [3/4] validate ===")
    report = validate_masks.validate(ctx)

    if not args.no_viz:
        print("\n=== [4/4] visualize samples ===")
        visualize_samples.visualize(ctx)
    else:
        print("\n=== [4/4] visualize samples (skipped) ===")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
