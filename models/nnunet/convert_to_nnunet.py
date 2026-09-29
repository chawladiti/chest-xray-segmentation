#!/usr/bin/env python3
"""Convert the *validated* CheXlocalize preprocessing into nnU-Net v2 raw format.

Design decisions (kept consistent with the project's methodology):

* CheXlocalize pathologies overlap ARBITRARILY (any combination; ~74% mask
  overlap). nnU-Net's region-based training only supports *nested* regions on a
  single-integer label map, which cannot represent arbitrary overlap. The correct
  representation is therefore **one BINARY nnU-Net dataset per pathology** — each
  pathology is an independent binary segmentation task, so all overlaps are
  preserved (handled independently).
* Splits: TRAIN -> imagesTr/labelsTr (nnU-Net trains + runs its own internal CV
  on these). INTERNAL_VAL -> imagesTs (+ ground-truth masks saved separately for
  our own Dice/IoU evaluation). The official TEST split is NEVER converted or
  shipped (held out).
* Target policy preserved: VALID_POSITIVE -> real mask; VALID_NEGATIVE -> all-zero
  label (true negative supervision); IGNORE -> excluded from supervision.
* Images resized to --size (bilinear); labels nearest-resized + binarized.

This runs LOCALLY (all source data is local) and produces a small dataset that is
shipped to the pod, so no raw data transfer / re-download is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

VALID_STATUSES = {"VALID_POSITIVE", "VALID_NEGATIVE"}


def _key(pathology: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", pathology.lower()).strip("_")


def _load_rows(csv_path: Path, pathology: str, split: str) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["pathology"] == pathology and row["split"] == split:
                rows.append(row)
    return rows


def _subsample_balanced(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    usable = [r for r in rows if r["target_status"] in VALID_STATUSES]
    usable.sort(key=lambda r: r["image_id"])  # deterministic
    if limit is None or len(usable) <= limit:
        return usable
    rng = np.random.default_rng(seed)
    pos = [r for r in usable if r["target_status"] == "VALID_POSITIVE"]
    neg = [r for r in usable if r["target_status"] == "VALID_NEGATIVE"]
    half = limit // 2
    take_pos = min(len(pos), max(half, limit - len(neg)))
    take_neg = limit - take_pos
    sel = list(rng.choice(pos, size=take_pos, replace=False)) if take_pos else []
    sel += list(rng.choice(neg, size=take_neg, replace=False)) if take_neg else []
    sel.sort(key=lambda r: r["image_id"])
    return sel


def _write_image(src: str, dst: Path, size: int) -> None:
    im = Image.open(src).convert("L").resize((size, size), Image.BILINEAR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)


def _write_label(row: dict, dst: Path, size: int) -> int:
    """Return #positive pixels (0 for negatives)."""
    if row["target_status"] == "VALID_POSITIVE" and row.get("mask_path"):
        m = Image.open(row["mask_path"]).convert("L").resize((size, size), Image.NEAREST)
        arr = (np.array(m) > 127).astype(np.uint8)
    else:  # VALID_NEGATIVE -> all background
        arr = np.zeros((size, size), dtype=np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(dst)
    return int(arr.sum())


def convert(args: argparse.Namespace) -> dict:
    pathology = args.pathology
    key = _key(pathology)
    ds_name = args.dataset_name or f"Dataset{args.dataset_id:03d}_{key}"
    raw_dir = Path(args.out_root) / ds_name
    imagesTr = raw_dir / "imagesTr"
    labelsTr = raw_dir / "labelsTr"
    imagesTs = raw_dir / "imagesTs"
    gt_dir = Path(args.gt_out) / ds_name  # internal_val ground truth for our eval

    train = _subsample_balanced(_load_rows(Path(args.targets_csv), pathology, "train"),
                                args.max_train, args.seed)
    val = _subsample_balanced(_load_rows(Path(args.targets_csv), pathology, "internal_val"),
                              args.max_val, args.seed + 1)

    stats = {"dataset": ds_name, "pathology": pathology, "train": {"pos": 0, "neg": 0},
             "val": {"pos": 0, "neg": 0}, "size": args.size}

    for r in train:
        cid = r["image_id"]
        _write_image(r["image_path"], imagesTr / f"{cid}_0000.png", args.size)
        n = _write_label(r, labelsTr / f"{cid}.png", args.size)
        stats["train"]["pos" if n > 0 else "neg"] += 1

    for r in val:
        cid = r["image_id"]
        _write_image(r["image_path"], imagesTs / f"{cid}_0000.png", args.size)
        n = _write_label(r, gt_dir / f"{cid}.png", args.size)  # gt for our evaluator
        stats["val"]["pos" if n > 0 else "neg"] += 1

    dataset_json = {
        "channel_names": {"0": "Xray"},
        "labels": {"background": 0, key: 1},
        "numTraining": len(train),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "description": f"CheXlocalize binary segmentation: {pathology} (arbitrary "
                       f"multi-label handled as independent binary task).",
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    stats["paths"] = {"raw_dir": str(raw_dir), "gt_dir": str(gt_dir),
                      "numTraining": len(train), "numVal": len(val)}
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets-csv",
                    default="data/processed/manifests/training_targets.csv")
    ap.add_argument("--pathology", required=True)
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--out-root", default="nnunet_data/nnUNet_raw")
    ap.add_argument("--gt-out", default="nnunet_data/internal_val_gt")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    stats = convert(args)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
