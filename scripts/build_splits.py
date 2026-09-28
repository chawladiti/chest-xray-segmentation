#!/usr/bin/env python3
"""Create the experimental train / internal-validation / test splits.

Methodology (see docs/experimental_split.md):
  * The official CheXlocalize TEST set (668 imgs) is the untouched benchmark and
    becomes test.csv verbatim — it is NEVER sampled/mixed into train/val.
  * The official CheXlocalize VALIDATION set (234 imgs) is the development pool,
    split at the PATIENT level into train / internal_val (~80/20 by images).
  * Negatives (images with no non-empty segmentation) are INCLUDED, not dropped.
  * CheXpert CSV labels are attached (chexpert_label) so "negative" pairs can be
    cross-checked against the official labels.
  * Original source files are read-only; nothing is renamed ("Airspace Opacity"
    stays canonical; "Lung Opacity" appears only as csv_alias).

Outputs into data/splits/: train.csv, internal_val.csv, test.csv,
split_summary.json. Runs automated leakage/integrity checks A–G and prints
PASS/FAIL. No models are implemented or run.

Run:
    python scripts/build_splits.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.config import (  # noqa: E402
    load_preprocessing_context,
    image_path_for,
)

ROW_FIELDS = [
    "image_id", "patient_id", "split", "image_path", "pathology",
    "mask_path", "has_mask", "csv_alias", "chexpert_label",
]


def parse_csv_path(path: str) -> Tuple[str, str, str, str]:
    """CSV Path -> (patient_id, study, view_stem, image_id)."""
    parts = path.replace("\\", "/").split("/")
    pi = next(i for i, t in enumerate(parts) if t.startswith("patient"))
    patient, study = parts[pi], parts[pi + 1]
    view_stem = Path(parts[-1]).stem
    return patient, study, view_stem, f"{patient}_{study}_{view_stem}"


def load_labels(csv_path: Path) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Return {image_id: {csv_column: value}} and the ordered image_id list."""
    labels: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            _, _, _, image_id = parse_csv_path(row["Path"])
            labels[image_id] = row
            order.append(image_id)
    return labels, order


def load_manifest(ctx) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """{(official_split, image_id, pathology): manifest_row}."""
    mpath = ctx.manifests_dir / ctx.prep["output"]["manifest_filename"]
    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}. Run scripts/preprocess.py first.")
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    with open(mpath, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["split"], r["image_id"], r["pathology"])] = r
    return out


def build_rows(ctx, image_ids, official_split, assigned_split, labels, manifest,
               canonical_to_csv) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    aliases = ctx.prep.get("pathology_csv_aliases", {})
    for image_id in image_ids:
        patient_id = image_id.split("_")[0]
        img_path = image_path_for(ctx, official_split, image_id)
        for pathology in ctx.pathologies:
            man = manifest.get((official_split, image_id, pathology))
            has_mask = str(man["has_mask"]).lower() in ("true", "1") if man else False
            mask_path = man["mask_path"] if (man and has_mask) else ""
            csv_col = canonical_to_csv[pathology]
            chexpert_label = labels.get(image_id, {}).get(csv_col, "")
            rows.append({
                "image_id": image_id,
                "patient_id": patient_id,
                "split": assigned_split,
                "image_path": str(img_path),
                "pathology": pathology,          # canonical (Airspace Opacity)
                "mask_path": mask_path,
                "has_mask": has_mask,
                "csv_alias": aliases.get(pathology, ""),
                "chexpert_label": chexpert_label,
            })
    return rows


def patient_level_split(val_image_ids, seed, train_fraction):
    """Group images by patient, shuffle patients, greedily fill train to ~fraction."""
    by_patient: Dict[str, List[str]] = defaultdict(list)
    for iid in val_image_ids:
        by_patient[iid.split("_")[0]].append(iid)
    patients = sorted(by_patient)
    rng = random.Random(seed)
    rng.shuffle(patients)

    total_images = len(val_image_ids)
    target_train = round(train_fraction * total_images)
    train_patients, val_patients = [], []
    running = 0
    for p in patients:
        # Assign to train until we reach the target, then everything else to val.
        if running < target_train:
            train_patients.append(p)
            running += len(by_patient[p])
        else:
            val_patients.append(p)
    train_ids = [iid for p in train_patients for iid in by_patient[p]]
    val_ids = [iid for p in val_patients for iid in by_patient[p]]
    return set(train_patients), set(val_patients), train_ids, val_ids, by_patient


def positive_image_ids(manifest, official_split) -> set:
    pos = set()
    for (split, iid, _path), r in manifest.items():
        if split == official_split and str(r["has_mask"]).lower() in ("true", "1"):
            pos.add(iid)
    return pos


def count_pos_neg(image_ids, pos_set):
    p = sum(1 for i in image_ids if i in pos_set)
    return p, len(image_ids) - p


def main() -> int:
    ctx = load_preprocessing_context()
    project = ctx.project
    scfg = project.raw.get("split", {})
    seed = int(scfg.get("seed", 42))
    train_fraction = float(scfg.get("train_fraction", 0.8))

    splits_dir = project.paths["splits_root"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    # canonical -> CSV column name (identity except Airspace Opacity -> Lung Opacity)
    aliases = ctx.prep.get("pathology_csv_aliases", {})
    canonical_to_csv = {p: aliases.get(p, p) for p in ctx.pathologies}

    # --- load official label lists (authoritative image lists) ---
    val_labels, val_order = load_labels(ctx.dataset_root / "CheXpert" / "val_labels.csv")
    test_labels, test_order = load_labels(ctx.dataset_root / "CheXpert" / "test_labels.csv")
    manifest = load_manifest(ctx)

    val_pos = positive_image_ids(manifest, "val")
    test_pos = positive_image_ids(manifest, "test")

    # --- patient-level split of the official VALIDATION set ---
    train_patients, ival_patients, train_ids, ival_ids, val_by_patient = \
        patient_level_split(val_order, seed, train_fraction)

    # --- build rows ---
    train_rows = build_rows(ctx, train_ids, "val", "train", val_labels, manifest, canonical_to_csv)
    ival_rows = build_rows(ctx, ival_ids, "val", "internal_val", val_labels, manifest, canonical_to_csv)
    test_rows = build_rows(ctx, test_order, "test", "test", test_labels, manifest, canonical_to_csv)

    def write_csv(name, rows):
        path = splits_dir / name
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
            w.writeheader()
            w.writerows(rows)
        return path

    write_csv("train.csv", train_rows)
    write_csv("internal_val.csv", ival_rows)
    write_csv("test.csv", test_rows)

    # --- positive/negative counts ---
    tr_pos, tr_neg = count_pos_neg(train_ids, val_pos)
    iv_pos, iv_neg = count_pos_neg(ival_ids, val_pos)
    te_pos, te_neg = count_pos_neg(test_order, test_pos)

    # --- negative label cross-check (requirement 9) ---
    # For (image, pathology) with has_mask=False, how many have CheXpert label == 1.0?
    def neg_label_conflicts(rows):
        pairs = []
        for r in rows:
            if not r["has_mask"]:
                v = str(r["chexpert_label"]).strip()
                if v in ("1", "1.0"):
                    pairs.append({"image_id": r["image_id"], "pathology": r["pathology"]})
        return pairs

    tr_conf = neg_label_conflicts(train_rows)
    iv_conf = neg_label_conflicts(ival_rows)
    te_conf = neg_label_conflicts(test_rows)
    neg_check = {
        "train_pairs_neg_mask_but_label_positive": len(tr_conf),
        "internal_val_pairs_neg_mask_but_label_positive": len(iv_conf),
        "test_pairs_neg_mask_but_label_positive": len(te_conf),
        "conflict_pairs": {"train": tr_conf, "internal_val": iv_conf, "test": te_conf},
        "note": ("Expected to be > 0: CheXlocalize only provides pixel masks where "
                 "radiologists localized a finding; a positive image-level CheXpert "
                 "label does not guarantee a segmentation. Informational, not a failure."),
    }

    # ------------------------------------------------------------------
    # Automated checks A–G
    # ------------------------------------------------------------------
    checks: Dict[str, bool] = {}
    train_id_set, ival_id_set, test_id_set = set(train_ids), set(ival_ids), set(test_order)

    # A. no patient overlap train vs internal_val
    checks["A_no_patient_overlap_train_ival"] = len(train_patients & ival_patients) == 0
    # B. no official test image in train
    checks["B_no_test_image_in_train"] = len(test_id_set & train_id_set) == 0
    # C. no official test image in internal_val
    checks["C_no_test_image_in_ival"] = len(test_id_set & ival_id_set) == 0
    # D. every split image exists
    all_rows = train_rows + ival_rows + test_rows
    unique_images = {(r["split"], r["image_id"], r["image_path"]) for r in all_rows}
    missing_images = [i for (_s, _id, i) in unique_images if not Path(i).is_file()]
    checks["D_all_images_exist"] = len(missing_images) == 0
    # E. every required mask path exists or is correctly marked empty
    mask_ok = True
    for r in all_rows:
        if r["has_mask"]:
            if not r["mask_path"] or not Path(r["mask_path"]).is_file():
                mask_ok = False
                break
        else:
            if r["mask_path"] != "":
                mask_ok = False
                break
    checks["E_masks_present_or_correctly_empty"] = mask_ok
    # F. pathology naming canonical "Airspace Opacity"
    path_values = {r["pathology"] for r in all_rows}
    checks["F_airspace_opacity_canonical"] = (
        "Airspace Opacity" in path_values and "Lung Opacity" not in path_values
    )
    # G. "Lung Opacity" only as csv_alias
    alias_ok = all(
        (r["csv_alias"] == "Lung Opacity") == (r["pathology"] == "Airspace Opacity")
        or r["pathology"] != "Airspace Opacity"
        for r in all_rows
    )
    # stricter: Airspace Opacity rows must have alias Lung Opacity; no other row may
    alias_ok = (
        all(r["csv_alias"] == "Lung Opacity" for r in all_rows if r["pathology"] == "Airspace Opacity")
        and all(r["csv_alias"] != "Lung Opacity" for r in all_rows if r["pathology"] != "Airspace Opacity")
    )
    checks["G_lung_opacity_only_alias"] = alias_ok

    passed = all(checks.values())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "seed": seed,
        "level": "patient",
        "train_fraction_target": train_fraction,
        "procedure": (
            "Official TEST set (668 imgs / 500 patients) copied verbatim to test.csv. "
            "Official VALIDATION set (234 imgs / 200 patients) grouped by patient; "
            f"patients shuffled with seed={seed}; patients added to train until "
            "cumulative train images reach round(train_fraction*234), remainder to "
            "internal_val. Split is patient-disjoint by construction."
        ),
        "splits": {
            "train": {
                "patients": len(train_patients), "images": len(train_ids),
                "positive_images": tr_pos, "negative_images": tr_neg,
                "rows": len(train_rows),
            },
            "internal_val": {
                "patients": len(ival_patients), "images": len(ival_ids),
                "positive_images": iv_pos, "negative_images": iv_neg,
                "rows": len(ival_rows),
            },
            "test": {
                "patients": len({i.split('_')[0] for i in test_order}),
                "images": len(test_order),
                "positive_images": te_pos, "negative_images": te_neg,
                "rows": len(test_rows),
                "note": "Official CheXlocalize benchmark — untouched.",
            },
        },
        "dev_pool_total_images": len(val_order),
        "actual_train_image_fraction": round(len(train_ids) / len(val_order), 4),
        "negative_label_cross_check": neg_check,
        "pathologies_canonical": ctx.pathologies,
        "pathology_csv_aliases": aliases,
        "checks": checks,
        "patient_overlap_train_ival": sorted(train_patients & ival_patients),
        "PASS": passed,
    }
    summary_path = splits_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Experimental split (patient-level; official test untouched)")
    print("=" * 70)
    print(f"seed: {seed} | target train fraction: {train_fraction}")
    print(f"train        : {len(train_patients):3d} patients | {len(train_ids):3d} images "
          f"| +{tr_pos}/-{tr_neg}")
    print(f"internal_val : {len(ival_patients):3d} patients | {len(ival_ids):3d} images "
          f"| +{iv_pos}/-{iv_neg}")
    print(f"test (official): {summary['splits']['test']['patients']:3d} patients | "
          f"{len(test_order):3d} images | +{te_pos}/-{te_neg}")
    print(f"actual train image fraction: {summary['actual_train_image_fraction']}")
    print("\nNegative-label cross-check (info):")
    for k, v in neg_check.items():
        if k != "note":
            print(f"  {k}: {v}")
    print("\nAutomated checks:")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if missing_images:
        print(f"  missing images (first 5): {missing_images[:5]}")
    print(f"\noutputs: {splits_dir}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
