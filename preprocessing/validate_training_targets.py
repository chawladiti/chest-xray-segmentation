"""TASK 5 — validate the segmentation training targets.

Checks:
  1. image/mask dimensional consistency (POSITIVE + NEGATIVE masks == image dims)
  2. no patient leakage (train vs internal_val patient-disjoint)
  3. no test-set training targets (no split == 'test')
  4. correct target_status assignment (recomputed from the rules)
  5. no positive-unlocalized case converted to a negative mask
     (every IGNORE pair has NO mask file)
  6. no duplicate (image_id, pathology) pairs

Writes processed/manifests/training_target_validation.json and
docs/training_target_validation.md; prints PASS/FAIL. Exit 0 iff all pass.

Run:
    python preprocessing/validate_training_targets.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.config import (
    PreprocessingContext,
    load_preprocessing_context,
    load_targets_config,
)
from preprocessing.build_training_targets import classify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = PROJECT_ROOT / "docs" / "training_target_validation.md"


def _read_manifest(ctx: PreprocessingContext, tcfg) -> List[Dict[str, str]]:
    path = ctx.manifests_dir / tcfg["output"]["manifest_filename"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_training_targets.py first.")
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def validate(ctx: PreprocessingContext) -> Dict[str, Any]:
    tcfg = load_targets_config()
    policy = tcfg["label_policy"]
    statuses = tcfg["statuses"]
    rows = _read_manifest(ctx, tcfg)

    dim_cache: Dict[str, tuple] = {}

    def image_dims(path: str):
        if path not in dim_cache:
            try:
                with Image.open(path) as im:
                    dim_cache[path] = (im.width, im.height)
            except Exception:
                dim_cache[path] = None
        return dim_cache[path]

    # 1. dimensional consistency
    dim_mismatches = []
    # 5. IGNORE must have no mask
    ignore_with_mask = []
    # 4. status correctness
    status_errors = []
    # test-set / duplicates / patients
    test_rows = 0
    pair_counts = Counter()
    patients_by_split = defaultdict(set)

    for r in rows:
        split = r["split"]
        pathology = r["pathology"]
        image_id = r["image_id"]
        status = r["target_status"]
        has_mask = str(r["has_mask"]).lower() in ("true", "1")
        mask_path = r["mask_path"]

        pair_counts[(image_id, pathology)] += 1
        patients_by_split[split].add(r["patient_id"])
        if split == "test":
            test_rows += 1

        # recompute expected status
        expected = classify(has_mask, r.get("chexpert_label", ""), policy, statuses)
        if expected != status:
            status_errors.append(f"{image_id}|{pathology}: got {status}, expected {expected}")

        if status == statuses["ignore"]:
            if mask_path:
                ignore_with_mask.append(f"{image_id}|{pathology}")
        else:
            # POSITIVE/NEGATIVE must have an existing mask file matching image dims
            if not mask_path or not Path(mask_path).is_file():
                dim_mismatches.append(f"{image_id}|{pathology}: missing mask file")
                continue
            idim = image_dims(r["image_path"])
            with Image.open(mask_path) as mim:
                mdim = (mim.width, mim.height)
            if idim is not None and idim != mdim:
                dim_mismatches.append(f"{image_id}|{pathology}: img{idim} != mask{mdim}")

    duplicates = [f"{a}|{b}" for (a, b), c in pair_counts.items() if c > 1]
    train_patients = patients_by_split.get("train", set())
    ival_patients = patients_by_split.get("internal_val", set())
    patient_overlap = sorted(train_patients & ival_patients)

    status_counts = Counter(r["target_status"] for r in rows)

    checks = {
        "dimensional_consistency": len(dim_mismatches) == 0,
        "no_patient_leakage": len(patient_overlap) == 0,
        "no_test_training_targets": test_rows == 0,
        "correct_status_assignment": len(status_errors) == 0,
        "no_positive_unlocalized_as_negative": len(ignore_with_mask) == 0,
        "no_duplicate_pairs": len(duplicates) == 0,
    }
    passed = all(checks.values())

    report = {
        "n_rows": len(rows),
        "status_counts": dict(status_counts),
        "checks": checks,
        "details": {
            "dimension_mismatches": dim_mismatches[:20],
            "n_dimension_mismatches": len(dim_mismatches),
            "patient_overlap_train_ival": patient_overlap,
            "test_rows": test_rows,
            "status_errors": status_errors[:20],
            "n_status_errors": len(status_errors),
            "ignore_with_mask": ignore_with_mask[:20],
            "n_ignore_with_mask": len(ignore_with_mask),
            "duplicates": duplicates[:20],
            "n_duplicates": len(duplicates),
        },
        "train_patients": len(train_patients),
        "internal_val_patients": len(ival_patients),
        "PASS": passed,
    }

    report_path = ctx.manifests_dir / tcfg["output"]["validation_report_filename"]
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _write_doc(report)

    print("=" * 66)
    print("Training-target validation")
    print("=" * 66)
    print(f"rows: {report['n_rows']}  status counts: {dict(status_counts)}")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"report: {report_path}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return report


def _write_doc(report: Dict[str, Any]) -> None:
    c = report["checks"]
    sc = report["status_counts"]
    lines = []
    a = lines.append
    a("# Training-target validation (TASK 5)\n")
    a("Automated verification of the segmentation training targets built from the "
      "development pool (train + internal_val). No models are run.\n")
    a("## Status counts\n")
    a("| target_status | pairs |")
    a("| --- | --- |")
    for k in ("VALID_POSITIVE", "VALID_NEGATIVE", "IGNORE"):
        a(f"| {k} | {sc.get(k, 0)} |")
    a(f"| **total** | {report['n_rows']} |\n")
    a("## Checks\n")
    a("| Check | Result |")
    a("| --- | --- |")
    a(f"| Image/mask dimensional consistency | {'✅ PASS' if c['dimensional_consistency'] else '❌ FAIL'} |")
    a(f"| No patient leakage (train ↔ internal_val) | {'✅ PASS' if c['no_patient_leakage'] else '❌ FAIL'} |")
    a(f"| No test-set training targets | {'✅ PASS' if c['no_test_training_targets'] else '❌ FAIL'} |")
    a(f"| Correct target_status assignment | {'✅ PASS' if c['correct_status_assignment'] else '❌ FAIL'} |")
    a(f"| No positive-unlocalized converted to negative mask | {'✅ PASS' if c['no_positive_unlocalized_as_negative'] else '❌ FAIL'} |")
    a(f"| No duplicate (image_id, pathology) pairs | {'✅ PASS' if c['no_duplicate_pairs'] else '❌ FAIL'} |")
    a(f"\n**Overall: {'PASS' if report['PASS'] else 'FAIL'}**\n")
    a("## Detail counts\n")
    d = report["details"]
    a(f"- dimension mismatches: {d['n_dimension_mismatches']}")
    a(f"- status assignment errors: {d['n_status_errors']}")
    a(f"- IGNORE pairs with a mask file (must be 0): {d['n_ignore_with_mask']}")
    a(f"- duplicate pairs: {d['n_duplicates']}")
    a(f"- test rows (must be 0): {d['test_rows']}")
    a(f"- train / internal_val patients: {report['train_patients']} / {report['internal_val_patients']}")
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ctx = load_preprocessing_context()
    report = validate(ctx)
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
