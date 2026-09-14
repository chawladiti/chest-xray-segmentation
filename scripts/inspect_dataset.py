#!/usr/bin/env python3
"""Inspect the CheXlocalize dataset AS DOWNLOADED — no assumptions.

This does NOT split, train, move, or modify anything. It only reports what is
physically present under the raw data root so we can decide an appropriate
train/eval strategy afterwards. It is safe to run before or after download.

Usage:
    python scripts/inspect_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import load_config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _summarize_dir(path: Path) -> tuple[int, int, Counter]:
    """Return (file_count, total_bytes, ext_counter) for a directory tree."""
    files = 0
    size = 0
    exts: Counter = Counter()
    for p in path.rglob("*"):
        if p.is_file():
            files += 1
            try:
                size += p.stat().st_size
            except OSError:
                pass
            exts[p.suffix.lower()] += 1
    return files, size, exts


def _peek_json(path: Path) -> str:
    """Report top-level structure of a JSON file without dumping contents."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # pragma: no cover
        return f"(could not parse: {exc})"
    if isinstance(data, dict):
        keys = list(data.keys())
        sample = keys[:3]
        detail = f"dict with {len(keys)} top-level keys; e.g. {sample}"
        # If values look like per-pathology dicts, report the inner keys once.
        if keys and isinstance(data[keys[0]], dict):
            inner = list(data[keys[0]].keys())
            detail += f"; inner keys (first entry): {inner[:12]}"
        return detail
    if isinstance(data, list):
        return f"list with {len(data)} items"
    return f"{type(data).__name__}"


def main() -> int:
    cfg = load_config()
    raw = cfg.paths["raw_data_root"]

    print("=" * 72)
    print("CheXlocalize dataset inspection (no assumptions, read-only)")
    print("=" * 72)
    print(f"raw_data_root: {raw}")

    if not raw.exists():
        print("\n[!] raw_data_root does not exist yet. Nothing downloaded.")
        print("    Run: bash scripts/download_chexlocalize.sh (after setting SAS URL)")
        return 0

    entries = sorted(raw.rglob("*"))
    if not entries:
        print("\n[!] raw_data_root is empty. Nothing downloaded yet.")
        return 0

    # Top-level layout.
    print("\nTop-level entries under raw_data_root:")
    for child in sorted(raw.iterdir()):
        kind = "dir " if child.is_dir() else "file"
        print(f"  [{kind}] {child.name}")

    # Directories with image content.
    print("\nDirectory summaries:")
    for d in sorted([p for p in raw.rglob("*") if p.is_dir()]):
        files, size, exts = _summarize_dir(d)
        if files == 0:
            continue
        img = sum(v for k, v in exts.items() if k in IMAGE_EXTS)
        rel = d.relative_to(raw)
        note = f" ({img} images)" if img else ""
        top_exts = ", ".join(f"{k or '<none>'}:{v}" for k, v in exts.most_common(4))
        print(f"  {rel}: {files} files, {_human(size)}{note} [{top_exts}]")

    # JSON annotation/segmentation files.
    json_files = sorted(raw.rglob("*.json"))
    if json_files:
        print("\nJSON files (annotations / segmentations):")
        for j in json_files:
            print(f"  {j.relative_to(raw)}: {_peek_json(j)}")

    # CSV label files.
    csv_files = sorted(raw.rglob("*.csv"))
    if csv_files:
        print("\nCSV files (labels):")
        for c in csv_files:
            try:
                with open(c, "r", encoding="utf-8") as fh:
                    header = fh.readline().strip()
                    rows = sum(1 for _ in fh)
                print(f"  {c.relative_to(raw)}: {rows} rows; header: {header[:120]}")
            except Exception as exc:  # pragma: no cover
                print(f"  {c.relative_to(raw)}: (could not read: {exc})")

    print("\n" + "=" * 72)
    print("Inspection complete. NO splits or training performed.")
    print("Next: decide train/eval strategy from this real structure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
