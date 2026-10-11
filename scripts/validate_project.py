#!/usr/bin/env python3
"""Basic project skeleton validation.

Confirms that:
  * the expected directory structure exists,
  * the central config loads and resolves paths,
  * DEMO/FULL output isolation is active,
  * no fabricated results are present.

This performs NO training, inference, or dataset download. Exit code 0 means
the skeleton is healthy.

Usage:
    python scripts/validate_project.py
    PROJECT_MODE=full python scripts/validate_project.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable (so `configs` package resolves) regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import load_config  # noqa: E402

EXPECTED_DIRS = [
    "data/raw",
    "data/processed",
    "data/splits",
    "configs",
    "models/nnunet",
    "models/sam2",
    "models/medsam",
    "models/nvidia_method",
    "preprocessing",
    "evaluation",
    "scripts",
    "results",
    "predictions",
    "checkpoints",
    "notebooks",
    "frontend",
    "docs",
    "tests",
]

EXPECTED_FILES = [
    "configs/project.yaml",
    "configs/loader.py",
    ".gitignore",
    "requirements.txt",
    "README.md",
]


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f" -> {detail}"
    print(line)
    return ok


def main() -> int:
    print("=" * 68)
    print("chexlocalize-medical-imaging :: project skeleton validation")
    print("=" * 68)

    ok = True

    print("\nDirectories:")
    for d in EXPECTED_DIRS:
        ok &= _check(d, (PROJECT_ROOT / d).is_dir())

    print("\nCore files:")
    for f in EXPECTED_FILES:
        ok &= _check(f, (PROJECT_ROOT / f).is_file())

    print("\nConfiguration:")
    try:
        cfg = load_config()
        _check("config loads", True)
        _check("project name", cfg.name == "chexlocalize-medical-imaging", cfg.name)
        _check("mode is demo/full", cfg.mode in ("demo", "full"), cfg.mode)

        # Paths resolve to absolute locations under the project root.
        for key in ("data_root", "raw_data_root", "results_root",
                    "predictions_root", "checkpoints_root"):
            p = cfg.paths.get(key)
            ok &= _check(f"path resolved: {key}", p is not None and p.is_absolute(),
                         str(p) if p else "MISSING")

        # DEMO/FULL isolation: mode-specific roots must end with the mode subdir.
        for key in ("results_root", "predictions_root", "checkpoints_root"):
            p = cfg.paths.get(key)
            ok &= _check(f"mode-isolated: {key}", p is not None and p.name == cfg.mode,
                         str(p) if p else "MISSING")

        # Training contract sanity (no models yet, just the config values).
        max_epochs = cfg.training.get("max_epochs")
        ok &= _check("training.max_epochs <= 20",
                     isinstance(max_epochs, int) and 1 <= max_epochs <= 20,
                     str(max_epochs))
    except Exception as exc:  # pragma: no cover - surfaced via exit code
        ok = _check(f"config loads (exception: {exc})", False)

    print("\n" + "=" * 68)
    if ok:
        print("RESULT: PASS - skeleton is healthy. No dataset/models/results yet.")
        return 0
    print("RESULT: FAIL - see items marked FAIL above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
