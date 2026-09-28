"""Evaluation configuration loader (model-independent)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import load_config  # noqa: E402


def load_evaluation_config(path: "str | None" = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else (PROJECT_ROOT / "configs" / "evaluation.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_config():
    return load_config()


def get_pathologies(eval_cfg: "Dict[str, Any] | None" = None) -> list[str]:
    eval_cfg = eval_cfg or load_evaluation_config()
    return list(eval_cfg["pathologies"])
