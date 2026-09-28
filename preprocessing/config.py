"""Shared preprocessing configuration + path helpers.

Loads configs/preprocessing.yaml and combines it with the central project config
(configs/project.yaml) for base data paths. Provides small utilities for
resolving image paths from CheXlocalize CXR IDs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import load_config  # noqa: E402


@dataclass
class PreprocessingContext:
    project: Any                      # ProjectConfig
    prep: Dict[str, Any]              # parsed preprocessing.yaml
    dataset_root: Path               # raw_data_root/<root_subdir>
    processed_root: Path
    masks_dir: Path
    images_dir: Path
    manifests_dir: Path
    visualizations_dir: Path

    @property
    def pathologies(self) -> list[str]:
        return list(self.prep["pathologies"])

    @property
    def splits(self) -> Dict[str, Any]:
        return self.prep["dataset"]["splits"]

    def ensure_output_dirs(self) -> None:
        for d in (self.masks_dir, self.images_dir, self.manifests_dir,
                  self.visualizations_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_preprocessing_context(prep_yaml: "str | None" = None) -> PreprocessingContext:
    project = load_config()
    prep_path = Path(prep_yaml) if prep_yaml else (PROJECT_ROOT / "configs" / "preprocessing.yaml")
    with open(prep_path, "r", encoding="utf-8") as fh:
        prep = yaml.safe_load(fh)

    raw_root = project.paths["raw_data_root"]
    dataset_root = raw_root / prep["dataset"]["root_subdir"]
    processed_root = project.paths["processed_data_root"]

    out = prep["output"]
    return PreprocessingContext(
        project=project,
        prep=prep,
        dataset_root=dataset_root,
        processed_root=processed_root,
        masks_dir=processed_root / out["masks_subdir"],
        images_dir=processed_root / out["images_subdir"],
        manifests_dir=processed_root / out["manifests_subdir"],
        visualizations_dir=processed_root / out["visualizations_subdir"],
    )


def load_targets_config(targets_yaml: "str | None" = None) -> Dict[str, Any]:
    """Load configs/targets.yaml."""
    path = Path(targets_yaml) if targets_yaml else (PROJECT_ROOT / "configs" / "targets.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_cxr_id(image_id: str) -> tuple[str, str, str]:
    """'patient64622_study1_view1_frontal' -> ('patient64622','study1','view1_frontal')."""
    parts = image_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected CXR ID format: {image_id!r}")
    return parts[0], parts[1], "_".join(parts[2:])


def image_path_for(ctx: PreprocessingContext, split: str, image_id: str) -> Path:
    """Resolve the original CXR JPG path for a CXR ID within a split."""
    patient, study, view = parse_cxr_id(image_id)
    images_subdir = ctx.splits[split]["images_subdir"]
    return ctx.dataset_root / images_subdir / patient / study / f"{view}.jpg"


def safe_pathology_filename(pathology: str) -> str:
    """Filesystem-safe token for a pathology name (spaces -> underscores)."""
    return pathology.replace(" ", "_").replace("/", "-")
