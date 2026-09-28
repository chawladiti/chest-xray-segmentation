"""Guardrail: the canonical CheXlocalize pathology list must be identical
everywhere — configs, committed split CSVs, and (when present) generated
manifests. Prevents silent drift (e.g. swapping Support Devices for Pneumonia,
or promoting the "Lung Opacity" CSV alias to a task name).
"""

import csv
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The single canonical task list + order (official CheXlocalize tasks).
CANONICAL = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Lung Lesion",
    "Airspace Opacity",
    "Pleural Effusion",
    "Pneumothorax",
    "Support Devices",
]
CANONICAL_SET = set(CANONICAL)


def _load_yaml(rel):
    with open(PROJECT_ROOT / rel, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_canonical_has_10_unique_tasks():
    assert len(CANONICAL) == 10
    assert len(CANONICAL_SET) == 10


def test_support_devices_present_pneumonia_absent():
    assert "Support Devices" in CANONICAL_SET
    assert "Pneumonia" not in CANONICAL_SET


def test_airspace_opacity_canonical_not_lung_opacity():
    assert "Airspace Opacity" in CANONICAL_SET
    assert "Lung Opacity" not in CANONICAL_SET


def test_preprocessing_yaml_matches_canonical_exactly():
    cfg = _load_yaml("configs/preprocessing.yaml")
    assert cfg["pathologies"] == CANONICAL  # same content AND order


def test_evaluation_yaml_matches_canonical_exactly():
    cfg = _load_yaml("configs/evaluation.yaml")
    assert cfg["pathologies"] == CANONICAL


def test_csv_alias_only_maps_airspace_to_lung_opacity():
    cfg = _load_yaml("configs/preprocessing.yaml")
    aliases = cfg.get("pathology_csv_aliases", {})
    assert aliases == {"Airspace Opacity": "Lung Opacity"}


def test_split_csvs_use_canonical_pathologies():
    """Committed split CSVs must use exactly the canonical names (no Pneumonia,
    no 'Lung Opacity' as a pathology)."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    for name in ("train.csv", "internal_val.csv", "test.csv"):
        path = splits_dir / name
        assert path.exists(), f"missing committed split: {name}"
        with open(path, "r", encoding="utf-8") as fh:
            pathologies = {r["pathology"] for r in csv.DictReader(fh)}
        assert pathologies == CANONICAL_SET, f"{name}: {pathologies ^ CANONICAL_SET}"
        assert "Pneumonia" not in pathologies
        assert "Lung Opacity" not in pathologies


def test_generated_manifests_use_canonical_when_present():
    """If the (git-ignored) processed manifests exist locally, they must also
    use the canonical names. Skipped on a fresh clone where they are absent."""
    import pytest

    manifests = PROJECT_ROOT / "data" / "processed" / "manifests"
    checked = 0
    for fname in ("manifest.csv", "training_targets.csv"):
        path = manifests / fname
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as fh:
            pathologies = {r["pathology"] for r in csv.DictReader(fh)}
        assert pathologies <= CANONICAL_SET, f"{fname}: unexpected {pathologies - CANONICAL_SET}"
        assert "Pneumonia" not in pathologies
        checked += 1
    if checked == 0:
        pytest.skip("no generated manifests present (fresh clone)")
