"""Validate that the project directory skeleton exists as designed."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


@pytest.mark.parametrize("rel", EXPECTED_DIRS)
def test_expected_directories_exist(rel):
    assert (PROJECT_ROOT / rel).is_dir(), f"Missing directory: {rel}"


@pytest.mark.parametrize("rel", EXPECTED_FILES)
def test_expected_files_exist(rel):
    assert (PROJECT_ROOT / rel).is_file(), f"Missing file: {rel}"


@pytest.mark.parametrize(
    "pkg",
    ["configs", "models", "preprocessing", "evaluation", "scripts", "tests"],
)
def test_packages_have_init(pkg):
    assert (PROJECT_ROOT / pkg / "__init__.py").is_file(), f"Missing __init__.py in {pkg}"
