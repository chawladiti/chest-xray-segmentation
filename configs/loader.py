"""Central configuration loader for chexlocalize-medical-imaging.

Responsibilities
----------------
* Load ``configs/project.yaml``.
* Expand ``${VAR:-default}`` environment-variable placeholders in any string
  value (optionally loading a local ``.env`` file first).
* Auto-detect ``project_root`` when it is not provided.
* Resolve every configured path to an absolute path (relative paths are
  resolved against ``project_root``).
* Apply DEMO/FULL mode isolation by appending the active mode's subdir to the
  results / predictions / checkpoints roots so the two modes never mix.

This module intentionally has NO heavy dependencies (only PyYAML, plus an
optional python-dotenv) so it can be imported anywhere in the project.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

# ${VAR} or ${VAR:-default}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

# Path keys under `paths:` that should be resolved to absolute paths.
_PATH_KEYS = (
    "project_root",
    "data_root",
    "raw_data_root",
    "processed_data_root",
    "splits_root",
    "results_root",
    "predictions_root",
    "checkpoints_root",
    "configs_root",
    "docs_root",
)

# Roots that are isolated per development mode (demo vs full).
_MODE_ISOLATED_KEYS = ("results_root", "predictions_root", "checkpoints_root")


def _find_project_root(start: Path) -> Path:
    """Walk upwards from ``start`` until a repo marker is found."""
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() or (candidate / "configs" / "project.yaml").exists():
            return candidate
    return start


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR:-default}`` placeholders in strings."""
    if isinstance(value, str):
        def repl(match: "re.Match[str]") -> str:
            var, default = match.group(1), match.group(2)
            return os.environ.get(var, default if default is not None else "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _maybe_load_dotenv(project_root: Path) -> None:
    """Load a local .env if python-dotenv is available (optional)."""
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(env_file, override=False)


@dataclass
class ProjectConfig:
    """Parsed, path-resolved project configuration."""

    raw: Dict[str, Any]
    project_root: Path
    mode: str
    paths: Dict[str, Path] = field(default_factory=dict)

    # -- convenience accessors -------------------------------------------------
    @property
    def name(self) -> str:
        return self.raw.get("project", {}).get("name", "chexlocalize-medical-imaging")

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw.get("evaluation", {})

    @property
    def dataset(self) -> Dict[str, Any]:
        return self.raw.get("dataset", {})

    def path(self, key: str) -> Path:
        return self.paths[key]

    def ensure_dirs(self) -> None:
        """Create the resolved output directories if they do not exist."""
        for key in ("results_root", "predictions_root", "checkpoints_root"):
            self.paths[key].mkdir(parents=True, exist_ok=True)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "project.yaml"


def load_config(
    config_path: "str | os.PathLike[str] | None" = None,
    *,
    mode: "str | None" = None,
    load_dotenv: bool = True,
) -> ProjectConfig:
    """Load and fully resolve the project configuration.

    Parameters
    ----------
    config_path:
        Path to the YAML config. Defaults to ``configs/project.yaml``.
    mode:
        Override the development mode ("demo" or "full"). Falls back to
        ``PROJECT_MODE`` / the YAML value / "demo".
    load_dotenv:
        Whether to load a local ``.env`` file before expanding variables.
    """
    cfg_path = Path(config_path) if config_path else _default_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # Establish project_root early so .env can be found relative to it.
    detected_root = _find_project_root(cfg_path.resolve().parent)
    if load_dotenv:
        _maybe_load_dotenv(detected_root)

    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw_yaml = yaml.safe_load(fh) or {}

    resolved = _expand_env(raw_yaml)

    # Resolve project_root (empty => auto-detect).
    paths_cfg = resolved.get("paths", {}) or {}
    root_value = (paths_cfg.get("project_root") or "").strip()
    project_root = Path(root_value).expanduser().resolve() if root_value else detected_root

    # Determine active mode.
    active_mode = (
        mode
        or resolved.get("project", {}).get("mode")
        or "demo"
    ).strip().lower()
    if active_mode not in ("demo", "full"):
        raise ValueError(f"Invalid mode '{active_mode}'. Expected 'demo' or 'full'.")

    mode_subdir = (
        resolved.get("modes", {}).get(active_mode, {}).get("subdir", active_mode)
    )

    # Resolve all path keys to absolute paths.
    resolved_paths: Dict[str, Path] = {"project_root": project_root}
    for key in _PATH_KEYS:
        if key == "project_root":
            continue
        raw_val = (paths_cfg.get(key) or "").strip()
        if not raw_val:
            continue
        p = Path(raw_val).expanduser()
        if not p.is_absolute():
            p = project_root / p
        # Isolate demo/full outputs.
        if key in _MODE_ISOLATED_KEYS:
            p = p / mode_subdir
        resolved_paths[key] = p.resolve()

    return ProjectConfig(
        raw=resolved,
        project_root=project_root,
        mode=active_mode,
        paths=resolved_paths,
    )


__all__ = ["ProjectConfig", "load_config"]
