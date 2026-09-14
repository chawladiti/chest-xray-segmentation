"""Tests for the central configuration loader."""

from pathlib import Path

from configs.loader import load_config


def test_config_loads_and_has_project_name():
    cfg = load_config()
    assert cfg.name == "chexlocalize-medical-imaging"


def test_default_mode_is_demo(monkeypatch):
    monkeypatch.delenv("PROJECT_MODE", raising=False)
    cfg = load_config()
    assert cfg.mode == "demo"


def test_explicit_full_mode():
    cfg = load_config(mode="full")
    assert cfg.mode == "full"


def test_paths_are_absolute_and_under_project_root():
    cfg = load_config()
    root = cfg.project_root
    for key in ("data_root", "raw_data_root", "results_root",
                "predictions_root", "checkpoints_root"):
        p = cfg.paths[key]
        assert p.is_absolute(), f"{key} should be absolute, got {p}"
        assert str(p).startswith(str(root)), f"{key} should be under project_root"


def test_demo_and_full_outputs_are_isolated():
    """DEMO and FULL results/predictions/checkpoints must never collide."""
    demo = load_config(mode="demo")
    full = load_config(mode="full")
    for key in ("results_root", "predictions_root", "checkpoints_root"):
        assert demo.paths[key] != full.paths[key]
        assert demo.paths[key].name == "demo"
        assert full.paths[key].name == "full"


def test_env_var_override_for_data_root(monkeypatch, tmp_path):
    custom = tmp_path / "external_data"
    monkeypatch.setenv("DATA_ROOT", str(custom))
    cfg = load_config()
    assert cfg.paths["data_root"] == Path(custom).resolve()


def test_training_contract_defaults():
    cfg = load_config()
    t = cfg.training
    assert t["max_epochs"] <= 20
    assert t["validate_every_epoch"] is True
    assert t["monitor_metric"] == "val_dice"
    assert t["early_stopping"]["enabled"] is True
    assert "patience" in t["early_stopping"]
    assert "learning_rate" in t
    assert "batch_size" in t


def test_dataset_safety_policy_present():
    """The anti-leakage policy must be encoded in config."""
    policy = load_config().dataset.get("policy", {})
    assert policy.get("do_not_train_on_official_test") is True
    assert policy.get("do_not_leak_test_annotations") is True
    assert policy.get("do_not_arbitrarily_split_official_test") is True
