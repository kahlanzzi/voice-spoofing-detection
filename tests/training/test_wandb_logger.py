"""Allow-list W&B logger (D40, D42). A fake wandb module records every call; nothing reaches the network."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.training.cnn_config import EXPECTED
from src.training.wandb_logger import CONFIG_KEYS, SETTINGS, RunLogger
from tests._torch import require_torch

PROVENANCE = {
    "seed": 0, "n_blocks": 3, "learning_rate": 0.001, "code_commit": "a" * 40,
    "profile_fingerprint": "b" * 64, "f2_manifest_sha256": "c" * 64,
    "train_cache_npz_sha256": "d" * 64, "val_a_cache_npz_sha256": "e" * 64,
}
EPOCH = {"epoch": 1, "train_loss": 0.6, "val_a_eer": 0.2, "val_a_auc": 0.9, "epoch_seconds": 3.5}
SUMMARY = {"best_epoch": 1, "best_val_a_eer": 0.2, "epochs_run": 6, "peak_gpu_memory_bytes": None}


def _config(**changes):
    cfg = {**EXPECTED, **PROVENANCE}
    cfg.update(changes)
    return cfg


class FakeWandb(types.ModuleType):
    """Records every call; save / Artifact / log_artifact are recorded so a test can prove they never run."""

    def __init__(self):
        super().__init__("wandb")
        self.calls = []
        fake = self

        class Settings:
            def __init__(self, **kwargs):
                fake.calls.append(("Settings", kwargs))
                self.kwargs = kwargs

        class Summary(dict):
            def update(self, data):
                fake.calls.append(("summary.update", dict(data)))
                super().update(data)

        class Run:
            def __init__(self):
                self.summary = Summary()

            def log(self, data, *args, **kwargs):
                fake.calls.append(("log", dict(data)))

            def finish(self):
                fake.calls.append(("finish", None))

            def save(self, *args, **kwargs):
                fake.calls.append(("run.save", args))

            def log_artifact(self, *args, **kwargs):
                fake.calls.append(("run.log_artifact", args))

        self.Settings = Settings
        self._Run = Run

    def init(self, **kwargs):
        self.calls.append(("init", kwargs))
        return self._Run()

    def save(self, *args, **kwargs):
        self.calls.append(("save", args))

    def Artifact(self, *args, **kwargs):  # noqa: N802 (wandb API name)
        self.calls.append(("Artifact", args))

    def log_artifact(self, *args, **kwargs):
        self.calls.append(("log_artifact", args))


@pytest.fixture()
def fake_wandb(monkeypatch):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_config_keys_are_cnn_yaml_fields_plus_provenance():
    assert CONFIG_KEYS == set(EXPECTED) | set(PROVENANCE)


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_full_run_sends_only_allowed_calls(fake_wandb, mode):
    logger = RunLogger(mode, _config(), "cnn-v1-b3-lr0.001-s0")
    logger.log_epoch(**EPOCH)
    logger.log_summary(**SUMMARY)
    logger.finish()
    logger.finish()  # second call does nothing
    names = [name for name, _ in fake_wandb.calls]
    assert names == ["Settings", "init", "log", "summary.update", "finish"]
    settings, init = fake_wandb.calls[0][1], fake_wandb.calls[1][1]
    assert settings == SETTINGS
    assert settings["save_code"] is False and settings["console"] == "off"
    assert init["mode"] == mode and init["name"] == "cnn-v1-b3-lr0.001-s0"
    assert (init["entity"], init["project"]) == ("voice-spoofing-detection-must",
                                                "voice-spoofing-detection")
    assert init["config"] == _config()
    assert fake_wandb.calls[2][1] == EPOCH
    assert fake_wandb.calls[3][1] == SUMMARY
    assert not {"save", "Artifact", "log_artifact", "run.save", "run.log_artifact"} & set(names)


def test_disabled_mode_never_imports_wandb(monkeypatch):
    class Exploding(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError(f"wandb.{name} used in disabled mode")

    monkeypatch.setitem(sys.modules, "wandb", Exploding("wandb"))
    logger = RunLogger("disabled", _config(), "run")
    logger.log_epoch(**EPOCH)
    logger.log_summary(**SUMMARY)
    logger.finish()


def test_logger_module_never_references_uploads():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "src" / "training" / "wandb_logger.py").read_text(
        encoding="utf-8")
    code = source.split('"""', 2)[2]  # skip the module docstring, which names what is never called
    for name in (".save(", "Artifact(", "log_artifact(", "upload"):
        assert name not in code


BAD_VALUES = [np.array([0.1, 0.2]), [0.1], {"a": 1}, "0.2", float("nan"), float("inf"),
              -float("inf"), None, np.float32(0.2), object()]


@pytest.mark.parametrize("value", BAD_VALUES, ids=[repr(v)[:20] for v in BAD_VALUES])
def test_bad_epoch_values_are_refused_before_sending(fake_wandb, value):
    logger = RunLogger("online", _config(), "run")
    with pytest.raises(ValueError):
        logger.log_epoch(**{**EPOCH, "val_a_eer": value})
    assert [n for n, _ in fake_wandb.calls] == ["Settings", "init"]


def test_tensor_is_refused():
    torch = require_torch()
    logger = RunLogger("disabled", _config(), "run")
    with pytest.raises(ValueError):
        logger.log_epoch(**{**EPOCH, "train_loss": torch.tensor(0.5)})
    with pytest.raises(ValueError):
        logger.log_summary(**{**SUMMARY, "peak_gpu_memory_bytes": torch.tensor(5)})


def test_media_objects_are_refused():
    class Image:  # stands in for wandb.Image / wandb.Table / wandb.Audio
        pass

    logger = RunLogger("disabled", _config(), "run")
    with pytest.raises(ValueError):
        logger.log_epoch(**{**EPOCH, "val_a_auc": Image()})
    with pytest.raises(ValueError):
        RunLogger("disabled", _config(seed=Image()), "run")
    with pytest.raises(ValueError):
        RunLogger("disabled", _config(grid={"n_blocks": np.array([3, 4]), "learning_rate": [0.001]}),
                  "run")


@pytest.mark.parametrize("key", ["val_b_eer", "VAL-B_auc", "test_eer", "external_eer", "momentum"])
def test_forbidden_or_unknown_keys_are_refused(fake_wandb, key):
    with pytest.raises(ValueError):
        RunLogger("online", _config(**{key: 0.1}), "run")
    assert fake_wandb.calls == []
    logger = RunLogger("online", _config(), "run")
    with pytest.raises(TypeError):
        logger.log_epoch(**EPOCH, **{key: 0.1})
    assert [n for n, _ in fake_wandb.calls] == ["Settings", "init"]


@pytest.mark.parametrize("key", ["val_b_eer", "val-B_eer", "test_auc", "external_eer"])
def test_forbidden_key_parts_win_even_over_an_allow_list_entry(fake_wandb, key):
    logger = RunLogger("online", _config(), "run")
    with pytest.raises(ValueError, match="never logged"):
        logger._send("epoch", {**EPOCH, key: 0.1}, (*EPOCH, key))
    assert [n for n, _ in fake_wandb.calls] == ["Settings", "init"]


def test_forbidden_nested_config_key_is_refused(fake_wandb):
    with pytest.raises(ValueError, match="test"):
        RunLogger("online", _config(early_stopping={"metric": "val_a_eer", "test_metric": "eer"}), "run")
    assert fake_wandb.calls == []


def test_missing_config_key_and_bad_arguments_are_refused():
    cfg = _config()
    del cfg["code_commit"]
    with pytest.raises(ValueError, match="missing"):
        RunLogger("disabled", cfg, "run")
    with pytest.raises(ValueError, match="mode"):
        RunLogger("dryrun", _config(), "run")
    with pytest.raises(ValueError, match="run_name"):
        RunLogger("disabled", _config(), " ")
    with pytest.raises(ValueError, match="dict"):
        RunLogger("disabled", list(_config().items()), "run")


def test_summary_allows_none_only_for_peak_memory():
    logger = RunLogger("disabled", _config(), "run")
    logger.log_summary(**{**SUMMARY, "peak_gpu_memory_bytes": 123})
    with pytest.raises(ValueError):
        logger.log_summary(**{**SUMMARY, "best_epoch": None})


def test_calls_after_finish_are_refused():
    logger = RunLogger("disabled", _config(), "run")
    logger.finish()
    with pytest.raises(ValueError, match="finished"):
        logger.log_epoch(**EPOCH)
