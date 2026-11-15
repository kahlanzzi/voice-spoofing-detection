"""Allow-list W&B logger for model v2 (D40): exact keys, forbidden parts, nulls, disabled mode."""
from __future__ import annotations

import sys

import pytest

from tests._torch import require_torch

torch = require_torch()  # v2_wandb_logger imports torch through v2_data

from src.training.v2_config import load_v2_config  # noqa: E402
from src.training.v2_wandb_logger import (  # noqa: E402
    CONFIG_KEYS,
    EPOCH_KEYS,
    RUN_KEYS,
    SUMMARY_KEYS,
    V2RunLogger,
)
from tests._v2_fakes import V2_CONFIG_PATH, v2_config_copy  # noqa: E402


@pytest.fixture(scope="module")
def base_config(tmp_path_factory):
    path = v2_config_copy(tmp_path_factory.mktemp("cfg") / "v2.yaml")
    return load_v2_config(path).to_dict()


def _config(base, candidate="crnn", **changes):
    cfg = {**base, "candidate": candidate, "seed": 0, "batch_size": 8, "num_workers": 2,
           "code_commit": "c" * 40, "profile_fingerprint": "1" * 64, "f2_manifest_sha256": "2" * 64,
           "raw_train_cache_npz_sha256": "3" * 64, "selection_set_sha256": "4" * 64,
           "augmentation_config_sha256": "5" * 64,
           "train_stats_npz_sha256": None if candidate == "aasist" else "6" * 64}
    cfg.update(changes)
    return cfg


def _epoch(**changes):
    data = {k: 1 for k in EPOCH_KEYS}
    data.update(changes)
    return data


def test_exact_key_sets():
    assert EPOCH_KEYS == (
        "epoch", "train_loss", "selection_m",
        "val_a_eer_clean", "val_a_eer_noise_10db", "val_a_eer_reverberation", "val_a_eer_opus_16",
        "val_a_eer_narrowband",
        "val_a_auc_clean", "val_a_auc_noise_10db", "val_a_auc_reverberation", "val_a_auc_opus_16",
        "val_a_auc_narrowband",
        "learning_rate", "epoch_seconds", "train_seconds", "selection_seconds",
        "family_clean", "family_recorded_noise", "family_reverberation", "family_codec",
        "family_gain", "family_rawboost",
        "codec_mp3_24", "codec_mp3_48", "codec_mp3_96", "codec_opus_8", "codec_opus_16",
        "codec_opus_24", "codec_narrowband_mulaw",
    )
    assert SUMMARY_KEYS == ("best_epoch", "best_selection_m", "epochs_run", "peak_gpu_memory_bytes")
    assert len(RUN_KEYS) == 11 and "train_stats_npz_sha256" in RUN_KEYS
    for key in (*CONFIG_KEYS, *EPOCH_KEYS, *SUMMARY_KEYS):
        assert not any(p in key.lower() for p in ("val_b", "val-b", "test", "external")), key


def test_disabled_mode_validates_and_sends_nothing(base_config, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # any import of wandb would fail
    logger = V2RunLogger("disabled", _config(base_config), "cnn-v2-crnn-s0")
    logger.log_epoch(_epoch(train_loss=0.5, selection_m=0.1))
    logger.log_summary({"best_epoch": 1, "best_selection_m": 0.1, "epochs_run": 1,
                        "peak_gpu_memory_bytes": None})
    logger.finish()
    logger.finish()
    with pytest.raises(ValueError, match="already finished"):
        logger.log_epoch(_epoch())


def test_train_stats_null_only_for_aasist(base_config):
    V2RunLogger("disabled", _config(base_config, "aasist"), "r")
    with pytest.raises(ValueError, match="only for aasist"):
        V2RunLogger("disabled", _config(base_config, "cnn-aug", train_stats_npz_sha256=None), "r")
    with pytest.raises(ValueError, match="must be null for aasist"):
        V2RunLogger("disabled", _config(base_config, "aasist", train_stats_npz_sha256="6" * 64), "r")
    with pytest.raises(ValueError, match="must be an int"):
        V2RunLogger("disabled", _config(base_config, selection_set_sha256=None), "r")
    with pytest.raises(ValueError, match="candidate must be one of"):
        V2RunLogger("disabled", _config(base_config, "lstm"), "r")


@pytest.mark.parametrize("changes, match", [
    ({"extra": 1}, "unknown keys"),
    ({"val_b_eer": 0.1}, "val_b"),
    ({"external_eer": 0.1}, "external"),
])
def test_config_refusals(base_config, changes, match):
    with pytest.raises(ValueError, match=match):
        V2RunLogger("disabled", _config(base_config, **changes), "r")
    cfg = _config(base_config)
    del cfg["num_workers"]
    with pytest.raises(ValueError, match="missing keys"):
        V2RunLogger("disabled", cfg, "r")


def test_epoch_and_summary_refusals(base_config):
    logger = V2RunLogger("disabled", _config(base_config), "r")
    bad_epoch = _epoch()
    del bad_epoch["codec_opus_16"]
    for data, match in ((bad_epoch, "missing keys"), ({**_epoch(), "test_eer": 0.1}, "test"),
                        (_epoch(selection_m=float("nan")), "finite"),
                        (_epoch(learning_rate=[0.1]), "int, float or bool"),
                        (_epoch(family_gain=None), "int, float or bool")):
        with pytest.raises(ValueError, match=match):
            logger.log_epoch(data)
    with pytest.raises(ValueError, match="int, float or bool"):
        logger.log_summary({"best_epoch": None, "best_selection_m": 0.1, "epochs_run": 1,
                            "peak_gpu_memory_bytes": None})
    with pytest.raises(ValueError, match="mode must be"):
        V2RunLogger("dryrun", _config(base_config), "r")


def test_committed_config_has_null_batch_sizes_refused(base_config):
    # The committed file keeps null batch sizes until V2b-2; a run config with null is refused.
    committed = load_v2_config(V2_CONFIG_PATH).to_dict()
    with pytest.raises(ValueError, match="must be an int"):
        V2RunLogger("disabled", _config(committed), "r")
