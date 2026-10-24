"""Allow-list W&B logger for CNN runs (D40, D42 "W&B").

W&B receives FoR training and validation metrics and the run config only (D40): per epoch the train
loss, val-A EER, val-A ROC-AUC and epoch time; at the end the best epoch and its val-A EER (D42).
Everything is checked before anything is sent, and a refused call raises ValueError:
- only the keys listed here (config: the cnn.yaml fields plus the run provenance keys);
- metric values must be finite scalars (bool, int, float); arrays, tensors, lists, images, tables
  and other media are refused. Config values may also be lists and mappings of such scalars (the
  cnn.yaml layout), but never arrays, tensors or media;
- no key (at any level) containing "val_b", "val-b", "test" or "external", so a val-B, test-split or
  DEEP-VOICE number cannot be logged even if a later allow-list edit tried to.
No file is ever uploaded: the module never calls wandb.save, wandb.Artifact or log_artifact, and
the run is started with code saving, console capture and requirements upload turned off.

wandb is imported only in "online" or "offline" mode; "disabled" validates and sends nothing.
The API key is read by wandb from the environment (WANDB_API_KEY, from .env); never from code.
"""
from __future__ import annotations

import math

from src.training.cnn_config import EXPECTED

MODES = ("online", "offline", "disabled")
CONFIG_KEYS = frozenset(EXPECTED) | {
    "seed", "n_blocks", "learning_rate", "code_commit", "profile_fingerprint",
    "f2_manifest_sha256", "train_cache_npz_sha256", "val_a_cache_npz_sha256",
}
EPOCH_KEYS = ("epoch", "train_loss", "val_a_eer", "val_a_auc", "epoch_seconds")
SUMMARY_KEYS = ("best_epoch", "best_val_a_eer", "epochs_run", "peak_gpu_memory_bytes")
NULLABLE_SUMMARY_KEYS = frozenset({"peak_gpu_memory_bytes"})  # None on CPU
FORBIDDEN_KEY_PARTS = ("val_b", "val-b", "test", "external")
# Turn off the code, console and requirements files wandb would otherwise send at init.
SETTINGS = {"save_code": False, "disable_code": True, "console": "off",
            "x_save_requirements": False, "disable_job_creation": True}


def _check_key(key) -> None:
    if not isinstance(key, str):
        raise ValueError(f"W&B key must be a string, got {key!r}")
    low = key.lower()
    for part in FORBIDDEN_KEY_PARTS:
        if part in low:
            raise ValueError(f"W&B key {key!r} contains {part!r}: val-B, test and external data are "
                             "never logged (D40, D42)")


def _check_scalar(key: str, value, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if isinstance(value, bool):
        return
    # int / float (and their subclasses such as numpy.float64); arrays, tensors and media are not.
    if not isinstance(value, (int, float)):
        raise ValueError(f"W&B value for {key!r} must be an int, float or bool, got "
                         f"{type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"W&B value for {key!r} must be finite, got {value!r}")


def _check_config_value(key: str, value) -> None:
    """A str, finite scalar, or list / mapping of those (recursively), with allowed nested keys."""
    if isinstance(value, str):
        return
    if isinstance(value, list):
        for item in value:
            _check_config_value(key, item)
        return
    if isinstance(value, dict):
        for sub, item in value.items():
            _check_key(sub)
            _check_config_value(f"{key}.{sub}", item)
        return
    _check_scalar(key, value)


def _check_exact(where: str, data: dict, keys) -> None:
    missing = sorted(set(keys) - set(data))
    unknown = sorted(set(data) - set(keys), key=str)
    if missing or unknown:
        raise ValueError(f"W&B {where}: missing keys {missing}, unknown keys {unknown}")


class RunLogger:
    """One W&B run: config at start, log_epoch per epoch, log_summary at the end, finish() closes it."""

    def __init__(self, mode: str, config_dict: dict, run_name: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if not isinstance(run_name, str) or not run_name.strip():
            raise ValueError(f"run_name must be a non-empty string, got {run_name!r}")
        if not isinstance(config_dict, dict):
            raise ValueError(f"config_dict must be a dict, got {type(config_dict).__name__}")
        for key in config_dict:
            _check_key(key)
        _check_exact("config", config_dict, CONFIG_KEYS)
        for key, value in config_dict.items():
            _check_config_value(key, value)
        self.mode = mode
        self.run_name = run_name
        self._run = None
        self._finished = False
        if mode != "disabled":
            import wandb  # lazy: only online / offline runs need it

            wandb_cfg = config_dict["wandb"]
            self._run = wandb.init(
                entity=wandb_cfg["entity"], project=wandb_cfg["project"], name=run_name,
                mode=mode, config=dict(config_dict), settings=wandb.Settings(**SETTINGS),
            )

    def _check_open(self) -> None:
        if self._finished:
            raise ValueError(f"W&B run {self.run_name!r} is already finished")

    def _send(self, where: str, data: dict, keys, nullable=frozenset()) -> dict:
        self._check_open()
        for key in data:
            _check_key(key)
        _check_exact(where, data, keys)
        for key, value in data.items():
            _check_scalar(key, value, nullable=key in nullable)
        return data

    def log_epoch(self, epoch, train_loss, val_a_eer, val_a_auc, epoch_seconds) -> None:
        """Send one epoch's train loss, val-A EER, val-A ROC-AUC and epoch time (D42)."""
        data = self._send("epoch", {"epoch": epoch, "train_loss": train_loss, "val_a_eer": val_a_eer,
                                    "val_a_auc": val_a_auc, "epoch_seconds": epoch_seconds},
                          EPOCH_KEYS)
        if self._run is not None:
            self._run.log(data)

    def log_summary(self, best_epoch, best_val_a_eer, epochs_run, peak_gpu_memory_bytes) -> None:
        """Send the final best epoch, its val-A EER, the epochs run and the peak GPU memory (or None)."""
        data = self._send("summary", {"best_epoch": best_epoch, "best_val_a_eer": best_val_a_eer,
                                      "epochs_run": epochs_run,
                                      "peak_gpu_memory_bytes": peak_gpu_memory_bytes},
                          SUMMARY_KEYS, NULLABLE_SUMMARY_KEYS)
        if self._run is not None:
            self._run.summary.update(data)

    def finish(self) -> None:
        """Close the run; calling it again does nothing."""
        if self._finished:
            return
        self._finished = True
        if self._run is not None:
            self._run.finish()
