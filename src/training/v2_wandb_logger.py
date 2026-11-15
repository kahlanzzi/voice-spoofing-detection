"""Allow-list W&B logger for model v2 runs (D40, D49).

Same guarantees as src.training.wandb_logger.RunLogger, whose check functions and settings are
reused (the v1 logger is unchanged): exact keys, finite scalars only, no media, no file upload, and
no key containing "val_b", "val-b", "test" or "external" at any level. Only FoR train and val-A
(through the selection set S) numbers are logged.

- config: every configs/v2_candidates.yaml field plus the run keys (RUN_KEYS).
  train_stats_npz_sha256 may be null only for aasist (no log-Mel statistics); nothing else is null.
- per epoch (EPOCH_KEYS): train loss, M, val-A EER and ROC-AUC per S condition, learning rate,
  timings, augmentation family counts and codec option counts.
- summary (SUMMARY_KEYS): best epoch, best M, epochs run, peak GPU memory (null on CPU).

wandb is imported only in "online" or "offline" mode; "disabled" validates and sends nothing.
"""
from __future__ import annotations

from src.models.v2_checkpoint import ARCHITECTURES, INPUTS
from src.training.selection_set import CONDITIONS
from src.training.v2_config import EXPECTED
from src.training.v2_data import CODEC_OPTION_IDS, COUNT_FAMILIES
from src.training.wandb_logger import (
    FORBIDDEN_KEY_PARTS,
    MODES,
    SETTINGS,
    _check_config_value,
    _check_exact,
    _check_key,
    _check_scalar,
)

RUN_KEYS = frozenset({
    "candidate", "seed", "batch_size", "num_workers", "code_commit", "profile_fingerprint",
    "f2_manifest_sha256", "raw_train_cache_npz_sha256", "selection_set_sha256",
    "augmentation_config_sha256", "train_stats_npz_sha256",
})
CONFIG_KEYS = frozenset(EXPECTED) | RUN_KEYS
EPOCH_KEYS = (
    "epoch", "train_loss", "selection_m",
    *(f"val_a_eer_{c}" for c in CONDITIONS),
    *(f"val_a_auc_{c}" for c in CONDITIONS),
    "learning_rate", "epoch_seconds", "train_seconds", "selection_seconds",
    *(f"family_{f}" for f in COUNT_FAMILIES),
    *(f"codec_{o}" for o in CODEC_OPTION_IDS),
)
SUMMARY_KEYS = ("best_epoch", "best_selection_m", "epochs_run", "peak_gpu_memory_bytes")
NULLABLE_SUMMARY_KEYS = frozenset({"peak_gpu_memory_bytes"})

__all__ = ["CONFIG_KEYS", "EPOCH_KEYS", "FORBIDDEN_KEY_PARTS", "MODES", "RUN_KEYS",
           "SUMMARY_KEYS", "V2RunLogger"]


def _check_config(config: dict) -> None:
    for key in config:
        _check_key(key)
    _check_exact("config", config, CONFIG_KEYS)
    candidate = config["candidate"]
    if candidate not in ARCHITECTURES:
        raise ValueError(f"W&B config candidate must be one of {ARCHITECTURES}, got {candidate!r}")
    stats = config["train_stats_npz_sha256"]
    if stats is None and INPUTS[candidate] != "waveform":
        raise ValueError(f"W&B config train_stats_npz_sha256 may be null only for aasist, not "
                         f"{candidate!r}")
    if stats is not None and INPUTS[candidate] == "waveform":
        raise ValueError("W&B config train_stats_npz_sha256 must be null for aasist")
    for key, value in config.items():
        if key == "train_stats_npz_sha256" and value is None:
            continue
        _check_config_value(key, value)


class V2RunLogger:
    """One v2 W&B run: config at start, log_epoch per epoch, log_summary at the end, then finish()."""

    def __init__(self, mode: str, config_dict: dict, run_name: str, *,
                 resume_id: str | None = None) -> None:
        """resume_id: re-open that existing run (wandb.init(id=..., resume="must")) after a
        per-epoch resume; refused in "disabled" mode, where there is no run."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if resume_id is not None and (mode == "disabled" or not isinstance(resume_id, str)
                                      or not resume_id.strip()):
            raise ValueError(f"resume_id must be a non-empty string in online/offline mode, got "
                             f"{resume_id!r} in mode {mode!r}")
        if not isinstance(run_name, str) or not run_name.strip():
            raise ValueError(f"run_name must be a non-empty string, got {run_name!r}")
        if not isinstance(config_dict, dict):
            raise ValueError(f"config_dict must be a dict, got {type(config_dict).__name__}")
        _check_config(config_dict)
        self.mode = mode
        self.run_name = run_name
        self._run = None
        self._finished = False
        if mode != "disabled":
            import wandb  # lazy: only online / offline runs need it

            target = config_dict["wandb"]
            resume = {} if resume_id is None else {"id": resume_id, "resume": "must"}
            self._run = wandb.init(
                entity=target["entity"], project=target["project"], name=run_name, mode=mode,
                config=dict(config_dict), settings=wandb.Settings(**SETTINGS), **resume,
            )

    @property
    def run_id(self) -> str | None:
        """The W&B run id (stored in the resume state); None in "disabled" mode."""
        return None if self._run is None else str(self._run.id)

    def _send(self, where: str, data: dict, keys, nullable=frozenset()) -> dict:
        if self._finished:
            raise ValueError(f"W&B run {self.run_name!r} is already finished")
        if not isinstance(data, dict):
            raise ValueError(f"W&B {where} must be a dict, got {type(data).__name__}")
        for key in data:
            _check_key(key)
        _check_exact(where, data, keys)
        for key, value in data.items():
            _check_scalar(key, value, nullable=key in nullable)
        return dict(data)

    def log_epoch(self, data: dict) -> None:
        """Send one epoch: exactly EPOCH_KEYS, finite scalars."""
        data = self._send("epoch", data, EPOCH_KEYS)
        if self._run is not None:
            self._run.log(data)

    def log_summary(self, data: dict) -> None:
        """Send the run summary: exactly SUMMARY_KEYS (peak_gpu_memory_bytes may be None)."""
        data = self._send("summary", data, SUMMARY_KEYS, NULLABLE_SUMMARY_KEYS)
        if self._run is not None:
            self._run.summary.update(data)

    def finish(self) -> None:
        """Close the run; calling it again does nothing."""
        if self._finished:
            return
        self._finished = True
        if self._run is not None:
            self._run.finish()
