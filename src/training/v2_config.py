"""configs/v2_candidates.yaml: the D49 candidates and training settings with D61, validated (Track V).

Every field has exactly one implemented value; the loader refuses anything else, including a value
of another type (True for 1, 30.0 for 30), so a change to the file needs a new decision and a code
change (drift guard, as configs/cnn.yaml for D30 + D42).

The one open field is each candidate's batch_size: null until the D49 compute benchmark (PR V2b)
writes a positive int. The loader accepts null; require_batch_sizes() raises while any is null, so
training code cannot start without them.

NumPy and torch are not needed here, so the config can be read and checked anywhere.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.training.cnn_config import _check_keys, _same

BATCH_SIZE = "batch_size"  # the only key not fixed by EXPECTED

# The only configuration the code implements (D49, D50, D61; D40 for the W&B target).
# Candidate entries list every key except batch_size, which is checked on its own.
EXPECTED = {
    "decision": ["D49", "D50", "D61"],
    "max_epochs": 30,
    "patience": 5,
    "loss": "bce_with_logits",
    "class_weights": "none",
    "dtype": "float32",
    "early_stopping": {"metric": "selection_m", "improvement": "strictly_lower",
                       "tie": "earliest_epoch"},
    "selection_metric": "mean_val_a_eer_over_S",
    "search_seed": 0,
    "final_seeds": [0, 1, 2],
    "tie_rule": ["trainable_parameters_ascending", "candidate_order"],
    "candidate_order": ["cnn-aug", "crnn", "aasist"],
    "logmel_stats": "clean_train",
    "wandb": {"entity": "voice-spoofing-detection-must", "project": "voice-spoofing-detection"},
    "candidates": {
        "cnn-aug": {
            "n_blocks": 4,
            "optimizer": {"name": "adam", "learning_rate": 0.001, "weight_decay": 0.0},
            "scheduler": "none",
        },
        "crnn": {
            "conv_channels": [16, 32, 64],
            "lstm": {"hidden": 64, "bidirectional": True, "layers": 1},
            "pooling": "mean_over_time",
            "dropout": 0.3,
            "optimizer": {"name": "adam", "learning_rate": 0.001, "weight_decay": 0.0},
            "scheduler": "none",
        },
        "aasist": {
            "reference": {"repository": "https://github.com/clovaai/aasist",
                          "commit": "a04c9863f63d44471dde8a6abcb3b082b07cd1d1",
                          "licence": "MIT"},
            "model_config": {
                "architecture": "AASIST",
                "nb_samp": 32000,
                "first_conv": 128,
                "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                "gat_dims": [64, 32],
                "pool_ratios": [0.5, 0.7, 0.5, 0.5],
                "temperatures": [2.0, 2.0, 100.0, 100.0],
            },
            "optimizer": {"name": "adam", "learning_rate": 0.0001, "betas": [0.9, 0.999],
                          "weight_decay": 0.0001, "amsgrad": False},
            "scheduler": {"name": "cosine_per_step", "lr_min": 0.000005, "total_epochs": 30},
            "swa": False,
            "freq_aug": False,
        },
    },
}


def _valid_batch_size(value) -> bool:
    return value is None or (type(value) is int and value > 0)


def _check(where: str, value, expected) -> None:
    """Exact keys at every mapping level, then exact values and types."""
    if isinstance(expected, dict) and isinstance(value, dict):
        _check_keys(where, value.keys(), expected)
        for key in expected:
            _check(f"{where}.{key}", value[key], expected[key])
        return
    if not _same(value, expected):
        raise ValueError(f"{where} must be {expected!r} (D49, D61; only this value is implemented), "
                         f"got {value!r}")


@dataclass(frozen=True)
class V2Config:
    """configs/v2_candidates.yaml, validated against D49 + D61."""

    settings: dict  # every field except candidates (a deep copy; read with to_dict)
    candidates: dict  # name -> candidate settings incl. batch_size (int or None)

    @property
    def candidate_order(self) -> tuple[str, ...]:
        return tuple(self.settings["candidate_order"])

    def candidate(self, name: str) -> dict:
        """One candidate's settings (a copy), including batch_size."""
        if name not in self.candidates:
            raise ValueError(f"unknown candidate {name!r}; candidates are {self.candidate_order}")
        return copy.deepcopy(self.candidates[name])

    def batch_sizes(self) -> dict[str, int | None]:
        return {name: self.candidates[name][BATCH_SIZE] for name in self.candidate_order}

    def require_batch_sizes(self) -> dict[str, int]:
        """The batch size per candidate; raise ValueError while any is null (D49 benchmark not done)."""
        sizes = self.batch_sizes()
        unset = [name for name, size in sizes.items() if size is None]
        if unset:
            raise ValueError(f"batch_size is null for {unset}: the D49 compute benchmark must write "
                             f"it to configs/v2_candidates.yaml before training")
        return sizes

    def to_dict(self) -> dict:
        """The config in the v2_candidates.yaml layout (plain lists and dicts)."""
        return {**copy.deepcopy(self.settings), "candidates": copy.deepcopy(self.candidates)}


def load_v2_config(path) -> V2Config:
    """Read configs/v2_candidates.yaml with yaml.safe_load; strict keys and values; raise ValueError."""
    path = Path(path)
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read v2 candidates config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must hold a YAML mapping")
    _check_keys(str(path), cfg.keys(), EXPECTED)
    candidates = cfg["candidates"]
    if not isinstance(candidates, dict):
        raise ValueError(f"candidates must be a mapping, got {candidates!r}")
    _check_keys("candidates", candidates.keys(), EXPECTED["candidates"])
    for name, expected in EXPECTED["candidates"].items():
        entry = candidates[name]
        if not isinstance(entry, dict):
            raise ValueError(f"candidates.{name} must be a mapping, got {entry!r}")
        _check_keys(f"candidates.{name}", entry.keys(), {**expected, BATCH_SIZE: None})
        if not _valid_batch_size(entry[BATCH_SIZE]):
            raise ValueError(f"candidates.{name}.batch_size must be null or an int > 0 (D49 "
                             f"benchmark), got {entry[BATCH_SIZE]!r}")
        _check(f"candidates.{name}", {k: v for k, v in entry.items() if k != BATCH_SIZE}, expected)
    for key, expected in EXPECTED.items():
        if key != "candidates":
            _check(key, cfg[key], expected)
    return V2Config(
        settings={k: copy.deepcopy(v) for k, v in cfg.items() if k != "candidates"},
        candidates=copy.deepcopy(candidates),
    )
