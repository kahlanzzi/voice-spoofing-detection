"""configs/cnn.yaml: the D30 search space and the D42 training settings, validated (Stage 8).

Every field has exactly one implemented value (or list of values); the loader refuses anything
else, including a value of another type (True for 1, 256.0 for 256), so a change to the file needs
a new decision and a code change (drift guard, as configs/baseline.yaml for D37).

NumPy and torch are not needed here, so the config can be read and checked anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# The only configuration the code implements (D30, D42; D40 for the W&B target).
EXPECTED = {
    "decision": ["D30", "D42"],
    "grid": {"n_blocks": [3, 4], "learning_rate": [0.001, 0.0003]},
    "optimizer": "adam",
    "weight_decay": 0.0,
    "loss": "bce_with_logits",
    "class_weights": "none",
    "dropout": 0.3,
    "max_epochs": 30,
    "patience": 5,
    "batch_size": 256,
    "early_stopping": {"metric": "val_a_eer", "improvement": "strictly_lower", "tie": "earliest_epoch"},
    "search_seed": 0,
    "final_seeds": [0, 1, 2],
    "tie_rule": ["n_blocks_ascending", "learning_rate_descending"],
    "wandb": {"entity": "voice-spoofing-detection-must", "project": "voice-spoofing-detection"},
}


def _same(value, expected) -> bool:
    """Equal with the same types at every level (so True != 1 and 256.0 != 256)."""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(_same(value[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return len(value) == len(expected) and all(_same(v, e) for v, e in zip(value, expected))
    return value == expected


def _check_keys(where: str, keys, expected) -> None:
    missing = sorted(set(expected) - set(keys))
    unknown = sorted(set(keys) - set(expected), key=str)
    if missing or unknown:
        raise ValueError(f"{where}: missing keys {missing}, unknown keys {unknown}")


@dataclass(frozen=True)
class CNNConfig:
    """configs/cnn.yaml, validated against D30 + D42."""

    decision: tuple[str, ...]
    n_blocks: tuple[int, ...]
    learning_rate: tuple[float, ...]
    optimizer: str
    weight_decay: float
    loss: str
    class_weights: str
    dropout: float
    max_epochs: int
    patience: int
    batch_size: int
    early_stopping_metric: str
    early_stopping_improvement: str
    early_stopping_tie: str
    search_seed: int
    final_seeds: tuple[int, ...]
    tie_rule: tuple[str, ...]
    wandb_entity: str
    wandb_project: str

    def grid(self) -> list[tuple[int, float]]:
        """The (n_blocks, learning_rate) pairs in grid order: n_blocks outer, learning_rate inner."""
        return [(n, lr) for n in self.n_blocks for lr in self.learning_rate]

    def to_dict(self) -> dict:
        """The config in the cnn.yaml layout (plain lists and dicts), for reports and the W&B config."""
        return {
            "decision": list(self.decision),
            "grid": {"n_blocks": list(self.n_blocks), "learning_rate": list(self.learning_rate)},
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "loss": self.loss,
            "class_weights": self.class_weights,
            "dropout": self.dropout,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "early_stopping": {"metric": self.early_stopping_metric,
                               "improvement": self.early_stopping_improvement,
                               "tie": self.early_stopping_tie},
            "search_seed": self.search_seed,
            "final_seeds": list(self.final_seeds),
            "tie_rule": list(self.tie_rule),
            "wandb": {"entity": self.wandb_entity, "project": self.wandb_project},
        }


def load_cnn_config(path) -> CNNConfig:
    """Read configs/cnn.yaml with yaml.safe_load; strict keys and values; raise ValueError if invalid."""
    path = Path(path)
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read CNN config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must hold a YAML mapping")
    _check_keys(str(path), cfg.keys(), EXPECTED)
    for key, expected in EXPECTED.items():
        if isinstance(expected, dict) and isinstance(cfg[key], dict):
            _check_keys(key, cfg[key].keys(), expected)
        if not _same(cfg[key], expected):
            raise ValueError(f"{key} must be {expected!r} (D30, D42; only this value is implemented), "
                             f"got {cfg[key]!r}")
    return CNNConfig(
        decision=tuple(cfg["decision"]),
        n_blocks=tuple(cfg["grid"]["n_blocks"]),
        learning_rate=tuple(cfg["grid"]["learning_rate"]),
        optimizer=cfg["optimizer"],
        weight_decay=cfg["weight_decay"],
        loss=cfg["loss"],
        class_weights=cfg["class_weights"],
        dropout=cfg["dropout"],
        max_epochs=cfg["max_epochs"],
        patience=cfg["patience"],
        batch_size=cfg["batch_size"],
        early_stopping_metric=cfg["early_stopping"]["metric"],
        early_stopping_improvement=cfg["early_stopping"]["improvement"],
        early_stopping_tie=cfg["early_stopping"]["tie"],
        search_seed=cfg["search_seed"],
        final_seeds=tuple(cfg["final_seeds"]),
        tie_rule=tuple(cfg["tie_rule"]),
        wandb_entity=cfg["wandb"]["entity"],
        wandb_project=cfg["wandb"]["project"],
    )
