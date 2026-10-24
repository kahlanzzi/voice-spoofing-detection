"""configs/cnn.yaml drift guard (D30, D42): the file must equal the approved values exactly."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.training.cnn_config import EXPECTED, load_cnn_config
from tests._torch import require_torch

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "cnn.yaml"

# D30 + D42 written out by hand, independent of EXPECTED in the module.
D30_D42 = {
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


def test_loaded_config_equals_d30_d42():
    config = load_cnn_config(CONFIG_PATH)
    assert config.to_dict() == D30_D42 == EXPECTED
    assert yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) == D30_D42
    assert config.grid() == [(3, 0.001), (3, 0.0003), (4, 0.001), (4, 0.0003)]
    assert config.search_seed in config.final_seeds


def test_config_matches_model_and_checkpoint_constants():
    require_torch()
    from src.models.cnn import DROPOUT, N_BLOCKS
    from src.models.cnn_checkpoint import MAX_EPOCHS

    config = load_cnn_config(CONFIG_PATH)
    assert config.n_blocks == N_BLOCKS
    assert config.dropout == DROPOUT
    assert config.max_epochs == MAX_EPOCHS


def _changes():
    """(label, path into the mapping, new value) for every field: each must be refused."""
    out = []
    for key, value in D30_D42.items():
        if isinstance(value, dict):
            for sub, v in value.items():
                other = v + ["x"] if isinstance(v, list) else (v + "x" if isinstance(v, str) else None)
                out.append((f"{key}.{sub}", (key, sub), other))
        elif isinstance(value, list):
            out.append((key, (key,), value[::-1]))
        elif isinstance(value, str):
            out.append((key, (key,), value + "x"))
        elif isinstance(value, float):
            out.append((key, (key,), value + 0.1))
        else:
            out.append((key, (key,), value + 1))
    # Same value, other type: must also be refused.
    out += [
        ("max_epochs as float", ("max_epochs",), 30.0),
        ("batch_size as float", ("batch_size",), 256.0),
        ("search_seed as bool", ("search_seed",), False),
        ("weight_decay as int", ("weight_decay",), 0),
        ("grid.n_blocks as floats", ("grid", "n_blocks"), [3.0, 4.0]),
        ("learning_rate reordered", ("grid", "learning_rate"), [0.0003, 0.001]),
        ("learning_rate as string", ("grid", "learning_rate"), ["1e-3", "3e-4"]),
        ("class_weights null", ("class_weights",), None),
        ("final_seeds subset", ("final_seeds",), [0, 1]),
    ]
    return out


@pytest.mark.parametrize("label,where,value", _changes(), ids=[c[0] for c in _changes()])
def test_every_changed_field_is_refused(tmp_path, label, where, value):
    cfg = copy.deepcopy(D30_D42)
    target = cfg
    for key in where[:-1]:
        target = target[key]
    target[where[-1]] = value
    path = tmp_path / "cnn.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cnn_config(path)


@pytest.mark.parametrize("change", ["missing", "unknown", "nested_missing", "nested_unknown"])
def test_missing_or_unknown_keys_are_refused(tmp_path, change):
    cfg = copy.deepcopy(D30_D42)
    if change == "missing":
        del cfg["patience"]
    elif change == "unknown":
        cfg["momentum"] = 0.9
    elif change == "nested_missing":
        del cfg["early_stopping"]["tie"]
    else:
        cfg["grid"]["dropout"] = [0.3]
    path = tmp_path / "cnn.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="keys"):
        load_cnn_config(path)


def test_unreadable_or_non_mapping_config_is_refused(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        load_cnn_config(tmp_path / "missing.yaml")
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_cnn_config(path)
