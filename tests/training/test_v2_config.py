"""configs/v2_candidates.yaml drift guard (D49, D61) and the D61 (3) cosine schedule. Torch-free,
except the one test that compares the config with the model constants."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import yaml

from src.training.v2_config import EXPECTED, load_v2_config
from src.training.v2_schedule import cosine_lr
from tests._torch import require_torch

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "v2_candidates.yaml"
NAMES = ("cnn-aug", "crnn", "aasist")

# D49 + D61 written out by hand, independent of EXPECTED in the module; batch sizes still null.
D49_D61 = {
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
            "batch_size": None,
        },
        "crnn": {
            "conv_channels": [16, 32, 64],
            "lstm": {"hidden": 64, "bidirectional": True, "layers": 1},
            "pooling": "mean_over_time",
            "dropout": 0.3,
            "optimizer": {"name": "adam", "learning_rate": 0.001, "weight_decay": 0.0},
            "scheduler": "none",
            "batch_size": None,
        },
        "aasist": {
            "reference": {"repository": "https://github.com/clovaai/aasist",
                          "commit": "a04c9863f63d44471dde8a6abcb3b082b07cd1d1", "licence": "MIT"},
            "model_config": {"architecture": "AASIST", "nb_samp": 32000, "first_conv": 128,
                             "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                             "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5, 0.5],
                             "temperatures": [2.0, 2.0, 100.0, 100.0]},
            "optimizer": {"name": "adam", "learning_rate": 0.0001, "betas": [0.9, 0.999],
                          "weight_decay": 0.0001, "amsgrad": False},
            "scheduler": {"name": "cosine_per_step", "lr_min": 0.000005, "total_epochs": 30},
            "swa": False,
            "freq_aug": False,
            "batch_size": None,
        },
    },
}


def _write(tmp_path, cfg) -> Path:
    path = tmp_path / "v2_candidates.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _set(cfg, where, value):
    target = cfg
    for key in where[:-1]:
        target = target[key]
    target[where[-1]] = value


def test_committed_config_equals_d49_d61():
    config = load_v2_config(CONFIG_PATH)
    assert config.to_dict() == D49_D61
    assert yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) == D49_D61
    expected_no_bs = copy.deepcopy(D49_D61)
    for name in NAMES:
        del expected_no_bs["candidates"][name]["batch_size"]
    assert expected_no_bs == EXPECTED
    assert config.candidate_order == NAMES
    assert config.settings["search_seed"] in config.settings["final_seeds"]


def test_config_matches_model_constants():
    # The config must describe the models the code builds, not a parallel copy that can drift.
    require_torch()
    from src.models.aasist import MODEL_CONFIG
    from src.models.crnn import CHANNELS, DROPOUT, LSTM_HIDDEN
    from src.models.v2_checkpoint import ARCHITECTURES, CNN_AUG_BLOCKS, MAX_EPOCHS

    config = load_v2_config(CONFIG_PATH)
    assert config.candidate_order == ARCHITECTURES
    assert config.settings["max_epochs"] == MAX_EPOCHS
    assert config.candidate("cnn-aug")["n_blocks"] == CNN_AUG_BLOCKS
    crnn = config.candidate("crnn")
    assert tuple(crnn["conv_channels"]) == CHANNELS
    assert crnn["lstm"]["hidden"] == LSTM_HIDDEN and crnn["dropout"] == DROPOUT
    assert config.candidate("aasist")["model_config"] == MODEL_CONFIG


def test_require_batch_sizes_raises_while_null(tmp_path):
    # Training code cannot start before the D49 benchmark writes every batch size.
    config = load_v2_config(CONFIG_PATH)
    assert config.batch_sizes() == {"cnn-aug": None, "crnn": None, "aasist": None}
    with pytest.raises(ValueError, match=r"batch_size is null for \['cnn-aug', 'crnn', 'aasist'\]"):
        config.require_batch_sizes()
    cfg = copy.deepcopy(D49_D61)
    cfg["candidates"]["cnn-aug"]["batch_size"] = 256
    cfg["candidates"]["crnn"]["batch_size"] = 128
    partial = load_v2_config(_write(tmp_path, cfg))
    with pytest.raises(ValueError, match=r"\['aasist'\]"):
        partial.require_batch_sizes()
    cfg["candidates"]["aasist"]["batch_size"] = 24
    full = load_v2_config(_write(tmp_path, cfg))
    assert full.require_batch_sizes() == {"cnn-aug": 256, "crnn": 128, "aasist": 24}


@pytest.mark.parametrize("value", [0, -8, 32.0, True, "64"])
def test_invalid_batch_size_is_refused(tmp_path, value):
    cfg = copy.deepcopy(D49_D61)
    cfg["candidates"]["crnn"]["batch_size"] = value
    with pytest.raises(ValueError, match="batch_size"):
        load_v2_config(_write(tmp_path, cfg))


def _changes():
    """(label, path into the mapping, new value): each must be refused."""
    return [
        ("decision", ("decision",), ["D49", "D50"]),
        ("max_epochs", ("max_epochs",), 31),
        ("max_epochs as float", ("max_epochs",), 30.0),
        ("patience", ("patience",), 6),
        ("loss", ("loss",), "cce"),
        ("class_weights", ("class_weights",), [0.1, 0.9]),
        ("dtype", ("dtype",), "float16"),
        ("early_stopping.metric", ("early_stopping", "metric"), "val_a_eer"),
        ("early_stopping.tie", ("early_stopping", "tie"), "latest_epoch"),
        ("selection_metric", ("selection_metric",), "clean_val_a_eer"),
        ("search_seed as bool", ("search_seed",), False),
        ("final_seeds subset", ("final_seeds",), [0, 1]),
        ("tie_rule reordered", ("tie_rule",), ["candidate_order", "trainable_parameters_ascending"]),
        ("candidate_order reordered", ("candidate_order",), ["crnn", "cnn-aug", "aasist"]),
        ("logmel_stats augmented", ("logmel_stats",), "augmented_train"),
        ("wandb.project", ("wandb", "project"), "other"),
        ("cnn-aug n_blocks 3", ("candidates", "cnn-aug", "n_blocks"), 3),
        ("cnn-aug learning_rate", ("candidates", "cnn-aug", "optimizer", "learning_rate"), 0.0003),
        ("cnn-aug weight_decay int", ("candidates", "cnn-aug", "optimizer", "weight_decay"), 0),
        ("cnn-aug scheduler", ("candidates", "cnn-aug", "scheduler"), "cosine"),
        ("crnn channels", ("candidates", "crnn", "conv_channels"), [16, 32, 64, 128]),
        ("crnn lstm hidden", ("candidates", "crnn", "lstm", "hidden"), 128),
        ("crnn unidirectional", ("candidates", "crnn", "lstm", "bidirectional"), False),
        ("crnn lstm layers", ("candidates", "crnn", "lstm", "layers"), 2),
        ("crnn pooling", ("candidates", "crnn", "pooling"), "last_step"),
        ("crnn dropout", ("candidates", "crnn", "dropout"), 0.5),
        ("aasist commit", ("candidates", "aasist", "reference", "commit"), "0" * 40),
        ("aasist nb_samp reference", ("candidates", "aasist", "model_config", "nb_samp"), 64600),
        ("aasist filts", ("candidates", "aasist", "model_config", "filts"),
         [70, [1, 32], [32, 32], [32, 64], [64, 128]]),
        ("aasist temperatures int", ("candidates", "aasist", "model_config", "temperatures"),
         [2, 2, 100, 100]),
        ("aasist lr", ("candidates", "aasist", "optimizer", "learning_rate"), 0.001),
        ("aasist lr as string", ("candidates", "aasist", "optimizer", "learning_rate"), "1e-4"),
        ("aasist betas", ("candidates", "aasist", "optimizer", "betas"), [0.9, 0.98]),
        ("aasist amsgrad", ("candidates", "aasist", "optimizer", "amsgrad"), True),
        ("aasist lr_min", ("candidates", "aasist", "scheduler", "lr_min"), 0.0),
        ("aasist total_epochs 100", ("candidates", "aasist", "scheduler", "total_epochs"), 100),
        ("aasist swa", ("candidates", "aasist", "swa"), True),
        ("aasist freq_aug", ("candidates", "aasist", "freq_aug"), True),
    ]


@pytest.mark.parametrize("label,where,value", _changes(), ids=[c[0] for c in _changes()])
def test_every_changed_value_is_refused(tmp_path, label, where, value):
    cfg = copy.deepcopy(D49_D61)
    _set(cfg, where, value)
    with pytest.raises(ValueError):
        load_v2_config(_write(tmp_path, cfg))


@pytest.mark.parametrize("change", [
    "missing", "unknown", "nested_missing", "nested_unknown", "candidate_missing",
    "candidate_unknown", "candidate_key_missing", "candidate_key_unknown", "batch_size_missing",
])
def test_missing_or_unknown_keys_are_refused(tmp_path, change):
    cfg = copy.deepcopy(D49_D61)
    cands = cfg["candidates"]
    if change == "missing":
        del cfg["patience"]
    elif change == "unknown":
        cfg["momentum"] = 0.9
    elif change == "nested_missing":
        del cfg["early_stopping"]["tie"]
    elif change == "nested_unknown":
        cfg["wandb"]["group"] = "v2"
    elif change == "candidate_missing":
        del cands["aasist"]
    elif change == "candidate_unknown":
        cands["wav2vec2"] = copy.deepcopy(cands["crnn"])  # scope lock: only D49's three
    elif change == "candidate_key_missing":
        del cands["crnn"]["pooling"]
    elif change == "candidate_key_unknown":
        cands["aasist"]["model_config"]["dropout"] = 0.5
    else:
        del cands["cnn-aug"]["batch_size"]
    with pytest.raises(ValueError, match="keys"):
        load_v2_config(_write(tmp_path, cfg))


def test_unreadable_or_non_mapping_config_is_refused(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        load_v2_config(tmp_path / "missing.yaml")
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_v2_config(path)
    cfg = copy.deepcopy(D49_D61)
    cfg["candidates"] = ["cnn-aug", "crnn", "aasist"]
    with pytest.raises(ValueError, match="mapping"):
        load_v2_config(_write(tmp_path, cfg))


def test_candidate_returns_a_copy():
    config = load_v2_config(CONFIG_PATH)
    config.candidate("crnn")["dropout"] = 0.9
    assert config.candidate("crnn")["dropout"] == 0.3
    with pytest.raises(ValueError, match="unknown candidate"):
        config.candidate("rawnet")


# Cosine schedule (D61 (3)) -----------------------------------------------------------------------

def test_cosine_lr_endpoints_and_midpoint():
    # Reference formula: base at step 0, minimum at total_steps, their mean halfway.
    base, minimum, total = 0.0001, 0.000005, 30 * 100
    assert cosine_lr(0, total, base, minimum) == pytest.approx(base, rel=1e-12)
    assert cosine_lr(total, total, base, minimum) == pytest.approx(minimum, rel=1e-12)
    assert cosine_lr(total // 2, total, base, minimum) == pytest.approx((base + minimum) / 2, rel=1e-12)
    quarter = minimum + (base - minimum) * 0.5 * (1 + math.cos(math.pi / 4))
    assert cosine_lr(total // 4, total, base, minimum) == pytest.approx(quarter, rel=1e-12)


def test_cosine_lr_decreases_monotonically():
    values = [cosine_lr(s, 50, 1.0, 0.0) for s in range(51)]
    assert all(a > b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("args", [
    (-1, 10, 1.0, 0.0), (11, 10, 1.0, 0.0), (1.0, 10, 1.0, 0.0), (0, 0, 1.0, 0.0),
    (0, True, 1.0, 0.0), (0, 10, 0.1, 0.2), (0, 10, float("nan"), 0.0), (0, 10, 1.0, -0.1),
])
def test_cosine_lr_refuses_invalid_arguments(args):
    with pytest.raises(ValueError):
        cosine_lr(*args)
