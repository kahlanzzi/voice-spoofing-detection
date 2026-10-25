"""train_one (D30, D42): early stopping, train-only statistics, no val-B / test input, CPU determinism.

Synthetic tensors only (seed 2026): 24 train and 12 val-A clips of shape (1, 64, 197).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from tests._torch import require_torch

torch = require_torch()

import src.training.train_cnn as train_module  # noqa: E402
from src.training.train_cnn import EpochRecord, TrainResult, train_one  # noqa: E402
from src.training.wandb_logger import RunLogger  # noqa: E402


def _data(n_train: int = 24, n_val: int = 12):
    rng = np.random.default_rng(2026)
    train_y = np.array([0, 1] * (n_train // 2), dtype=np.int8)
    val_y = np.array([0, 1] * (n_val // 2), dtype=np.int8)
    train_x = rng.normal(-5.0, 2.0, (n_train, 1, 64, 197)).astype(np.float32)
    val_x = rng.normal(-5.0, 2.0, (n_val, 1, 64, 197)).astype(np.float32)
    train_x[train_y == 1] += 0.5  # a weak class difference so training has something to learn
    val_x[val_y == 1] += 0.5
    return train_x, train_y, val_x, val_y


class RecordingLogger:
    def __init__(self):
        self.epochs, self.summaries = [], []

    def log_epoch(self, *args):
        self.epochs.append(args)

    def log_summary(self, *args):
        self.summaries.append(args)


def _train(data=None, *, logger=None, **kw):
    train_x, train_y, val_x, val_y = data or _data()
    params = {"n_blocks": 3, "learning_rate": 0.001, "seed": 0, "max_epochs": 2, "patience": 2,
              "batch_size": 8}
    params.update(kw)
    return train_one(train_x, train_y, val_x, val_y, logger=logger or RecordingLogger(), **params)


@pytest.fixture()
def scripted(monkeypatch):
    """Replace eer with a scripted sequence and snapshot the model's weights at every val-A scoring."""
    state = {"script": [], "snapshots": [], "model": None}
    real_build = train_module.build_cnn

    def build(n_blocks):
        state["model"] = real_build(n_blocks)
        return state["model"]

    def fake_eer(labels, scores):
        state["snapshots"].append({k: v.detach().clone() for k, v in state["model"].state_dict().items()})
        return state["script"][len(state["snapshots"]) - 1], 0.0

    monkeypatch.setattr(train_module, "build_cnn", build)
    monkeypatch.setattr(train_module, "eer", fake_eer)
    return state


def _equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_early_stopping_is_strict_keeps_earliest_tie_and_stops_after_patience(scripted):
    scripted["script"] = [0.3, 0.2, 0.2, 0.25, 0.1, 0.1, 0.1, 0.05]
    logger = RecordingLogger()
    result = _train(logger=logger, max_epochs=8, patience=3)
    # 0.3 best(1) · 0.2 best(2) · 0.2 tie: bad 1 · 0.25: bad 2 · 0.1 best(5) · 0.1: bad 1 · 0.1: bad 2
    # · 0.05 would improve, but max_epochs=8 is reached anyway; patience 3 is never hit here.
    assert (result.best_epoch, result.best_val_a_eer, result.epochs_run) == (8, 0.05, 8)

    scripted["snapshots"].clear()
    scripted["script"] = [0.3, 0.2, 0.2, 0.25, 0.1, 0.1, 0.1, 0.1, 0.05]
    logger = RecordingLogger()
    result = _train(logger=logger, max_epochs=20, patience=3)
    # best at 5 (0.1); epochs 6, 7, 8 tie -> 3 non-improving epochs -> stop after epoch 8.
    assert (result.best_epoch, result.best_val_a_eer, result.epochs_run) == (5, 0.1, 8)
    assert [r.val_a_eer for r in result.history] == scripted["script"][:8]
    assert [r.epoch for r in result.history] == list(range(1, 9))
    assert len(logger.epochs) == 8 and logger.summaries == [(5, 0.1, 8, None)]
    assert _equal(result.state_dict, scripted["snapshots"][4])  # the best epoch's weights
    assert not _equal(result.state_dict, scripted["snapshots"][-1])  # not the last epoch's


def test_tie_with_best_never_replaces_it(scripted):
    scripted["script"] = [0.2, 0.2, 0.2]
    result = _train(max_epochs=3, patience=5)
    assert (result.best_epoch, result.epochs_run) == (1, 3)
    assert _equal(result.state_dict, scripted["snapshots"][0])


def test_stops_after_exactly_patience_non_improving_epochs(scripted):
    scripted["script"] = [0.3, 0.4, 0.5, 0.1]
    result = _train(max_epochs=10, patience=2)
    assert (result.best_epoch, result.epochs_run) == (1, 3)


def test_max_epochs_is_respected(scripted):
    scripted["script"] = [0.5, 0.4, 0.3, 0.2, 0.1]
    result = _train(max_epochs=3, patience=5)
    assert (result.best_epoch, result.epochs_run, len(scripted["snapshots"])) == (3, 3, 3)


def test_fit_bin_stats_is_called_once_on_train_only(monkeypatch):
    calls = []
    real = train_module.fit_bin_stats

    def spy(x):
        calls.append(x)
        return real(x)

    monkeypatch.setattr(train_module, "fit_bin_stats", spy)
    data = _data()
    result = _train(data, max_epochs=1)
    assert len(calls) == 1 and calls[0] is data[0]
    mean, std = real(data[0])
    assert np.array_equal(result.mean, mean) and np.array_equal(result.std, std)


def test_signature_has_no_val_b_test_or_external_input():
    params = list(inspect.signature(train_one).parameters)
    assert params == ["train_x", "train_y", "val_a_x", "val_a_y", "n_blocks", "learning_rate", "seed",
                      "max_epochs", "patience", "batch_size", "logger", "device"]
    for name in params:
        assert not any(part in name.lower() for part in ("val_b", "test", "external"))


def test_same_seed_gives_identical_results_on_cpu():
    a, b = _train(max_epochs=2), _train(max_epochs=2)
    assert _equal(a.state_dict, b.state_dict)
    strip = [(r.epoch, r.train_loss, r.val_a_eer, r.val_a_auc) for r in a.history]
    assert strip == [(r.epoch, r.train_loss, r.val_a_eer, r.val_a_auc) for r in b.history]
    c = _train(max_epochs=2, seed=1)
    assert not _equal(a.state_dict, c.state_dict)


def test_result_fields_and_logging_with_a_real_disabled_logger():
    from src.training.cnn_config import EXPECTED

    config = {**EXPECTED, "seed": 0, "n_blocks": 4, "learning_rate": 0.0003,
              "code_commit": "a" * 40, "profile_fingerprint": "b" * 64, "f2_manifest_sha256": "c" * 64,
              "train_cache_npz_sha256": "d" * 64, "val_a_cache_npz_sha256": "e" * 64}
    logger = RunLogger("disabled", config, "run")
    result = _train(logger=logger, n_blocks=4, learning_rate=0.0003, max_epochs=2)
    assert isinstance(result, TrainResult) and all(isinstance(r, EpochRecord) for r in result.history)
    assert result.peak_gpu_memory_bytes is None
    assert result.determinism["seed"] == 0 and result.determinism["cudnn_deterministic"] is True
    assert result.mean.shape == result.std.shape == (64,)
    assert all(v.device.type == "cpu" for v in result.state_dict.values())
    assert 1 <= result.best_epoch <= result.epochs_run <= 2
    assert result.best_val_a_eer == min(r.val_a_eer for r in result.history)
    for r in result.history:
        assert np.isfinite([r.train_loss, r.val_a_eer, r.val_a_auc, r.epoch_seconds]).all()
        assert 0 <= r.val_a_eer <= 1 and 0 <= r.val_a_auc <= 1 and r.epoch_seconds > 0


@pytest.mark.parametrize("change,match", [
    ({"learning_rate": 0}, "learning_rate"),
    ({"learning_rate": float("nan")}, "learning_rate"),
    ({"max_epochs": 0}, "max_epochs"),
    ({"patience": True}, "patience"),
    ({"batch_size": 8.0}, "batch_size"),
    ({"n_blocks": 5}, "n_blocks"),
    ({"seed": -1}, "seed"),
])
def test_invalid_hyperparameters_raise(change, match):
    with pytest.raises(ValueError, match=match):
        _train(**change)


def test_invalid_labels_raise():
    train_x, train_y, val_x, val_y = _data()
    with pytest.raises(ValueError, match="one class"):
        _train((train_x, train_y, val_x, np.zeros_like(val_y)))
    with pytest.raises(ValueError, match="one label per clip"):
        _train((train_x, train_y[:-2], val_x, val_y))


def test_cuda_requested_without_cuda_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        _train(device="cuda")
