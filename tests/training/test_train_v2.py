"""One model v2 run (D49, D61): early stopping on M, schedule, inputs, determinism. Synthetic, CPU."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from tests._torch import require_torch

torch = require_torch()

import src.training.train_v2 as tv  # noqa: E402
from src.augmentation import Augmenter, load_augmentation_config  # noqa: E402
from src.features.logmel_cache import load_logmel_cache  # noqa: E402
from src.features.raw_train_cache import load_raw_train_cache  # noqa: E402
from src.training.normalization import fit_bin_stats  # noqa: E402
from src.training.selection_set import CONDITIONS, load_selection_set  # noqa: E402
from src.training.v2_config import load_v2_config  # noqa: E402
from src.training.v2_schedule import cosine_lr  # noqa: E402
from src.training.v2_wandb_logger import EPOCH_KEYS, V2RunLogger  # noqa: E402
from tests._v2_fakes import (  # noqa: E402
    AUG_CONFIG_PATH,
    S_COUNTS,
    FakeCodec,
    build_caches,
    fake_resources,
    profile as load_test_profile,
    selection_arrays,
    sha,
    v2_config_copy,
    write_dataset,
    write_s,
)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("trainv2")
    prof = load_test_profile()
    root, manifest, manifest_path = write_dataset(base)
    raw_dir, lm = build_caches(base, root, manifest, manifest_path, prof)
    f2 = sha(manifest_path)
    write_s(base / "S", selection_arrays(root, manifest, prof), prof, f2, sha(AUG_CONFIG_PATH))
    s_args = {"expected_n_rows": 10, "expected_counts": S_COUNTS, "recompute_rows": 1}
    features = load_logmel_cache(lm / "train", "train", prof, expected_f2_manifest_sha256=f2)[0]
    mean, std = fit_bin_stats(features)
    config = load_v2_config(v2_config_copy(base / "v2.yaml"))
    return {
        "profile": prof, "config": config, "mean": mean, "std": std,
        "raw": load_raw_train_cache(raw_dir, expected_f2_manifest_sha256=f2),
        "S": {form: load_selection_set(base / "S", prof, form=form, **s_args)
              for form in ("logmel", "waveform")},
        "augmenter": Augmenter(load_augmentation_config(AUG_CONFIG_PATH), fake_resources(), FakeCodec()),
    }


class RecordingLogger:
    def __init__(self):
        self.epochs, self.summary = [], None

    def log_epoch(self, data):
        assert tuple(data) == EPOCH_KEYS
        self.epochs.append(dict(data))

    def log_summary(self, data):
        self.summary = dict(data)


def _train(env, candidate="cnn-aug", logger=None, **kw):
    logmel = candidate != "aasist"
    args = dict(config=env["config"], run_seed=0, raw_cache=env["raw"],
                selection=env["S"]["logmel" if logmel else "waveform"], augmenter=env["augmenter"],
                profile=env["profile"], mean=env["mean"] if logmel else None,
                std=env["std"] if logmel else None, max_epochs=1, patience=5, batch_size=8,
                num_workers=0, logger=logger or RecordingLogger(), device="cpu")
    args.update(kw)
    return tv.train_v2(candidate, **args)


def _script(monkeypatch, ms, states):
    """evaluate_selection returns scripted EERs (mean = ms[epoch]); keeps each epoch's weights."""
    calls = iter(ms)

    def fake(model, tensors, labels, batch_size, device):
        m = next(calls)
        states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        spread = [m - 0.02, m - 0.01, m, m + 0.01, m + 0.02]
        return dict(zip(CONDITIONS, spread)), dict.fromkeys(CONDITIONS, 0.5)

    monkeypatch.setattr(tv, "evaluate_selection", fake)


def test_tie_keeps_earliest_and_patience_stops(env, monkeypatch):
    states = []
    _script(monkeypatch, [0.3, 0.2, 0.2, 0.25, 0.1], states)
    logger = RecordingLogger()
    result = _train(env, max_epochs=10, patience=2, logger=logger)
    assert result.epochs_run == 4 and result.best_epoch == 2  # 0.2 tie at epoch 3 does not improve
    assert result.best_selection_m == pytest.approx(0.2)
    assert all(torch.equal(result.state_dict[k], states[1][k]) for k in states[1])
    assert not all(torch.equal(result.state_dict[k], states[3][k]) for k in states[3])
    assert [r["selection_m"] for r in result.history] == pytest.approx([0.3, 0.2, 0.2, 0.25])
    assert logger.epochs == list(result.history)
    assert logger.summary == {"best_epoch": 2, "best_selection_m": result.best_selection_m,
                              "epochs_run": 4, "peak_gpu_memory_bytes": None}


def test_max_epochs_stops_and_last_improvement_wins(env, monkeypatch):
    _script(monkeypatch, [0.5, 0.4, 0.3], [])
    result = _train(env, max_epochs=3, patience=5)
    assert (result.epochs_run, result.best_epoch) == (3, 3)
    with pytest.raises(ValueError, match="exceeds the D49 maximum"):
        _train(env, max_epochs=31)


def test_m_is_mean_of_five_eers(env, monkeypatch):
    _script(monkeypatch, [0.3], [])
    record = _train(env).history[0]
    eers = [record[f"val_a_eer_{c}"] for c in CONDITIONS]
    assert eers == pytest.approx([0.28, 0.29, 0.3, 0.31, 0.32])
    assert record["selection_m"] == math.fsum(eers) / 5
    assert tv.selection_m(dict(zip(CONDITIONS, [0.0, 0.1, 0.2, 0.3, 0.4]))) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="in order"):
        tv.selection_m(dict(zip(reversed(CONDITIONS), [0.1] * 5)))


def test_learning_rate_schedule_d61(env):
    aasist = env["config"].candidate("aasist")
    spe = 7
    total = 30 * spe  # D61 (3): 30 epochs, whatever early stopping does
    assert tv.learning_rate(aasist, 0, spe) == pytest.approx(1e-4)
    assert tv.learning_rate(aasist, 1, spe) == cosine_lr(1, total, 0.0001, 0.000005)
    assert tv.learning_rate(aasist, 1, spe) < 1e-4
    assert tv.learning_rate(aasist, total - 1, spe) == cosine_lr(total - 1, total, 0.0001, 0.000005)
    assert 5e-6 < tv.learning_rate(aasist, total - 1, spe) < 5.01e-6  # last step is total - 1
    for name in ("cnn-aug", "crnn"):
        settings = env["config"].candidate(name)
        assert {tv.learning_rate(settings, s, spe) for s in (0, 1, 50, total - 1)} == {0.001}


def test_optimizers_follow_config(env):
    for name, expected in (("cnn-aug", (0.001, 0.0, (0.9, 0.999), False)),
                           ("aasist", (0.0001, 0.0001, (0.9, 0.999), False))):
        model = tv.build_model(name)
        group = tv.build_optimizer(model, env["config"].candidate(name)).param_groups[0]
        assert (group["lr"], group["weight_decay"], group["betas"], group["amsgrad"]) == expected


def test_mean_std_and_form_rules(env):
    with pytest.raises(ValueError, match="mean and std must be None"):
        _train(env, "aasist", mean=env["mean"], std=env["std"])
    for name in ("cnn-aug", "crnn"):
        with pytest.raises(ValueError, match="clean-train mean and std are required"):
            _train(env, name, mean=None, std=None)
    with pytest.raises(ValueError, match="form 'waveform'"):
        _train(env, "aasist", selection=env["S"]["logmel"])
    with pytest.raises(ValueError, match="candidate must be one of"):
        _train(env, "lstm")


def test_signature_has_no_val_b_test_or_external_input():
    names = list(inspect.signature(tv.train_v2).parameters)
    assert not [n for n in names if any(p in n.lower() for p in ("val_b", "valb", "test", "external"))]
    assert names == ["candidate", "config", "run_seed", "raw_cache", "selection", "augmenter",
                     "profile", "mean", "std", "max_epochs", "patience", "batch_size",
                     "num_workers", "logger", "device", "resume_dir", "provenance"]


def test_same_seed_gives_identical_result(env):
    a, b = _train(env), _train(env)
    assert all(torch.equal(a.state_dict[k], b.state_dict[k]) for k in a.state_dict)
    keep = [k for k in EPOCH_KEYS if not k.endswith("_seconds")]
    assert [{k: r[k] for k in keep} for r in a.history] == [{k: r[k] for k in keep} for r in b.history]


@pytest.mark.parametrize("candidate", ["cnn-aug", "crnn", "aasist"])
def test_real_one_epoch_run(env, candidate, monkeypatch):
    steps = []
    real_lr = tv.learning_rate

    def spy(settings, step, spe):
        steps.append(step)
        return real_lr(settings, step, spe)

    monkeypatch.setattr(tv, "learning_rate", spy)
    cfg = {**env["config"].to_dict(), "candidate": candidate, "seed": 0, "batch_size": 8,
           "num_workers": 0, "code_commit": "c" * 40, "profile_fingerprint": "1" * 64,
           "f2_manifest_sha256": "2" * 64, "raw_train_cache_npz_sha256": "3" * 64,
           "selection_set_sha256": "4" * 64, "augmentation_config_sha256": "5" * 64,
           "train_stats_npz_sha256": None if candidate == "aasist" else "6" * 64}
    result = _train(env, candidate, logger=V2RunLogger("disabled", cfg, f"t-{candidate}"))
    record = result.history[0]
    assert tuple(record) == EPOCH_KEYS and result.best_epoch == 1 and result.epochs_run == 1
    assert 0.0 <= record["selection_m"] <= 1.0 and np.isfinite(record["train_loss"])
    assert sum(record[k] for k in EPOCH_KEYS if k.startswith("family_")) == 24
    assert sum(record[k] for k in EPOCH_KEYS if k.startswith("codec_")) == record["family_codec"]
    assert steps == [0, 1, 2]  # 24 clips / batch 8, global step from 0
    assert record["learning_rate"] == real_lr(env["config"].candidate(candidate), 2, 3)
    assert result.determinism["seed"] == 0 and result.determinism["num_workers"] == 0
    assert result.trainable_parameters == tv.trainable_parameters(tv.build_model(candidate))
    assert all(v.device.type == "cpu" for v in result.state_dict.values())
