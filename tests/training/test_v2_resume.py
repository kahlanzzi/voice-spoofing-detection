"""Per-epoch resume of one model v2 run (Execution Plan v3.0 §9 V3): a run stopped after an epoch and
resumed equals the uninterrupted run (CPU), and invalid states are refused. Synthetic, CPU."""
from __future__ import annotations

import ast
import json
import random

import numpy as np
import pytest

from tests._torch import require_torch

torch = require_torch()

import src.training.train_v2 as tv  # noqa: E402
import src.training.v2_resume as vr  # noqa: E402
from src.augmentation import Augmenter, load_augmentation_config  # noqa: E402
from src.features.logmel_cache import load_logmel_cache  # noqa: E402
from src.features.raw_train_cache import load_raw_train_cache  # noqa: E402
from src.training.normalization import fit_bin_stats  # noqa: E402
from src.training.selection_set import CONDITIONS, load_selection_set  # noqa: E402
from src.training.v2_config import load_v2_config  # noqa: E402
from src.training.v2_wandb_logger import EPOCH_KEYS  # noqa: E402
from tests._v2_fakes import (  # noqa: E402
    AUG_CONFIG_PATH,
    S_COUNTS,
    FakeCodec,
    SubsetRawCache,
    SubsetSelection,
    build_caches,
    fake_resources,
    profile as load_test_profile,
    selection_arrays,
    sha,
    v2_config_copy,
    write_dataset,
    write_s,
)

PROV = {"code_commit": "c" * 40, "raw_train_cache_npz_sha256": "3" * 64}
N_CLIPS, BATCH = 4, 2  # 2 optimizer steps per epoch
KEEP = [k for k in EPOCH_KEYS if not k.endswith("_seconds")]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("resume")
    prof = load_test_profile()
    root, manifest, manifest_path = write_dataset(base)
    raw_dir, lm = build_caches(base, root, manifest, manifest_path, prof)
    f2 = sha(manifest_path)
    write_s(base / "S", selection_arrays(root, manifest, prof), prof, f2, sha(AUG_CONFIG_PATH))
    s_args = {"expected_n_rows": 10, "expected_counts": S_COUNTS, "recompute_rows": 1}
    features = load_logmel_cache(lm / "train", "train", prof, expected_f2_manifest_sha256=f2)[0]
    mean, std = fit_bin_stats(features)
    raw = load_raw_train_cache(raw_dir, expected_f2_manifest_sha256=f2)
    return {
        "profile": prof, "mean": mean, "std": std,
        "config": load_v2_config(v2_config_copy(base / "v2.yaml")),
        "raw": SubsetRawCache(raw, N_CLIPS),
        "S": {form: SubsetSelection(load_selection_set(base / "S", prof, form=form, **s_args), 1)
              for form in ("logmel", "waveform")},
        "augmenter": Augmenter(load_augmentation_config(AUG_CONFIG_PATH), fake_resources(),
                               FakeCodec()),
    }


class RecordingLogger:
    def __init__(self, run_id=None):
        self.epochs, self.summary, self.run_id = [], None, run_id

    def log_epoch(self, data):
        self.epochs.append(dict(data))

    def log_summary(self, data):
        self.summary = dict(data)


class Crash(Exception):
    """Stands for a Colab disconnect right after a resume state is written."""


def _train(env, candidate="cnn-aug", logger=None, **kw):
    logmel = candidate != "aasist"
    args = dict(config=env["config"], run_seed=0, raw_cache=env["raw"],
                selection=env["S"]["logmel" if logmel else "waveform"], augmenter=env["augmenter"],
                profile=env["profile"], mean=env["mean"] if logmel else None,
                std=env["std"] if logmel else None, max_epochs=3, patience=5, batch_size=BATCH,
                num_workers=0, logger=logger or RecordingLogger(), device="cpu")
    args.update(kw)
    return tv.train_v2(candidate, **args)


def _crash_after(monkeypatch, epoch):
    real = tv.write_state

    def writing_then_crashing(resume_dir, meta, tensors):
        out = real(resume_dir, meta, tensors)
        if meta["last_epoch"] == epoch:
            raise Crash(f"stopped after the epoch-{epoch} state was written")
        return out
    monkeypatch.setattr(tv, "write_state", writing_then_crashing)


def _assert_same(a, b):
    assert set(a.state_dict) == set(b.state_dict)
    assert all(torch.equal(a.state_dict[k], b.state_dict[k]) for k in a.state_dict)
    assert (a.best_epoch, a.best_selection_m, a.epochs_run) == (b.best_epoch, b.best_selection_m,
                                                                b.epochs_run)
    assert [{k: r[k] for k in KEEP} for r in a.history] == [{k: r[k] for k in KEEP} for r in b.history]


def _crash_and_resume(env, monkeypatch, tmp_path, candidate, **kw):
    resume = tmp_path / f"{candidate}.partial" / "resume"
    with monkeypatch.context() as m:
        _crash_after(m, 2)
        with pytest.raises(Crash):
            _train(env, candidate, resume_dir=resume, provenance=PROV, **kw)
    assert json.loads((resume / vr.JSON_NAME).read_text(encoding="utf-8"))["last_epoch"] == 2
    random.random(), np.random.rand(), torch.rand(3)  # the process moves on; restore must undo it
    logger = RecordingLogger()
    result = _train(env, candidate, logger=logger, resume_dir=resume, provenance=PROV, **kw)
    return result, logger, resume


@pytest.fixture(scope="module")
def straight(env):
    """The uninterrupted 3-epoch runs, without resume_dir, computed once per candidate."""
    cache = {}

    def get(candidate):
        if candidate not in cache:
            cache[candidate] = _train(env, candidate)
        return cache[candidate]
    return get


@pytest.mark.parametrize("candidate", ["cnn-aug", "crnn", "aasist"])
def test_crash_after_epoch_2_and_resume_equals_straight_run(env, straight, monkeypatch, tmp_path,
                                                            candidate):
    uninterrupted = straight(candidate)
    steps = []
    real_lr = tv.learning_rate

    def spy(settings, step, spe):
        steps.append((step, real_lr(settings, step, spe)))
        return steps[-1][1]

    monkeypatch.setattr(tv, "learning_rate", spy)
    resumed, logger, resume = _crash_and_resume(env, monkeypatch, tmp_path, candidate)
    _assert_same(resumed, uninterrupted)
    # The D61 schedule continues from the saved global step: steps 0-3 before, 4-5 after.
    assert [s for s, _ in steps] == [0, 1, 2, 3, 4, 5]
    settings = env["config"].candidate(candidate)
    assert steps[4] == (4, real_lr(settings, 4, 2))
    if candidate == "aasist":
        assert steps[4][1] < steps[3][1]  # cosine: the first resumed step is lower, not reset
    assert [r["epoch"] for r in logger.epochs] == [3]  # epochs 1-2 are not logged again
    assert logger.summary["epochs_run"] == 3
    assert json.loads((resume / vr.JSON_NAME).read_text(encoding="utf-8"))["last_epoch"] == 3


def test_resume_with_two_workers_equals_straight_run(env, straight, monkeypatch, tmp_path):
    resumed, _, _ = _crash_and_resume(env, monkeypatch, tmp_path, "cnn-aug", num_workers=2)
    _assert_same(resumed, straight("cnn-aug"))  # straight run: num_workers 0
    assert resumed.determinism["num_workers"] == 2


def _scripted(monkeypatch, ms):
    calls = iter(ms)

    def fake(model, tensors, labels, batch_size, device):
        m = next(calls)
        return dict.fromkeys(CONDITIONS, m), dict.fromkeys(CONDITIONS, 0.5)
    monkeypatch.setattr(tv, "evaluate_selection", fake)


@pytest.mark.parametrize("ms, max_epochs, patience, reason", [
    ([0.3, 0.4, 0.5], 5, 2, "patience"),
    ([0.3, 0.2], 2, 5, "max_epochs"),
])
def test_stopped_run_resumes_without_training(env, monkeypatch, tmp_path, ms, max_epochs, patience,
                                              reason):
    resume = tmp_path / "resume"
    _scripted(monkeypatch, ms)
    first = _train(env, max_epochs=max_epochs, patience=patience, resume_dir=resume,
                   provenance=PROV)
    assert first.epochs_run == len(ms)
    _scripted(monkeypatch, [])  # any scoring call would raise StopIteration
    logger = RecordingLogger()
    again = _train(env, logger=logger, max_epochs=max_epochs, patience=patience,
                   resume_dir=resume, provenance=PROV)
    assert logger.epochs == [] and logger.summary["epochs_run"] == len(ms)
    _assert_same(again, first)


# --- refusals ------------------------------------------------------------------------------------

@pytest.fixture()
def saved(env, monkeypatch, tmp_path):
    """A valid state after epoch 1 of a cnn-aug run (scripted M)."""
    resume = tmp_path / "run.partial" / "resume"
    _scripted(monkeypatch, [0.3])
    with monkeypatch.context() as m:
        _crash_after(m, 1)
        with pytest.raises(Crash):
            _train(env, resume_dir=resume, provenance=PROV)
    return resume


def _rewrite(resume, change):
    path = resume / vr.JSON_NAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    change(meta)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_refuses_tampered_tensors(env, saved):
    data = bytearray((saved / vr.TENSORS_NAME).read_bytes())
    data[-1] ^= 1
    (saved / vr.TENSORS_NAME).write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA-256"):
        _train(env, resume_dir=saved, provenance=PROV)


@pytest.mark.parametrize("candidate, kw, match", [
    ("cnn-aug", {"provenance": {**PROV, "code_commit": "d" * 40}}, "provenance differs"),
    ("crnn", {}, "candidate differs"),
    ("cnn-aug", {"run_seed": 1}, "run_seed differs"),
    ("cnn-aug", {"batch_size": 1}, "run settings differ"),
    ("cnn-aug", {"logger": RecordingLogger(run_id="abc123")}, "W&B run id"),
])
def test_refuses_other_run(env, saved, candidate, kw, match):
    args = {"resume_dir": saved, "provenance": PROV, **kw}
    with pytest.raises(ValueError, match=match):
        _train(env, candidate, **args)


def test_refuses_missing_key_and_nan(env, saved):
    keep = (saved / vr.JSON_NAME).read_text(encoding="utf-8")
    _rewrite(saved, lambda meta: meta.pop("patience_counter"))
    with pytest.raises(ValueError, match=r"missing \['patience_counter'\]"):
        _train(env, resume_dir=saved, provenance=PROV)
    (saved / vr.JSON_NAME).write_text(keep.replace('"best_selection_m": 0.3', '"best_selection_m": NaN'),
                                      encoding="utf-8")
    with pytest.raises(ValueError, match="NaN"):
        _train(env, resume_dir=saved, provenance=PROV)
    (saved / vr.JSON_NAME).write_text(keep.replace("\n  ", "\n "), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        _train(env, resume_dir=saved, provenance=PROV)


def test_refuses_missing_tensor_and_needs_provenance(env, saved):
    from safetensors.torch import load, save
    tensors = load((saved / vr.TENSORS_NAME).read_bytes())
    del tensors["rng.sampler"]
    data = save(tensors)
    (saved / vr.TENSORS_NAME).write_bytes(data)
    import hashlib
    _rewrite(saved, lambda meta: meta.update(state_safetensors_sha256=hashlib.sha256(data).hexdigest()))
    with pytest.raises(ValueError, match="RNG tensors missing"):
        _train(env, resume_dir=saved, provenance=PROV)
    with pytest.raises(ValueError, match="given together"):
        _train(env, resume_dir=saved)


# --- atomic write ---------------------------------------------------------------------------------

def _state(epoch):
    meta = {"candidate": "cnn-aug", "run_seed": 0, "last_epoch": epoch, "global_step": 2 * epoch,
            "best_selection_m": 0.1, "best_epoch": 1, "patience_counter": epoch - 1,
            "history": [{"epoch": e} for e in range(1, epoch + 1)],
            "numpy_random_state": vr.numpy_state_json(), "python_random_state": vr.python_state_json(),
            "determinism": {}, "wandb_run_id": None, "provenance": PROV,
            "settings": {"max_epochs": 3, "patience": 5, "batch_size": 3, "steps_per_epoch": 2,
                         "n_train": 6, "device_type": "cpu"},
            "peak_gpu_memory_bytes": None}
    tensors = {"model.w": torch.full((2,), float(epoch)), "best.w": torch.zeros(2),
               "rng.torch_cpu": torch.get_rng_state(), "rng.sampler": torch.Generator().get_state()}
    return meta, tensors


def _read(resume):
    return vr.read_state(resume, candidate="cnn-aug", run_seed=0, provenance=PROV)


def test_write_replaces_the_state_and_interrupted_swaps_recover(tmp_path):
    resume = tmp_path / "resume"
    vr.write_state(resume, *_state(1))
    vr.write_state(resume, *_state(2))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["resume"]
    assert _read(resume)[0]["last_epoch"] == 2
    # Stopped between the two renames: only resume.old/ (the previous complete state) exists.
    resume.rename(tmp_path / "resume.old")
    (tmp_path / "resume.tmp").mkdir()
    assert vr.has_state(resume)
    meta, tensors = _read(resume)
    assert meta["last_epoch"] == 2 and torch.equal(tensors["model.w"], torch.full((2,), 2.0))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["resume"]
    # Stopped while writing the very first state: only resume.tmp/ -> no state.
    other = tmp_path / "other" / "resume"
    other.with_name("resume.tmp").mkdir(parents=True)
    assert not vr.has_state(other)
    vr.remove_state(resume)
    assert not resume.exists()


def test_python_and_numpy_states_round_trip_through_json():
    random.seed(5)
    np.random.seed(5)
    np.random.standard_normal()  # leaves a cached Gaussian in the legacy state
    py = json.loads(vr.json_text(vr.python_state_json()))
    npy = json.loads(vr.json_text(vr.numpy_state_json()))
    expected = (random.random(), np.random.standard_normal(), np.random.rand())
    random.seed(9)
    np.random.seed(9)
    vr.set_python_state(py)
    vr.set_numpy_state(npy)
    assert (random.random(), np.random.standard_normal(), np.random.rand()) == expected


def test_optimizer_state_round_trip():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.randn(4, 3)).sum().backward()
    opt.step()
    flat = vr.flatten_optimizer(opt)
    assert sorted(flat) == ["opt.0.exp_avg", "opt.0.exp_avg_sq", "opt.0.step", "opt.1.exp_avg",
                            "opt.1.exp_avg_sq", "opt.1.step"]
    fresh = torch.optim.Adam(model.parameters(), lr=1e-3)
    vr.restore_optimizer(fresh, flat)
    for a, b in zip(opt.state_dict()["state"].values(), fresh.state_dict()["state"].values()):
        assert all(torch.equal(a[k], b[k]) for k in a)
    with pytest.raises(ValueError, match="expected"):
        vr.restore_optimizer(torch.optim.Adam(model.parameters()),
                             {k: v for k, v in flat.items() if k != "opt.1.step"})


def test_resume_state_holds_no_pickle(env, saved):
    names = sorted(p.name for p in saved.iterdir())
    assert names == ["state.json", "state.safetensors"]
    meta = json.loads((saved / vr.JSON_NAME).read_text(encoding="utf-8"))
    assert set(meta) == vr.STATE_KEYS and meta["provenance"] == PROV
    for path in (vr.__file__, tv.__file__):
        tree = ast.parse(open(path, encoding="utf-8").read())
        modules = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        modules |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not {m for m in modules if m and "pickle" in m}, path
        calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert not calls & {"torch.save", "torch.load"}, path
