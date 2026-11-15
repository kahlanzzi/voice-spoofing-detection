"""load_bundle_model dispatch on the checkpoint format (D59 (7), D61 (6)): F5 / model v2 bundles.

Synthetic only (random weights, seed 2026, the real configs/preprocessing.yaml); no dataset path.
Includes the hand-over test with the Stage 13 runner (verify_f5_artifacts + build_frozen_scorers).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests._torch import require_torch

torch = require_torch()
from safetensors.torch import load as load_tensors  # noqa: E402
from safetensors.torch import save as save_tensors  # noqa: E402

import src.bundle.model_loader as loader_module  # noqa: E402
from src.bundle.bundle import (  # noqa: E402
    CHECKPOINT_DIR,
    MANIFEST_NAME,
    build_bundle,
    load_bundle,
)
from src.bundle.model_loader import load_bundle_model  # noqa: E402
from src.evaluation import external_run  # noqa: E402
from src.models import cnn_checkpoint, v2_checkpoint  # noqa: E402
from src.models.cnn import INPUT_SHAPE, build_cnn  # noqa: E402
from src.models.v2_checkpoint import ARCHITECTURES, build_model, save_v2_checkpoint  # noqa: E402
from src.preprocessing.profile import load_profile  # noqa: E402
from src.risk.calibration import PlattCalibration  # noqa: E402
from src.risk.policy import RiskPolicy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROFILE_SRC = ROOT / "configs" / "preprocessing.yaml"
LEDGER = "c9fc2e230c0883028459d0b2b8de4f6ee537676cd5a4e259e9d0f545c60943cc"
FINGERPRINT = load_profile(PROFILE_SRC).fingerprint()
PARAMETERS = {"cnn-aug": 97_761, "crnn": 319_585, "aasist": 297_705}  # pinned in Entry 170
INPUT_SHAPES = {"cnn-aug": (1, 64, 197), "crnn": (1, 64, 197), "aasist": (32000,)}


def _v2_meta(arch: str, **changes) -> dict:
    logmel = arch != "aasist"
    meta = {
        "schema_version": 1, "architecture": arch, "version": f"{arch}-v2",
        "input": "logmel" if logmel else "waveform",
        "input_shape": [1, 64, 197] if logmel else [32000],
        "seed": 0, "best_epoch": 9, "selection_m": 0.0425, "code_commit": "a" * 40,
        "environment": {"python": "3.13.15", "torch": "2.11.0+cpu"},
        "profile_fingerprint": FINGERPRINT, "f2_manifest_sha256": "c" * 64,
        "augmentation_config_sha256": "d" * 64, "selection_set_sha256": "e" * 64,
        "train_stats_npz_sha256": "f" * 64 if logmel else None,
        "determinism": {"seed": 0, "cudnn_deterministic": True},
        "trainable_parameters": PARAMETERS[arch],
    }
    meta.update(changes)
    return meta


def _v1_meta() -> dict:
    # As tests/bundle/test_model_loader.py.
    return {
        "schema_version": 1, "version": "cnn-v1", "n_blocks": 3, "channels": [16, 32, 64],
        "dropout": 0.3, "input_shape": [1, 64, 197], "learning_rate": 0.001, "seed": 0,
        "best_epoch": 7, "val_a_eer": 0.0125, "code_commit": "a" * 40,
        "environment": {"python": "3.13.15", "torch": "2.11.0+cpu"},
        "profile_fingerprint": FINGERPRINT, "f2_manifest_sha256": "c" * 64,
        "train_cache_npz_sha256": "d" * 64, "val_a_cache_npz_sha256": "e" * 64,
        "determinism": {"seed": 0, "cudnn_deterministic": True},
    }


def _stats():
    rng = np.random.default_rng(2026)
    return rng.normal(-6.0, 2.0, 64).astype(np.float32), rng.uniform(0.5, 3.0, 64).astype(np.float32)


def _inputs(arch: str, n: int = 2):
    rng = np.random.default_rng(7)
    return torch.from_numpy(rng.normal(size=(n, *INPUT_SHAPES[arch])).astype(np.float32))


def _v2_model(arch: str):
    """Random weights with non-default BatchNorm running statistics, so buffers are covered too."""
    torch.manual_seed(2026)
    model = build_model(arch).train()
    with torch.no_grad():
        model(_inputs(arch, 3))
    return model.eval()


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    policy = src / "risk_policy.json"
    policy.write_text(RiskPolicy(
        version="policy-v2", calibration=PlattCalibration(a=1.0, b=0.0, n_spoof=10, n_bonafide=12),
        threshold=0.5, half_width=0.1, target_deferral=0.10, n_spoof_fit=10, n_bonafide_fit=12,
        deferral_on_fit=0.0909090909090909,
    ).to_json(), encoding="utf-8")
    profile = src / "preprocessing.yaml"
    profile.write_bytes(PROFILE_SRC.read_bytes())
    return policy, profile


def _build(tmp_path: Path, ckpt: Path, *, name: str, model_version: str, bundle_version: str) -> Path:
    policy, profile = _sources(tmp_path)
    directory = tmp_path / name
    build_bundle(
        directory, checkpoint_dir=ckpt,
        expected_checkpoint_sha256=v2_checkpoint.v2_checkpoint_sha256(ckpt),
        profile=profile, policy=policy, bundle_version=bundle_version, model_version=model_version,
        profile_version="preprocessing-v1", code_commit="0" * 40, protocol_version="1.0",
        f2_ledger_sha256=LEDGER, created_at="2026-11-15T12:00:00Z",
    )
    return directory


def _v2_bundle(tmp_path: Path, arch: str):
    model = _v2_model(arch)
    mean, std = _stats() if arch != "aasist" else (None, None)
    ckpt = save_v2_checkpoint(model, _v2_meta(arch), tmp_path / "src" / f"{arch}_ckpt", mean=mean, std=std)
    directory = _build(tmp_path, ckpt, name=f"bundle-{arch}", model_version=f"{arch}-v2",
                       bundle_version="bundle-v2")
    return directory, model, mean, std


def _v1_bundle(tmp_path: Path):
    torch.manual_seed(2026)
    model = build_cnn(3).train()
    with torch.no_grad():
        model(torch.randn(4, *INPUT_SHAPE))
    model.eval()
    mean, std = _stats()
    ckpt = cnn_checkpoint.save_checkpoint(model, mean, std, _v1_meta(), tmp_path / "src" / "v1_ckpt")
    directory = _build(tmp_path, ckpt, name="bundle-v1", model_version="cnn-v1", bundle_version="bundle-v1")
    return directory, model, mean, std


def _crafted_checkpoint(tmp_path: Path, arch: str, edit) -> Path:
    """A saved v2 checkpoint whose tensors are edited by edit(dict) and re-written (bypasses the saver)."""
    model = _v2_model(arch)
    mean, std = _stats() if arch != "aasist" else (None, None)
    ckpt = save_v2_checkpoint(model, _v2_meta(arch), tmp_path / "src" / f"{arch}_crafted", mean=mean, std=std)
    weights = ckpt / v2_checkpoint.WEIGHTS_NAME
    tensors = dict(load_tensors(weights.read_bytes()))
    edit(tensors)
    weights.write_bytes(save_tensors(tensors))
    return ckpt


# Dispatch and round trip -------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_v2_bundle_loads_by_architecture(tmp_path, arch):
    directory, model, mean, std = _v2_bundle(tmp_path, arch)
    bm = load_bundle_model(directory)
    assert bm.architecture == arch == bm.metadata["architecture"]
    assert type(bm.model) is type(model)
    assert not bm.model.training
    assert all(p.device.type == "cpu" for p in bm.model.parameters())
    assert bm.metadata["version"] == bm.bundle.manifest.components["checkpoint"].version == f"{arch}-v2"
    if arch == "aasist":
        assert bm.mean is None
        assert bm.std is None
    else:
        assert bm.mean.dtype == np.float32
        assert bm.mean.shape == (64,)
        np.testing.assert_array_equal(bm.mean, mean)
        np.testing.assert_array_equal(bm.std, std)
    x = _inputs(arch)
    with torch.no_grad():
        assert torch.equal(model(x), bm.model(x))


def test_v1_bundle_dispatches_to_cnn_loader_unchanged(tmp_path, monkeypatch):
    directory, model, mean, std = _v1_bundle(tmp_path)
    called = []
    real = loader_module.v2_checkpoint.load_v2_checkpoint
    monkeypatch.setattr(loader_module.v2_checkpoint, "load_v2_checkpoint",
                        lambda *a, **k: called.append(1) or real(*a, **k))
    bm = load_bundle_model(directory)
    assert called == []
    assert bm.architecture == "cnn"
    assert "architecture" not in bm.metadata
    # Same values as the F4 path: cnn_checkpoint.load_checkpoint on the same folder and anchor.
    ref_model, ref_mean, ref_std, ref_meta = cnn_checkpoint.load_checkpoint(
        bm.bundle.checkpoint_dir, expected_sha256=bm.bundle.checkpoint_sha256)
    assert bm.metadata == ref_meta
    np.testing.assert_array_equal(bm.mean, ref_mean)
    np.testing.assert_array_equal(bm.std, ref_std)
    for key, value in model.state_dict().items():
        assert torch.equal(bm.model.state_dict()[key], value), key
    assert not bm.model.training


# Refusals ----------------------------------------------------------------------------------------

def test_v2_weights_replaced_after_build_are_refused(tmp_path):
    directory, *_ = _v2_bundle(tmp_path, "crnn")
    weights = directory / CHECKPOINT_DIR / v2_checkpoint.WEIGHTS_NAME
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0x01
    weights.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA-256 mismatch for checkpoint/weights.safetensors"):
        load_bundle_model(directory)


def test_v2_weights_replaced_after_load_bundle_are_refused(tmp_path, monkeypatch):
    # Check-then-use: load_v2_checkpoint hashes the bytes it loads against the manifest anchor.
    directory, *_ = _v2_bundle(tmp_path, "aasist")
    weights = directory / CHECKPOINT_DIR / v2_checkpoint.WEIGHTS_NAME
    real_load_bundle = loader_module.load_bundle

    def load_then_tamper(path):
        bundle = real_load_bundle(path)
        data = bytearray(weights.read_bytes())
        data[-1] ^= 0x01
        weights.write_bytes(bytes(data))
        return bundle

    monkeypatch.setattr(loader_module, "load_bundle", load_then_tamper)
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        load_bundle_model(directory)


def test_architecture_edited_in_checkpoint_json_is_refused(tmp_path):
    directory, *_ = _v2_bundle(tmp_path, "crnn")
    path = directory / CHECKPOINT_DIR / v2_checkpoint.JSON_NAME
    path.write_text(path.read_text(encoding="utf-8").replace('"crnn"', '"cnn-aug"'), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch for checkpoint/checkpoint.json"):
        load_bundle_model(directory)


def test_architecture_edited_after_load_bundle_is_refused(tmp_path, monkeypatch):
    # The dispatch read is itself hashed against the manifest before it is parsed.
    directory, *_ = _v2_bundle(tmp_path, "crnn")
    path = directory / CHECKPOINT_DIR / v2_checkpoint.JSON_NAME
    real_load_bundle = loader_module.load_bundle

    def load_then_tamper(p):
        bundle = real_load_bundle(p)
        path.write_text(path.read_text(encoding="utf-8").replace('"crnn"', '"aasist"'), encoding="utf-8")
        return bundle

    monkeypatch.setattr(loader_module, "load_bundle", load_then_tamper)
    with pytest.raises(ValueError, match="SHA-256 mismatch for checkpoint.json"):
        load_bundle_model(directory)


def test_unknown_architecture_is_refused_by_schema(tmp_path):
    # A re-anchored bundle (hashes consistent) naming an architecture outside D49.
    ckpt = save_v2_checkpoint(_v2_model("crnn"), _v2_meta("crnn"), tmp_path / "src" / "ck", mean=_stats()[0],
                              std=_stats()[1])
    meta_path = ckpt / v2_checkpoint.JSON_NAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["architecture"] = "wav2vec2"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    directory = _build(tmp_path, ckpt, name="b", model_version="crnn-v2", bundle_version="bundle-v2")
    with pytest.raises(ValueError, match="architecture must be one of"):
        load_bundle_model(directory)


def test_profile_fingerprint_mismatch_is_refused(tmp_path):
    model = _v2_model("cnn-aug")
    mean, std = _stats()
    ckpt = save_v2_checkpoint(model, _v2_meta("cnn-aug", profile_fingerprint="b" * 64),
                              tmp_path / "src" / "ck", mean=mean, std=std)
    with pytest.raises(ValueError, match="profile_fingerprint mismatch"):
        _build(tmp_path, ckpt, name="b", model_version="cnn-aug-v2", bundle_version="bundle-v2")
    # And at load time, on a bundle whose manifest was re-anchored to the edited checkpoint.json.
    directory, *_ = _v2_bundle(tmp_path / "ok", "cnn-aug")
    json_path = directory / CHECKPOINT_DIR / v2_checkpoint.JSON_NAME
    json_path.write_text(json_path.read_text(encoding="utf-8").replace(FINGERPRINT, "b" * 64), encoding="utf-8")
    manifest_path = directory / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["components"]["checkpoint"]["files"]
    files[v2_checkpoint.JSON_NAME] = hashlib.sha256(json_path.read_bytes()).hexdigest()
    manifest["components"]["checkpoint"]["sha256"] = v2_checkpoint.v2_checkpoint_sha256(json_path.parent)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="profile_fingerprint mismatch"):
        load_bundle_model(directory)


def test_aasist_checkpoint_with_norm_tensors_is_refused(tmp_path):
    mean, std = _stats()

    def add_norm(t):
        t["norm.mean"], t["norm.std"] = torch.from_numpy(mean), torch.from_numpy(std)

    ckpt = _crafted_checkpoint(tmp_path, "aasist", add_norm)
    directory = _build(tmp_path, ckpt, name="b", model_version="aasist-v2", bundle_version="bundle-v2")
    with pytest.raises(ValueError, match="takes the waveform and must not hold"):
        load_bundle_model(directory)


@pytest.mark.parametrize("arch", ["cnn-aug", "crnn"])
def test_logmel_checkpoint_without_norm_tensors_is_refused(tmp_path, arch):
    def drop_norm(t):
        del t["norm.mean"], t["norm.std"]

    ckpt = _crafted_checkpoint(tmp_path, arch, drop_norm)
    directory = _build(tmp_path, ckpt, name="b", model_version=f"{arch}-v2", bundle_version="bundle-v2")
    with pytest.raises(ValueError, match=r"needs tensors \['norm.mean', 'norm.std'\]"):
        load_bundle_model(directory)


def _patched_v2_loader(monkeypatch, change):
    real = loader_module.v2_checkpoint.load_v2_checkpoint

    def fake(*args, **kwargs):
        return change(*real(*args, **kwargs))

    monkeypatch.setattr(loader_module.v2_checkpoint, "load_v2_checkpoint", fake)


def test_loader_checks_loaded_architecture_itself(tmp_path, monkeypatch):
    directory, *_ = _v2_bundle(tmp_path, "crnn")
    _patched_v2_loader(monkeypatch, lambda m, mu, sd, meta: (m, mu, sd, {**meta, "architecture": "cnn-aug"}))
    with pytest.raises(ValueError, match=r"architecture mismatch \(dispatched 'crnn', loaded 'cnn-aug'\)"):
        load_bundle_model(directory)


@pytest.mark.parametrize("arch, change, message", [
    ("crnn", lambda m, mu, sd, meta: (m.train(), mu, sd, meta), "must be in eval mode"),
    ("crnn", lambda m, mu, sd, meta: (m, None, sd, meta), r"needs mean as float32 \(64,\)"),
    ("cnn-aug", lambda m, mu, sd, meta: (m, mu, sd.astype(np.float64), meta), r"needs std as float32"),
    ("aasist", lambda m, mu, sd, meta: (m, np.zeros(64, np.float32), sd, meta), "mean and std must be None"),
])
def test_loader_checks_eval_mode_and_statistics_itself(tmp_path, monkeypatch, arch, change, message):
    directory, *_ = _v2_bundle(tmp_path, arch)
    _patched_v2_loader(monkeypatch, change)
    with pytest.raises(ValueError, match=message):
        load_bundle_model(directory)


def test_loader_checks_v2_version_against_manifest(tmp_path, monkeypatch):
    directory, *_ = _v2_bundle(tmp_path, "aasist")
    _patched_v2_loader(monkeypatch, lambda m, mu, sd, meta: (m, mu, sd, {**meta, "version": "other"}))
    with pytest.raises(ValueError, match=r"version mismatch \(manifest 'aasist-v2', loaded 'other'\)"):
        load_bundle_model(directory)


# Hand-over to the Stage 13 runner ----------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_handover_to_stage13_runner(tmp_path, arch):
    v2_dir, *_ = _v2_bundle(tmp_path, arch)
    verified = load_bundle(v2_dir)
    record = tmp_path / "f5_record.json"
    record.write_text(json.dumps({
        "checks": [{"name": "synthetic", "ok": True}],
        "manifest_sha256": verified.manifest_sha256,
        "bundle_version": verified.manifest.bundle_version,
    }), encoding="utf-8")
    v2_model, v2_sha = external_run.verify_f5_artifacts(
        v2_dir, record, bundle_verifier=load_bundle, bundle_loader=load_bundle_model)
    assert v2_sha == verified.manifest_sha256
    assert v2_model.architecture == arch

    v1_dir, *_ = _v1_bundle(tmp_path)
    v1_model = load_bundle_model(v1_dir)
    baseline = SimpleNamespace(score=lambda features: np.zeros(len(features)))
    artifacts = external_run.FrozenArtifacts(
        v1_model, baseline, v1_model.bundle.manifest_sha256, "b" * 64, v2_model, v2_sha)
    scorers = external_run.build_frozen_scorers(artifacts)
    assert set(scorers) == {"bundle-v1", "baseline-v1", "bundle-v2"}

    profile = v2_model.bundle.profile
    waveform = np.random.default_rng(11).uniform(-0.5, 0.5, profile.samples).astype(np.float32)
    score = scorers["bundle-v2"](waveform)
    assert isinstance(score, float)
    assert np.isfinite(score)
