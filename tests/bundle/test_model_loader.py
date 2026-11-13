"""load_bundle_model (D59 (5)): a tiny real D42 checkpoint in a bundle, loaded and verified. Synthetic only."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests._torch import require_torch

torch = require_torch()

import src.bundle.model_loader as loader_module  # noqa: E402
from src.bundle.bundle import (  # noqa: E402
    CHECKPOINT_DIR,
    CHECKPOINT_FILES,
    build_bundle,
    checkpoint_dir_sha256,
)
from src.bundle.model_loader import load_bundle_model  # noqa: E402
from src.models import cnn_checkpoint  # noqa: E402
from src.models.cnn import INPUT_SHAPE, build_cnn  # noqa: E402
from src.preprocessing.profile import load_profile  # noqa: E402
from src.risk.calibration import PlattCalibration  # noqa: E402
from src.risk.policy import RiskPolicy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROFILE_SRC = ROOT / "configs" / "preprocessing.yaml"
LEDGER = "c9fc2e230c0883028459d0b2b8de4f6ee537676cd5a4e259e9d0f545c60943cc"


def _meta(**changes) -> dict:
    # As tests/models/test_cnn_checkpoint.py, with the real profile's fingerprint.
    meta = {
        "schema_version": 1, "version": "cnn-v1", "n_blocks": 3, "channels": [16, 32, 64],
        "dropout": 0.3, "input_shape": [1, 64, 197], "learning_rate": 0.001, "seed": 0,
        "best_epoch": 7, "val_a_eer": 0.0125, "code_commit": "a" * 40,
        "environment": {"python": "3.13.15", "torch": "2.11.0+cpu"},
        "profile_fingerprint": load_profile(PROFILE_SRC).fingerprint(),
        "f2_manifest_sha256": "c" * 64, "train_cache_npz_sha256": "d" * 64,
        "val_a_cache_npz_sha256": "e" * 64, "determinism": {"seed": 0, "cudnn_deterministic": True},
    }
    meta.update(changes)
    return meta


def _model():
    """A CNN with non-default BatchNorm running statistics, so buffers are covered too."""
    torch.manual_seed(2026)
    model = build_cnn(3).train()
    with torch.no_grad():
        for _ in range(2):
            model(torch.randn(4, *INPUT_SHAPE))
    return model.eval()


def _stats():
    rng = np.random.default_rng(2026)
    return rng.normal(-6.0, 2.0, 64).astype(np.float32), rng.uniform(0.5, 3.0, 64).astype(np.float32)


@pytest.fixture
def built(tmp_path: Path):
    model = _model()
    mean, std = _stats()
    ckpt = cnn_checkpoint.save_checkpoint(model, mean, std, _meta(), tmp_path / "src" / "seed_0")
    policy = tmp_path / "src" / "risk_policy.json"
    policy.write_text(RiskPolicy(
        version="policy-v1", calibration=PlattCalibration(a=1.0, b=0.0, n_spoof=10, n_bonafide=12),
        threshold=0.5, half_width=0.1, target_deferral=0.10, n_spoof_fit=10, n_bonafide_fit=12,
        deferral_on_fit=0.0909090909090909,
    ).to_json(), encoding="utf-8")
    profile = tmp_path / "src" / "preprocessing.yaml"
    profile.write_bytes(PROFILE_SRC.read_bytes())
    directory = tmp_path / "bundle"
    build_bundle(
        directory, checkpoint_dir=ckpt, expected_checkpoint_sha256=cnn_checkpoint.checkpoint_sha256(ckpt),
        profile=profile, policy=policy, bundle_version="bundle-v1", model_version="cnn-v1",
        profile_version="preprocessing-v1", code_commit="0" * 40, protocol_version="1.0",
        f2_ledger_sha256=LEDGER, created_at="2026-11-10T12:00:00Z",
    )
    return directory, ckpt, model, mean, std


def test_checkpoint_constants_and_hash_rule_match_cnn_checkpoint(built):
    # bundle.py restates the D42 names and hash rule to stay torch-free; they must be the same.
    _, ckpt, *_ = built
    assert CHECKPOINT_FILES == cnn_checkpoint.FILES
    files = {n: hashlib.sha256((ckpt / n).read_bytes()).hexdigest() for n in CHECKPOINT_FILES}
    assert checkpoint_dir_sha256(files) == cnn_checkpoint.checkpoint_sha256(ckpt)


def test_build_then_load_bundle_model(built):
    # The loaded CNN is the saved one: eval mode, identical state_dict (weights and BatchNorm
    # buffers), identical normalization statistics, and metadata whose version is the manifest's.
    directory, ckpt, model, mean, std = built
    bm = load_bundle_model(directory)
    assert bm.bundle.checkpoint_sha256 == cnn_checkpoint.checkpoint_sha256(ckpt)
    assert not bm.model.training
    saved, loaded = model.state_dict(), bm.model.state_dict()
    assert saved.keys() == loaded.keys()
    for key in saved:
        assert torch.equal(saved[key], loaded[key]), key
    np.testing.assert_array_equal(bm.mean, mean)
    np.testing.assert_array_equal(bm.std, std)
    assert bm.metadata["version"] == bm.bundle.manifest.components["checkpoint"].version == "cnn-v1"


def test_weight_changed_after_load_bundle_is_refused(built, monkeypatch):
    # Check-then-use: one weight byte changes after load_bundle verified the folder and before
    # load_checkpoint reads it. load_checkpoint hashes the bytes it loads against the manifest's
    # checkpoint_sha256, so the change is refused instead of being loaded.
    directory = built[0]
    weights = directory / CHECKPOINT_DIR / "weights.safetensors"
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


def test_checkpoint_version_mismatch_is_refused(built):
    # Through the public path: the manifest claims another model version than checkpoint.json.
    directory = built[0]
    path = directory / "bundle_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["components"]["checkpoint"]["version"] = "cnn-v2"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint: version mismatch"):
        load_bundle_model(directory)


def test_model_loader_checks_loaded_version_itself(built, monkeypatch):
    # The model loader's own check, independent of load_bundle: a verified bundle whose manifest
    # names another version than the metadata load_checkpoint returns is refused.
    directory = built[0]
    real_load_bundle = loader_module.load_bundle

    def other_version(path):
        b = real_load_bundle(path)
        ck = dataclasses.replace(b.manifest.components["checkpoint"], version="cnn-v2")
        manifest = dataclasses.replace(b.manifest, components={**b.manifest.components, "checkpoint": ck})
        return dataclasses.replace(b, manifest=manifest)

    monkeypatch.setattr(loader_module, "load_bundle", other_version)
    with pytest.raises(ValueError, match=r"checkpoint: version mismatch \(manifest 'cnn-v2', loaded 'cnn-v1'\)"):
        load_bundle_model(directory)


def test_model_loader_does_not_import_research_code():
    # Used by the API (no Evaluation or Tuning service, SDD Fig. 1).
    tree = ast.parse((ROOT / "src" / "bundle" / "model_loader.py").read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert [n for n in names if n.startswith(("src.evaluation", "src.tuning"))] == []
