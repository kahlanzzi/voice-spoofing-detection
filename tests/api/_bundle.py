"""A synthetic bundle (build_bundle, tiny real CNN, no dataset) and analyzers that run in spawned workers.

Importable without torch, as a spawned worker and the collection of other tests need."""
from __future__ import annotations

import json
import os
from pathlib import Path

from api.bundle_analyzer import BundleAnalyzer

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs" / "preprocessing.yaml"
LEDGER = "c9fc2e230c0883028459d0b2b8de4f6ee537676cd5a4e259e9d0f545c60943cc"
BAND = {"threshold": 0.5, "half_width": 0.2}


def build_synthetic_bundle(folder: Path, a: float = 1.0, b: float = 0.0) -> Path:
    """A verified v1-format bundle; a and b are its calibration."""
    import numpy as np
    import torch
    from src.bundle.bundle import build_bundle
    from src.models.cnn import INPUT_SHAPE, build_cnn
    from src.models.cnn_checkpoint import checkpoint_sha256, save_checkpoint
    from src.preprocessing.profile import load_profile
    from src.risk.calibration import PlattCalibration
    from src.risk.policy import RiskPolicy

    torch.manual_seed(2026)
    model = build_cnn(3).train()
    with torch.no_grad():
        for _ in range(2):
            model(torch.randn(4, *INPUT_SHAPE))  # non-default BatchNorm statistics
    rng = np.random.default_rng(2026)
    mean, std = rng.normal(-6.0, 2.0, 64).astype(np.float32), rng.uniform(0.5, 3.0, 64).astype(np.float32)
    meta = {"schema_version": 1, "version": "cnn-test", "n_blocks": 3, "channels": [16, 32, 64], "dropout": 0.3,
            "input_shape": [1, 64, 197], "learning_rate": 0.001, "seed": 0, "best_epoch": 7, "val_a_eer": 0.0125,
            "code_commit": "a" * 40, "environment": {"python": "test"},
            "profile_fingerprint": load_profile(PROFILE).fingerprint(), "f2_manifest_sha256": "c" * 64,
            "train_cache_npz_sha256": "d" * 64, "val_a_cache_npz_sha256": "e" * 64,
            "determinism": {"seed": 0}}
    folder.mkdir(parents=True)
    checkpoint = save_checkpoint(model.eval(), mean, std, meta, folder / "seed_0")
    policy = folder / "risk_policy.json"
    policy.write_text(RiskPolicy(
        version="policy-test", calibration=PlattCalibration(a=a, b=b, n_spoof=10, n_bonafide=12), **BAND,
        target_deferral=0.10, n_spoof_fit=10, n_bonafide_fit=12, deferral_on_fit=0.0909).to_json(), encoding="utf-8")
    build_bundle(folder / "bundle", checkpoint_dir=checkpoint, expected_checkpoint_sha256=checkpoint_sha256(checkpoint),
                 profile=PROFILE, policy=policy, bundle_version="bundle-test", model_version="cnn-test",
                 profile_version="preprocessing-v1", code_commit="0" * 40, protocol_version="1.0",
                 f2_ledger_sha256=LEDGER, created_at="2026-11-24T12:00:00Z")
    return folder / "bundle"


class RecordingAnalyzer(BundleAnalyzer):
    """Writes one file per bundle load and one per job (with the worker's torch thread count)."""

    def __init__(self, bundle_dir: str, expected_manifest_sha256: str, control: str):
        self.control = Path(control)
        super().__init__(bundle_dir, expected_manifest_sha256)

    def _load(self, bundle_dir):
        (self.control / f"load_{os.getpid()}_{len(list(self.control.glob('load_*')))}").touch()
        return super()._load(bundle_dir)

    def analyze(self, path):
        import torch
        outcome = super().analyze(path)
        (self.control / f"job_{len(list(self.control.glob('job_*')))}").write_text(
            json.dumps({"pid": os.getpid(), "threads": torch.get_num_threads()}))
        return outcome


class FailingWarmup(BundleAnalyzer):
    """The model raises on the first input it sees, which is the warm-up's."""

    def __init__(self, bundle_dir: str, expected_manifest_sha256: str, control: str):
        (Path(control) / f"build_{os.getpid()}").write_text(str(os.getpid()))
        super().__init__(bundle_dir, expected_manifest_sha256)

    def _load(self, bundle_dir):
        import dataclasses

        class Broken:
            def eval(self): return self

            def __call__(self, x): raise ArithmeticError("no such kernel")

        return dataclasses.replace(super()._load(bundle_dir), model=Broken())
