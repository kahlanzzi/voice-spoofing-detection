"""The real analyzer (api/bundle_analyzer.py) on a synthetic bundle: refusals at start, scoring parity with the
research path, one load per worker, and what the API reports and stores. Nothing here reads F4 or a dataset."""
from __future__ import annotations

import asyncio
import dataclasses
import functools
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tests._torch import require_torch

torch = require_torch()

import api.bundle_analyzer  # noqa: E402
from api.bundle_analyzer import BundleAnalyzer  # noqa: E402
from api.main import create_app  # noqa: E402
from api.pool import NEEDS_SELECTOR_LOOP, AnalysisPool, PoolStartError  # noqa: E402
from src.preprocessing.features import extract_features  # noqa: E402
from src.risk.decisions import three_way_decisions  # noqa: E402
from src.tuning.risk_run import score_clips  # noqa: E402
from tests.api._bundle import ROOT, FailingWarmup, RecordingAnalyzer, build_synthetic_bundle  # noqa: E402
from tests.api.conftest import signed_in_client  # noqa: E402

posix_only = pytest.mark.skipif(sys.platform == "win32", reason=NEEDS_SELECTOR_LOOP)
SCORING_MODULES = ("api.bundle_analyzer", "src.bundle.model_loader", "src.preprocessing.waveform",
                   "src.preprocessing.features", "src.risk.policy", "src.training.normalization")


@pytest.fixture(autouse=True)
def restore_torch_threads():
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


def manifest_sha256(bundle: Path) -> str:
    import hashlib
    return hashlib.sha256((bundle / "bundle_manifest.json").read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path) -> Path:
    return build_synthetic_bundle(tmp_path / "made")


@pytest.fixture
def pin(bundle) -> str:
    return manifest_sha256(bundle)


def recording(seconds=3.0, rate=44100, channels=2, fmt="WAV", seed=0, level=0.2) -> bytes:
    rng = np.random.default_rng(seed)
    samples = rng.uniform(-level, level, (int(seconds * rate), channels))
    samples *= np.linspace(0.2, 1.0, len(samples))[:, None]
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format=fmt)
    return buffer.getvalue()


def write(folder: Path, name: str, data: bytes) -> Path:
    path = folder / name
    path.write_bytes(data)
    return path


# --- start-up refusals ----------------------------------------------------------------------------

@posix_only
def test_wrong_manifest_hash_refuses_to_start(config, bundle, pin):
    wrong = ("0" if pin[0] != "0" else "1") + pin[1:]
    app = create_app(config, analyzer_factory=functools.partial(BundleAnalyzer, str(bundle), wrong))
    with pytest.raises(PoolStartError, match="is not the pinned"):
        with TestClient(app):
            pass
    assert app.state.pool.worker_pids() == set()


@posix_only
def test_a_changed_weight_refuses_to_start(config, bundle, pin):
    weights = bundle / "checkpoint" / "weights.safetensors"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 1
    weights.write_bytes(bytes(data))
    with pytest.raises(PoolStartError, match="SHA-256 mismatch"):
        with TestClient(create_app(config, analyzer_factory=functools.partial(BundleAnalyzer, str(bundle), pin))):
            pass


@posix_only
def test_failing_warmup_refuses_to_start_with_a_clear_message(config, bundle, pin, tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    factory = functools.partial(FailingWarmup, str(bundle), pin, str(control))
    with pytest.raises(PoolStartError, match="bundle warm-up failed: ArithmeticError: no such kernel"):
        with TestClient(create_app(config, analyzer_factory=factory)):
            pass
    pids = [int(p.read_text()) for p in control.glob("build_*")]
    assert pids
    if sys.platform == "linux":
        assert not [pid for pid in pids if Path(f"/proc/{pid}").exists()]


def test_v2_architecture_is_refused(bundle, pin, monkeypatch):
    load = BundleAnalyzer._load
    monkeypatch.setattr(BundleAnalyzer, "_load",
                        lambda self, d: dataclasses.replace(load(self, d), architecture="crnn"))
    with pytest.raises(ValueError, match="architecture 'crnn' is not served"):
        BundleAnalyzer(str(bundle), pin)


# --- scoring --------------------------------------------------------------------------------------

CLIPS = [("WAV", 44100, 2), ("WAV", 16000, 1), ("FLAC", 48000, 2), ("MP3", 44100, 1), ("MP3", 16000, 2)]


def test_logits_and_decisions_equal_score_clips(bundle, pin, tmp_path):
    """The API's logit is score_clips' at batch 1, bit for bit; at batch 256 within float32 rounding. The policy
    outcome is recomputed here from the calibration and three_way_decisions."""
    analyzer = BundleAnalyzer(str(bundle), pin)
    loaded = analyzer._load(str(bundle))
    policy = loaded.bundle.policy
    paths, features, logits = [], [], []
    for i, (fmt, rate, channels) in enumerate(CLIPS):
        path = write(tmp_path, f"clip{i}.{fmt.lower()}", recording(1.0 + i, rate, channels, fmt, seed=i, level=0.05 * (i + 1)))
        waveform, read_rate = sf.read(path, dtype="float32", always_2d=False)
        features.append(extract_features(waveform, read_rate, loaded.bundle.profile)[1])
        logits.append(analyzer.logit(waveform, read_rate))
        paths.append(path)
    features = np.stack(features)
    assert len(set(logits)) == len(logits)  # the clips differ, so a constant score would not pass
    one = score_clips(loaded.model, loaded.mean, loaded.std, features, 1)
    many = score_clips(loaded.model, loaded.mean, loaded.std, features, 256)
    np.testing.assert_array_equal(np.array(logits), one)
    np.testing.assert_allclose(many, one, rtol=0, atol=5e-5)
    p = policy.calibration.transform(one)
    decisions = three_way_decisions(p, policy.threshold, policy.half_width)
    for path, pi, decision in zip(paths, p, decisions):
        outcome = analyzer.analyze(path)
        assert outcome.prediction == decision
        assert outcome.confidence == round(max(pi, 1 - pi), 2)
        assert outcome.risk_level == {"authentic": "LOW", "uncertain": "MEDIUM", "synthetic": "HIGH"}[decision]


def test_batch_256_gives_the_same_predictions_and_risk_levels(tmp_path):
    """300 random feature arrays, calibrated so that all three risk levels occur (else equality says nothing)."""
    plain = build_synthetic_bundle(tmp_path / "plain")
    loaded = api.bundle_analyzer.BundleAnalyzer._load(None, str(plain))
    rng = np.random.default_rng(0)
    level = rng.uniform(-3, 3, (300, 1, 1, 1))
    features = ((level + rng.normal(size=(300, 1, 64, 197))) * loaded.std[None, None, :, None]
                + loaded.mean[None, None, :, None]).astype(np.float32)
    one = score_clips(loaded.model, loaded.mean, loaded.std, features, 1)
    many = score_clips(loaded.model, loaded.mean, loaded.std, features, 256)
    np.testing.assert_allclose(many, one, rtol=0, atol=5e-5)
    # Standardize the logits through the calibration: p then spans the band.
    a = float(2.0 / one.std())
    policy = dataclasses.replace(loaded.bundle.policy, calibration=dataclasses.replace(
        loaded.bundle.policy.calibration, a=a, b=float(-a * one.mean())))
    first, second = policy.decide(one), policy.decide(many)
    assert {d.risk_level for d in first} == {"LOW", "MEDIUM", "HIGH"}
    assert [d.prediction for d in first] == [d.prediction for d in second]
    assert [d.risk_level for d in first] == [d.risk_level for d in second]


def test_profile_path_is_never_opened_after_load(bundle, pin, tmp_path):
    path = write(tmp_path, "clip.wav", recording())
    analyzer = BundleAnalyzer(str(bundle), pin)
    before = analyzer.analyze(path)
    for file in [p for p in bundle.rglob("*") if p.is_file()]:
        file.write_bytes(b"not the verified bytes")
    assert analyzer.analyze(path) == before
    for file in [p for p in bundle.rglob("*") if p.is_file()]:
        file.unlink()
    assert analyzer.analyze(path) == before


# --- in a worker ----------------------------------------------------------------------------------

@posix_only
def test_real_analyzer_runs_in_a_spawned_worker(bundle, pin, tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    path = write(tmp_path, "clip.wav", recording())

    async def main():
        pool = AnalysisPool(functools.partial(RecordingAnalyzer, str(bundle), pin, str(control)), 1, 120)
        await pool.start()
        try:
            outcomes = []
            for _ in range(2):
                async with pool.slot() as job:
                    outcomes.append(await pool.run(job, path, 60))
            return outcomes
        finally:
            await pool.close()

    outcomes = asyncio.run(main())
    assert outcomes[0] == outcomes[1] == BundleAnalyzer(str(bundle), pin).analyze(path)
    jobs = [json.loads(p.read_text()) for p in control.glob("job_*")]
    assert len(jobs) == 2 and len(list(control.glob("load_*"))) == 1  # one load for two jobs
    assert {j["threads"] for j in jobs} == {1}  # D53 (1)


def test_scoring_path_never_imports_validation_or_benchmark(bundle, pin, tmp_path):
    path = write(tmp_path, "clip.wav", recording())
    code = f"""
import importlib, sys
for name in {SCORING_MODULES!r}: importlib.import_module(name)
from pathlib import Path
from api.bundle_analyzer import BundleAnalyzer
BundleAnalyzer({str(bundle)!r}, {pin!r}).analyze(Path({str(path)!r}))
bad = ("src.preprocessing.validation", "src.preprocessing.benchmark", "src.tuning", "src.evaluation", "src.audit",
       "src.training.train_cnn", "pandas")
print([m for m in bad if m in sys.modules])
"""
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_parent_process_never_imports_torch():
    code = ("import sys, api.main, api.bundle_analyzer; "
            "print([m for m in ('torch', 'librosa', 'numba', 'safetensors', 'src') if m in sys.modules])")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


# --- through the API ------------------------------------------------------------------------------

@posix_only
def test_status_and_record_of_a_real_result(config, bundle, pin):
    config = dataclasses.replace(config, analysis_worker_start_timeout_s=120)
    factory = functools.partial(BundleAnalyzer, str(bundle), pin)
    with signed_in_client(create_app(config, analyzer_factory=factory)) as client:
        status = client.get("/api/status").json()
        response = client.post("/api/analyses", files={"file": ("call.wav", recording(), "audio/wav")})
        assert response.status_code == 201, response.text
        stored = client.app.state.repository.get(response.json()["request_id"])
    versions = {"model_version": "cnn-test", "profile_version": "preprocessing-v1", "policy_version": "policy-test"}
    assert status["versions"] == versions
    assert (status["mock"], status["bundle_manifest_sha256"], status["deferral_on_fit"]) == (False, pin, 0.0909)
    assert {k: response.json()[k] for k in versions} == versions
    assert stored.is_mock is False and stored.model_version == "cnn-test"
