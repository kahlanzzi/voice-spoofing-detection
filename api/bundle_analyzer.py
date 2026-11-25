"""The real analyzer: a frozen ModelBundle behind the §8 seam (D48 (1), D59 (5), D60; docs/api_design.md §8).

Built once per pool worker by the start-up hook (api/pool.py); raising here refuses the start. Each worker
verifies the whole bundle with load_bundle_model, checks its manifest SHA-256 against the pinned value and runs
one warm-up, so a file, library or cache-folder problem stops the start instead of failing the first upload
(librosa loads numba, which needs a writable cache folder, only at its first use).

Scoring is that of src.tuning.risk_run.score_clips at batch 1, which the image does not hold (research code):
decode with soundfile (float32), preprocess_waveform and log-Mel with LoadedBundle.profile (never profile_path),
apply_bin_stats, one forward in eval mode under no_grad; the logit then goes to the bundled policy, whose
decide() applies the calibration and three_way_decisions. tests/api/test_api_bundle_analyzer.py compares both.

The API process imports this module to name the factory, so nothing heavy is imported at the top: torch, librosa
and src load inside the worker only.
"""
from __future__ import annotations

from pathlib import Path

from api.analyzer import AnalysisOutcome, AnalyzerInfo

ARCHITECTURE = "cnn"  # model v1 (F4); the F5 architectures get their scoring in the F5 change
WARMUP_RATE = 44100   # not the profile's rate, so the warm-up also runs the resampler
WARMUP_SECONDS = 2


class BundleAnalyzer:
    def __init__(self, bundle_dir: str, expected_manifest_sha256: str):
        import torch

        torch.set_num_threads(1)  # D53 (1)
        loaded = self._load(bundle_dir)
        actual = loaded.bundle.manifest_sha256
        if actual != expected_manifest_sha256:
            raise ValueError(f"bundle {bundle_dir}: manifest SHA-256 {actual} is not the pinned "
                             f"{expected_manifest_sha256}")
        if loaded.architecture != ARCHITECTURE:
            raise ValueError(f"bundle {bundle_dir}: architecture {loaded.architecture!r} is not served "
                             f"(only {ARCHITECTURE!r})")
        self._model = loaded.model
        self._mean, self._std = loaded.mean, loaded.std
        self._profile = loaded.bundle.profile
        self._policy = loaded.bundle.policy
        components = loaded.bundle.manifest.components
        self._versions = {"model_version": components["checkpoint"].version,
                          "profile_version": components["profile"].version,
                          "policy_version": components["policy"].version}
        self._manifest_sha256 = actual
        self._warm_up()

    def _load(self, bundle_dir: str):
        from src.bundle.model_loader import load_bundle_model

        return load_bundle_model(Path(bundle_dir))

    def _warm_up(self) -> None:
        import numpy as np

        try:
            self.logit(np.zeros(WARMUP_RATE * WARMUP_SECONDS, dtype=np.float32), WARMUP_RATE)
        except Exception as exc:
            raise RuntimeError(f"bundle warm-up failed: {type(exc).__name__}: {exc}") from exc

    def logit(self, waveform, sample_rate: int) -> float:
        """The raw score of one decoded recording: score_clips at batch 1."""
        import torch
        from src.preprocessing.features import log_mel_spectrogram
        from src.preprocessing.waveform import preprocess_waveform
        from src.training.normalization import apply_bin_stats

        prepared = preprocess_waveform(waveform, sample_rate, self._profile)
        features = log_mel_spectrogram(prepared.waveform, self._profile)[None]  # (1, 1, 64, 197)
        x = torch.from_numpy(apply_bin_stats(features, self._mean, self._std))
        self._model.eval()
        with torch.no_grad():
            score = self._model(x).cpu()
        return float(score.numpy().astype("float64")[0])

    def analyze(self, path: Path) -> AnalysisOutcome:
        import soundfile as sf

        # As src.preprocessing.validation.read_verified_audio decodes the training and evaluation clips.
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        (decision,) = self._policy.decide([self.logit(waveform, int(sample_rate))])
        return AnalysisOutcome(prediction=decision.prediction, confidence=decision.confidence,
                               risk_level=decision.risk_level, **self._versions)

    def info(self) -> AnalyzerInfo:
        return AnalyzerInfo(ready=True, deferral_on_fit=self._policy.deferral_on_fit,
                            bundle_manifest_sha256=self._manifest_sha256, mock=False, **self._versions)
