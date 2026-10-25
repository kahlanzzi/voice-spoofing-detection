"""Strict loader for the Stage 6 preprocessing profile (Protocol v1.0 section 3)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


F2_LEDGER_SHA256 = "c9fc2e230c0883028459d0b2b8de4f6ee537676cd5a4e259e9d0f545c60943cc"


class PreprocessingError(RuntimeError):
    """Raised when a profile or waveform violates the frozen Stage 6 contract."""


def _keys(where: str, value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreprocessingError(f"{where} must be a mapping")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise PreprocessingError(f"{where}: missing keys {missing}, unknown keys {unknown}")
    return value


def _number(where: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreprocessingError(f"{where} must be numeric, got {value!r}")
    result = float(value)
    if positive and result <= 0:
        raise PreprocessingError(f"{where} must be positive, got {value!r}")
    return result


@dataclass(frozen=True)
class PreprocessingProfile:
    schema_version: int
    profile_version: str
    protocol_version: str
    f2_ledger_sha256: str
    waveform: Mapping[str, Any]
    resampling: Mapping[str, Any]
    silence: Mapping[str, Any]
    loudness: Mapping[str, Any]
    features: Mapping[str, Any]
    validation: Mapping[str, Any]
    benchmark: Mapping[str, Any]

    @property
    def sample_rate(self) -> int:
        return int(self.waveform["sample_rate_hz"])

    @property
    def samples(self) -> int:
        return int(self.waveform["samples"])

    @property
    def expected_log_mel_frames(self) -> int:
        return 1 + (self.samples - int(self.features["n_fft"])) // int(
            self.features["hop_length"]
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_version": self.profile_version,
            "protocol_version": self.protocol_version,
            "f2_ledger_sha256": self.f2_ledger_sha256,
            "waveform": dict(self.waveform),
            "resampling": dict(self.resampling),
            "silence": dict(self.silence),
            "loudness": dict(self.loudness),
            "features": dict(self.features),
            "validation": dict(self.validation),
            "benchmark": dict(self.benchmark),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_profile(path: Path) -> PreprocessingProfile:
    """Load and fully validate a preprocessing profile YAML file."""
    path = Path(path)
    if not path.is_file():
        raise PreprocessingError(f"Preprocessing profile does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PreprocessingError(f"Cannot read preprocessing profile {path}: {exc}") from exc

    top = _keys(
        "profile",
        raw,
        {
            "schema_version", "profile_version", "protocol_version", "f2_ledger_sha256",
            "waveform", "resampling", "silence", "loudness", "features", "validation",
            "benchmark",
        },
    )
    if top["schema_version"] != 1:
        raise PreprocessingError("schema_version must be 1")
    if top["profile_version"] != "preprocessing-v1" or top["protocol_version"] != "1.0":
        raise PreprocessingError("Stage 6 requires profile preprocessing-v1 and Protocol v1.0")
    if top["f2_ledger_sha256"] != F2_LEDGER_SHA256:
        raise PreprocessingError("Profile does not reference the approved D28 F2 hash ledger")

    waveform = _keys(
        "waveform", top["waveform"],
        {"sample_rate_hz", "duration_seconds", "samples", "mono_method", "length_policy"},
    )
    if waveform != {
        "sample_rate_hz": 16000,
        "duration_seconds": 2.0,
        "samples": 32000,
        "mono_method": "mean",
        "length_policy": "center_pad_or_crop",
    }:
        raise PreprocessingError("waveform settings must match D5, D18 and Protocol v1.0")

    resampling = _keys("resampling", top["resampling"], {"method", "fix_length", "scale"})
    if resampling != {"method": "soxr_hq", "fix_length": True, "scale": False}:
        raise PreprocessingError("resampling settings must be soxr_hq/fix_length/no-scale")

    silence = _keys(
        "silence", top["silence"],
        {
            "detector", "frame_ms", "hop_ms", "threshold_dbfs", "trim_edges",
            "all_silent_policy", "external_min_speech_fraction",
        },
    )
    if silence["detector"] != "frame_rms_dbfs" or silence["all_silent_policy"] != "preserve":
        raise PreprocessingError("Stage 6 uses the deterministic frame-RMS detector and preserve policy")
    if silence["trim_edges"] is not True:
        raise PreprocessingError("silence.trim_edges must be true")
    if _number("silence.frame_ms", silence["frame_ms"], positive=True) != 20.0:
        raise PreprocessingError("silence.frame_ms must be 20.0")
    if _number("silence.hop_ms", silence["hop_ms"], positive=True) != 20.0:
        raise PreprocessingError("silence.hop_ms must be 20.0")
    if _number("silence.threshold_dbfs", silence["threshold_dbfs"]) != -50.0:
        raise PreprocessingError("silence.threshold_dbfs must be -50.0")
    if _number("silence.external_min_speech_fraction", silence["external_min_speech_fraction"]) != 0.5:
        raise PreprocessingError("D17 requires external_min_speech_fraction 0.50")

    loudness = _keys(
        "loudness", top["loudness"],
        {
            "mode", "selection_split", "separability_auc_threshold", "target_dbfs",
            "max_gain_db", "peak_ceiling",
        },
    )
    if loudness["mode"] not in {"none", "speech_rms"} or loudness["selection_split"] != "train":
        raise PreprocessingError("loudness mode must be none/speech_rms and selected on train")
    if _number("loudness.separability_auc_threshold", loudness["separability_auc_threshold"]) != 0.75:
        raise PreprocessingError("loudness separability threshold must be 0.75")
    if not 0 < _number("loudness.peak_ceiling", loudness["peak_ceiling"]) <= 1:
        raise PreprocessingError("loudness.peak_ceiling must be in (0, 1]")

    features = _keys(
        "features", top["features"],
        {
            "n_fft", "win_length", "hop_length", "window", "center", "n_mels", "fmin_hz",
            "fmax_hz", "mel_power", "log_epsilon", "mfcc_count", "mfcc_dct_type", "mfcc_norm",
        },
    )
    fixed_features = {
        "n_fft": 512, "win_length": 400, "hop_length": 160, "window": "hann",
        "center": False, "n_mels": 64, "fmin_hz": 0.0, "fmax_hz": 7600.0,
        "mel_power": 1.0, "log_epsilon": 1.0e-6, "mfcc_count": 20,
        "mfcc_dct_type": 2, "mfcc_norm": "ortho",
    }
    if features != fixed_features:
        raise PreprocessingError("feature settings differ from the approved Stage 6 profile")

    validation = _keys(
        "validation", top["validation"],
        {
            "invariance_sample_per_subgroup", "invariance_sample_rates_hz",
            "max_relative_l2_error", "min_cosine_similarity", "shortcut_auc_flag",
            "shortcut_auc_high",
        },
    )
    if validation["invariance_sample_rates_hz"] != [40000, 44100, 48000]:
        raise PreprocessingError("D18 requires invariance sample rates 40/44.1/48 kHz")
    if int(validation["invariance_sample_per_subgroup"]) != 10:
        raise PreprocessingError("invariance sample size must be 10 per subgroup")
    if _number("validation.max_relative_l2_error", validation["max_relative_l2_error"]) != 0.01:
        raise PreprocessingError("invariance relative-L2 tolerance must be 0.01")
    if _number("validation.min_cosine_similarity", validation["min_cosine_similarity"]) != 0.999:
        raise PreprocessingError("invariance cosine tolerance must be 0.999")

    benchmark = _keys(
        "benchmark", top["benchmark"],
        {
            "preprocessing_samples", "candidate_batch_sizes", "max_vram_fraction",
            "warmup_steps", "measured_steps", "mode",
        },
    )
    if benchmark["mode"] != "dry_run_no_optimizer":
        raise PreprocessingError("Stage 6 benchmark must not fit or save a model")
    batches = benchmark["candidate_batch_sizes"]
    if not isinstance(batches, list) or not batches or any(
        isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in batches
    ):
        raise PreprocessingError("benchmark candidate_batch_sizes must be positive integers")

    frozen = {
        name: MappingProxyType(dict(value))
        for name, value in {
            "waveform": waveform, "resampling": resampling, "silence": silence,
            "loudness": loudness, "features": features, "validation": validation,
            "benchmark": benchmark,
        }.items()
    }
    return PreprocessingProfile(
        schema_version=1,
        profile_version=top["profile_version"],
        protocol_version=top["protocol_version"],
        f2_ledger_sha256=top["f2_ledger_sha256"],
        **frozen,
    )
