"""Deterministic Stage 14 robustness conditions (Protocol section 9; D3, D41, D58).

This module contains no dataset access and no model scoring.  It strictly loads the reviewed
condition grid and applies one degradation to one already-verified 16 kHz mono waveform.  The
caller remains responsible for reading FoR through the Stage 14 runner and for applying unchanged
F3 preprocessing after this function returns.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import yaml
from scipy.signal import butter, sosfiltfilt

from src.augmentation.codec import FFmpegCodecBackend
from src.evaluation.clip_scores import ClipScores
from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.metrics import eer


SAMPLE_RATE = 16_000
NOISE_SNRS = (30, 20, 10)
MP3_BITRATES = (128, 64, 32)
LOW_PASS_CUTOFFS = (6000, 4000, 3400)
CONDITION_IDS = (
    *(f"white_gaussian_noise_{value}db" for value in NOISE_SNRS),
    *(f"mp3_{value}kbps" for value in MP3_BITRATES),
    *(f"low_pass_{value}hz" for value in LOW_PASS_CUTOFFS),
)
_NOISE_SEED_RULE = "sha256_global_seed_clip_id_condition_id_uint64_little"
_LOW_PASS_DESIGN = "butterworth_sosfiltfilt"


class RobustnessError(RuntimeError):
    """The Stage 14 config, condition, or waveform violates the reviewed contract."""


def _mapping(where: str, value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RobustnessError(f"{where} must be a mapping")
    missing, unknown = sorted(expected - set(value)), sorted(set(value) - expected)
    if missing or unknown:
        raise RobustnessError(f"{where}: missing keys {missing}, unknown keys {unknown}")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class RobustnessConfig:
    schema_version: int
    config_version: str
    sample_rate: int
    conditions: Mapping[str, Any]
    bootstrap: Mapping[str, Any]
    endpoints: Mapping[str, Any]


def load_robustness_config(path: str | Path) -> RobustnessConfig:
    """Load the exact Stage 14 v1 grid; reject missing, extra, reordered, or changed values."""
    path = Path(path)
    if not path.is_file():
        raise RobustnessError(f"robustness config does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RobustnessError(f"cannot read robustness config {path}: {error}") from error
    top = _mapping(
        "robustness", raw,
        {"schema_version", "config_version", "decisions", "sample_rate_hz", "conditions",
         "bootstrap", "endpoints"},
    )
    if (top["schema_version"], top["config_version"], top["decisions"],
            top["sample_rate_hz"]) != (1, "robustness-v1", ["D3", "D41", "D58"], SAMPLE_RATE):
        raise RobustnessError("schema, decisions, or sample rate differ from Stage 14 v1")

    conditions = _mapping(
        "conditions", top["conditions"], {"white_gaussian_noise", "mp3", "low_pass"})
    noise = _mapping("conditions.white_gaussian_noise", conditions["white_gaussian_noise"],
                     {"snr_db", "realization"})
    realization = _mapping("conditions.white_gaussian_noise.realization", noise["realization"],
                           {"global_seed", "per_clip_seed", "power"})
    if noise["snr_db"] != list(NOISE_SNRS) or realization != {
        "global_seed": 2026, "per_clip_seed": _NOISE_SEED_RULE, "power": "mean_square"}:
        raise RobustnessError("white-Gaussian-noise grid or realization policy drifted")
    mp3 = _mapping("conditions.mp3", conditions["mp3"], {"bitrate_kbps"})
    if mp3["bitrate_kbps"] != list(MP3_BITRATES):
        raise RobustnessError("MP3 bitrate grid drifted")
    low_pass = _mapping("conditions.low_pass", conditions["low_pass"],
                        {"cutoff_hz", "design", "order"})
    if low_pass != {"cutoff_hz": list(LOW_PASS_CUTOFFS), "design": _LOW_PASS_DESIGN, "order": 8}:
        raise RobustnessError("low-pass grid or filter design drifted")

    bootstrap = _mapping("bootstrap", top["bootstrap"],
                         {"unit", "n_resamples", "seed", "confidence"})
    if bootstrap != {"unit": "d14_group", "n_resamples": 2000, "seed": 2026,
                     "confidence": 0.95}:
        raise RobustnessError("bootstrap policy differs from D41")
    endpoints = _mapping("endpoints", top["endpoints"],
                         {"meaningful_increase", "baseline_primary", "cnn_h6"})
    if endpoints != {
        "meaningful_increase": 0.05,
        "baseline_primary": "eer_increase_low_pass_3400_vs_clean",
        "cnn_h6": "real_false_synthetic_rate_increase_forced_binary_low_pass_3400_vs_clean",
    }:
        raise RobustnessError("Stage 14 v1 endpoint definitions drifted")
    return RobustnessConfig(1, "robustness-v1", SAMPLE_RATE, _freeze(conditions),
                            _freeze(bootstrap), _freeze(endpoints))


def _waveform(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 1 or array.size == 0:
        raise RobustnessError(
            f"waveform must be one non-empty float32 vector, got {array.dtype} {array.shape}")
    if not np.isfinite(array).all():
        raise RobustnessError("waveform contains NaN or infinity")
    return array


def _noise_seed(global_seed: int, clip_id: str, condition_id: str) -> int:
    if not isinstance(clip_id, str) or not clip_id:
        raise RobustnessError("clip_id must be non-empty text")
    payload = f"{global_seed}\n{clip_id}\n{condition_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _white_noise(x: np.ndarray, snr_db: int, clip_id: str, global_seed: int) -> np.ndarray:
    condition_id = f"white_gaussian_noise_{snr_db}db"
    signal_power = float(np.mean(np.square(x, dtype=np.float64)))
    if signal_power <= 0.0:
        raise RobustnessError(f"{clip_id}: cannot define SNR for a zero-power waveform")
    rng = np.random.default_rng(_noise_seed(global_seed, clip_id, condition_id))
    noise = rng.standard_normal(x.size)
    noise -= noise.mean()
    noise_power = float(np.mean(np.square(noise)))
    scale = np.sqrt(signal_power / ((10.0 ** (snr_db / 10.0)) * noise_power))
    return np.asarray(x.astype(np.float64) + noise * scale, dtype=np.float32)


def _low_pass(x: np.ndarray, cutoff_hz: int, order: int) -> np.ndarray:
    sos = butter(order, cutoff_hz, btype="lowpass", fs=SAMPLE_RATE, output="sos")
    try:
        output = sosfiltfilt(sos, x.astype(np.float64))
    except ValueError as error:
        raise RobustnessError(f"low-pass filtering failed: {error}") from error
    return np.asarray(output, dtype=np.float32)


def apply_condition(
    waveform: np.ndarray,
    *,
    condition_id: str,
    clip_id: str,
    config: RobustnessConfig,
    codec_backend: FFmpegCodecBackend | None = None,
) -> np.ndarray:
    """Apply one reviewed Stage 14 condition and return finite float32 with unchanged length."""
    x = _waveform(waveform)
    if config.sample_rate != SAMPLE_RATE:
        raise RobustnessError(f"config sample rate must be {SAMPLE_RATE}, got {config.sample_rate}")
    if condition_id not in CONDITION_IDS:
        raise RobustnessError(f"unknown condition_id {condition_id!r}")
    if condition_id.startswith("white_gaussian_noise_"):
        snr = int(condition_id.removeprefix("white_gaussian_noise_").removesuffix("db"))
        out = _white_noise(x, snr, clip_id,
                           int(config.conditions["white_gaussian_noise"]["realization"]["global_seed"]))
    elif condition_id.startswith("mp3_"):
        if codec_backend is None:
            raise RobustnessError("MP3 condition requires the pinned FFmpeg codec backend")
        bitrate = int(condition_id.removeprefix("mp3_").removesuffix("kbps"))
        out = codec_backend.round_trip(x, {"kind": "mp3", "bitrate_kbps": bitrate})
    else:
        cutoff = int(condition_id.removeprefix("low_pass_").removesuffix("hz"))
        out = _low_pass(x, cutoff, int(config.conditions["low_pass"]["order"]))
    out = np.asarray(out)
    if out.dtype != np.float32 or out.shape != x.shape or not np.isfinite(out).all():
        raise RobustnessError(
            f"{condition_id}: output must be finite float32 {x.shape}, got {out.dtype} {out.shape}")
    return out


def _paired_rows(clean: ClipScores, degraded: ClipScores) -> None:
    """Require the exact Stage 11 row set and model identity before any paired statistic."""
    for name in ("clip_id", "label", "group_id"):
        if getattr(clean, name) != getattr(degraded, name):
            raise RobustnessError(f"clean and degraded {name} rows differ")
    for name in ("model_id", "model_role", "model_artifact_sha256", "manifest_sha256"):
        if clean.metadata[name] != degraded.metadata[name]:
            raise RobustnessError(f"clean and degraded {name} differ")
    if clean.metadata["evaluation_purpose"] != "stage11_test":
        raise RobustnessError("clean scores are not from stage11_test")
    if degraded.metadata["evaluation_purpose"] != "stage14_robustness":
        raise RobustnessError("degraded scores are not from stage14_robustness")
    if clean.metadata["model_role"] not in {"baseline", "cnn_bundle_seed"}:
        raise RobustnessError(
            f"Stage 14 v1 does not compare model role {clean.metadata['model_role']!r}")


def _ci_record(value) -> dict[str, Any]:
    return asdict(value)


def compare_condition(
    clean: ClipScores,
    degraded: ClipScores,
    *,
    condition_id: str,
    config: RobustnessConfig,
    forced_binary_threshold: float | None = None,
) -> dict[str, Any]:
    """Compute paired clean-to-condition endpoints with the frozen D14 group bootstrap."""
    if condition_id not in CONDITION_IDS:
        raise RobustnessError(f"unknown condition_id {condition_id!r}")
    _paired_rows(clean, degraded)
    labels = np.array([1 if value == "FAKE" else 0 for value in clean.label], dtype=np.int8)
    n = len(labels)
    settings = config.bootstrap

    def eer_delta(indices: np.ndarray) -> float:
        return eer(labels[indices], degraded.raw_score[indices])[0] - eer(
            labels[indices], clean.raw_score[indices])[0]

    clean_eer = eer(labels, clean.raw_score)[0]
    degraded_eer = eer(labels, degraded.raw_score)[0]
    result: dict[str, Any] = {
        "schema_version": 1,
        "condition_id": condition_id,
        "model_id": clean.metadata["model_id"],
        "model_role": clean.metadata["model_role"],
        "n_rows": n,
        "clean_eer": clean_eer,
        "degraded_eer": degraded_eer,
        "eer_increase": _ci_record(bootstrap_ci(
            eer_delta, n, clusters=clean.group_id,
            n_resamples=int(settings["n_resamples"]), seed=int(settings["seed"]),
            confidence=float(settings["confidence"]),
        )),
    }
    if clean.metadata["model_role"] == "cnn_bundle_seed":
        if clean.p is None or degraded.p is None:
            raise RobustnessError("CNN bundle H6 scores require calibrated probabilities")
        if (not isinstance(forced_binary_threshold, (int, float))
                or not np.isfinite(forced_binary_threshold)
                or not 0.0 < float(forced_binary_threshold) < 1.0):
            raise RobustnessError(
                "CNN bundle H6 requires its finite frozen threshold strictly inside (0, 1)")
        if clean.prediction is None or degraded.prediction is None:
            raise RobustnessError("CNN bundle descriptive rates require three-way predictions")
        threshold_value = float(forced_binary_threshold)
        clean_p = np.asarray(clean.p, dtype=np.float64)
        degraded_p = np.asarray(degraded.p, dtype=np.float64)
        if (clean_p.shape != (n,) or degraded_p.shape != (n,)
                or not np.isfinite(clean_p).all() or not np.isfinite(degraded_p).all()):
            raise RobustnessError("CNN bundle probabilities must be finite and match the row set")
        clean_forced_synthetic = clean_p >= threshold_value
        degraded_forced_synthetic = degraded_p >= threshold_value
        clean_prediction = np.asarray(clean.prediction, dtype=object)
        degraded_prediction = np.asarray(degraded.prediction, dtype=object)
        if not np.any(labels == 0):
            raise RobustnessError("CNN H6 comparison requires at least one REAL clip")

        def false_synthetic_delta(indices: np.ndarray) -> float:
            real = indices[labels[indices] == 0]
            if not len(real):
                raise ValueError("resample contains no REAL clips")
            return float(np.mean(degraded_forced_synthetic[real])
                         - np.mean(clean_forced_synthetic[real]))

        real = labels == 0
        result.update({
            "forced_binary_threshold": threshold_value,
            "clean_real_false_synthetic_rate_forced_binary": float(
                np.mean(clean_forced_synthetic[real])),
            "degraded_real_false_synthetic_rate_forced_binary": float(
                np.mean(degraded_forced_synthetic[real])),
            "real_false_synthetic_rate_increase_forced_binary": _ci_record(bootstrap_ci(
                false_synthetic_delta, n, clusters=clean.group_id,
                n_resamples=int(settings["n_resamples"]), seed=int(settings["seed"]),
                confidence=float(settings["confidence"]),
            )),
            "descriptive_three_way_rates": {
                "clean_real_synthetic_rate": float(
                    np.mean(clean_prediction[real] == "synthetic")),
                "degraded_real_synthetic_rate": float(
                    np.mean(degraded_prediction[real] == "synthetic")),
                "clean_real_uncertain_rate": float(
                    np.mean(clean_prediction[real] == "uncertain")),
                "degraded_real_uncertain_rate": float(
                    np.mean(degraded_prediction[real] == "uncertain")),
            },
        })
    threshold = float(config.endpoints["meaningful_increase"])
    if condition_id == "low_pass_3400hz":
        if clean.metadata["model_role"] == "baseline":
            interval = result["eer_increase"]
            result["primary_supported"] = bool(
                interval["estimate"] >= threshold and interval["low"] > 0.0)
        elif clean.metadata["model_role"] == "cnn_bundle_seed":
            interval = result["real_false_synthetic_rate_increase_forced_binary"]
            result["h6_supported"] = bool(
                interval["estimate"] >= threshold and interval["low"] > 0.0)
    return result
