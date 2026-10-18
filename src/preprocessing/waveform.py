"""Waveform stage of preprocessing profile v1."""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .profile import PreprocessingError, PreprocessingProfile


_EPS = 1.0e-12


@dataclass(frozen=True)
class WaveformResult:
    waveform: np.ndarray
    speech_mask: np.ndarray
    speech_fraction: float
    input_sample_rate: int
    trimmed: bool


def _mono(waveform: np.ndarray, channel_axis: int) -> np.ndarray:
    data = np.asarray(waveform)
    if data.ndim == 1:
        mono = data
    elif data.ndim == 2:
        axis = channel_axis if channel_axis >= 0 else data.ndim + channel_axis
        if axis not in (0, 1):
            raise PreprocessingError(f"channel_axis must identify axis 0 or 1, got {channel_axis}")
        if data.shape[axis] < 1:
            raise PreprocessingError("waveform has zero channels")
        mono = data.mean(axis=axis)
    else:
        raise PreprocessingError(f"waveform must be one- or two-dimensional, got shape {data.shape}")
    mono = np.asarray(mono, dtype=np.float32)
    if mono.size == 0:
        raise PreprocessingError("waveform is empty")
    if not np.isfinite(mono).all():
        raise PreprocessingError("waveform contains NaN or infinity")
    return mono


def _resample(mono: np.ndarray, sample_rate: int, profile: PreprocessingProfile) -> np.ndarray:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise PreprocessingError(f"sample_rate must be a positive integer, got {sample_rate!r}")
    if sample_rate == profile.sample_rate:
        return mono.astype(np.float32, copy=True)
    result = librosa.resample(
        mono,
        orig_sr=int(sample_rate),
        target_sr=profile.sample_rate,
        res_type=str(profile.resampling["method"]),
        fix=bool(profile.resampling["fix_length"]),
        scale=bool(profile.resampling["scale"]),
    )
    return np.asarray(result, dtype=np.float32)


def speech_mask(waveform: np.ndarray, profile: PreprocessingProfile) -> np.ndarray:
    """Return one Boolean speech decision per deterministic RMS frame."""
    frame = round(float(profile.silence["frame_ms"]) * profile.sample_rate / 1000.0)
    hop = round(float(profile.silence["hop_ms"]) * profile.sample_rate / 1000.0)
    if waveform.size <= frame:
        count = 1
    else:
        count = 1 + int(np.ceil((waveform.size - frame) / hop))
    total = (count - 1) * hop + frame
    padded = np.pad(waveform, (0, max(0, total - waveform.size)))
    starts = np.arange(count) * hop
    frames = np.stack([padded[start:start + frame] for start in starts])
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    dbfs = 20.0 * np.log10(rms + _EPS)
    return dbfs > float(profile.silence["threshold_dbfs"])


def _trim_edges(waveform: np.ndarray, mask: np.ndarray, profile: PreprocessingProfile) -> tuple[np.ndarray, bool]:
    if not bool(profile.silence["trim_edges"]) or not mask.any():
        return waveform, False
    frame = round(float(profile.silence["frame_ms"]) * profile.sample_rate / 1000.0)
    hop = round(float(profile.silence["hop_ms"]) * profile.sample_rate / 1000.0)
    voiced = np.flatnonzero(mask)
    start = int(voiced[0] * hop)
    end = min(waveform.size, int(voiced[-1] * hop + frame))
    return waveform[start:end], start > 0 or end < waveform.size


def _normalize_loudness(waveform: np.ndarray, profile: PreprocessingProfile) -> np.ndarray:
    if profile.loudness["mode"] == "none":
        return waveform
    rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
    if rms <= _EPS:
        return waveform
    current_dbfs = 20.0 * np.log10(rms)
    gain_db = min(
        float(profile.loudness["target_dbfs"]) - current_dbfs,
        float(profile.loudness["max_gain_db"]),
    )
    scaled = waveform * np.float32(10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(scaled)))
    ceiling = float(profile.loudness["peak_ceiling"])
    if peak > ceiling:
        scaled = scaled * np.float32(ceiling / peak)
    return np.asarray(scaled, dtype=np.float32)


def _fix_length(waveform: np.ndarray, samples: int) -> np.ndarray:
    if waveform.size < samples:
        missing = samples - waveform.size
        left = missing // 2
        return np.pad(waveform, (left, missing - left)).astype(np.float32, copy=False)
    if waveform.size > samples:
        excess = waveform.size - samples
        left = excess // 2
        return waveform[left:left + samples].astype(np.float32, copy=False)
    return waveform.astype(np.float32, copy=False)


def preprocess_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    profile: PreprocessingProfile,
    *,
    channel_axis: int = -1,
) -> WaveformResult:
    """Apply the shared mono/resample/silence/level/length pipeline."""
    mono = _mono(waveform, channel_axis)
    resampled = _resample(mono, sample_rate, profile)
    mask = speech_mask(resampled, profile)
    fraction = float(mask.mean())
    trimmed, did_trim = _trim_edges(resampled, mask, profile)
    normalized = _normalize_loudness(trimmed, profile)
    fixed = _fix_length(normalized, profile.samples)
    if fixed.shape != (profile.samples,) or fixed.dtype != np.float32 or not np.isfinite(fixed).all():
        raise PreprocessingError("waveform preprocessing violated the output contract")
    return WaveformResult(
        waveform=fixed,
        speech_mask=mask,
        speech_fraction=fraction,
        input_sample_rate=int(sample_rate),
        trimmed=did_trim,
    )
