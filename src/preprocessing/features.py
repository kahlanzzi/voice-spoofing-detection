"""Log-Mel and MFCC feature transforms frozen by preprocessing profile v1."""
from __future__ import annotations

import librosa
import numpy as np

from .profile import PreprocessingError, PreprocessingProfile
from .waveform import WaveformResult, preprocess_waveform


def _require_waveform(waveform: np.ndarray, profile: PreprocessingProfile) -> np.ndarray:
    data = np.asarray(waveform, dtype=np.float32)
    if data.shape != (profile.samples,):
        raise PreprocessingError(
            f"model-ready waveform must have shape ({profile.samples},), got {data.shape}"
        )
    if not np.isfinite(data).all():
        raise PreprocessingError("model-ready waveform contains NaN or infinity")
    return data


def log_mel_spectrogram(waveform: np.ndarray, profile: PreprocessingProfile) -> np.ndarray:
    """Return CNN input with shape (1, 64, 197) for profile v1."""
    data = _require_waveform(waveform, profile)
    f = profile.features
    mel = librosa.feature.melspectrogram(
        y=data,
        sr=profile.sample_rate,
        n_fft=int(f["n_fft"]),
        win_length=int(f["win_length"]),
        hop_length=int(f["hop_length"]),
        window=str(f["window"]),
        center=bool(f["center"]),
        power=float(f["mel_power"]),
        n_mels=int(f["n_mels"]),
        fmin=float(f["fmin_hz"]),
        fmax=float(f["fmax_hz"]),
    )
    result = np.log(np.maximum(mel, float(f["log_epsilon"]))).astype(np.float32)[None, :, :]
    expected = (1, int(f["n_mels"]), profile.expected_log_mel_frames)
    if result.shape != expected or not np.isfinite(result).all():
        raise PreprocessingError(f"log-Mel output must be finite with shape {expected}, got {result.shape}")
    return result


def mfcc_summary(waveform: np.ndarray, profile: PreprocessingProfile) -> np.ndarray:
    """Return 20 MFCC means followed by 20 standard deviations."""
    log_mel = log_mel_spectrogram(waveform, profile)[0]
    f = profile.features
    mfcc = librosa.feature.mfcc(
        S=log_mel,
        n_mfcc=int(f["mfcc_count"]),
        dct_type=int(f["mfcc_dct_type"]),
        norm=str(f["mfcc_norm"]),
        lifter=0,
    )
    result = np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1))).astype(np.float32)
    expected = (2 * int(f["mfcc_count"]),)
    if result.shape != expected or not np.isfinite(result).all():
        raise PreprocessingError(f"MFCC summary must be finite with shape {expected}, got {result.shape}")
    return result


def extract_features(
    waveform: np.ndarray,
    sample_rate: int,
    profile: PreprocessingProfile,
    *,
    channel_axis: int = -1,
) -> tuple[WaveformResult, np.ndarray, np.ndarray]:
    """Shared entry point: waveform, CNN log-Mel and baseline MFCC outputs."""
    prepared = preprocess_waveform(
        waveform, sample_rate, profile, channel_axis=channel_axis
    )
    return (
        prepared,
        log_mel_spectrogram(prepared.waveform, profile),
        mfcc_summary(prepared.waveform, profile),
    )
