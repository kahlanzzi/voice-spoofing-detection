"""Train-only waveform augmentation orchestrator for model v2 (D50/D56)."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from scipy import signal

from .codec import FFmpegCodecBackend
from .config import AugmentationConfig, AugmentationError
from .rawboost import rawboost_algorithm_1


class CodecBackend(Protocol):
    def round_trip(self, waveform: np.ndarray, option: Mapping[str, Any]) -> np.ndarray: ...


@dataclass(frozen=True)
class AugmentationResources:
    """Already-audited training-half waveforms supplied by the future data loader."""

    noise_waveforms: Sequence[np.ndarray] = field(default_factory=tuple)
    rir_waveforms: Sequence[np.ndarray] = field(default_factory=tuple)


@dataclass(frozen=True)
class AugmentationResult:
    waveform: np.ndarray
    family: str
    parameters: Mapping[str, Any]


def _input_waveform(waveform: np.ndarray) -> np.ndarray:
    value = np.asarray(waveform)
    if value.ndim != 1 or value.size == 0:
        raise AugmentationError(f"waveform must be a non-empty vector, got {value.shape}")
    if not np.isfinite(value).all():
        raise AugmentationError("waveform contains NaN or infinity")
    return value.astype(np.float32, copy=True)


def _resource(value: np.ndarray, name: str) -> np.ndarray:
    item = _input_waveform(value)
    if not np.any(item):
        raise AugmentationError(f"{name} resource must not be all zero")
    return item


class Augmenter:
    """Apply exactly one uniformly selected family, or leave the training item clean."""

    def __init__(
        self,
        config: AugmentationConfig,
        resources: AugmentationResources | None = None,
        codec_backend: CodecBackend | None = None,
    ) -> None:
        self.config = config
        self.resources = resources or AugmentationResources()
        self.codec_backend = codec_backend or FFmpegCodecBackend(config.ffmpeg, config.sample_rate)

    def sample_family(self, rng: np.random.RandomState) -> str:
        return self.config.families[int(rng.randint(0, len(self.config.families)))]

    def sample_codec_option(self, rng: np.random.RandomState) -> Mapping[str, Any]:
        return self.config.codec_options[int(rng.randint(0, len(self.config.codec_options)))]

    @staticmethod
    def _fit_resource(resource: np.ndarray, samples: int, rng: np.random.RandomState) -> np.ndarray:
        if resource.size < samples:
            repeats = int(np.ceil(samples / resource.size))
            return np.tile(resource, repeats)[:samples]
        if resource.size > samples:
            start = int(rng.randint(0, resource.size - samples + 1))
            return resource[start:start + samples]
        return resource

    def _recorded_noise(self, x: np.ndarray, rng: np.random.RandomState) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.resources.noise_waveforms:
            raise AugmentationError("recorded_noise selected but no audited training-half noise is loaded")
        index = int(rng.randint(0, len(self.resources.noise_waveforms)))
        noise = self._fit_resource(
            _resource(self.resources.noise_waveforms[index], "noise"), x.size, rng,
        ).astype(np.float64)
        noise -= float(np.mean(noise))
        signal_rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        noise_rms = float(np.sqrt(np.mean(noise ** 2)))
        if signal_rms <= 0 or noise_rms <= 0:
            raise AugmentationError("recorded-noise mixing requires non-silent signal and noise")
        snr = self.config.recorded_noise["snr_db"]
        snr_db = float(rng.uniform(float(snr["min"]), float(snr["max"])))
        scale = signal_rms / (noise_rms * 10.0 ** (snr_db / 20.0))
        return np.asarray(x.astype(np.float64) + noise * scale, dtype=np.float32), {
            "snr_db": snr_db, "resource_index": index,
        }

    def _reverberation(self, x: np.ndarray, rng: np.random.RandomState) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.resources.rir_waveforms:
            raise AugmentationError("reverberation selected but no audited training-half RIR is loaded")
        index = int(rng.randint(0, len(self.resources.rir_waveforms)))
        impulse = _resource(self.resources.rir_waveforms[index], "RIR").astype(np.float64)
        result = signal.fftconvolve(x.astype(np.float64), impulse, mode="full")[:x.size]
        return np.asarray(result, dtype=np.float32), {"resource_index": index}

    def _gain(self, x: np.ndarray, rng: np.random.RandomState) -> tuple[np.ndarray, dict[str, Any]]:
        settings = self.config.gain
        gain_db = float(rng.uniform(float(settings["min_db"]), float(settings["max_db"])))
        result = x.astype(np.float64) * 10.0 ** (gain_db / 20.0)
        ceiling = 10.0 ** (float(settings["peak_ceiling_dbfs"]) / 20.0)
        peak = float(np.max(np.abs(result)))
        limited = peak > ceiling
        if limited:
            result *= ceiling / peak
        return np.asarray(result, dtype=np.float32), {
            "gain_db": gain_db, "peak_limited": limited,
        }

    def augment(self, waveform: np.ndarray, *, split: str, label: str, seed: int) -> AugmentationResult:
        """Augment one item; refuses every split except train and every unknown label."""
        if split not in self.config.training["allowed_splits"]:
            raise AugmentationError(f"D50 forbids augmentation on split {split!r}")
        if label not in {"REAL", "FAKE"}:
            raise AugmentationError(f"label must be REAL or FAKE, got {label!r}")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise AugmentationError("seed must be an integer")
        x = _input_waveform(waveform)
        rng = np.random.RandomState(int(seed))
        if float(rng.uniform()) >= float(self.config.training["apply_probability"]):
            return AugmentationResult(x, "clean", MappingProxyType({}))

        family = self.sample_family(rng)
        if family == "recorded_noise":
            result, parameters = self._recorded_noise(x, rng)
        elif family == "reverberation":
            result, parameters = self._reverberation(x, rng)
        elif family == "codec":
            option = self.sample_codec_option(rng)
            result = self.codec_backend.round_trip(x, option)
            parameters = {"option": option["id"]}
        elif family == "gain":
            result, parameters = self._gain(x, rng)
        elif family == "rawboost":
            result = rawboost_algorithm_1(x, self.config.rawboost, rng)
            parameters = {"algorithm": 1}
        else:  # strict config makes this unreachable, kept as a fail-loud boundary
            raise AugmentationError(f"unknown augmentation family: {family}")
        output = np.asarray(result, dtype=np.float32)
        if output.shape != x.shape or not np.isfinite(output).all():
            raise AugmentationError(f"{family} violated the waveform output contract")
        return AugmentationResult(output, family, MappingProxyType(parameters))
