"""Independent implementation of RawBoost algorithm 1 (D56).

The equations and parameter names follow Tak et al. (2022), section 3.1.  The
pinned official implementation is used only as a synthetic parity oracle in
the tests; no reference source code is copied into this module.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy import signal

from .config import AugmentationError


def _waveform(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value)
    if x.ndim != 1 or x.size == 0:
        raise AugmentationError(f"waveform must be a non-empty vector, got {x.shape}")
    if not np.isfinite(x).all():
        raise AugmentationError("waveform contains NaN or infinity")
    # Keep float32 powers, matching the F3 waveform and the pinned oracle.
    return x.astype(np.float32, copy=True)


def _uniform(rng: Any, low: float, high: float) -> float:
    # RandomState is intentional: the pinned reference uses NumPy's legacy
    # global RNG.  Supplying a RandomState gives exact, isolated parity.
    return float(np.asarray(rng.uniform(low=low, high=high, size=(1,))).item())


def _notch_filter(
    rng: Any,
    *,
    bands: int,
    min_frequency: float,
    max_frequency: float,
    min_bandwidth: float,
    max_bandwidth: float,
    min_coefficients: int,
    max_coefficients: int,
    min_gain_db: float,
    max_gain_db: float,
    sample_rate: int,
) -> np.ndarray:
    coefficients: np.ndarray | float = 1.0
    for _ in range(bands):
        centre = _uniform(rng, min_frequency, max_frequency)
        bandwidth = _uniform(rng, min_bandwidth, max_bandwidth)
        count = int(_uniform(rng, min_coefficients, max_coefficients))
        if count % 2 == 0:
            count += 1
        low = max(0.001, centre - bandwidth / 2.0)
        high = min(sample_rate / 2.0 - 0.001, centre + bandwidth / 2.0)
        if not low < high:
            raise AugmentationError("RawBoost sampled an invalid notch band")
        bandstop = signal.firwin(
            count, [float(low), float(high)], window="hamming", fs=sample_rate,
        )
        coefficients = np.convolve(coefficients, bandstop)
    gain_db = _uniform(rng, min_gain_db, max_gain_db)
    _, response = signal.freqz(coefficients, 1, fs=sample_rate)
    peak = float(np.max(np.abs(response)))
    if peak <= 0 or not np.isfinite(peak):
        raise AugmentationError("RawBoost generated an invalid FIR response")
    return np.asarray(10.0 ** (gain_db / 20.0) * coefficients / peak, dtype=np.float64)


def _filter_fir(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    # Same centred, length-preserving FIR convention as the pinned oracle.
    width = coefficients.shape[0] + 1
    padded = np.pad(x, (0, width), mode="constant")
    filtered = signal.lfilter(coefficients, 1, padded)
    return np.asarray(filtered[width // 2: filtered.shape[0] - width // 2], dtype=np.float64)


def rawboost_algorithm_1(
    waveform: np.ndarray,
    settings: Mapping[str, Any],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Apply D56 linear/non-linear convolutive distortion to one waveform."""
    x = _waveform(waveform)
    if not isinstance(rng, np.random.RandomState):
        raise AugmentationError("RawBoost parity requires an isolated numpy RandomState")
    if settings["algorithm"] != 1 or settings["sample_rate_hz"] != 16000:
        raise AugmentationError("only D56 RawBoost algorithm 1 at 16 kHz is allowed")

    frequency = settings["centre_frequency_hz"]
    bandwidth = settings["bandwidth_hz"]
    taps = settings["fir_coefficients"]
    offset = settings["higher_order_gain_offset_db"]
    output = np.zeros_like(x, dtype=np.float64)
    minimum_gain = maximum_gain = float(settings["linear_gain_db"])

    for index in range(int(settings["nonlinearity_order"])):
        if index == 1:
            # The paper/reference express the higher-order range as the
            # linear gain minus biases 5..20 dB.  Retaining that direction
            # also preserves the pinned oracle's legacy-RNG sequence.
            minimum_gain -= abs(float(offset["max"]))
            maximum_gain -= abs(float(offset["min"]))
        coefficients = _notch_filter(
            rng,
            bands=int(settings["notch_bands"]),
            min_frequency=float(frequency["min"]),
            max_frequency=float(frequency["max"]),
            min_bandwidth=float(bandwidth["min"]),
            max_bandwidth=float(bandwidth["max"]),
            min_coefficients=int(taps["min"]),
            max_coefficients=int(taps["max"]),
            min_gain_db=minimum_gain,
            max_gain_db=maximum_gain,
            sample_rate=int(settings["sample_rate_hz"]),
        )
        output += _filter_fir(np.power(x, index + 1), coefficients)

    output -= float(np.mean(output))
    peak = float(np.max(np.abs(output)))
    if peak > 1.0:
        output /= peak
    result = np.asarray(output, dtype=np.float32)
    if result.shape != x.shape or not np.isfinite(result).all():
        raise AugmentationError("RawBoost violated the waveform output contract")
    return result
