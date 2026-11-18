"""Pure deterministic recipe used by the frozen V1b selection-set manifest (D50, D56).

The functions in this module are the reusable form of notebook 07's completed recipe.  They do not
know about Colab or Drive paths: callers supply the audited inventory rows, corpus roots and codec
backend.  Resource bytes are hash-checked immediately before decoding.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy import signal

from src.preprocessing.validation import sha256_file

SELECTION_SEED = 2026
CONDITIONS = (
    "clean",
    "recorded_noise_10db",
    "reverberation",
    "opus_16",
    "narrowband_mulaw",
)
RECIPE_COLUMNS = (
    "source_path", "label", "subgroup", "group", "source_file_sha256", "condition",
    "condition_seed", "resource_path", "resource_file_sha256", "resource_channel",
    "parameters_json", "sample_rate_hz", "samples", "waveform_float32_sha256",
)


def waveform_sha256(waveform: np.ndarray) -> str:
    """SHA-256 of canonical little-endian float32 sample bytes."""
    canonical = np.asarray(waveform, dtype=np.float32).astype("<f4", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def derived_seed(source_path: str, condition: str, *, seed: int = SELECTION_SEED) -> int:
    """Notebook-07 seed: SHA-256(seed NUL source path NUL condition), first four bytes."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown selection condition: {condition!r}")
    payload = f"{seed}\0{source_path}\0{condition}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def fit_noise(noise: np.ndarray, samples: int, rng: np.random.RandomState):
    """Tile, crop or retain noise exactly as notebook 07 did; return samples and metadata."""
    noise = np.asarray(noise)
    if noise.ndim != 1 or noise.size == 0:
        raise ValueError(f"noise must be a non-empty vector, got {noise.shape}")
    if noise.size < samples:
        repeats = int(np.ceil(samples / noise.size))
        return np.tile(noise, repeats)[:samples], {
            "fit": "tile", "repeats": repeats, "start_frame": 0,
        }
    if noise.size > samples:
        start = int(rng.randint(0, noise.size - samples + 1))
        return noise[start:start + samples], {
            "fit": "crop", "repeats": 1, "start_frame": start,
        }
    return noise, {"fit": "exact", "repeats": 1, "start_frame": 0}


def _row(rows: Sequence[Mapping[str, Any]], index: int, half: str) -> Mapping[str, Any]:
    try:
        row = rows[index]
    except IndexError as exc:
        raise ValueError(f"resource index {index} is outside the audited inventory") from exc
    if row.get("half") != half:
        raise ValueError(
            f"resource {row.get('rel_path')!r} is in half {row.get('half')!r}, expected {half!r}"
        )
    return row


def _read_resource(root: Path, row: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    rel = str(row["rel_path"])
    path = Path(root) / rel
    expected = str(row["sha256"])
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{rel}: SHA-256 {actual} differs from audited inventory {expected}")
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    if not np.isfinite(data).all():
        raise ValueError(f"{rel}: decoded resource contains NaN or infinity")
    return np.asarray(data, dtype=np.float32), int(rate)


def apply_condition(
    source: np.ndarray,
    source_path: str,
    condition: str,
    *,
    noise_inventory: Sequence[Mapping[str, Any]],
    rir_inventory: Sequence[Mapping[str, Any]],
    noise_root: Path,
    rirs_root: Path,
    codec_backend,
    codec_options: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply one frozen S condition and return the waveform plus complete recipe metadata."""
    source = np.asarray(source, dtype=np.float32)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("source must be one finite, non-empty float32 waveform")
    condition_seed = derived_seed(source_path, condition)
    rng = np.random.RandomState(condition_seed)
    resource_path = ""
    resource_sha256 = ""
    resource_channel: int | str = ""
    parameters: dict[str, Any] = {}

    if condition == "clean":
        output = source.copy()
    elif condition == "recorded_noise_10db":
        resource_index = int(rng.randint(0, len(noise_inventory)))
        resource_row = _row(noise_inventory, resource_index, "selection")
        resource_path, resource_sha256 = str(resource_row["rel_path"]), str(resource_row["sha256"])
        noise_data, noise_rate = _read_resource(noise_root, resource_row)
        if noise_rate != 16000 or noise_data.shape[1] != 1:
            raise ValueError(f"{resource_path}: selection noise must be 16 kHz mono")
        resource_channel = 0
        noise, fit_parameters = fit_noise(noise_data[:, 0], source.size, rng)
        noise = noise.astype(np.float64)
        noise -= float(np.mean(noise))
        source_rms = float(np.sqrt(np.mean(source.astype(np.float64) ** 2)))
        noise_rms = float(np.sqrt(np.mean(noise ** 2)))
        if source_rms <= 0 or noise_rms <= 0:
            raise ValueError("recorded-noise recipe requires non-silent source and noise")
        snr_db = 10.0
        scale = source_rms / (noise_rms * 10.0 ** (snr_db / 20.0))
        output = np.asarray(source.astype(np.float64) + noise * scale, dtype=np.float32)
        parameters = {"snr_db": snr_db, "resource_index": resource_index, **fit_parameters}
    elif condition == "reverberation":
        resource_index = int(rng.randint(0, len(rir_inventory)))
        resource_row = _row(rir_inventory, resource_index, "selection")
        resource_path, resource_sha256 = str(resource_row["rel_path"]), str(resource_row["sha256"])
        rir_data, rir_rate = _read_resource(rirs_root, resource_row)
        if rir_rate != 16000:
            raise ValueError(f"{resource_path}: selection RIR must be 16 kHz")
        resource_channel = int(rng.randint(0, rir_data.shape[1]))
        impulse = rir_data[:, resource_channel]
        output = signal.fftconvolve(
            source.astype(np.float64), impulse.astype(np.float64), mode="full"
        )[:source.size].astype(np.float32)
        parameters = {
            "resource_index": resource_index,
            "channel_policy": "seeded_single_channel",
            "convolution": "full_then_crop_to_input",
        }
    elif condition == "opus_16":
        output = codec_backend.round_trip(source, codec_options["opus_16"])
        parameters = {"codec_option": "opus_16", "bitrate_kbps": 16, "vbr": False,
                      "application": "voip"}
    elif condition == "narrowband_mulaw":
        output = codec_backend.round_trip(source, codec_options["narrowband_mulaw"])
        parameters = {"codec_option": "narrowband_mulaw", "intermediate_sample_rate_hz": 8000,
                      "codec": "pcm_mulaw", "output_sample_rate_hz": 16000}
    else:  # derived_seed has already rejected it; retained as a fail-loud boundary
        raise ValueError(f"unknown selection condition: {condition!r}")

    output = np.asarray(output, dtype=np.float32)
    if output.shape != source.shape or not np.isfinite(output).all():
        raise ValueError(f"{condition}: output violated the waveform contract")
    return {
        "output": output,
        "condition_seed": condition_seed,
        "resource_path": resource_path,
        "resource_file_sha256": resource_sha256,
        "resource_channel": resource_channel,
        "parameters": parameters,
    }
