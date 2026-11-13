"""Strict loader for the D50/D56 augmentation contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


FAMILIES = ("recorded_noise", "reverberation", "codec", "gain", "rawboost")
CODEC_OPTIONS = (
    ("mp3_24", "mp3", 24),
    ("mp3_48", "mp3", 48),
    ("mp3_96", "mp3", 96),
    ("opus_8", "opus", 8),
    ("opus_16", "opus", 16),
    ("opus_24", "opus", 24),
    ("narrowband_mulaw", "narrowband_mulaw", None),
)
RAWBOOST_REFERENCE_COMMIT = "4f161a8b4d0d9f4a8431509ddd86c645869ef6c4"


class AugmentationError(RuntimeError):
    """Raised when augmentation configuration or input violates D50/D56."""


def _mapping(where: str, value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AugmentationError(f"{where} must be a mapping")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise AugmentationError(f"{where}: missing keys {missing}, unknown keys {unknown}")
    return value


def _number(where: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AugmentationError(f"{where} must be numeric, got {value!r}")
    return float(value)


def _range(where: str, value: Any, low: float, high: float, distribution: str) -> None:
    item = _mapping(where, value, {"distribution", "min", "max"})
    if item != {"distribution": distribution, "min": low, "max": high}:
        raise AugmentationError(f"{where} must be {distribution} over [{low}, {high}]")


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _freeze_value(dict(value))


@dataclass(frozen=True)
class AugmentationConfig:
    schema_version: int
    config_version: str
    sample_rate: int
    seed: int
    training: Mapping[str, Any]
    recorded_noise: Mapping[str, Any]
    reverberation: Mapping[str, Any]
    codec: Mapping[str, Any]
    gain: Mapping[str, Any]
    rawboost: Mapping[str, Any]
    held_out: Mapping[str, Any]
    corpora: Mapping[str, Any]
    ffmpeg: Mapping[str, Any]

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(self.training["families"])

    @property
    def codec_options(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.codec["options"])

    @property
    def corpus_pins_complete(self) -> bool:
        return all(
            self.corpora[k][field]
            for k in ("noise", "rir")
            for field in ("archive_sha256", "license_sha256")
        )

    @property
    def ffmpeg_pin_complete(self) -> bool:
        return bool(self.ffmpeg["expected_version"])


def load_augmentation_config(path: Path) -> AugmentationConfig:
    """Load and reject any drift from the approved D50/D56 V1a contract."""
    path = Path(path)
    if not path.is_file():
        raise AugmentationError(f"Augmentation config does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AugmentationError(f"Cannot read augmentation config {path}: {exc}") from exc
    top = _mapping(
        "augmentation",
        raw,
        {
            "schema_version", "config_version", "decisions", "sample_rate_hz", "seed",
            "training", "recorded_noise", "reverberation", "codec", "gain", "rawboost",
            "held_out_from_training", "corpora", "ffmpeg",
        },
    )
    if top["schema_version"] != 1 or top["config_version"] != "augmentation-v1":
        raise AugmentationError("schema/config version must be 1/augmentation-v1")
    if top["decisions"] != ["D50", "D56"]:
        raise AugmentationError("augmentation config must be governed by D50 and D56")
    if top["sample_rate_hz"] != 16000 or top["seed"] != 2026:
        raise AugmentationError("D50/D56 require 16 kHz and seed 2026")

    training = _mapping(
        "training", top["training"],
        {"allowed_splits", "apply_probability", "family_sampling", "families"},
    )
    if training != {
        "allowed_splits": ["train"], "apply_probability": 0.8,
        "family_sampling": "uniform", "families": list(FAMILIES),
    }:
        raise AugmentationError("training policy differs from D50")

    noise = _mapping("recorded_noise", top["recorded_noise"], {"snr_db"})
    _range("recorded_noise.snr_db", noise["snr_db"], 5.0, 30.0, "uniform")
    reverb = _mapping("reverberation", top["reverberation"], {"convolution"})
    if reverb["convolution"] != "full_then_crop_to_input":
        raise AugmentationError("reverberation must preserve input length")

    codec = _mapping("codec", top["codec"], {"option_sampling", "options"})
    if codec["option_sampling"] != "uniform" or not isinstance(codec["options"], list):
        raise AugmentationError("D56 requires uniform codec-option sampling")
    got_options = tuple(
        (x.get("id"), x.get("kind"), x.get("bitrate_kbps"))
        if isinstance(x, dict) and set(x) == {"id", "kind", "bitrate_kbps"} else None
        for x in codec["options"]
    )
    if got_options != CODEC_OPTIONS:
        raise AugmentationError("codec options must be the exact seven D56 options in order")

    gain = _mapping("gain", top["gain"], {"distribution", "min_db", "max_db", "peak_ceiling_dbfs"})
    if gain != {"distribution": "uniform", "min_db": -12.0, "max_db": 6.0,
                "peak_ceiling_dbfs": -1.0}:
        raise AugmentationError("gain policy differs from D50")

    rb = _mapping(
        "rawboost", top["rawboost"],
        {
            "algorithm", "implementation", "sample_rate_hz", "notch_bands", "fir_coefficients",
            "nonlinearity_order", "centre_frequency_hz", "bandwidth_hz", "linear_gain_db",
            "higher_order_gain_offset_db", "reference",
        },
    )
    if (rb["algorithm"], rb["implementation"], rb["sample_rate_hz"], rb["notch_bands"],
            rb["nonlinearity_order"], _number("rawboost.linear_gain_db", rb["linear_gain_db"])) != (
            1, "independent_from_paper", 16000, 5, 5, 0.0):
        raise AugmentationError("RawBoost must be paper-defined algorithm 1 with D56 fixed values")
    _range("rawboost.fir_coefficients", rb["fir_coefficients"], 10, 100, "uniform_integer")
    _range("rawboost.centre_frequency_hz", rb["centre_frequency_hz"], 20.0, 8000.0, "uniform")
    _range("rawboost.bandwidth_hz", rb["bandwidth_hz"], 100.0, 1000.0, "uniform")
    _range(
        "rawboost.higher_order_gain_offset_db", rb["higher_order_gain_offset_db"],
        -20.0, -5.0, "uniform",
    )
    reference = _mapping("rawboost.reference", rb["reference"], {"repository", "commit", "file", "license"})
    if reference != {
        "repository": "https://github.com/TakHemlata/RawBoost-antispoofing",
        "commit": RAWBOOST_REFERENCE_COMMIT, "file": "RawBoost.py", "license": "MIT",
    }:
        raise AugmentationError("RawBoost reference must match the D56 pin")

    held = _mapping(
        "held_out_from_training", top["held_out_from_training"],
        {"additive_noise_families", "mp3_bitrates_kbps", "plain_low_pass"},
    )
    if held != {"additive_noise_families": ["white_gaussian"],
                "mp3_bitrates_kbps": [32, 64, 128], "plain_low_pass": True}:
        raise AugmentationError("Stage 14 held-out conditions drifted from D50/H9")

    corpora = _mapping("corpora", top["corpora"], {"split_seed", "training_fraction", "noise", "rir"})
    if corpora["split_seed"] != 2026 or corpora["training_fraction"] != 0.5:
        raise AugmentationError("corpus split must be seeded 2026 and 50/50 by file")
    for key, name, slr in (("noise", "MUSAN", 17), ("rir", "RIRS_NOISES", 28)):
        item = _mapping(f"corpora.{key}", corpora[key],
                        {"name", "open_slr_id", "archive_sha256", "license_sha256"})
        if item["name"] != name or item["open_slr_id"] != slr:
            raise AugmentationError(f"corpora.{key} must identify {name}/OpenSLR {slr}")
        for field in ("archive_sha256", "license_sha256"):
            value = item[field]
            if value is not None and (not isinstance(value, str) or len(value) != 64):
                raise AugmentationError(f"corpora.{key}.{field} must be null or a SHA-256")

    ffmpeg = _mapping(
        "ffmpeg", top["ffmpeg"],
        {"executable", "expected_version", "expected_codec_libraries"},
    )
    if ffmpeg["executable"] != "ffmpeg" or not isinstance(ffmpeg["expected_codec_libraries"], dict):
        raise AugmentationError("ffmpeg executable/libraries pin is invalid")
    if ffmpeg["expected_version"] is not None and not isinstance(ffmpeg["expected_version"], str):
        raise AugmentationError("ffmpeg.expected_version must be null or text")

    return AugmentationConfig(
        schema_version=1, config_version="augmentation-v1", sample_rate=16000, seed=2026,
        training=_freeze(training), recorded_noise=_freeze(noise), reverberation=_freeze(reverb),
        codec=_freeze(codec), gain=_freeze(gain), rawboost=_freeze(rb), held_out=_freeze(held),
        corpora=_freeze(corpora), ffmpeg=_freeze(ffmpeg),
    )
