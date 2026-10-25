"""Safe, synthetic-testable primitives for the guarded Stage 13 runner.

The external runner must split a recording into complete protocol windows first and then call
the frozen Stage 6 preprocessing function once per window. This module keeps the windowing
primitive independent of any audio decoder, checkpoint or external-data path: those pieces
are injected only after frozen prerequisites have been verified.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.preprocessing import PreprocessingProfile, WaveformResult, preprocess_waveform


@dataclass(frozen=True)
class WindowScore:
    """One complete protocol window after the shared preprocessing call."""

    window_index: int
    raw_score: float
    speech_fraction: float
    padding_samples: int
    kept_by_filter: bool


@dataclass(frozen=True)
class WindowRun:
    """Scores and accounting from one in-memory recording pass."""

    scores: tuple[WindowScore, ...]
    n_dropped_samples: int

    @property
    def n_complete_windows(self) -> int:
        return len(self.scores)

    @property
    def filtered_scores(self) -> tuple[WindowScore, ...]:
        """The primary view: windows retained by the frozen speech filter."""
        return tuple(row for row in self.scores if row.kept_by_filter)

    @property
    def n_filter_dropped(self) -> int:
        return sum(not row.kept_by_filter for row in self.scores)


@dataclass(frozen=True)
class ExternalRunConfig:
    """Validated paths and identity for a future guarded external run."""

    schema_version: int
    dataset: str
    registry_path: Path
    results_path: Path
    bundle_path: Path
    manifest_path: Path
    profile_path: Path


def _sample_axis(waveform: np.ndarray, channel_axis: int) -> int:
    if waveform.ndim == 1:
        return 0
    if waveform.ndim != 2:
        raise ValueError(f"waveform must be one- or two-dimensional, got shape {waveform.shape}")
    axis = channel_axis if channel_axis >= 0 else waveform.ndim + channel_axis
    if axis not in (0, 1):
        raise ValueError(f"channel_axis must identify axis 0 or 1, got {channel_axis}")
    return 1 - axis


def _window_count(sample_count: int, sample_rate: int, profile: PreprocessingProfile) -> tuple[int, int]:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate!r}")
    duration = float(profile.waveform["duration_seconds"])
    window_samples = round(duration * int(sample_rate))
    if window_samples <= 0 or not np.isclose(window_samples, duration * int(sample_rate)):
        raise ValueError("profile duration does not map to a whole number of input samples")
    return sample_count // window_samples, window_samples


def iter_complete_windows(
    waveform: np.ndarray,
    sample_rate: int,
    profile: PreprocessingProfile,
    *,
    channel_axis: int = -1,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, raw_window) for complete non-overlapping protocol windows.

    The short remainder is intentionally not yielded. No trimming, padding, resampling or feature
    work happens here; those operations belong to the shared per-window call below.
    """
    array = np.asarray(waveform)
    sample_axis = _sample_axis(array, channel_axis)
    count, window_samples = _window_count(array.shape[sample_axis], sample_rate, profile)
    for index in range(count):
        start = index * window_samples
        stop = start + window_samples
        slices = [slice(None)] * array.ndim
        slices[sample_axis] = slice(start, stop)
        yield index, array[tuple(slices)]


def score_windows(
    waveform: np.ndarray,
    sample_rate: int,
    profile: PreprocessingProfile,
    scorer: Callable[[np.ndarray], Real],
    *,
    channel_axis: int = -1,
) -> WindowRun:
    """Preprocess and score every complete window exactly once.

    scorer receives the fixed-length waveform returned by the real Stage 6 function. The caller
    can inject a fake scorer in tests or a frozen model adapter later without duplicating
    preprocessing logic. Both retained and unfiltered views use these same scores.
    """
    if not callable(scorer):
        raise ValueError("scorer must be callable")
    array = np.asarray(waveform)
    sample_axis = _sample_axis(array, channel_axis)
    count, window_samples = _window_count(array.shape[sample_axis], sample_rate, profile)
    scores: list[WindowScore] = []
    for index, raw_window in iter_complete_windows(
        array, sample_rate, profile, channel_axis=channel_axis
    ):
        prepared: WaveformResult = preprocess_waveform(
            raw_window, sample_rate, profile, channel_axis=channel_axis
        )
        value = scorer(prepared.waveform)
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"scorer must return one finite numeric score, got {value!r}")
        scores.append(WindowScore(
            window_index=index,
            raw_score=float(value),
            speech_fraction=float(prepared.speech_fraction),
            padding_samples=int(prepared.padding_samples),
            kept_by_filter=(
                prepared.speech_fraction
                >= float(profile.silence["external_min_speech_fraction"])
            ),
        ))
    return WindowRun(
        scores=tuple(scores),
        n_dropped_samples=array.shape[sample_axis] - count * window_samples,
    )


def load_external_config(path: str | Path) -> ExternalRunConfig:
    """Load the exact, path-driven Stage 13 runner configuration template."""
    path = Path(path)
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read external-run config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("external-run config must be a mapping")
    expected = {
        "schema_version", "dataset", "registry_path", "results_path", "bundle_path",
        "manifest_path", "profile_path",
    }
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"external-run config keys: missing {missing}, unknown {unknown}")
    if (
        raw["schema_version"] != 1
        or not isinstance(raw["dataset"], str)
        or not raw["dataset"].strip()
    ):
        raise ValueError("schema_version must be 1 and dataset must be non-empty")
    values = {}
    for name in ("registry_path", "results_path", "bundle_path", "manifest_path", "profile_path"):
        value = raw[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty path string")
        values[name] = Path(value)
    return ExternalRunConfig(schema_version=1, dataset=raw["dataset"], **values)
