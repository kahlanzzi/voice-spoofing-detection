"""D52 checkpoint decisions from an explicitly ordered simulated call's scores.

Pure calculation only: no dataset access, calibration fitting or sequence sampling.
The caller supplies calibrated probabilities from one frozen bundle. None means
a window rejected by D17, not a model deferral. All kept windows enter the mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Real
from typing import Iterable

from src.risk.decisions import three_way_decisions


@dataclass(frozen=True)
class WarmupCheckpoint:
    kept_windows: int
    reached: bool
    elapsed_windows: int | None
    mean_p: float | None
    prediction: str | None


def warmup_checkpoints(
    probabilities: Iterable[float | None], *, threshold: float, half_width: float
) -> tuple[WarmupCheckpoint, ...]:
    """Return D52's fixed 3/5/10-kept-window checkpoints for a 30-second call.

    Exactly fifteen 2-second slots are required. A missing checkpoint is explicit,
    never silently discarded or labelled uncertain. Validate every slot, including
    slots after the tenth kept window, before returning any result.
    """
    # Use the shared policy validation and boundaries; never calibrate p again.
    three_way_decisions([0.5], threshold, half_width)
    slots = tuple(probabilities)
    if len(slots) != 15:
        raise ValueError("a D52 simulated call requires exactly fifteen 2-second slots")
    for value in slots:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("each p must be finite in [0, 1], or None for no speech")
    found = {}
    kept = []
    for elapsed, value in enumerate(slots, start=1):
        if value is None:
            continue
        kept.append(float(value))
        if len(kept) in (3, 5, 10):
            mean = fsum(kept) / len(kept)
            label = str(three_way_decisions([mean], threshold, half_width)[0])
            found[len(kept)] = WarmupCheckpoint(len(kept), True, elapsed, mean, label)
    return tuple(
        found.get(n, WarmupCheckpoint(n, False, None, None, None)) for n in (3, 5, 10)
    )
