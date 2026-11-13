"""Per-step cosine learning-rate schedule for the aasist candidate (D61 (3)).

The reference (clovaai/aasist utils.cosine_annealing at commit a04c986) computes
    lr_min + (lr_max - lr_min) * 0.5 * (1 + cos(pi * step / total_steps))
per optimizer step. D61 sets total_steps = 30 (D49's maximum epochs) x steps per epoch; the
schedule does not change if early stopping ends the run sooner. Standard library only.
"""
from __future__ import annotations

import math
from numbers import Real


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def cosine_lr(step: int, total_steps: int, base: float, minimum: float) -> float:
    """Learning rate at optimizer step `step` in [0, total_steps]: base at 0, minimum at total_steps."""
    if not _is_int(total_steps) or total_steps <= 0:
        raise ValueError(f"total_steps must be an int > 0, got {total_steps!r}")
    if not _is_int(step) or not 0 <= step <= total_steps:
        raise ValueError(f"step must be an int in [0, {total_steps}], got {step!r}")
    if not (_is_number(base) and _is_number(minimum)) or not 0 <= minimum <= base:
        raise ValueError(f"need finite 0 <= minimum <= base, got base={base!r}, minimum={minimum!r}")
    return float(minimum + (base - minimum) * 0.5 * (1 + math.cos(math.pi * step / total_steps)))
