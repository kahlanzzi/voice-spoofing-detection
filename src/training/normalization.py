"""Per-Mel-bin standardization of the CNN input (D42 "Input").

The CNN input is the F3 log-Mel (n, 1, 64, 197) (D39). D42 standardizes each of the 64 Mel bins
with one mean and one standard deviation, computed over every clip and every frame of the FoR
**train** split only, stored in the checkpoint and applied unchanged to every split and to the
external windows. There is no per-clip normalization (D42: it would remove level information).

NumPy only, so the statistics can be computed and checked without torch.
"""
from __future__ import annotations

import numpy as np

N_MELS = 64
FEATURE_SHAPE = (1, N_MELS, 197)  # D39 log-Mel shape per clip


def _features(x, name: str, *, min_rows: int) -> np.ndarray:
    """Return x as a float32 array (n, 1, 64, 197) with n >= min_rows and finite values, or raise."""
    a = np.asarray(x)
    if a.dtype != np.float32:
        raise ValueError(f"{name} must be float32, got dtype {a.dtype}")
    if a.ndim != 4 or a.shape[1:] != FEATURE_SHAPE or a.shape[0] < min_rows:
        raise ValueError(
            f"{name} must have shape (n, {', '.join(map(str, FEATURE_SHAPE))}) with n >= {min_rows},"
            f" got {a.shape}"
        )
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains {int((~np.isfinite(a)).sum())} NaN or infinite values")
    return a


def _stat(name: str, value) -> np.ndarray:
    a = np.asarray(value)
    if a.dtype != np.float32 or a.shape != (N_MELS,):
        raise ValueError(f"{name} must be float32 with shape ({N_MELS},), got {a.dtype} {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return a


def fit_bin_stats(train_features) -> tuple[np.ndarray, np.ndarray]:
    """Per-Mel-bin mean and std (ddof=0) over clips and frames; two float32 arrays of shape (64,).

    The statistics must come from the FoR train split only (D42): pass the train log-Mel cache, never
    val-A, val-B, test or external features. Refuses a wrong shape or dtype, a non-finite value,
    fewer than 2 clips, or a constant bin (std == 0). Accumulated in float64, returned as float32.
    One bin at a time: the float64 copy is then n x 197 values (about 20 MB for the 12,404 train
    clips), not the whole n x 64 x 197 array (about 1.25 GB).
    """
    x = _features(train_features, "train_features", min_rows=2)
    mean = np.empty(N_MELS)
    std = np.empty(N_MELS)
    for b in range(N_MELS):
        values = x[:, 0, b, :].astype(np.float64)
        mean[b] = values.mean()
        std[b] = values.std(ddof=0)
    constant = np.flatnonzero(std == 0)
    if constant.size:
        raise ValueError(f"constant Mel bins (std == 0) at indices {constant.tolist()}")
    mean32, std32 = mean.astype(np.float32), std.astype(np.float32)
    if (std32 == 0).any():  # a std too small for float32
        raise ValueError("a Mel-bin std underflows to 0 in float32")
    return mean32, std32


def apply_bin_stats(features, mean, std) -> np.ndarray:
    """Standardize any (n, 1, 64, 197) float32 array with stored statistics; return a new float32 array.

    mean and std (float32, shape (64,), std > 0) are read only, never refitted or modified.
    """
    x = _features(features, "features", min_rows=1)
    mean, std = _stat("mean", mean), _stat("std", std)
    if (std <= 0).any():
        raise ValueError("std must be > 0 for every Mel bin")
    out = (x - mean[None, None, :, None]) / std[None, None, :, None]
    return out.astype(np.float32, copy=False)
