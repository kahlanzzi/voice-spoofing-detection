"""Seeding and deterministic-algorithm settings for CNN runs (D42 "Determinism", D11, D30).

set_seed seeds Python's random, NumPy's global generator and torch (CPU and every CUDA device),
forces cuDNN into deterministic mode without autotuning, and asks torch for deterministic
algorithms where they exist (warn_only=True: an operation without a deterministic version warns
instead of stopping the run). D42 does not promise byte-identical GPU runs; reproducibility is
reported as observed, and the returned dict records what was set, for the run metadata.

torch is imported inside set_seed, so importing this module does not need torch.
"""
from __future__ import annotations

import os
import random

import numpy as np

CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_VALUE = ":4096:8"  # required by cuBLAS for deterministic results (CUDA >= 10.2)


def set_seed(seed: int) -> dict:
    """Seed random, NumPy and torch with seed and turn on deterministic settings; return what was set.

    CUBLAS_WORKSPACE_CONFIG is set to ":4096:8" only if it is unset; an existing value is kept and
    recorded. It must be set before the first cuBLAS call, so set_seed is called at the start of a run.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError(f"seed must be an int in [0, 2**32), got {seed!r}")
    import torch

    cublas_was_set = CUBLAS_ENV in os.environ
    if not cublas_was_set:
        os.environ[CUBLAS_ENV] = CUBLAS_VALUE
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "seeded": ["random", "numpy", "torch", "torch.cuda"],
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ[CUBLAS_ENV],
        "cublas_workspace_config_preset": cublas_was_set,
    }
