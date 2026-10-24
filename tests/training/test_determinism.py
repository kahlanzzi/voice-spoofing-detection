"""set_seed (D42 "Determinism"): seeds and deterministic settings, recorded for the run metadata."""
from __future__ import annotations

import os
import random

import numpy as np
import pytest

from src.training.determinism import CUBLAS_ENV, CUBLAS_VALUE, set_seed
from tests._torch import require_torch

torch = require_torch()


@pytest.fixture(autouse=True)
def _restore_torch_flags(monkeypatch):
    """set_seed changes process-wide torch flags; put them back so other tests are not affected."""
    det = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_det, cudnn_bench = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    monkeypatch.setenv(CUBLAS_ENV, "placeholder")  # records the original state for teardown
    monkeypatch.delenv(CUBLAS_ENV)
    yield
    torch.use_deterministic_algorithms(det, warn_only=warn)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_det, cudnn_bench


def test_set_seed_sets_flags_and_reports_them():
    info = set_seed(3)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert os.environ[CUBLAS_ENV] == CUBLAS_VALUE
    assert info == {
        "seed": 3,
        "seeded": ["random", "numpy", "torch", "torch.cuda"],
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": True,
        "cublas_workspace_config": CUBLAS_VALUE,
        "cublas_workspace_config_preset": False,
    }


def test_existing_cublas_config_is_kept(monkeypatch):
    monkeypatch.setenv(CUBLAS_ENV, ":16:8")
    info = set_seed(0)
    assert os.environ[CUBLAS_ENV] == ":16:8"
    assert info["cublas_workspace_config"] == ":16:8"
    assert info["cublas_workspace_config_preset"] is True


def test_same_seed_same_draws_different_seed_different_draws():
    def draws(seed):
        set_seed(seed)
        return random.random(), np.random.random(), torch.rand(3).tolist()

    assert draws(5) == draws(5)
    a, b = draws(5), draws(6)
    assert a[0] != b[0] and a[1] != b[1] and a[2] != b[2]


@pytest.mark.parametrize("bad", [-1, 2**32, 1.0, "0", True, None])
def test_invalid_seed_raises(bad):
    with pytest.raises(ValueError, match="seed"):
        set_seed(bad)
