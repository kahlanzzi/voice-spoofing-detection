"""The torch-test guard (tests/_torch.py): fail, not skip, when CI requires torch and it is missing."""
from __future__ import annotations

import sys
import types

import pytest

from tests._torch import ENV_VAR, require_torch


def _hide_torch(monkeypatch):
    # sys.modules[name] = None makes "import name" raise ImportError, as if torch were not installed.
    monkeypatch.setitem(sys.modules, "torch", None)


def test_guard_fails_not_skips_when_required_and_torch_missing(monkeypatch):
    _hide_torch(monkeypatch)
    monkeypatch.setenv(ENV_VAR, "1")
    # pytest.fail raises Failed; a skip would raise Skipped, which this check would not catch.
    with pytest.raises(pytest.fail.Exception, match="torch is required"):
        require_torch()


@pytest.mark.parametrize("value", [None, "0", "", "true"])
def test_guard_skips_when_not_required_and_torch_missing(monkeypatch, value):
    _hide_torch(monkeypatch)
    if value is None:
        monkeypatch.delenv(ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ENV_VAR, value)  # only the exact value "1" makes it a failure
    with pytest.raises(pytest.skip.Exception):
        require_torch()


def test_guard_returns_torch_when_importable(monkeypatch):
    fake = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setenv(ENV_VAR, "1")
    assert require_torch() is fake
