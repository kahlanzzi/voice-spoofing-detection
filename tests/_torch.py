"""Import guard for torch tests (Stage 8, D42).

CI installs torch and sets REQUIRE_TORCH_TESTS=1, so a torch test there must fail, never skip, if
torch is missing: a silently skipped CNN test would look like a pass. Locally, without torch, the
tests are skipped so the rest of the suite still runs.

Every torch test module starts with: torch = require_torch()
"""
from __future__ import annotations

import importlib
import os

import pytest

ENV_VAR = "REQUIRE_TORCH_TESTS"


def require_torch():
    """Return the torch module; if it cannot be imported, fail when REQUIRE_TORCH_TESTS == "1", else skip."""
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        if os.environ.get(ENV_VAR) == "1":
            pytest.fail(f"torch is required ({ENV_VAR}=1) but cannot be imported: {exc}",
                        pytrace=False)
        pytest.skip(f"torch not installed ({exc}); set {ENV_VAR}=1 to make this a failure",
                    allow_module_level=True)
