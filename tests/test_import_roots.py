"""Import roots: `tests` and `src` must resolve to this repo, not to a same-named site-packages package.

Colab ships packages that install a top-level `tests/` into dist-packages; without tests/__init__.py
that regular package beat our namespace package and broke `from tests._torch import ...`.
The check is the exact folder, not "somewhere under the repo root": the local .venv lives inside it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import src
import tests

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("module", [tests, src], ids=["tests", "src"])
def test_package_resolves_to_repo_folder(module):
    assert module.__file__ is not None, (
        f"`import {module.__name__}` is a namespace package at {list(module.__path__)}: "
        f"{module.__name__}/__init__.py is missing, so a same-named regular package on sys.path can shadow it"
    )
    location = Path(module.__file__).resolve()
    assert location.parent == REPO_ROOT / module.__name__, (
        f"`import {module.__name__}` resolves to {location}, not {REPO_ROOT / module.__name__}: "
        f"a same-named package on sys.path is shadowing the repo's {module.__name__}/"
    )
