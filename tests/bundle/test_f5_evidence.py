"""f5 layering and the state of the REAL committed evidence. No torch, no Drive, no data.

The v2 evidence (search, risk-v2, v2 Stage 11) is not committed yet; until it is, F5 must be refused
by expected_from_evidence. The freeze PR replaces that test with one pinning the real values (as
test_f4_evidence.py does for F4).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.bundle import f5

ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path, *, top_level_only: bool = False) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    names = [a.name for n in nodes if isinstance(n, ast.Import) for a in n.names]
    return names + [n.module or "" for n in nodes if isinstance(n, ast.ImportFrom)]


def test_f5_does_not_import_research_code():
    # Shared core (SDD Fig. 1): the API imports src.bundle, so no Evaluation or Tuning import anywhere.
    names = _imports(ROOT / "src" / "bundle" / "f5.py")
    assert [n for n in names if n.startswith(("src.evaluation", "src.tuning"))] == []
    assert "src.bundle.f4" in names  # the F4 helpers are reused, not copied


def test_f5_module_level_is_torch_free():
    # torch and the model loader are imported inside build/verify only.
    top = _imports(ROOT / "src" / "bundle" / "f5.py", top_level_only=True)
    assert [n for n in top if n.split(".")[0] in {"torch", "safetensors"} or n.startswith(
        ("src.models", "src.bundle.model_loader"))] == []


@pytest.mark.skipif((ROOT / f5.SEARCH_REPORT).exists(), reason="v2 evidence committed: pin its values instead")
def test_real_repository_refuses_f5_before_the_v2_evidence():
    with pytest.raises(ValueError, match="v2_search/run_report.json is missing"):
        f5.expected_from_evidence(ROOT)
