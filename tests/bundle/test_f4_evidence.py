"""F4 expected values from the REAL committed evidence, and f4 layering. No torch, no Drive, no data.

Runs everywhere: if a committed evidence file changed, F4 would be built against other values, so
this pins every value the Stage 8, 10, 6 and 7 evidence records.
"""
from __future__ import annotations

import ast
from pathlib import Path

from src.bundle import f4

ROOT = Path(__file__).resolve().parents[2]


def test_expected_from_real_evidence():
    e = f4.expected_from_evidence(ROOT)
    assert e["bundle_seed"] == 0
    ck = e["checkpoint"]
    assert ck["sha256"] == "c33eb82086d3bc40f3dedf1356addc6e5c1924ca2e517b0d4527b447e1958932"
    assert ck["version"] == "cnn-v1"
    assert ck["source"] == "checkpoints/cnn/cnn-v1/seed_0"
    assert ck["files"] == {
        "checkpoint.json": {
            "sha256": "dc0f8b0e9a8549583478db7d1a94e24d01668af9459bfd692895abcc83c98932", "size": 1350},
        "weights.safetensors": {
            "sha256": "6722f8587b54aaea649456a04ec3074fb30883d549bf48ea6a6ffa9111dcf835", "size": 396084},
    }
    assert e["policy"]["sha256"] == "a74f1f512b2e7c527df5f34e56acaf5fbe2ceb58071e3d46a441cc2f3f4966c1"
    assert e["policy"]["version"] == "risk-v1"
    assert e["profile"]["sha256"] == "7bd35bfdcb9270dd55ea9e73449c7bf6c6a359a80335b2551c0c89dde77c8630"
    assert e["profile"]["fingerprint"] == "5b4f5849a9bb63f9016dd79969ff5bfb9e37006acbff1e26bbfee4ee2bad68f2"
    assert e["profile"]["version"] == "preprocessing-v1"
    assert e["f2_ledger_sha256"] == "c9fc2e230c0883028459d0b2b8de4f6ee537676cd5a4e259e9d0f545c60943cc"
    assert e["baseline"]["version"] == "baseline-v1"
    assert e["baseline"]["artifact_sha256"] == "42cc4d53a7cd881f56e4fa54fcf4797de8885b404d0a2b31c293555c868df10a"
    assert e["protocol_version"] == "1.0"
    assert all(c["ok"] for c in e["checks"]) and len(e["checks"]) >= 9


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    return names + [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]


def test_f4_does_not_import_research_code():
    # Shared core (SDD Fig. 1): the API side may run verify, so no Evaluation or Tuning import.
    names = _imports(ROOT / "src" / "bundle" / "f4.py")
    assert [n for n in names if n.startswith(("src.evaluation", "src.tuning"))] == []


def test_f4_module_level_is_torch_free():
    # torch and the model loader are imported inside build/verify only, so this module (and the
    # evidence check above) runs without torch.
    tree = ast.parse((ROOT / "src" / "bundle" / "f4.py").read_text(encoding="utf-8"))
    top = [a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names]
    top += [n.module or "" for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert [n for n in top if n.split(".")[0] in {"torch", "safetensors"} or n.startswith(
        ("src.models", "src.bundle.model_loader"))] == []
