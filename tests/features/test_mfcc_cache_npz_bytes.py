"""MFCC cache .npz bytes do not depend on the code commit (D46 (4), the reproduction gate).

The Stage 11 reproduction gate compares the rebuilt val-A MFCC npz_sha256 with the one recorded at
Stage 7, which was built from another commit, so the .npz bytes must not depend on the commit:
code_commit is written into the .json only, and the .npz writer stamps fixed zip metadata. The
counterpart of tests/features/test_logmel_cache_npz_bytes.py.
Synthetic WAV files and a synthetic manifest only, seed 2026.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.features.mfcc_cache import build_mfcc_cache
from src.preprocessing import load_profile
from tests.features.test_logmel_cache_npz_bytes import COMMITS, PROVENANCE, _corpus

REPO = Path(__file__).resolve().parents[2]


def test_npz_bytes_are_independent_of_code_commit(tmp_path):
    root = tmp_path / "for"
    manifest = _corpus(root)
    profile = load_profile(REPO / "configs" / "preprocessing.yaml")
    outs = [tmp_path / f"cache_{i}" for i in range(2)]
    metas = [build_mfcc_cache(root, manifest, profile, "val-A", out, **PROVENANCE, code_commit=commit)
             for out, commit in zip(outs, COMMITS)]

    npz = [(out / "val-A.npz").read_bytes() for out in outs]
    assert npz[0] == npz[1]
    assert metas[0]["npz_sha256"] == metas[1]["npz_sha256"] == hashlib.sha256(npz[0]).hexdigest()

    stored = [json.loads((out / "val-A.json").read_text(encoding="utf-8")) for out in outs]
    assert stored == metas
    assert [m["code_commit"] for m in stored] == list(COMMITS)
    assert {k for k in stored[0] if stored[0][k] != stored[1][k]} == {"code_commit"}
    assert set(stored[0]) == set(stored[1])
