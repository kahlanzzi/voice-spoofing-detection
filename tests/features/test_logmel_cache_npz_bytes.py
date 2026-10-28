"""Log-Mel cache .npz bytes do not depend on the code commit (Stage 10, docs/risk_run.md step 3).

The Stage 10 runner compares the rebuilt val-A npz_sha256 with the one recorded in the checkpoint at
Stage 8, which was built from another commit, so the .npz bytes must not depend on the commit:
code_commit is written into the .json only, and the .npz writer stamps fixed zip metadata.
Synthetic WAV files and a synthetic manifest only, seed 2026.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.features.logmel_cache import build_logmel_cache
from src.preprocessing import load_profile

REPO = Path(__file__).resolve().parents[2]
PROVENANCE = {"f2_manifest_sha256": "a" * 64, "profile_config_sha256": "b" * 64}
COMMITS = ("c" * 40, "d" * 40)


def _corpus(root: Path) -> pd.DataFrame:
    """Three val-A clips (tone, noise, 22.05 kHz stereo) and one train row that is not built."""
    rng = np.random.default_rng(2026)
    specs = [  # (path, label, split, sample_rate, kind)
        ("val/b.wav", "FAKE", "val-A", 16000, "tone"),
        ("val/a.wav", "REAL", "val-A", 16000, "noise"),
        ("val/c.wav", "REAL", "val-A", 22050, "stereo"),
        ("train/a.wav", "FAKE", "train", 16000, "tone"),
    ]
    rows = []
    for i, (rel, label, split, rate, kind) in enumerate(specs):
        n = int(rate * (1.5 + 0.2 * i))
        t = np.arange(n) / rate
        data = 0.1 * rng.standard_normal(n) if kind == "noise" else 0.3 * np.sin(2 * np.pi * (220 + 50 * i) * t)
        if kind == "stereo":
            data = np.stack([data, 0.5 * data], axis=1)
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, data.astype(np.float32), rate, subtype="PCM_16")
        rows.append({
            "path": rel, "label": label, "subgroup": "REAL" if label == "REAL" else "FAKE_other",
            "group": f"g{i}", "split": split, "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return pd.DataFrame(rows, dtype=str)


def test_npz_bytes_are_independent_of_code_commit(tmp_path):
    root = tmp_path / "for"
    manifest = _corpus(root)
    profile = load_profile(REPO / "configs" / "preprocessing.yaml")
    outs = [tmp_path / f"cache_{i}" for i in range(2)]
    metas = [build_logmel_cache(root, manifest, profile, "val-A", out, **PROVENANCE, code_commit=commit)
             for out, commit in zip(outs, COMMITS)]

    npz = [(out / "val-A.npz").read_bytes() for out in outs]
    assert npz[0] == npz[1]
    assert metas[0]["npz_sha256"] == metas[1]["npz_sha256"] == hashlib.sha256(npz[0]).hexdigest()

    stored = [json.loads((out / "val-A.json").read_text(encoding="utf-8")) for out in outs]
    assert stored == metas
    assert [m["code_commit"] for m in stored] == list(COMMITS)
    assert {k for k in stored[0] if stored[0][k] != stored[1][k]} == {"code_commit"}
    assert set(stored[0]) == set(stored[1])
