"""Drift guard: configs/v2_candidates.yaml batch sizes against the D49 benchmark evidence (V2b-2).

Torch-free. The evidence is docs/audit_evidence/v2b2_benchmark/ (Entry 173): two GPU runs on a Colab
T4 and one CPU throughput run, copied byte for byte with their SHA-256 recorded in the README.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from src.preprocessing.benchmark import choose_batch_size
from src.training.v2_config import load_v2_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "v2_candidates.yaml"
EVIDENCE = ROOT / "docs" / "audit_evidence" / "v2b2_benchmark"
GPU_RUNS = ("gpu_run1.json", "gpu_run2.json")
FILES = (*GPU_RUNS, "cpu_throughput.json")
FRACTION = 0.80  # D49 / Stage 6: 80 % of VRAM


def _load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _readme_hashes() -> dict[str, str]:
    text = (EVIDENCE / "README.md").read_text(encoding="utf-8")
    return dict(re.findall(r"^\| `([\w.]+\.json)` \| `([0-9a-f]{64})` \|$", text, flags=re.M))


def test_committed_batch_sizes_equal_both_gpu_runs():
    committed = load_v2_config(CONFIG_PATH).require_batch_sizes()
    for name in GPU_RUNS:
        report = _load(name)
        assert report["device"] == "cuda" and report["choice_errors"] == {}, name
        assert report["chosen_batch_size"] == committed, name


def test_gpu_runs_have_identical_status_and_peaks():
    run1, run2 = (_load(name) for name in GPU_RUNS)
    key = lambda r: (r["candidate"], r["batch_size"], r["status"], r["peak_allocated_bytes"])  # noqa: E731
    assert [key(r) for r in run1["rows"]] == [key(r) for r in run2["rows"]]
    assert run1["total_vram_bytes"] == run2["total_vram_bytes"]
    assert run1["benchmark"] == run2["benchmark"]
    assert run1["benchmark"]["max_vram_fraction"] == FRACTION


@pytest.mark.parametrize("name", GPU_RUNS)
def test_each_chosen_size_follows_the_stage6_rule(name):
    report = _load(name)
    committed = load_v2_config(CONFIG_PATH).require_batch_sizes()
    for candidate, size in committed.items():
        rows = [r for r in report["rows"] if r["candidate"] == candidate]
        assert rows, candidate
        assert choose_batch_size(rows, report["total_vram_bytes"], FRACTION) == size, candidate


def test_evidence_hashes_equal_the_readme():
    recorded = _readme_hashes()
    assert set(recorded) == set(FILES)
    for name in FILES:
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == recorded[name], name
