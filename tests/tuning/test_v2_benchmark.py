"""D49 batch-size benchmark and CPU throughput (Track V): CPU dry run, codec pin rules, import isolation."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests._torch import require_torch

torch = require_torch()

import src.tuning.v2_benchmark as vb  # noqa: E402
from src.augmentation import load_augmentation_config  # noqa: E402
from src.augmentation.codec import FFmpegCodecBackend  # noqa: E402
from tests._v2_fakes import COMMIT, REPO, FakeCodec, aug_config_copy  # noqa: E402

TRAINING_MODULES = ("src/training/train_v2.py", "src/training/v2_data.py", "src/tuning/v2_search.py")


@pytest.fixture()
def quick(monkeypatch):
    monkeypatch.setattr(vb, "code_version", lambda: {"git_commit": COMMIT, "git_dirty": False})
    real = vb.benchmark_settings
    monkeypatch.setattr(vb, "benchmark_settings",
                        lambda p: {**real(p), "warmup_steps": 1, "measured_steps": 1})


def test_cpu_dry_run_schema(quick, tmp_path):
    report = vb.run_benchmark(tmp_path / "out", device="cpu", batch_sizes=[1, 2], throughput_items=1,
                              codec_backend=FakeCodec())
    assert json.loads((tmp_path / "out" / "benchmark.json").read_text(encoding="utf-8")) == report
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["benchmark.json"]
    assert [(r["candidate"], r["batch_size"]) for r in report["rows"]] == [
        (c, b) for c in ("cnn-aug", "crnn", "aasist") for b in (1, 2)]
    for row in report["rows"]:
        assert set(row) == {"candidate", "batch_size", "status", "mean_step_seconds", "peak_allocated_bytes"}
        assert row["status"] == "ok" and row["mean_step_seconds"] > 0 and row["peak_allocated_bytes"] is None
    assert report["chosen_batch_size"] == {"cnn-aug": None, "crnn": None, "aasist": None}  # CPU
    assert report["benchmark"]["optimizer_step"] is True
    assert (report["weights_saved"], report["data_read"], report["code_commit"]) == (False, False, COMMIT)
    assert report["codec_timing"] == "injected" and any("ffmpeg -version" in n for n in report["notes"])
    table = report["throughput"]
    assert set(table["families"]) == {"clean", "recorded_noise", "reverberation", "codec", "gain", "rawboost"}
    assert all(v["n"] == 1 for v in table["families"].values())
    assert set(table["stages"]) == {"augment", "f3", "logmel_stats"}
    assert table["family_share"]["clean"] == pytest.approx(0.2)
    assert table["family_share"]["codec"] == pytest.approx(0.16)
    assert set(table["estimated_epoch_seconds"]) == {"waveform", "logmel"}
    est = table["estimated_epoch_seconds"]["logmel"]
    assert est["2"] == pytest.approx(est["1"] / 2)
    assert est["1"] == pytest.approx(12404 * table["per_item_mean_seconds"]["logmel"])


def test_throughput_skipped_and_existing_out_refused(quick, tmp_path):
    report = vb.run_benchmark(tmp_path / "a", device="cpu", candidates=["cnn-aug"], batch_sizes=[1],
                              throughput_items=0)
    assert report["throughput"] is None and report["codec_timing"] is None
    with pytest.raises(ValueError, match="already exists"):
        vb.run_benchmark(tmp_path / "a", device="cpu", candidates=["cnn-aug"], batch_sizes=[1],
                         throughput_items=0)
    with pytest.raises(ValueError, match="inside the repository"):
        vb.run_benchmark(REPO / "benchmark_should_not_exist", device="cpu", throughput_items=0)
    with pytest.raises(ValueError, match="unknown candidates"):
        vb.run_benchmark(tmp_path / "b", device="cpu", candidates=["lstm"], throughput_items=0)


def test_unpinned_codec_timing_needs_the_flag(quick, tmp_path, monkeypatch):
    unpinned = load_augmentation_config(REPO / "configs" / "augmentation.yaml")
    assert not unpinned.ffmpeg_pin_complete
    with pytest.raises(ValueError, match="--codec-unpinned-timing"):
        vb.codec_backend_for(unpinned, unpinned_timing=False)
    with pytest.raises(ValueError, match="--codec-unpinned-timing"):
        vb.run_benchmark(tmp_path / "out", device="cpu", batch_sizes=[1], throughput_items=1)
    assert not (tmp_path / "out").exists()
    backend, label = vb.codec_backend_for(unpinned, unpinned_timing=True)
    assert label == "unpinned" and isinstance(backend, vb.UnpinnedTimingCodecBackend)
    monkeypatch.setattr(vb.UnpinnedTimingCodecBackend, "version_text",
                        lambda self: "ffmpeg version 9.9-synthetic\nlibopus 1.0\n")
    backend.verify_pin()  # records instead of checking
    assert backend.observed_version == "ffmpeg version 9.9-synthetic"


def test_pinned_codec_is_verified(tmp_path, monkeypatch):
    pinned = load_augmentation_config(aug_config_copy(tmp_path / "aug.yaml"))
    calls = []
    monkeypatch.setattr(FFmpegCodecBackend, "verify_pin", lambda self: calls.append(self))
    backend, label = vb.codec_backend_for(pinned, unpinned_timing=True)
    assert label == "pinned" and type(backend) is FFmpegCodecBackend and len(calls) == 1


def test_cli(quick, tmp_path, capsys):
    vb.main(["--out", str(tmp_path / "cli"), "--device", "cpu", "--candidates", "cnn-aug",
             "--batch-sizes", "1", "--throughput-items", "0"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["chosen_batch_size"] == {"cnn-aug": None} and printed["codec_timing"] is None


def test_training_code_never_imports_the_benchmark():
    for rel in TRAINING_MODULES:
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not [n for n in names if "v2_benchmark" in n], rel
    code = ("import sys; import src.training.train_v2, src.training.v2_data, src.tuning.v2_search; "
            "assert 'src.tuning.v2_benchmark' not in sys.modules, 'benchmark imported'")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
