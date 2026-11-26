"""Synthetic export tests; no dataset or model access."""
import hashlib
import json

import pytest

from src.evaluation import live_warmup_export as exporter


@pytest.fixture
def calls(monkeypatch):
    recorded = []
    monkeypatch.setattr(exporter, "version", lambda name: "test-version")

    def score(**kwargs):
        recorded.append(kwargs)
        return {"undefined": float("nan"), "provenance": {"code_commit": "a" * 40}}

    monkeypatch.setattr(exporter, "score_call_plan", score)
    return recorded


def test_export_preserves_provenance_and_hashes_exact_bytes(tmp_path, calls):
    output = tmp_path / "report.json"
    digest = exporter.export_report(output, code_commit="a" * 40)
    payload = output.read_bytes()
    result = json.loads(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert result["undefined"] is None
    assert result["provenance"]["code_commit"] == "a" * 40
    assert result["runtime"]["torch"] == "test-version"
    assert result["runtime"]["device"] == "cpu"
    assert result["started_at_utc"] <= result["finished_at_utc"]
    assert calls == [{"code_commit": "a" * 40}]


def test_existing_output_rejected_before_scoring(tmp_path, calls):
    output = tmp_path / "report.json"
    output.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        exporter.export_report(output)
    assert output.read_bytes() == b"original"
    assert calls == []


def test_failure_leaves_reservation_not_success(tmp_path, calls, monkeypatch):
    def fail(**kwargs):
        raise ValueError("bad pin")

    monkeypatch.setattr(exporter, "score_call_plan", fail)
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="bad pin"):
        exporter.export_report(output)
    assert output.read_bytes() == b""


def test_cli_forwards_all_pins_and_prints_digest(tmp_path, calls, capsys):
    output = tmp_path / "report.json"
    exporter.main(["--output", str(output), "--plan-path", "plan.json",
                   "--manifest-path", "manifest.csv", "--audio-root", "audio",
                   "--bundle-dir", "bundle", "--expected-plan-sha256", "b" * 64,
                   "--expected-bundle-sha256", "c" * 64, "--code-commit", "a" * 40])
    assert calls[0]["expected_plan_sha256"] == "b" * 64
    assert calls[0]["expected_bundle_sha256"] == "c" * 64
    assert calls[0]["audio_root"].name == "audio"
    assert hashlib.sha256(output.read_bytes()).hexdigest() in capsys.readouterr().out
