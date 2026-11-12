"""F4 build/verify (src.bundle.f4) on a synthetic evidence tree. Synthetic only: no Drive, no real checkpoint.

The tree mirrors the real evidence file names and fields; the checkpoint is a tiny real D42
checkpoint (as in test_model_loader.py) whose real hashes are written into the fake stage8/stage10
reports. git is injected, so no repository is needed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests._torch import require_torch

torch = require_torch()

import src.bundle.model_loader as loader_module  # noqa: E402
from src.bundle import f4  # noqa: E402
from src.bundle.bundle import CHECKPOINT_FILES, MANIFEST_NAME, load_bundle  # noqa: E402
from src.bundle.model_loader import load_bundle_model  # noqa: E402
from src.models import cnn_checkpoint  # noqa: E402
from src.models.cnn import INPUT_SHAPE, build_cnn  # noqa: E402
from src.preprocessing.profile import load_profile  # noqa: E402
from src.risk.calibration import PlattCalibration  # noqa: E402
from src.risk.policy import RiskPolicy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROFILE_SRC = ROOT / "configs" / "preprocessing.yaml"
LEDGER = "c9fc2e230c0883028459d0b2b8de4f6ee537676cd5a4e259e9d0f545c60943cc"
COMMIT = "1" * 40
CREATED_AT = "2026-11-12T10:00:00Z"
BASELINE_SHA = "f" * 64


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean_git(_root):
    return COMMIT, ""


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _edit_json(path: Path, **changes) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    _write_json(path, data)


def _checkpoint(directory: Path) -> Path:
    torch.manual_seed(2026)
    model = build_cnn(3).train()
    with torch.no_grad():
        model(torch.randn(4, *INPUT_SHAPE))
    rng = np.random.default_rng(2026)
    mean = rng.normal(-6.0, 2.0, 64).astype(np.float32)
    std = rng.uniform(0.5, 3.0, 64).astype(np.float32)
    meta = {
        "schema_version": 1, "version": "cnn-v1", "n_blocks": 3, "channels": [16, 32, 64],
        "dropout": 0.3, "input_shape": [1, 64, 197], "learning_rate": 0.001, "seed": 0,
        "best_epoch": 7, "val_a_eer": 0.0125, "code_commit": "a" * 40,
        "environment": {"python": "3.13.15", "torch": "2.11.0+cpu"},
        "profile_fingerprint": load_profile(PROFILE_SRC).fingerprint(),
        "f2_manifest_sha256": "c" * 64, "train_cache_npz_sha256": "d" * 64,
        "val_a_cache_npz_sha256": "e" * 64, "determinism": {"seed": 0, "cudnn_deterministic": True},
    }
    return cnn_checkpoint.save_checkpoint(model.eval(), mean, std, meta, directory)


@pytest.fixture
def tree(tmp_path: Path):
    """(repo root, checkpoint folder) of a synthetic evidence tree with the real file names."""
    root = tmp_path / "repo"
    ckpt = _checkpoint(tmp_path / "drive" / "checkpoints" / "cnn" / "cnn-v1" / "seed_0")
    combined = cnn_checkpoint.checkpoint_sha256(ckpt)
    profile = root / f4.PROFILE_SOURCE
    profile.parent.mkdir(parents=True)
    profile.write_bytes(PROFILE_SRC.read_bytes())
    policy = root / f4.POLICY_SOURCE
    policy.parent.mkdir(parents=True)
    policy.write_text(RiskPolicy(
        version="risk-v1", calibration=PlattCalibration(a=1.0, b=0.0, n_spoof=10, n_bonafide=12),
        threshold=0.5, half_width=0.1, target_deferral=0.10, n_spoof_fit=10, n_bonafide_fit=12,
        deferral_on_fit=0.0909090909090909,
    ).to_json(), encoding="utf-8")
    fingerprint = load_profile(profile).fingerprint()
    _write_json(root / f4.STAGE8_REPORT, {
        "bundle_seed": 0, "checkpoint_sha256": {"0": combined, "1": "1" * 64}, "version": "cnn-v1",
        "profile_config_sha256": _sha(profile.read_bytes()), "profile_fingerprint": fingerprint,
    })
    _write_json(root / f4.STAGE8_EXPORT, {"files": [
        {"path": f"checkpoints/cnn/cnn-v1/seed_1/{n}", "sha256": "2" * 64, "size": 1} for n in CHECKPOINT_FILES
    ] + [
        {"path": f"checkpoints/cnn/cnn-v1/seed_0/{n}", "sha256": _sha((ckpt / n).read_bytes()),
         "size": (ckpt / n).stat().st_size} for n in CHECKPOINT_FILES
    ]})
    _write_json(root / f4.STAGE10_REPORT, {
        "bundle_seed": 0, "checkpoint_sha256": combined, "version": "risk-v1",
        "risk_policy_sha256": _sha(policy.read_bytes()),
    })
    _write_json(root / f4.STAGE6_REPORT, {"f2": {"ledger_sha256": LEDGER, "manifest_sha256": "3" * 64}})
    _write_json(root / f4.STAGE7_REPORT, {"version": "baseline-v1", "artifact_sha256": BASELINE_SHA})
    return root, ckpt


def _build(tree, out: Path, **kw):
    root, ckpt = tree
    kw.setdefault("created_at", CREATED_AT)
    kw.setdefault("git", _clean_git)
    return f4.build(ckpt, out, root=root, **kw)


def _nothing_written(out: Path) -> None:
    assert not out.exists() or not any(out.iterdir())
    assert not (out.parent / f4.RECORD_NAME).exists()
    assert not (out.parent / MANIFEST_NAME).exists()


def test_build_then_verify(tree, tmp_path):
    # The happy path: the record carries the evidence values, manifest_sha256 is the SHA-256 of the
    # manifest file, the record and the manifest copy sit next to (not inside) the bundle, and the
    # bundle passes the API's loader; verify on the same folder agrees.
    root, ckpt = tree
    out = tmp_path / "out" / "bundle-v1"
    record = _build(tree, out)
    expected = f4.expected_from_evidence(root)
    manifest_bytes = (out / MANIFEST_NAME).read_bytes()
    assert record["manifest_sha256"] == _sha(manifest_bytes)
    assert record["bundle_version"] == "bundle-v1" and record["protocol_version"] == "1.0"
    assert record["code_commit"] == COMMIT and record["created_at"] == CREATED_AT
    assert record["components"]["checkpoint"]["sha256"] == cnn_checkpoint.checkpoint_sha256(ckpt)
    assert record["components"]["policy"]["sha256"] == expected["policy"]["sha256"]
    assert record["components"]["profile"]["fingerprint"] == expected["profile"]["fingerprint"]
    assert record["baseline"] == {"version": "baseline-v1", "artifact_sha256": BASELINE_SHA,
                                  "source": f4.STAGE7_REPORT}
    assert record["f2_ledger_sha256"] == LEDGER
    assert set(record["evidence_sha256"]) == {f4.STAGE8_REPORT, f4.STAGE8_EXPORT, f4.STAGE10_REPORT,
                                              f4.POLICY_SOURCE, f4.PROFILE_SOURCE, f4.STAGE6_REPORT,
                                              f4.STAGE7_REPORT}
    assert set(record["environment"]) == {"python", "platform", "torch", "numpy", "safetensors"}
    assert record["checks"] and all(c["ok"] for c in record["checks"])
    # Record on disk: strict JSON, sorted keys, 2-space indent, trailing newline; equal to the return.
    text = (out.parent / f4.RECORD_NAME).read_text(encoding="utf-8")
    assert text == json.dumps(record, indent=2, sort_keys=True) + "\n"
    assert (out.parent / MANIFEST_NAME).read_bytes() == manifest_bytes
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["bundle_manifest.json", "checkpoint", "policy.json", "preprocessing.yaml"])
    assert load_bundle_model(out).bundle.manifest_sha256 == record["manifest_sha256"]
    report = f4.verify(out, out.parent / f4.RECORD_NAME)
    assert report["manifest_sha256"] == record["manifest_sha256"]
    assert all(c["ok"] for c in report["checks"])


def test_build_is_deterministic(tree, tmp_path):
    # Same inputs, created_at and commit -> identical manifest bytes and manifest_sha256, so F4 can
    # be reproduced and its hash re-derived.
    r1 = _build(tree, tmp_path / "a" / "bundle-v1")
    r2 = _build(tree, tmp_path / "b" / "bundle-v1")
    assert r1["manifest_sha256"] == r2["manifest_sha256"]
    assert (tmp_path / "a" / "bundle-v1" / MANIFEST_NAME).read_bytes() == \
        (tmp_path / "b" / "bundle-v1" / MANIFEST_NAME).read_bytes()


def test_default_created_at_is_utc_seconds(tree, tmp_path):
    # Without --created-at the build stamps the current UTC time to the second.
    record = _build(tree, tmp_path / "out" / "bundle-v1", created_at=None)
    assert f4._CREATED_AT.fullmatch(record["created_at"])


@pytest.mark.parametrize("git, match", [
    (lambda r: (COMMIT, " M src/x.py"), "dirty"),
    (lambda r: ("abc", ""), "40 lowercase hex"),
    (lambda r: ("AB" * 20, ""), "40 lowercase hex"),
])
def test_build_refuses_bad_provenance(tree, tmp_path, git, match):
    # The manifest's code_commit must name a clean commit, or the bundle cannot be traced to code.
    out = tmp_path / "out" / "bundle-v1"
    with pytest.raises(ValueError, match=match):
        _build(tree, out, git=git)
    _nothing_written(out)


@pytest.mark.parametrize("rel, changes, match", [
    (f4.STAGE10_REPORT, {"bundle_seed": 1}, "bundle_seed: mismatch"),
    (f4.STAGE10_REPORT, {"checkpoint_sha256": "4" * 64}, "checkpoint_sha256: mismatch"),
    (f4.STAGE10_REPORT, {"risk_policy_sha256": "5" * 64}, "policy sha256: mismatch"),
    (f4.STAGE10_REPORT, {"version": "risk-v2"}, "policy version: mismatch"),
    (f4.STAGE8_REPORT, {"profile_config_sha256": "6" * 64}, "profile sha256: mismatch"),
    (f4.STAGE8_REPORT, {"profile_fingerprint": "7" * 64}, "profile fingerprint: mismatch"),
    (f4.STAGE6_REPORT, {"f2": {"ledger_sha256": "8" * 64}}, "f2_ledger_sha256: mismatch"),
    (f4.STAGE8_REPORT, {"bundle_seed": True}, "non-negative int"),
    (f4.STAGE7_REPORT, {"artifact_sha256": "XYZ"}, "64 lowercase hex"),
])
def test_build_refuses_evidence_disagreement(tree, tmp_path, rel, changes, match):
    # The sources must agree with each other; the error names the files and fields involved and
    # nothing is built.
    _edit_json(tree[0] / rel, **changes)
    out = tmp_path / "out" / "bundle-v1"
    with pytest.raises(ValueError, match=match) as info:
        _build(tree, out)
    assert rel in str(info.value)
    _nothing_written(out)


def test_build_refuses_export_manifest_inconsistent_with_run_report(tree, tmp_path):
    # Per-file hashes in the export manifest must give the run report's combined hash.
    path = tree[0] / f4.STAGE8_EXPORT
    data = json.loads(path.read_text(encoding="utf-8"))
    data["files"][-1]["sha256"] = "9" * 64
    _write_json(path, data)
    out = tmp_path / "out" / "bundle-v1"
    with pytest.raises(ValueError, match="checkpoint_sha256 from per-file hashes: mismatch"):
        _build(tree, out)
    _nothing_written(out)


def test_build_refuses_missing_export_entry(tree, tmp_path):
    path = tree[0] / f4.STAGE8_EXPORT
    data = json.loads(path.read_text(encoding="utf-8"))
    data["files"] = data["files"][:-1]
    _write_json(path, data)
    with pytest.raises(ValueError, match="exactly one files"):
        _build(tree, tmp_path / "out" / "bundle-v1")


def test_build_refuses_policy_file_changed(tree, tmp_path):
    # The committed policy file is not the one Stage 10 recorded.
    policy = tree[0] / f4.POLICY_SOURCE
    policy.write_bytes(policy.read_bytes() + b"\n")
    out = tmp_path / "out" / "bundle-v1"
    with pytest.raises(ValueError, match="policy sha256: mismatch"):
        _build(tree, out)
    _nothing_written(out)


@pytest.mark.parametrize("name", CHECKPOINT_FILES)
def test_build_refuses_flipped_checkpoint_bit(tree, tmp_path, name):
    # One bit flipped in a downloaded checkpoint file: the per-file check names the file.
    path = tree[1] / name
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    out = tmp_path / "out" / "bundle-v1"
    with pytest.raises(ValueError, match=f"checkpoint file {name} sha256: mismatch"):
        _build(tree, out)
    _nothing_written(out)


def test_build_refuses_extra_checkpoint_file(tree, tmp_path):
    (tree[1] / "notes.txt").write_text("x", encoding="utf-8")
    out = tmp_path / "out" / "bundle-v1"
    with pytest.raises(ValueError, match="must hold exactly"):
        _build(tree, out)
    _nothing_written(out)


def test_build_refuses_missing_checkpoint_file(tree, tmp_path):
    (tree[1] / "weights.safetensors").unlink()
    out = tmp_path / "out" / "bundle-v1"
    with pytest.raises(ValueError, match="must hold exactly"):
        _build(tree, out)
    _nothing_written(out)


def test_build_never_overwrites_a_record(tree, tmp_path):
    # An existing record (an earlier F4) is left untouched and nothing is built.
    out = tmp_path / "out" / "bundle-v1"
    out.parent.mkdir()
    (out.parent / f4.RECORD_NAME).write_text("earlier", encoding="utf-8")
    with pytest.raises(ValueError, match="never overwritten"):
        _build(tree, out)
    assert (out.parent / f4.RECORD_NAME).read_text(encoding="utf-8") == "earlier"
    assert not out.exists()


def test_build_refuses_non_empty_out_dir(tree, tmp_path):
    out = tmp_path / "out" / "bundle-v1"
    out.mkdir(parents=True)
    (out / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        _build(tree, out)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]
    assert not (out.parent / f4.RECORD_NAME).exists()


@pytest.mark.parametrize("value", ["2026-11-12T10:00:00.5Z", "2026-11-12 10:00:00", "2026-11-12T10:00:00+00:00"])
def test_build_refuses_bad_created_at(tree, tmp_path, value):
    out = tmp_path / "out" / "bundle-v1"
    with pytest.raises(ValueError, match="created_at"):
        _build(tree, out, created_at=value)
    _nothing_written(out)


def test_failure_after_build_removes_the_bundle(tree, tmp_path, monkeypatch):
    # A failure in the verify step (after build_bundle wrote the folder) leaves nothing behind.
    def boom(_directory):
        raise ValueError("loader refused")
    monkeypatch.setattr(loader_module, "load_bundle_model", boom)
    out = tmp_path / "out" / "bundle-v1"
    with pytest.raises(ValueError, match="loader refused"):
        _build(tree, out)
    _nothing_written(out)


def test_verify_refuses_other_manifest_sha256(tree, tmp_path):
    # A record of another F4 (other manifest_sha256) does not verify this bundle.
    out = tmp_path / "out" / "bundle-v1"
    _build(tree, out)
    record = out.parent / f4.RECORD_NAME
    data = json.loads(record.read_text(encoding="utf-8"))
    data["manifest_sha256"] = "0" * 64
    record.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_sha256: mismatch"):
        f4.verify(out, record)


def test_verify_refuses_component_difference(tree, tmp_path):
    out = tmp_path / "out" / "bundle-v1"
    _build(tree, out)
    record = out.parent / f4.RECORD_NAME
    data = json.loads(record.read_text(encoding="utf-8"))
    data["components"]["policy"]["sha256"] = "0" * 64
    record.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="policy sha256: mismatch"):
        f4.verify(out, record)


def test_verify_refuses_tampered_bundle(tree, tmp_path):
    # A bundle changed after the build fails in the verifying loader.
    out = tmp_path / "out" / "bundle-v1"
    _build(tree, out)
    policy = out / "policy.json"
    policy.write_bytes(policy.read_bytes().replace(b"0.5", b"0.4", 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        f4.verify(out, out.parent / f4.RECORD_NAME)


def test_verify_refuses_record_of_wrong_schema(tree, tmp_path):
    out = tmp_path / "out" / "bundle-v1"
    _build(tree, out)
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="not an F4 record"):
        f4.verify(out, bad)


def test_cli_build_and_verify(tree, tmp_path, capsys):
    # CLI: build prints the record path, manifest_sha256 and "F4 build OK"; verify prints
    # manifest_sha256 and "F4 verify OK"; a failure exits non-zero with the message.
    root, ckpt = tree
    out = tmp_path / "out" / "bundle-v1"
    assert f4.main(["build", "--checkpoint-dir", str(ckpt), "--out", str(out), "--created-at", CREATED_AT],
                   root=root, git=_clean_git) == 0
    text = capsys.readouterr().out
    sha = load_bundle(out).manifest_sha256
    assert f"manifest_sha256: {sha}" in text and "F4 build OK" in text and f4.RECORD_NAME in text
    record = out.parent / f4.RECORD_NAME
    assert f4.main(["verify", "--bundle", str(out), "--record", str(record)], root=root) == 0
    text = capsys.readouterr().out
    assert f"manifest_sha256: {sha}" in text and "F4 verify OK" in text
    assert f4.main(["build", "--checkpoint-dir", str(ckpt), "--out", str(out)], root=root, git=_clean_git) == 1
    assert "F4 build FAILED" in capsys.readouterr().err
