"""F5 build/verify (src.bundle.f5) on synthetic evidence written by the real writers. CPU, no real data.

The evidence comes from the Stage 11 v2 test world (tests/evaluation/test_stage11_v2_run.py, reused,
not copied): seeded untrained crnn (log-Mel) and aasist (waveform) checkpoints, the v2 search reports
written by v2_search's report writers, risk-v2 written by v2_risk_run.run, and here the v2 Stage 11
claim -> score on those fakes with a RunRegistry holding the v1 pair. Their outputs are copied into a
fake repository tree with the real committed file names; git is injected.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tests._torch import require_torch

torch = require_torch()

import src.bundle.model_loader as loader_module  # noqa: E402
import src.evaluation.run_registry as run_registry  # noqa: E402
import src.evaluation.stage11_run as v1  # noqa: E402
import src.evaluation.stage11_v2_run as m  # noqa: E402
import src.tuning.risk_run as stage10  # noqa: E402
import src.tuning.v2_risk_run as vr  # noqa: E402
import src.tuning.v2_search as vs  # noqa: E402
from src.bundle import f4, f5  # noqa: E402
from src.bundle.bundle import CHECKPOINT_FILES, MANIFEST_NAME, checkpoint_dir_sha256, load_bundle  # noqa: E402
from src.bundle.model_loader import load_bundle_model  # noqa: E402
from src.evaluation import external_run  # noqa: E402
from src.evaluation.run_registry import RunRegistry  # noqa: E402
from src.models import v2_checkpoint  # noqa: E402
from tests._v2_fakes import COMMIT as FAKE_COMMIT  # noqa: E402
from tests._v2_fakes import PROFILE_PATH, sha  # noqa: E402
from tests.evaluation.test_stage11_v2_run import (  # noqa: E402,F401  (world is a fixture)
    ARCHS,
    RESULTS_REF,
    _inputs,
    _manifest_reader,
    _v1_pair,
    load_s,
    world,
)

REPO = Path(__file__).resolve().parents[2]
LEDGER = "c9fc2e230c0883028459d0b2b8de4f6ee537676cd5a4e259e9d0f545c60943cc"
COMMIT = "1" * 40
CREATED_AT = "2026-12-01T10:00:00Z"


def _clean_git(_root):
    return COMMIT, ""


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _stage11_v2(world, arch: str, folder: Path) -> tuple[Path, Path]:
    """The v2 Stage 11 claim -> score on the fakes (the patches of that module's own env fixture)."""
    a = world["arch"][arch]
    folder.mkdir(parents=True)
    registry = folder / "run_registry.jsonl"
    _v1_pair(registry)
    out = folder / "out"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "code_version", lambda: {"git_commit": FAKE_COMMIT, "git_dirty": False})
        mp.setattr(m, "load_f2_manifest", _manifest_reader)
        mp.setattr(m, "F2_MANIFEST_SHA256", world["f2"])
        mp.setattr(v1, "F2_MANIFEST_SHA256", world["f2"])
        mp.setattr(v1, "REQUIRED_PYTHON", ".".join(v1.platform.python_version_tuple()[:2]))
        mp.setattr(v1, "REQUIRED_TORCH", torch.__version__.split("+")[0])
        mp.setattr(m, "environment_record", lambda: ({"stub": True}, "pkg==1.0\n"))
        mp.setattr(m, "load_selection_set", load_s)
        mp.setattr(m, "V2_SEARCH_EVIDENCE", a["search"])
        mp.setattr(m, "V2_RISK_EVIDENCE", a["risk"])
        mp.setattr(m, "STAGE8_EVIDENCE", world["stage8"])
        mp.setattr(m, "V1_RUN_REPORT", world["v1_report"])
        record = m.claim(root=world["root"], registry_path=registry, out_dir=out, results_ref=RESULTS_REF,
                         **_inputs(world, arch))
        m.score(root=world["root"], registry_path=registry, out_dir=out, run_id=record["run_id"],
                **_inputs(world, arch))
    return out / m.RUN_REPORT, registry


@pytest.fixture(scope="module")
def templates(world, tmp_path_factory):
    """arch -> (repo template with the committed F5 evidence names, bundle-seed checkpoint folder)."""
    out = {}
    for arch in ARCHS:
        a = world["arch"][arch]
        base = tmp_path_factory.mktemp(f"f5_{arch}")
        report, registry = _stage11_v2(world, arch, base / "stage11")
        repo = base / "repo"
        for rel, src in ((f5.SEARCH_REPORT, a["search"] / vs.RUN_REPORT),
                         (f5.SELECTION_REPORT, a["search"] / vs.SELECTION_REPORT),
                         (f5.RISK_REPORT, a["risk"] / stage10.RUN_REPORT),
                         (f5.POLICY_SOURCE, a["risk"] / stage10.RISK_POLICY),
                         (f5.STAGE11_V2_REPORT, report),
                         (f5.RUN_REGISTRY, registry),
                         (f5.PROFILE_SOURCE, PROFILE_PATH)):
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, repo / rel)
        _write_json(repo / f5.STAGE6_REPORT, {"f2": {"ledger_sha256": LEDGER, "manifest_sha256": world["f2"]}})
        ckpt = a["runs"] / vs.run_dir_name(arch, a["bundle_seed"]) / vs.CHECKPOINT_DIR
        out[arch] = (repo, ckpt)
    return out


@pytest.fixture
def tree(templates, tmp_path):
    """A fresh copy of one architecture's tree: tree(arch) -> (root, checkpoint folder copy)."""
    def make(arch="crnn"):
        repo, ckpt = templates[arch]
        root = tmp_path / f"repo_{arch}"
        shutil.copytree(repo, root)
        ck = tmp_path / "drive" / arch / "checkpoint"
        shutil.copytree(ckpt, ck)
        return root, ck
    return make


def _build(root, ckpt, out: Path, **kw):
    kw.setdefault("created_at", CREATED_AT)
    kw.setdefault("git", _clean_git)
    return f5.build(ckpt, out, root=root, **kw)


def _nothing_written(out: Path) -> None:
    assert not out.exists() or not any(out.iterdir())
    assert not (out.parent / f5.RECORD_NAME).exists()
    assert not (out.parent / MANIFEST_NAME).exists()


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _edit(root: Path, rel: str, fn, *, rebind: bool = True) -> None:
    """Apply fn to one evidence JSON file; rebind re-signs risk-v2's v2_search_evidence so only the
    edited value disagrees."""
    data = _load(root / rel)
    fn(data)
    _write_json(root / rel, data)
    if rebind and rel in (f5.SEARCH_REPORT, f5.SELECTION_REPORT):
        risk = _load(root / f5.RISK_REPORT)
        risk["v2_search_evidence"] = {vs.RUN_REPORT: sha(root / f5.SEARCH_REPORT),
                                      vs.SELECTION_REPORT: sha(root / f5.SELECTION_REPORT)}
        _write_json(root / f5.RISK_REPORT, risk)


def _set(**values):
    return lambda d: d.update(values)


def _repoint(root: Path, ckpt: Path, arch: str, seed: int) -> None:
    """Point every evidence file at the (edited) checkpoint folder's combined hash."""
    old = _load(root / f5.RISK_REPORT)["checkpoint_sha256"]
    new = checkpoint_dir_sha256({n: sha(ckpt / n) for n in CHECKPOINT_FILES})
    _edit(root, f5.SEARCH_REPORT, lambda d: d["checkpoint_sha256"].update({vs.run_dir_name(arch, seed): new}))
    _edit(root, f5.RISK_REPORT, _set(checkpoint_sha256=new))
    bundle_id = f5.model_ids_v2(arch)[seed]
    _edit(root, f5.STAGE11_V2_REPORT, lambda d: d["model_set"][bundle_id].update(artifact_sha256=new))
    assert old != new


# --- Names shared with the research code ------------------------------------------------------------

def test_names_equal_the_research_definitions():
    # f5 restates these (shared core: no src.evaluation / src.tuning import); they must not drift.
    assert f5.ARCHITECTURES == v2_checkpoint.ARCHITECTURES
    assert f5.SEEDS == m.SEEDS
    for arch in f5.ARCHITECTURES:
        assert f5.model_ids_v2(arch) == m.model_ids_v2(arch)
        for seed in f5.SEEDS:
            assert f5.run_dir_name(arch, seed) == vs.run_dir_name(arch, seed)
    with pytest.raises(ValueError, match="winner must be one of"):
        f5.model_ids_v2("cnn")
    assert (f5.SEARCH_RUN_REPORT_NAME, f5.SEARCH_SELECTION_REPORT_NAME) == (vs.RUN_REPORT, vs.SELECTION_REPORT)
    assert (f5.RISK_RUN_REPORT_NAME, f5.RISK_POLICY_NAME) == (stage10.RUN_REPORT, stage10.RISK_POLICY)
    assert f5.STAGE11_RUN_REPORT_NAME == m.RUN_REPORT
    assert (f5.SEARCH_RUNS_DIR, f5.CHECKPOINT_DIR) == (vs.RUNS_DIR, vs.CHECKPOINT_DIR)
    assert REPO / f5.V2_SEARCH_DIR == vr.V2_SEARCH_EVIDENCE == m.V2_SEARCH_EVIDENCE
    assert REPO / f5.V2_RISK_DIR == m.V2_RISK_EVIDENCE
    assert (f5.STAGE11_DATASET, f5.STAGE11_SPLIT, f5.STAGE11_PURPOSE) == (m.DATASET, m.SPLIT, m.PURPOSE)
    assert (f5.REGISTRY_STARTED, f5.REGISTRY_COMPLETED, f5.REGISTRY_GENESIS) == (
        run_registry.STARTED, run_registry.COMPLETED, run_registry.GENESIS_SHA256)
    assert (REPO / f5.RUN_REGISTRY).is_file()  # the real registry's committed path


# --- Happy path and hand-over -----------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHS)
def test_build_then_verify_and_hand_over(world, tree, tmp_path, arch):
    # crnn takes the log-Mel, aasist the waveform: both go through load_bundle_model's v2 dispatch.
    # The record carries the evidence values; the record and manifest copy sit next to (not inside)
    # the bundle; verify and Stage 13's verify_f5_artifacts accept it.
    root, ckpt = tree(arch)
    a = world["arch"][arch]
    out = tmp_path / "out" / "bundle-v2"
    record = _build(root, ckpt, out)
    expected = f5.expected_from_evidence(root)
    manifest_bytes = (out / MANIFEST_NAME).read_bytes()
    assert record["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert record["bundle_version"] == "bundle-v2" and record["protocol_version"] == "1.0"
    assert record["code_commit"] == COMMIT and record["created_at"] == CREATED_AT
    assert (record["architecture"], record["bundle_seed"]) == (arch, a["bundle_seed"])
    ck = record["components"]["checkpoint"]
    assert ck["sha256"] == a["hashes"][a["bundle_seed"]] == v2_checkpoint.v2_checkpoint_sha256(ckpt)
    assert ck["files"] == {n: {"sha256": sha(ckpt / n), "size": (ckpt / n).stat().st_size} for n in CHECKPOINT_FILES}
    assert ck["version"] == "cnn-v2-test" and ck["source"] == f"runs/{arch}_seed{a['bundle_seed']}/checkpoint"
    assert record["components"]["policy"] == {
        "file": "policy.json", "source": f5.POLICY_SOURCE, "sha256": sha(root / f5.POLICY_SOURCE),
        "version": "risk-v2-test"}
    assert record["components"]["profile"]["sha256"] == sha(PROFILE_PATH)
    assert record["selection"] == {"winner": arch,
                                   "best_selection_m": {str(k): a["ms"][k] for k in f5.SEEDS}}
    report = _load(root / f5.STAGE11_V2_REPORT)
    assert record["stage11_v2"] == {"run_id": report["run_id"], "model_set": sorted(m.model_ids_v2(arch)),
                                    "source": f5.STAGE11_V2_REPORT}
    assert record["provenance"] == {"f2_manifest_sha256": world["f2"], "selection_set_sha256": world["s_sha"]}
    assert "baseline" not in record and record["f2_ledger_sha256"] == LEDGER
    assert set(record["evidence_sha256"]) == {f5.SEARCH_REPORT, f5.SELECTION_REPORT, f5.RISK_REPORT,
                                              f5.POLICY_SOURCE, f5.STAGE11_V2_REPORT, f5.RUN_REGISTRY,
                                              f5.PROFILE_SOURCE, f5.STAGE6_REPORT}
    assert record["evidence_sha256"] == expected["evidence_sha256"]
    names = [c["name"] for c in record["checks"]]
    assert all(c["ok"] is True for c in record["checks"]) and len(names) == len(set(names))
    for name in ("registry v2 stage11_test records", "loaded architecture", "loaded model device",
                 "loaded model eval mode", "checkpoint.json seed", "manifest_sha256"):
        assert name in names
    text = (out.parent / f5.RECORD_NAME).read_text(encoding="utf-8")
    assert text == json.dumps(record, indent=2, sort_keys=True) + "\n"
    for local in (str(tmp_path), str(ckpt)):  # committed later: no local path in it
        assert local not in text and json.dumps(local)[1:-1] not in text
    assert (out.parent / MANIFEST_NAME).read_bytes() == manifest_bytes
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["bundle_manifest.json", "checkpoint", "policy.json", "preprocessing.yaml"])

    bm = load_bundle_model(out)
    assert bm.architecture == arch and bm.bundle.manifest_sha256 == record["manifest_sha256"]
    assert (bm.mean is None) == (arch == "aasist")
    verified = f5.verify(out, out.parent / f5.RECORD_NAME)
    assert verified["manifest_sha256"] == record["manifest_sha256"] and verified["architecture"] == arch
    assert all(c["ok"] for c in verified["checks"])

    # Hand-over: Stage 13 (Khalil) verifies the bundle and the record with the reviewed loaders.
    model, anchor = external_run.verify_f5_artifacts(out, out.parent / f5.RECORD_NAME,
                                                     bundle_verifier=load_bundle, bundle_loader=load_bundle_model)
    assert anchor == record["manifest_sha256"] and model.architecture == arch


def test_build_is_deterministic(tree, tmp_path):
    root, ckpt = tree()
    r1 = _build(root, ckpt, tmp_path / "a" / "bundle-v2")
    r2 = _build(root, ckpt, tmp_path / "b" / "bundle-v2")
    assert r1["manifest_sha256"] == r2["manifest_sha256"]


# --- Evidence refusals (before anything is written) -------------------------------------------------

def _other(seed: int) -> int:
    return (seed + 1) % 3


EVIDENCE_CASES = {  # case: (file, edit(data, world) or None, rebind, message)
    "selection_winner": (f5.SELECTION_REPORT, lambda d, w: d.update(selected="cnn-aug"), True,
                         "selected: mismatch"),
    "selection_bundle_seed": (f5.SELECTION_REPORT, lambda d, w: d.update(bundle_seed=_other(d["bundle_seed"])),
                              True, "bundle_seed: mismatch"),
    "search_selected_unknown": (f5.SEARCH_REPORT, lambda d, w: d.update(selected="cnn-v1"), True,
                                "selected must be one of"),
    "search_seed_hash_missing": (f5.SEARCH_REPORT, lambda d, w: d["checkpoint_sha256"].pop(
        f"crnn_seed{_other(d['bundle_seed'])}"), True, "checkpoint_sha256 has no entry crnn_seed"),
    "selection_seeds": (f5.SELECTION_REPORT, lambda d, w: d["seeds"].pop("2"), True, "seeds must have exactly"),
    "risk_architecture": (f5.RISK_REPORT, lambda d, w: d.update(architecture="aasist"), False,
                          "risk architecture: mismatch"),
    "risk_bundle_seed": (f5.RISK_REPORT, lambda d, w: d.update(bundle_seed=_other(d["bundle_seed"])), False,
                         "risk bundle_seed: mismatch"),
    "risk_checkpoint": (f5.RISK_REPORT, lambda d, w: d.update(checkpoint_sha256="0" * 64), False,
                        "risk checkpoint_sha256: mismatch"),
    "risk_profile": (f5.RISK_REPORT, lambda d, w: d.update(profile_fingerprint="5" * 64), False,
                     "risk profile_fingerprint: mismatch"),
    "risk_f2": (f5.RISK_REPORT, lambda d, w: d.update(f2_manifest_sha256="7" * 64), False,
                "risk f2_manifest_sha256: mismatch"),
    "risk_search_binding": (f5.SEARCH_REPORT, lambda d, w: d.update(note="edited"), False,
                            "risk v2_search_evidence: mismatch"),
    "risk_policy_sha256": (f5.RISK_REPORT, lambda d, w: d.update(risk_policy_sha256="5" * 64), False,
                           "policy sha256: mismatch"),
    "risk_policy_version": (f5.RISK_REPORT, lambda d, w: d.update(version="risk-v9"), False,
                            "policy version: mismatch"),
    "stage11_architecture": (f5.STAGE11_V2_REPORT, lambda d, w: d.update(architecture="aasist"), False,
                             "stage11_v2 architecture: mismatch"),
    "stage11_bundle_seed": (f5.STAGE11_V2_REPORT, lambda d, w: d.update(bundle_seed=_other(d["bundle_seed"])),
                            False, "stage11_v2 bundle_seed: mismatch"),
    "stage11_model_set": (f5.STAGE11_V2_REPORT, lambda d, w: d.update(model_set={
        k.replace("crnn", "cnn-aug"): v for k, v in d["model_set"].items()}), False,
        "stage11_v2 model_set: mismatch"),
    "stage11_bundle_artifact": (f5.STAGE11_V2_REPORT, lambda d, w: d["model_set"][
        f"crnn-v2-seed{d['bundle_seed']}"].update(artifact_sha256="0" * 64), False,
        "stage11_v2 bundle artifact_sha256: mismatch"),
    "stage11_other_artifact": (f5.STAGE11_V2_REPORT, lambda d, w: d["model_set"][
        f"crnn-v2-seed{_other(d['bundle_seed'])}"].update(artifact_sha256="0" * 64), False,
        "artifact_sha256: mismatch"),
    "stage11_other_run_id": (f5.STAGE11_V2_REPORT, lambda d, w: d.update(run_id="f" * 32), False,
                             "registry v2 stage11_test records: mismatch"),
    "stage11_purpose": (f5.STAGE11_V2_REPORT, lambda d, w: d.update(evaluation_purpose="stage14_robustness"),
                        False, "stage11_v2 evaluation_purpose: mismatch"),
    "stage6_ledger": (f5.STAGE6_REPORT, lambda d, w: d["f2"].update(ledger_sha256="8" * 64), False,
                      "f2_ledger_sha256: mismatch"),
    "stage6_manifest": (f5.STAGE6_REPORT, lambda d, w: d["f2"].update(manifest_sha256="8" * 64), False,
                        "f2_manifest_sha256: mismatch"),
}


@pytest.mark.parametrize("case", sorted(EVIDENCE_CASES))
def test_build_refuses_evidence_disagreement(world, tree, tmp_path, case):
    # Every source must agree; the error names the files (and fields) involved; nothing is built.
    rel, edit, rebind, message = EVIDENCE_CASES[case]
    root, ckpt = tree()
    _edit(root, rel, lambda d: edit(d, world), rebind=rebind)
    out = tmp_path / "out" / "bundle-v2"
    with pytest.raises(ValueError, match=message) as info:
        _build(root, ckpt, out)
    assert rel in str(info.value)
    _nothing_written(out)


def _registry_lines(root: Path) -> list[bytes]:
    return (root / f5.RUN_REGISTRY).read_bytes().splitlines(keepends=True)


@pytest.mark.parametrize("case, message", [
    ("started_only", "registry v2 stage11_test records: mismatch"),    # scored, never completed
    ("v1_pair_only", "registry v2 stage11_test records: mismatch"),    # no v2 claim at all
    ("edited_line", "breaks the hash chain"),
    ("missing_report", "stage11_v2_run/run_report.json is missing"),
    ("missing_registry", "run_registry.jsonl is missing"),
])
def test_build_refuses_without_a_completed_v2_stage11_run(tree, tmp_path, case, message):
    # D48 (7): the v2 Stage 11 run is claimed and completed before F5; otherwise F5 is refused.
    root, ckpt = tree()
    lines = _registry_lines(root)
    assert len(lines) == 4  # v1 started + completed, v2 started + completed
    path = root / f5.RUN_REGISTRY
    if case == "started_only":
        path.write_bytes(b"".join(lines[:3]))
        assert [r.status for r in RunRegistry(path).records()][-1] == "started"  # still a valid registry
    elif case == "v1_pair_only":
        path.write_bytes(b"".join(lines[:2]))
    elif case == "edited_line":
        path.write_bytes(b"".join([lines[0].replace(b"8888", b"9999", 1), *lines[1:]]))
    elif case == "missing_report":
        (root / f5.STAGE11_V2_REPORT).unlink()
    else:
        path.unlink()
    out = tmp_path / "out" / "bundle-v2"
    with pytest.raises(ValueError, match=message):
        _build(root, ckpt, out)
    _nothing_written(out)


def test_build_refuses_policy_file_changed(tree, tmp_path):
    root, ckpt = tree()
    policy = root / f5.POLICY_SOURCE
    policy.write_bytes(policy.read_bytes() + b"\n")
    out = tmp_path / "out" / "bundle-v2"
    with pytest.raises(ValueError, match="policy sha256: mismatch"):
        _build(root, ckpt, out)
    _nothing_written(out)


def test_build_refuses_profile_file_changed(tree, tmp_path):
    # The committed profile is not the one the v2 checkpoints were trained with.
    root, ckpt = tree()
    profile = root / f5.PROFILE_SOURCE
    profile.write_bytes(profile.read_bytes() + b"# edited\n")
    out = tmp_path / "out" / "bundle-v2"
    with pytest.raises(ValueError, match="profile sha256: mismatch"):
        _build(root, ckpt, out)
    _nothing_written(out)


# --- Checkpoint folder refusals ---------------------------------------------------------------------

def test_build_refuses_extra_or_missing_checkpoint_file(tree, tmp_path):
    root, ckpt = tree()
    out = tmp_path / "out" / "bundle-v2"
    (ckpt / "run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must hold exactly"):
        _build(root, ckpt, out)
    (ckpt / "run.json").unlink()
    (ckpt / "weights.safetensors").unlink()
    with pytest.raises(ValueError, match="must hold exactly"):
        _build(root, ckpt, out)
    _nothing_written(out)


def test_build_refuses_wrong_combined_hash(tree, tmp_path):
    # One byte flipped in the weights: the folder is not the checkpoint the evidence names.
    root, ckpt = tree()
    path = ckpt / "weights.safetensors"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    out = tmp_path / "out" / "bundle-v2"
    with pytest.raises(ValueError, match="checkpoint_sha256 of the checkpoint folder: mismatch"):
        _build(root, ckpt, out)
    _nothing_written(out)


def test_build_refuses_another_seed_checkpoint(world, tree, tmp_path):
    # A non-bundle seed's folder of the same winner: a real v2 checkpoint, but not F5's.
    root, _ = tree()
    a = world["arch"]["crnn"]
    other = a["runs"] / vs.run_dir_name("crnn", _other(a["bundle_seed"])) / vs.CHECKPOINT_DIR
    copy = tmp_path / "drive" / "other"
    shutil.copytree(other, copy)
    out = tmp_path / "out" / "bundle-v2"
    with pytest.raises(ValueError, match="checkpoint_sha256 of the checkpoint folder: mismatch"):
        _build(root, copy, out)
    _nothing_written(out)


@pytest.mark.parametrize("field, value, message", [
    ("architecture", "cnn-aug", "checkpoint.json architecture: mismatch"),
    ("seed", "other", "checkpoint.json seed: mismatch"),
    ("version", "cnn-v2-other", "checkpoint.json version: mismatch"),
])
def test_build_refuses_checkpoint_metadata_disagreement(world, tree, tmp_path, field, value, message):
    # checkpoint.json edited and every evidence file re-pointed at the new combined hash, so only the
    # metadata check can catch it (before build_bundle and the loader).
    root, ckpt = tree()
    seed = world["arch"]["crnn"]["bundle_seed"]
    meta = _load(ckpt / "checkpoint.json")
    meta[field] = _other(seed) if value == "other" else value
    (ckpt / "checkpoint.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _repoint(root, ckpt, "crnn", seed)
    out = tmp_path / "out" / "bundle-v2"
    with pytest.raises(ValueError, match=message):
        _build(root, ckpt, out)
    _nothing_written(out)


# --- Output, provenance and failure handling --------------------------------------------------------

@pytest.mark.parametrize("parts", [("bundle-v2",), ("x", "bundle-v2")])
def test_build_refuses_out_dir_inside_repository(tree, parts):
    root, ckpt = tree()
    out = root.joinpath(*parts)
    with pytest.raises(ValueError, match="output: .* is inside the repository"):
        _build(root, ckpt, out)
    _nothing_written(out)


def test_build_refuses_checkpoint_dir_inside_repository(tree, tmp_path):
    root, ckpt = tree()
    inside = root / "checkpoints" / "checkpoint"
    shutil.copytree(ckpt, inside)
    out = tmp_path / "out" / "bundle-v2"
    with pytest.raises(ValueError, match="checkpoint: .* is inside the repository"):
        _build(root, inside, out)
    _nothing_written(out)


@pytest.mark.parametrize("git, match", [
    (lambda r: (COMMIT, " M src/x.py"), "dirty"),
    (lambda r: ("abc", ""), "40 lowercase hex"),
])
def test_build_refuses_bad_provenance(tree, tmp_path, git, match):
    root, ckpt = tree()
    out = tmp_path / "out" / "bundle-v2"
    with pytest.raises(ValueError, match=match):
        _build(root, ckpt, out, git=git)
    _nothing_written(out)


def test_build_never_overwrites_a_record_or_a_bundle(tree, tmp_path):
    root, ckpt = tree()
    out = tmp_path / "out" / "bundle-v2"
    out.parent.mkdir()
    (out.parent / f5.RECORD_NAME).write_text("earlier", encoding="utf-8")
    with pytest.raises(ValueError, match="never overwritten"):
        _build(root, ckpt, out)
    assert (out.parent / f5.RECORD_NAME).read_text(encoding="utf-8") == "earlier" and not out.exists()
    (out.parent / f5.RECORD_NAME).unlink()
    out.mkdir()
    (out / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        _build(root, ckpt, out)
    assert [p.name for p in out.iterdir()] == ["keep.txt"] and not (out.parent / f5.RECORD_NAME).exists()


def test_failure_after_build_removes_the_bundle(tree, tmp_path, monkeypatch):
    # The loader fails after build_bundle wrote the folder: nothing is left behind.
    root, ckpt = tree()

    def boom(_directory):
        raise ValueError("loader refused")
    monkeypatch.setattr(loader_module, "load_bundle_model", boom)
    out = tmp_path / "out" / "bundle-v2"
    with pytest.raises(ValueError, match="loader refused"):
        _build(root, ckpt, out)
    _nothing_written(out)


@pytest.mark.parametrize("change, message", [
    ({"architecture": "cnn"}, "loaded architecture: mismatch"),
    ("train", "loaded model eval mode: mismatch"),
])
def test_loaded_model_must_be_the_v2_winner_in_eval_mode(tree, tmp_path, monkeypatch, change, message):
    # f5 checks the loader's result itself: v2 dispatch to the winner, eval mode (CPU checked alike).
    root, ckpt = tree()
    real = loader_module.load_bundle_model

    def altered(directory):
        bm = real(directory)
        if change == "train":
            bm.model.train()
            return bm
        return dataclasses.replace(bm, **change)
    monkeypatch.setattr(loader_module, "load_bundle_model", altered)
    out = tmp_path / "out" / "bundle-v2"
    with pytest.raises(ValueError, match=message):
        _build(root, ckpt, out)
    _nothing_written(out)


# --- Verify -----------------------------------------------------------------------------------------

def test_verify_passes_on_a_copy_and_fails_on_a_changed_file(tree, tmp_path):
    root, ckpt = tree()
    out = tmp_path / "out" / "bundle-v2"
    record = _build(root, ckpt, out)
    copy = tmp_path / "copy" / "bundle-v2"
    shutil.copytree(out, copy)
    assert f5.verify(copy, out.parent / f5.RECORD_NAME)["manifest_sha256"] == record["manifest_sha256"]
    policy = copy / "policy.json"
    policy.write_bytes(policy.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        f5.verify(copy, out.parent / f5.RECORD_NAME)


def test_verify_refuses_record_differences(tree, tmp_path):
    root, ckpt = tree()
    out = tmp_path / "out" / "bundle-v2"
    _build(root, ckpt, out)
    record = out.parent / f5.RECORD_NAME
    for edit, message in ((lambda d: d.update(manifest_sha256="0" * 64), "manifest_sha256: mismatch"),
                          (lambda d: d.update(architecture="aasist"), "architecture: mismatch"),
                          (lambda d: d["components"]["policy"].update(sha256="0" * 64), "policy sha256: mismatch"),
                          (lambda d: d.update(bundle_version="bundle-v1"), "not an F5 record")):
        data = _load(record)
        edit(data)
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            f5.verify(out, bad)


def test_cli_build_and_verify(tree, tmp_path, capsys):
    root, ckpt = tree()
    out = tmp_path / "out" / "bundle-v2"
    assert f5.main(["build", "--checkpoint-dir", str(ckpt), "--out", str(out), "--created-at", CREATED_AT],
                   root=root, git=_clean_git) == 0
    text = capsys.readouterr().out
    anchor = load_bundle(out).manifest_sha256
    assert f"manifest_sha256: {anchor}" in text and "F5 build OK" in text and f5.RECORD_NAME in text
    assert f5.main(["verify", "--bundle", str(out), "--record", str(out.parent / f5.RECORD_NAME)], root=root) == 0
    assert "F5 verify OK" in capsys.readouterr().out
    assert f5.main(["build", "--checkpoint-dir", str(ckpt), "--out", str(out)], root=root, git=_clean_git) == 1
    assert "F5 build FAILED" in capsys.readouterr().err


def test_f4_record_is_not_an_f5_record(tree, tmp_path):
    # The F4 tool is untouched and its record format is refused by f5.verify (bundle-v1).
    root, ckpt = tree()
    out = tmp_path / "out" / "bundle-v2"
    _build(root, ckpt, out)
    f4_record = REPO / "docs" / "audit_evidence" / "stage12_bundle" / f4.RECORD_NAME
    with pytest.raises(ValueError, match="not an F5 record"):
        f5.verify(out, f4_record)
