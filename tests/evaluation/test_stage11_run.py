"""Stage 11 runner: claim phase (pre-claim checks, claim, evidence), score phase (marker, re-checks,
features and scoring twice, independent prediction checks, per-clip files, run report, registry), and
the validation-only rehearsal.

Synthetic only: a synthetic manifest, fake audio reads and fake features (monkeypatched, derived
from each path with a fixed seed), real log-Mel and MFCC caches built from those fakes, three tiny
real CNN checkpoints (save_checkpoint, untrained, seeds 0-2), a tiny real baseline artifact, a
real risk policy file and synthetic Stage 7, 8 and 10 evidence. CPU; no dataset, no W&B.
Convention: label 1 = spoof (FAKE), 0 = bonafide (REAL).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._torch import require_torch

torch = require_torch()

import src.evaluation.stage11_run as run_module  # noqa: E402
import src.features.logmel_cache as logmel_module  # noqa: E402
import src.features.mfcc_cache as mfcc_module  # noqa: E402
from src.evaluation.clip_scores import read_clip_scores  # noqa: E402
from src.evaluation.metrics import eer  # noqa: E402
from src.evaluation.run_registry import COMPLETED, STARTED, RunRegistry  # noqa: E402
from src.features.logmel_cache import build_logmel_cache, load_logmel_cache  # noqa: E402
from src.features.mfcc_cache import build_mfcc_cache, load_mfcc_cache  # noqa: E402
from src.models.baseline import BaselineModel, _write_npz, artifact_sha256  # noqa: E402
from src.models.baseline import save as save_baseline  # noqa: E402
from src.models.cnn import build_cnn  # noqa: E402
from src.models.cnn_checkpoint import checkpoint_sha256, load_checkpoint, save_checkpoint  # noqa: E402
from src.preprocessing import load_profile  # noqa: E402
from src.preprocessing.profile import PreprocessingError  # noqa: E402
from src.risk.calibration import PlattCalibration  # noqa: E402
from src.risk.decisions import AUTHENTIC, SYNTHETIC, UNCERTAIN, three_way_decisions  # noqa: E402
from src.risk.policy import RiskPolicy  # noqa: E402
from src.tuning.risk_run import score_clips  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO / "configs" / "preprocessing.yaml"
CONFIG_PATH = REPO / "configs" / "cnn.yaml"
EVIDENCE = REPO / "docs" / "audit_evidence"
COMMIT = "c" * 40
# Rows per subgroup (REAL, FAKE_mp3, FAKE_other) per split.
SIZES = {"val-A": (3, 2, 2), "val-B": (3, 2, 2), "test": (4, 2, 2)}
N_TEST = sum(SIZES["test"])
ORIGINAL_ENVIRONMENT_RECORD = run_module.environment_record
FAILS = (ValueError, PreprocessingError)
RESULTS_REF = "stage11/run_report.json"
ENV_FILES = ("environment.json", "pip_freeze.txt")
CLAIM_FILES = (*ENV_FILES, "claim.json")
OTHER_RUN_ID = "f" * 32
SCORE_START_FILES = ("scoring_started.json", "environment_score.json")
IMPOSSIBLE_EER = 0.123456  # no EER of 7 clips (3 REAL, 4 FAKE) takes this value


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fake_features(path: str, perturb: float = 0.0):
    """Deterministic float32 log-Mel (1, 64, 197) and MFCC (40,) from the path; FAKE shifted up."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(path.encode()).digest()[:8], "little"))
    shift = 0.5 if "/FAKE" in path else 0.0
    log_mel = (rng.standard_normal((1, 64, 197)) + shift + perturb).astype(np.float32)
    mfcc = (rng.standard_normal(40) + shift + perturb).astype(np.float32)
    return log_mel, mfcc


def _fake_read(root, row):
    return row["path"], 16000  # the "waveform" carries the path to _fake_extract


def _fake_extract(perturb=0.0):
    def extract(waveform, sample_rate, profile):
        return (None, *_fake_features(waveform, perturb))
    return extract


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _checkpoint_meta(seed: int, val_a_eer: float, profile, manifest_sha: str) -> dict:
    return {
        "schema_version": 1, "version": "cnn-v1", "n_blocks": 3, "channels": [16, 32, 64],
        "dropout": 0.3, "input_shape": [1, 64, 197], "learning_rate": 0.001, "seed": seed,
        "best_epoch": 1, "val_a_eer": val_a_eer, "code_commit": COMMIT,
        "environment": {"python": "3.13.16"}, "profile_fingerprint": profile.fingerprint(),
        "f2_manifest_sha256": manifest_sha, "train_cache_npz_sha256": "a" * 64,
        "val_a_cache_npz_sha256": "b" * 64, "determinism": {"seed": seed},
    }


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Manifest, caches (build A, build B, a differing build), checkpoints, baseline, policy, evidence."""
    base = tmp_path_factory.mktemp("stage11")
    profile = load_profile(PROFILE_PATH)
    rows = []
    for split, counts in SIZES.items():
        for subgroup, n in zip(("REAL", "FAKE_mp3", "FAKE_other"), counts):
            for i in range(n):
                rows.append({"path": f"{split}/{subgroup}/{i:02d}.wav",
                             "label": "REAL" if subgroup == "REAL" else "FAKE", "subgroup": subgroup,
                             "group": f"g-{split}-{subgroup}-{i}", "split": split, "hash": "0" * 64})
    manifest = pd.DataFrame(rows, dtype=str)
    manifest_path = base / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    manifest_sha, profile_sha = _sha(manifest_path), _sha(PROFILE_PATH)

    builders = {"log_mel": build_logmel_cache, "mfcc": build_mfcc_cache}

    def build(folder, splits, perturb=0.0):
        with pytest.MonkeyPatch.context() as mp:
            for module in (logmel_module, mfcc_module):
                mp.setattr(module, "read_verified_audio", _fake_read)
                mp.setattr(module, "extract_features", _fake_extract(perturb))
            for feature, builder in builders.items():
                for split in splits:
                    builder(base, manifest, profile, split, folder / feature / split,
                            f2_manifest_sha256=manifest_sha, profile_config_sha256=profile_sha,
                            code_commit=COMMIT)
        return folder

    cache_a = build(base / "cache_a", ("val-A", "val-B"))
    cache_b = build(base / "cache_b", ("val-A",))
    cache_bad = build(base / "cache_bad", ("val-A",), perturb=1e-3)

    def load(feature, split):
        loader = load_logmel_cache if feature == "log_mel" else load_mfcc_cache
        return loader(cache_a / feature / split, split, profile, expected_f2_manifest_sha256=manifest_sha)

    lm_a, y_a, _, lm_meta_a = load("log_mel", "val-A")
    lm_b, y_b, paths_b, lm_meta_b = load("log_mel", "val-B")
    mf_a, _, _, mf_meta_a = load("mfcc", "val-A")
    mf_b, _, _, mf_meta_b = load("mfcc", "val-B")

    # Three untrained 3-block CNNs; val_a_eer is the EER of the runner's own scoring path.
    checkpoints, hashes, cnn = base / "checkpoints", {}, {}
    for seed in run_module.SEEDS:
        torch.manual_seed(seed)
        model = build_cnn(3).eval()
        rng = np.random.default_rng(100 + seed)
        mean = rng.standard_normal(64).astype(np.float32)
        std = rng.uniform(0.5, 1.5, 64).astype(np.float32)
        val_a_eer = eer(y_a, score_clips(model, mean, std, lm_a, 256))[0]
        folder = save_checkpoint(model, mean, std, _checkpoint_meta(seed, val_a_eer, profile, manifest_sha),
                                 checkpoints / f"seed_{seed}")
        hashes[str(seed)] = checkpoint_sha256(folder)
        cnn[seed] = (model, mean, std)

    rng = np.random.default_rng(7)
    proto = dict(version="baseline-v1", mean=np.zeros(40), std=np.ones(40),
                 support_vectors=rng.standard_normal((6, 40)), dual_coef=rng.standard_normal(6),
                 intercept=0.1, gamma=0.02, C=1.0)
    val_b_rate, val_b_threshold = eer(y_b, BaselineModel(**proto, threshold=0.0).score(mf_b))
    baseline = BaselineModel(**proto, threshold=val_b_threshold)
    baseline_dir = save_baseline(baseline, base / "baseline")

    policy = RiskPolicy(version="risk-test", calibration=PlattCalibration(a=1.0, b=0.0, n_spoof=4, n_bonafide=3),
                        threshold=0.5, half_width=0.1, target_deferral=0.5, n_spoof_fit=4,
                        n_bonafide_fit=3, deferral_on_fit=0.2)
    policy_path = base / "risk_policy.json"
    policy_path.write_bytes(policy.to_json().encode("utf-8"))

    bundle_scores = score_clips(*cnn[0], lm_b, 256)
    scores_path = base / "val_b_scores.npz"
    _write_npz(scores_path, {"paths": paths_b, "labels": y_b, "score": bundle_scores,
                             "p": policy.calibration.transform(bundle_scores)})

    shared = {"profile_fingerprint": profile.fingerprint(), "profile_config_sha256": profile_sha,
              "f2_manifest_sha256": manifest_sha}
    evidence = base / "evidence"
    _write_json(evidence / "stage7" / "run_report.json", {
        **shared, "version": "baseline-v1", "artifact_sha256": artifact_sha256(baseline_dir),
        "caches": {"val-A": {"npz_sha256": mf_meta_a["npz_sha256"]}, "val-B": {"npz_sha256": mf_meta_b["npz_sha256"]}},
        "val_a_eer": eer(y_a, baseline.score(mf_a))[0], "val_b_eer": val_b_rate, "val_b_threshold": val_b_threshold,
    })
    _write_json(evidence / "stage8" / "run_report.json", {
        **shared, "bundle_seed": 0, "checkpoint_sha256": hashes, "cnn_config_sha256": _sha(CONFIG_PATH),
        "caches": {"val-A": {"npz_sha256": lm_meta_a["npz_sha256"]}},
        "val_a_cache_npz_sha256": lm_meta_a["npz_sha256"],
    })
    _write_json(evidence / "stage8" / "selection_report.json", {"bundle_seed": 0})
    _write_json(evidence / "stage10" / "run_report.json", {
        **shared, "bundle_seed": 0, "checkpoint_sha256": hashes["0"], "risk_policy_sha256": _sha(policy_path),
        "val_b_scores_sha256": _sha(scores_path), "caches": {"val-B": {"npz_sha256": lm_meta_b["npz_sha256"]}},
        "scoring": {"device": "cpu", "batch_size": 256},
    })
    return {"base": base, "profile": profile, "manifest": manifest, "manifest_path": manifest_path,
            "cache_a": cache_a, "cache_b": cache_b, "cache_bad": cache_bad, "checkpoints": checkpoints,
            "hashes": hashes, "baseline_dir": baseline_dir, "baseline": baseline, "policy": policy,
            "policy_path": policy_path, "scores_path": scores_path, "evidence": evidence}


def _inputs(world, **overrides) -> dict:
    ev = world["evidence"]
    inputs = {
        "checkpoints": world["checkpoints"], "baseline_dir": world["baseline_dir"],
        "risk_policy": world["policy_path"], "profile_path": PROFILE_PATH, "config_path": CONFIG_PATH,
        "manifest_path": world["manifest_path"], "stage7_evidence": ev / "stage7",
        "stage8_evidence": ev / "stage8", "stage10_evidence": ev / "stage10",
        "cache_a": world["cache_a"], "cache_b": world["cache_b"],
    }
    inputs.update(overrides)
    return inputs


@pytest.fixture()
def patched(world, monkeypatch, tmp_path):
    """Clean commit, the synthetic manifest as F2, the synthetic baseline as D41, spies on audio reads."""
    state = {"reads": [], "claimed_at_first_read": None, "perturb_after": None, "extracts": 0,
             "registry": tmp_path / "registry.jsonl", "commits": 0, "out": None,
             "env_files_at_first_read": None, "score_env_at_first_read": None, "crash_on_read": False,
             "perturb_only": None}

    def code_version():
        state["commits"] += 1
        return {"git_commit": COMMIT, "git_dirty": False}

    def read(root, row):
        if not state["reads"]:
            lines = state["registry"].read_bytes().splitlines() if state["registry"].exists() else []
            state["claimed_at_first_read"] = bool(lines) and json.loads(lines[-1])["status"] == STARTED
            out = state["out"]
            state["env_files_at_first_read"] = out is not None and all((out / f).is_file() for f in ENV_FILES)
            state["score_env_at_first_read"] = out is not None and (out / "environment_score.json").is_file()
        if state["crash_on_read"]:
            raise OSError("simulated crash on the first audio read")
        state["reads"].append(row["path"])
        return _fake_read(root, row)

    def extract(waveform, sample_rate, profile):
        state["extracts"] += 1
        after = state["perturb_after"]
        delta = 1e-3 if after is not None and state["extracts"] > after else 0.0
        log_mel, mfcc = _fake_features(waveform, delta)
        if state["perturb_only"] == "mfcc":
            log_mel, mfcc = _fake_features(waveform)[0], _fake_features(waveform, 1e-3)[1]
        return None, log_mel, mfcc

    monkeypatch.setattr(run_module, "code_version", code_version)
    monkeypatch.setattr(run_module, "load_f2_manifest", lambda p: pd.read_csv(p, dtype=str, keep_default_na=False))
    monkeypatch.setattr(run_module, "F2_MANIFEST_SHA256", _sha(world["manifest_path"]))
    monkeypatch.setattr(run_module, "BASELINE_ARTIFACT_SHA256", artifact_sha256(world["baseline_dir"]))
    monkeypatch.setattr(run_module, "REQUIRED_PYTHON", ".".join(run_module.platform.python_version_tuple()[:2]))
    monkeypatch.setattr(run_module, "REQUIRED_TORCH", torch.__version__.split("+")[0])
    monkeypatch.setattr(run_module, "read_verified_audio", read)
    monkeypatch.setattr(run_module, "extract_features", extract)
    monkeypatch.setattr(run_module, "environment_record", lambda: ({"stub": True}, "pkg==1.0\n"))
    return state


def _claim(world, state, out, results_ref=RESULTS_REF, **overrides):
    state["out"] = out
    return run_module.claim(registry_path=state["registry"], out_dir=out, results_ref=results_ref,
                            **_inputs(world, **overrides))


def _score(world, state, out, run_id, registry=None, **overrides):
    state["out"] = out
    return run_module.score(root=world["base"], registry_path=registry or state["registry"], out_dir=out,
                            run_id=run_id, **_inputs(world, **overrides))


def _run(world, state, out, results_ref=RESULTS_REF, **overrides):
    """Both phases, as the notebook calls them (claim, flush Drive, score)."""
    record = _claim(world, state, out, results_ref, **overrides)
    return _score(world, state, out, record["run_id"], **overrides)


def _rehearse(world, out, splits=run_module.REHEARSAL_SPLITS, **overrides):
    return run_module.rehearse(root=world["base"], stage10_scores=overrides.pop("stage10_scores", world["scores_path"]),
                               out_dir=out, splits=splits, **_inputs(world, **overrides))


def _edited_evidence(world, tmp_path, edits: dict) -> dict:
    """A copy of the evidence with {stage: function(report)} edits; returns the input overrides."""
    folder = tmp_path / "evidence"
    shutil.copytree(world["evidence"], folder)
    for stage, edit in edits.items():
        path = folder / stage / "run_report.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        edit(data)
        _write_json(path, data)
    return {f"{stage}_evidence": folder / stage for stage in ("stage7", "stage8", "stage10")}


def _set(**values):
    def edit(data):
        data.update(values)
    return edit


def _all_stages(**values) -> dict:
    return {stage: _set(**values) for stage in ("stage7", "stage8", "stage10")}


def _swap_seed_1_for_seed_2(world, tmp_path, monkeypatch):
    # seed_1/ holds seed 2's checkpoint, and the evidence names its hash: it loads, metadata seed 2.
    folder = tmp_path / "checkpoints"
    shutil.copytree(world["checkpoints"], folder)
    shutil.rmtree(folder / "seed_1")
    shutil.copytree(world["checkpoints"] / "seed_2", folder / "seed_1")
    hashes = {**world["hashes"], "1": world["hashes"]["2"]}
    return {"checkpoints": folder, **_edited_evidence(world, tmp_path, {"stage8": _set(checkpoint_sha256=hashes)})}


def _other_baseline(world, tmp_path, monkeypatch):
    folder = tmp_path / "baseline"
    shutil.copytree(world["baseline_dir"], folder)
    meta = json.loads((folder / "baseline.json").read_text(encoding="utf-8"))
    meta["threshold"] += 1.0
    (folder / "baseline.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"baseline_dir": folder}


def _other_policy(world, tmp_path, monkeypatch):
    path = tmp_path / "risk_policy.json"
    path.write_bytes(world["policy_path"].read_bytes() + b"\n")
    return {"risk_policy": path}


def _dirty(world, tmp_path, monkeypatch):
    def refuse():
        raise PreprocessingError("Official Stage 6 validation requires a clean Git worktree")
    monkeypatch.setattr(run_module, "code_version", refuse)
    return {}


PRE_CLAIM_FAILURES = {  # case: make(world, tmp_path, monkeypatch) -> input overrides
    "dirty_worktree": _dirty,
    "short_commit": lambda w, t, mp: mp.setattr(run_module, "code_version", lambda: {"git_commit": "abc1234"}) or {},
    "python_version": lambda w, t, mp: mp.setattr(run_module, "REQUIRED_PYTHON", "3.12") or {},
    "torch_version": lambda w, t, mp: mp.setattr(run_module, "REQUIRED_TORCH", "0.0.0") or {},
    "checkpoint_sha256": lambda w, t, mp: _edited_evidence(
        w, t, {"stage8": _set(checkpoint_sha256={**w["hashes"], "1": "1" * 64})}),
    "checkpoint_metadata_seed": _swap_seed_1_for_seed_2,
    "baseline_artifact": _other_baseline,
    "baseline_not_d41": lambda w, t, mp: mp.setattr(run_module, "BASELINE_ARTIFACT_SHA256", "4" * 64) or {},
    "risk_policy_sha256": _other_policy,
    "profile_fingerprint": lambda w, t, mp: _edited_evidence(w, t, _all_stages(profile_fingerprint="5" * 64)),
    "profile_config_sha256": lambda w, t, mp: _edited_evidence(w, t, _all_stages(profile_config_sha256="6" * 64)),
    "f2_manifest_sha256": lambda w, t, mp: _edited_evidence(w, t, _all_stages(f2_manifest_sha256="7" * 64)),
    "evidence_disagrees": lambda w, t, mp: _edited_evidence(w, t, {"stage10": _set(profile_fingerprint="5" * 64)}),
    "stage10_other_checkpoint": lambda w, t, mp: _edited_evidence(w, t, {"stage10": _set(checkpoint_sha256="8" * 64)}),
    "cnn_config_sha256": lambda w, t, mp: _edited_evidence(w, t, {"stage8": _set(cnn_config_sha256="9" * 64)}),
    "cache_builds_differ": lambda w, t, mp: {"cache_b": w["cache_bad"]},
    "same_cache_twice": lambda w, t, mp: {"cache_b": w["cache_a"]},
    "cache_vs_stage8": lambda w, t, mp: _edited_evidence(w, t, {"stage8": _set(
        caches={"val-A": {"npz_sha256": "a" * 64}}, val_a_cache_npz_sha256="a" * 64)}),
    "cache_vs_stage7": lambda w, t, mp: _edited_evidence(w, t, {"stage7": lambda d: d["caches"]["val-A"].update(
        npz_sha256="a" * 64)}),
}


# The message each case must fail with, so no case passes by failing somewhere else.
PRE_CLAIM_MESSAGES = {
    "dirty_worktree": "clean Git worktree",
    "short_commit": "full 40-hex SHA",
    "python_version": "environment.python",
    "torch_version": "environment.torch",
    "checkpoint_sha256": "check failed: checkpoint.seed_1.sha256",
    "checkpoint_metadata_seed": "checkpoint.seed_1.metadata_seed",
    "baseline_artifact": "baseline.artifact_sha256",
    "baseline_not_d41": "differs from the D41 value",
    "risk_policy_sha256": "risk_policy.sha256",
    "profile_fingerprint": "check failed: profile_fingerprint",
    "profile_config_sha256": "check failed: profile_config_sha256",
    "f2_manifest_sha256": "check failed: f2_manifest_sha256:",
    "evidence_disagrees": "evidence disagrees on profile_fingerprint",
    "stage10_other_checkpoint": "Stage 10 policy was fitted",
    "cnn_config_sha256": "check failed: cnn_config_sha256",
    "cache_builds_differ": "log_mel.val-A.builds_identical",
    "same_cache_twice": "the same folder",
    "cache_vs_stage8": "log_mel.val-A.npz_sha256",
    "cache_vs_stage7": "mfcc.val-A.npz_sha256",
}


def test_every_pre_claim_case_names_its_message():
    assert set(PRE_CLAIM_MESSAGES) == set(PRE_CLAIM_FAILURES)


@pytest.mark.parametrize("case", sorted(PRE_CLAIM_FAILURES))
def test_pre_claim_failure_reads_no_audio_and_writes_nothing(world, patched, tmp_path, monkeypatch, case):
    overrides = PRE_CLAIM_FAILURES[case](world, tmp_path, monkeypatch)
    out = tmp_path / "out"
    with pytest.raises(FAILS, match=PRE_CLAIM_MESSAGES[case]):
        _run(world, patched, out, **overrides)
    assert patched["reads"] == []
    assert not patched["registry"].exists()
    assert not out.exists()


def test_existing_stage11_claim_is_refused_before_any_read(world, patched, tmp_path):
    # Any (for, test, stage11_test) record blocks the run, whatever its model set.
    RunRegistry(patched["registry"]).claim("for", "test", ["baseline-v1"], COMMIT, evaluation_purpose="stage11_test")
    before = patched["registry"].read_bytes()
    with pytest.raises(ValueError, match="already holds a started stage11_test record"):
        _run(world, patched, tmp_path / "out")
    assert patched["reads"] == []
    assert patched["registry"].read_bytes() == before


def test_existing_out_dir_is_refused_first(world, patched, tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _run(world, patched, tmp_path / "out")
    assert patched["commits"] == 0 and patched["reads"] == [] and not patched["registry"].exists()


@pytest.mark.parametrize("group", [1.5, np.nan, "", "  ", " g "])
def test_bad_group_is_refused_before_the_claim(world, patched, tmp_path, monkeypatch, group):
    def manifest(path):
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df["group"] = df["group"].astype(object)  # another loader could give a float or NaN
        df.loc[df.index[df["split"] == "test"][0], "group"] = group
        return df

    monkeypatch.setattr(run_module, "load_f2_manifest", manifest)
    with pytest.raises(ValueError, match="group_id must be a non-empty trimmed string"):
        _claim(world, patched, tmp_path / "out")
    assert patched["reads"] == [] and not patched["registry"].exists()


@pytest.mark.parametrize("path", ["test/REAL/./00.wav", "test//REAL/00.wav", "test\\REAL\\00.wav",
                                  " test/REAL/00.wav"])
def test_non_canonical_test_path_is_refused_before_the_claim(world, patched, tmp_path, monkeypatch, path):
    def manifest(p):
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        df.loc[df["path"] == "test/REAL/00.wav", "path"] = path
        return df

    monkeypatch.setattr(run_module, "load_f2_manifest", manifest)
    with pytest.raises(ValueError, match="clip_id"):
        _claim(world, patched, tmp_path / "out")
    assert patched["reads"] == [] and not patched["registry"].exists() and not (tmp_path / "out").exists()


def test_missing_subgroup_column_is_refused_before_the_claim(world, patched, tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "load_f2_manifest",
                        lambda p: pd.read_csv(p, dtype=str, keep_default_na=False).drop(columns="subgroup"))
    with pytest.raises(ValueError, match=r"missing columns \['subgroup'\]"):
        _claim(world, patched, tmp_path / "out")
    assert patched["reads"] == [] and not patched["registry"].exists() and not (tmp_path / "out").exists()


def test_claim_is_on_disk_before_the_first_audio_read(world, patched, tmp_path):
    _run(world, patched, tmp_path / "out")
    assert patched["claimed_at_first_read"] is True
    # Two passes over exactly the test rows, sorted by path; no other split is read.
    test_paths = sorted(world["manifest"].loc[world["manifest"]["split"] == "test", "path"])
    assert patched["reads"] == test_paths * 2


def _assert_started_only(state, out):
    # D46 (8): the started record, the claim evidence and the marker stay; no score file was written.
    records = RunRegistry(state["registry"]).records()
    assert [r.status for r in records] == [STARTED]
    assert sorted(p.name for p in out.iterdir()) == sorted([*CLAIM_FILES, *SCORE_START_FILES])


@pytest.mark.parametrize("results_ref", ["", "   ", "/abs/run_report.json", "C:/run_report.json",
                                         "..", "../run_report.json", "a/../run_report.json",
                                         "a\\run_report.json", "./run_report.json", None])
def test_bad_results_ref_is_refused_before_the_claim(world, patched, tmp_path, results_ref):
    with pytest.raises(ValueError, match="results_ref"):
        _run(world, patched, tmp_path / "out", results_ref=results_ref)
    assert patched["commits"] == 0 and patched["reads"] == [] and not patched["registry"].exists()
    assert not (tmp_path / "out").exists()


def test_missing_results_ref_is_refused(world, patched, tmp_path):
    with pytest.raises(TypeError, match="results_ref"):
        run_module.claim(registry_path=patched["registry"], out_dir=tmp_path / "out", **_inputs(world))
    assert patched["commits"] == 0 and patched["reads"] == [] and not patched["registry"].exists()


def test_environment_files_are_written_between_claim_and_first_read(world, patched, tmp_path):
    _run(world, patched, tmp_path / "out")
    assert patched["claimed_at_first_read"] is True
    assert patched["env_files_at_first_read"] is True
    assert patched["score_env_at_first_read"] is True


def test_crash_on_first_audio_read_leaves_claim_and_environment_files(world, patched, tmp_path):
    patched["crash_on_read"] = True
    out = tmp_path / "out"
    with pytest.raises(OSError, match="simulated crash"):
        _run(world, patched, out)
    _assert_started_only(patched, out)
    assert json.loads((out / "environment.json").read_text(encoding="utf-8")) == {"stub": True}
    assert json.loads((out / "claim.json").read_text(encoding="utf-8"))["run_id"] == \
        RunRegistry(patched["registry"]).records()[0].run_id
    assert (out / "pip_freeze.txt").read_bytes() == b"pkg==1.0\n"


def test_feature_pass_2_differs_raises_after_claim(world, patched, tmp_path):
    patched["perturb_after"] = N_TEST  # pass 2 extracts different features
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="feature passes differ"):
        _run(world, patched, out)
    _assert_started_only(patched, out)


def test_cnn_scoring_pass_2_differs_raises(world, patched, tmp_path, monkeypatch):
    calls = []

    def flaky(*args):
        calls.append(1)
        scores = score_clips(*args)
        return scores + 1e-9 if len(calls) == 2 else scores

    monkeypatch.setattr(run_module, "score_clips", flaky)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="cnn-v1-seed0: scores differ"):
        _run(world, patched, out)
    _assert_started_only(patched, out)


def test_baseline_scoring_pass_2_differs_raises(world, patched, tmp_path, monkeypatch):
    calls, real = [], BaselineModel.score

    def flaky(self, features):
        calls.append(1)
        scores = real(self, features)
        return scores + 1e-9 if len(calls) == 2 else scores

    monkeypatch.setattr(BaselineModel, "score", flaky)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="baseline-v1: scores differ"):
        _run(world, patched, out)
    _assert_started_only(patched, out)


def _flip(value: str) -> str:
    return AUTHENTIC if value != AUTHENTIC else SYNTHETIC


def test_injected_bundle_prediction_mismatch_raises_before_writing(world, patched, tmp_path, monkeypatch):
    real = RiskPolicy.decide

    def wrong(self, scores):
        decisions = list(real(self, scores))
        first = decisions[0]
        decisions[0] = type(first)(p=first.p, prediction=_flip(first.prediction),
                                   confidence=first.confidence, risk_level=first.risk_level)
        return tuple(decisions)

    monkeypatch.setattr(RiskPolicy, "decide", wrong)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="bundle seed: 1 predictions differ"):
        _run(world, patched, out)
    _assert_started_only(patched, out)


def test_injected_baseline_prediction_mismatch_raises_before_writing(world, patched, tmp_path, monkeypatch):
    real = BaselineModel.predict

    def wrong(self, features):
        out = real(self, features).copy()
        out[0] = _flip(str(out[0]))
        return out

    monkeypatch.setattr(BaselineModel, "predict", wrong)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="baseline: 1 predictions differ"):
        _run(world, patched, out)
    _assert_started_only(patched, out)


def test_claim_reads_no_audio_and_writes_claim_evidence(world, patched, tmp_path):
    out = tmp_path / "out"
    record = _claim(world, patched, out)
    assert patched["reads"] == []
    assert sorted(p.name for p in out.iterdir()) == sorted(CLAIM_FILES)
    assert (out / "claim.json").read_bytes() == run_module._json_text(record).encode("utf-8")
    [started] = RunRegistry(patched["registry"]).records()
    assert (started.status, started.run_id, started.code_commit) == (STARTED, record["run_id"], COMMIT)
    assert record["registry_sha256"] == _sha(patched["registry"])
    assert (record["results_ref"], record["code_commit"]) == (RESULTS_REF, COMMIT)
    assert record["environment_files"] == {name: _sha(out / name) for name in ENV_FILES}
    assert record["pre_claim_checks"]["registry.for.test.stage11_test.records"] == {
        "expected": 0, "actual": 0, "passed": True}
    assert record["counts"] == {"label": {"FAKE": 4, "REAL": 4},
                                "subgroup": {"FAKE_mp3": 2, "FAKE_other": 2, "REAL": 4}}


def test_claim_refuses_an_existing_claim(world, patched, tmp_path):
    _claim(world, patched, tmp_path / "out")
    before = patched["registry"].read_bytes()
    with pytest.raises(ValueError, match="already holds a started stage11_test record"):
        _claim(world, patched, tmp_path / "out2")
    assert patched["registry"].read_bytes() == before and not (tmp_path / "out2").exists()


def test_score_without_a_claim_folder_is_refused(world, patched, tmp_path):
    with pytest.raises(ValueError, match="score needs the folder written by claim"):
        _score(world, patched, tmp_path / "out", OTHER_RUN_ID)
    assert patched["reads"] == [] and patched["commits"] == 0 and not patched["registry"].exists()


def test_score_without_a_registry_claim_is_refused(world, patched, tmp_path):
    # claim.json exists, but the registry passed to score holds no claim.
    out = tmp_path / "out"
    record = _claim(world, patched, out)
    with pytest.raises(ValueError, match="exactly one started stage11_test record"):
        _score(world, patched, out, record["run_id"], registry=tmp_path / "empty.jsonl")
    assert patched["reads"] == [] and (out / "scoring_started.json").is_file()


def test_score_with_another_run_id_is_refused_without_a_marker(world, patched, tmp_path):
    # A mistyped run_id is refused from claim.json alone; nothing is written, so the correct
    # run_id then scores normally.
    out = tmp_path / "out"
    record = _claim(world, patched, out)
    with pytest.raises(ValueError, match="claim.json is for run_id .* nothing was written"):
        _score(world, patched, out, OTHER_RUN_ID)
    assert sorted(p.name for p in out.iterdir()) == sorted(CLAIM_FILES) and patched["reads"] == []
    report = _score(world, patched, out, record["run_id"])
    assert report["run_id"] == record["run_id"]
    assert [r.status for r in RunRegistry(patched["registry"]).records()] == [STARTED, COMPLETED]


def test_score_at_another_commit_is_refused_without_a_marker(world, patched, tmp_path, monkeypatch):
    out = tmp_path / "out"
    record = _claim(world, patched, out)
    monkeypatch.setattr(run_module, "code_version", lambda: {"git_commit": "d" * 40, "git_dirty": False})
    with pytest.raises(ValueError, match="nothing was written"):
        _score(world, patched, out, record["run_id"])
    assert sorted(p.name for p in out.iterdir()) == sorted(CLAIM_FILES) and patched["reads"] == []


def test_score_called_twice_is_refused_even_after_a_crash_before_any_read(world, patched, tmp_path, monkeypatch):
    out = tmp_path / "out"
    record = _claim(world, patched, out)
    real = run_module.load_evidence

    def crash(*args):
        raise OSError("simulated crash before any audio read")

    monkeypatch.setattr(run_module, "load_evidence", crash)  # inside preflight, after the marker
    with pytest.raises(OSError, match="simulated crash"):
        _score(world, patched, out, record["run_id"])
    monkeypatch.setattr(run_module, "load_evidence", real)
    with pytest.raises(ValueError, match="scoring was already started"):
        _score(world, patched, out, record["run_id"])
    assert patched["reads"] == []
    _assert_started_only(patched, out)


@pytest.mark.parametrize("same", [True, False])
def test_environment_same_as_claim_flag(world, patched, tmp_path, monkeypatch, same):
    calls = []

    def record():
        calls.append(1)
        return ({"session": 1 if same else len(calls)}, "pkg==1.0\n")

    monkeypatch.setattr(run_module, "environment_record", record)
    out = tmp_path / "out"
    report = _run(world, patched, out)
    assert len(calls) == 2  # claim, then score
    assert json.loads((out / "environment_score.json").read_text(encoding="utf-8")) == {"session": 1 if same else 2}
    assert report["environment_same_as_claim"] is same


def test_score_after_completion_is_refused(world, patched, tmp_path):
    out = tmp_path / "out"
    report = _run(world, patched, out)
    reads = len(patched["reads"])
    with pytest.raises(ValueError, match="scoring was already started"):
        _score(world, patched, out, report["run_id"])
    for name in SCORE_START_FILES:  # even without the marker, the registry refuses
        (out / name).unlink()
    with pytest.raises(ValueError, match="exactly one started stage11_test record.*completed"):
        _score(world, patched, out, report["run_id"])
    assert len(patched["reads"]) == reads


def test_score_refuses_a_registry_changed_since_the_claim(world, patched, tmp_path):
    out = tmp_path / "out"
    record = _claim(world, patched, out)
    RunRegistry(patched["registry"]).claim("for", "test", ["x"], COMMIT, evaluation_purpose="stage14_robustness")
    with pytest.raises(ValueError, match="the registry changed since the claim"):
        _score(world, patched, out, record["run_id"])
    assert patched["reads"] == []


@pytest.mark.parametrize("name", ["environment.json", "pip_freeze.txt"])
def test_score_refuses_an_environment_file_changed_since_the_claim(world, patched, tmp_path, name):
    out = tmp_path / "out"
    record = _claim(world, patched, out)
    (out / name).write_bytes(b"other bytes\n")
    with pytest.raises(ValueError, match=f"{name} changed since the claim"):
        _score(world, patched, out, record["run_id"])
    assert patched["reads"] == []


def test_score_refuses_test_rows_changed_since_the_claim(world, patched, tmp_path, monkeypatch):
    out = tmp_path / "out"
    record = _claim(world, patched, out)

    def manifest(path):
        # Same canonical paths, labels and groups; one FAKE_mp3 test row becomes FAKE_other.
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.loc[df["path"] == "test/FAKE_mp3/00.wav", "subgroup"] = "FAKE_other"
        return df

    monkeypatch.setattr(run_module, "load_f2_manifest", manifest)
    with pytest.raises(ValueError, match="test rows changed since the claim"):
        _score(world, patched, out, record["run_id"])
    assert patched["reads"] == []


class _TieBaseline:
    threshold = 0.25

    def __init__(self, at_tie):
        self.at_tie = at_tie

    def predict(self, mfcc):
        return np.array([self.at_tie, AUTHENTIC, SYNTHETIC])


def test_baseline_tie_rule_is_synthetic():
    # D29 forced binary via three_way_decisions(..., 0.0): score == threshold is "synthetic".
    scores = np.array([0.25, 0.1, 0.4])
    assert three_way_decisions(scores, 0.25, 0.0).tolist() == [SYNTHETIC, AUTHENTIC, SYNTHETIC]
    assert run_module.baseline_outputs(_TieBaseline(SYNTHETIC), None, scores) == [SYNTHETIC, AUTHENTIC, SYNTHETIC]
    with pytest.raises(ValueError, match="baseline: 1 predictions differ"):
        run_module.baseline_outputs(_TieBaseline(AUTHENTIC), None, scores)


def test_happy_path_files_report_and_registry(world, patched, tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "environment_record", ORIGINAL_ENVIRONMENT_RECORD)
    out = tmp_path / "out"
    report = _run(world, patched, out)
    assert sorted(p.name for p in out.iterdir()) == ["claim.json", "clip_scores", "environment.json",
                                                    "environment_score.json", "pip_freeze.txt",
                                                    "run_report.json", "scoring_started.json"]
    assert (out / "run_report.json").read_bytes() == run_module._json_text(report).encode("utf-8")
    assert set(report) == {
        "schema_version", "code_commit", "run_id", "dataset", "split", "evaluation_purpose", "model_set",
        "bundle_seed", "claim_sha256", "scoring_marker_sha256", "pre_scoring_checks", "environment_files",
        "environment_score_sha256", "environment_same_as_claim", "scoring", "features", "clip_scores", "counts",
    }
    assert report["environment_score_sha256"] == _sha(out / "environment_score.json")
    assert report["environment_same_as_claim"] is (
        (out / "environment_score.json").read_bytes() == (out / "environment.json").read_bytes())
    assert report["claim_sha256"] == _sha(out / "claim.json")
    assert report["scoring_marker_sha256"] == _sha(out / "scoring_started.json")
    assert (report["dataset"], report["split"], report["evaluation_purpose"]) == ("for", "test", "stage11_test")
    assert (report["code_commit"], report["bundle_seed"]) == (COMMIT, 0)
    assert report["scoring"] == {"device": "cpu", "batch_size": 256, "passes": 2}
    checks = report["pre_scoring_checks"]
    assert all(c["passed"] is True for c in checks.values())
    assert {"environment.python", "environment.torch", "reproduction.log_mel.val-A.builds_identical",
            "reproduction.mfcc.val-A.npz_sha256", "checkpoint.seed_2.metadata_seed", "baseline.artifact_sha256",
            "risk_policy.sha256"} <= set(checks)
    # The recorded checkpoint hash is computed, not copied: the repository's definition by hand.
    for seed in (0, 1, 2):
        folder = world["checkpoints"] / f"seed_{seed}"
        lines = "".join(f"{name}:{_sha(folder / name)}\n" for name in ("checkpoint.json", "weights.safetensors"))
        independent = hashlib.sha256(lines.encode("ascii")).hexdigest()
        assert checks[f"checkpoint.seed_{seed}.sha256"] == {
            "expected": world["hashes"][str(seed)], "actual": independent, "passed": True}
    assert "metrics" not in json.dumps(report) and "eer" not in report
    assert report["environment_files"] == {name: _sha(out / name) for name in ENV_FILES}
    env = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    assert (out / "environment.json").read_bytes() == run_module._json_text(env).encode("utf-8")
    assert set(env["packages"]) == set(run_module.ENV_PACKAGES)
    assert env["pip_freeze_sha256"] == _sha(out / "pip_freeze.txt")
    assert report["counts"] == {"label": {"FAKE": 4, "REAL": 4},
                                "subgroup": {"FAKE_mp3": 2, "FAKE_other": 2, "REAL": 4}}

    # Feature hashes: the canonical digest of the expected fake arrays, sorted paths and labels.
    test = world["manifest"][world["manifest"]["split"] == "test"].sort_values("path")
    arrays = [np.stack(a) for a in zip(*(_fake_features(p) for p in test["path"]))]
    hashes = {"log_mel": run_module.array_sha256("log_mel", arrays[0]),
              "mfcc": run_module.array_sha256("mfcc", arrays[1])}
    assert report["features"] == {"arrays": hashes, "n_clips": N_TEST, "features_sha256": run_module.features_sha256(
        test["path"].tolist(), test["label"].tolist(), hashes)}

    roles = {"baseline-v1": "baseline", "cnn-v1-seed0": "cnn_bundle_seed", "cnn-v1-seed1": "cnn_other_seed",
             "cnn-v1-seed2": "cnn_other_seed"}
    artifacts = {"baseline-v1": artifact_sha256(world["baseline_dir"]),
                 **{f"cnn-v1-seed{s}": world["hashes"][str(s)] for s in (0, 1, 2)}}
    assert report["model_set"] == {m: {"role": roles[m], "artifact_sha256": artifacts[m]} for m in roles}
    for model_id, role in roles.items():
        scores = read_clip_scores(out / "clip_scores", model_id)
        meta = scores.metadata
        assert (meta["model_role"], meta["run_id"], meta["evaluation_purpose"]) == (role, report["run_id"], "stage11_test")
        assert meta["model_artifact_sha256"] == artifacts[model_id]
        assert meta["manifest_sha256"] == _sha(world["manifest_path"])
        assert list(scores.clip_id) == test["path"].tolist() and list(scores.label) == test["label"].tolist()
        assert list(scores.group_id) == test["group"].tolist()
        files = report["clip_scores"]["files"][model_id]
        assert files == {"csv_sha256": _sha(out / "clip_scores" / f"{model_id}.csv"),
                         "sidecar_sha256": _sha(out / "clip_scores" / f"{model_id}.json"), "n_rows": N_TEST}
        if role == "cnn_bundle_seed":
            assert scores.p.tobytes() == world["policy"].calibration.transform(scores.raw_score).tobytes()
            assert list(scores.prediction) == three_way_decisions(scores.p, 0.5, 0.1).tolist()
        elif role == "baseline":
            assert scores.p is None
            assert list(scores.prediction) == np.where(
                scores.raw_score >= world["baseline"].threshold, SYNTHETIC, AUTHENTIC).tolist()
        else:
            assert scores.p is None and scores.prediction is None

    records = RunRegistry(patched["registry"]).records()
    assert [r.status for r in records] == [STARTED, COMPLETED]
    assert {r.run_id for r in records} == {report["run_id"]}
    assert records[1].results_ref == RESULTS_REF
    assert records[0].model_set == tuple(sorted(run_module.MODEL_IDS))

    # A second claim is refused before any audio read.
    reads = len(patched["reads"])
    with pytest.raises(ValueError, match="already holds a started stage11_test record"):
        _claim(world, patched, tmp_path / "out2")
    assert len(patched["reads"]) == reads


def test_bundle_predictions_use_the_band():
    # bundle_outputs keeps the three-way rule: p inside the band is "uncertain".
    policy = RiskPolicy(version="t", calibration=PlattCalibration(a=1.0, b=0.0, n_spoof=1, n_bonafide=1),
                        threshold=0.5, half_width=0.1, target_deferral=0.5, n_spoof_fit=1, n_bonafide_fit=1,
                        deferral_on_fit=0.2)
    p, prediction = run_module.bundle_outputs(policy, np.array([-3.0, 0.0, 3.0]))
    assert prediction == [AUTHENTIC, UNCERTAIN, SYNTHETIC]
    assert p.tobytes() == policy.calibration.transform(np.array([-3.0, 0.0, 3.0])).tobytes()


def test_shifted_decide_prediction_is_caught(monkeypatch):
    # decide() applying the band around a shifted threshold disagrees with the explicit rule.
    policy = RiskPolicy(version="t", calibration=PlattCalibration(a=1.0, b=0.0, n_spoof=1, n_bonafide=1),
                        threshold=0.5, half_width=0.1, target_deferral=0.5, n_spoof_fit=1, n_bonafide_fit=1,
                        deferral_on_fit=0.2)
    real = RiskPolicy.decide

    def shifted(self, scores):
        p = self.calibration.transform(scores)
        moved = three_way_decisions(p, self.threshold + 0.3, self.half_width)
        return tuple(type(d)(p=d.p, prediction=str(m), confidence=d.confidence, risk_level=d.risk_level)
                     for d, m in zip(real(self, scores), moved))

    monkeypatch.setattr(RiskPolicy, "decide", shifted)
    with pytest.raises(ValueError, match="bundle seed: 1 predictions differ"):
        run_module.bundle_outputs(policy, np.array([-3.0, 0.0, 3.0]))  # p = 0.5 is uncertain, not authentic


def test_array_digest_is_header_plus_c_order_bytes():
    a = np.arange(2 * 40, dtype=np.float32).reshape(2, 40)
    header = b'{"dtype":"<f4","name":"mfcc","shape":[2,40]}\n'
    assert run_module.array_sha256("mfcc", a) == hashlib.sha256(header + a.tobytes(order="C")).hexdigest()
    with pytest.raises(ValueError, match="float32"):
        run_module.array_sha256("mfcc", a.astype(np.float64))


def test_committed_evidence_loads_with_the_d46_values():
    ev = run_module.load_evidence(EVIDENCE / "stage7_baseline", EVIDENCE / "stage8_cnn", EVIDENCE / "stage10_risk")
    assert ev.bundle_seed == 0
    assert ev.checkpoint_sha256[0] == "c33eb82086d3bc40f3dedf1356addc6e5c1924ca2e517b0d4527b447e1958932"
    assert ev.baseline_artifact_sha256 == run_module.BASELINE_ARTIFACT_SHA256
    assert ev.cache_npz_sha256[("log_mel", "val-A")].startswith("4502eea2")
    assert ev.cache_npz_sha256[("mfcc", "val-A")].startswith("04227c58")
    assert ev.cache_npz_sha256[("log_mel", "val-B")].startswith("2cff949d")
    assert ev.cache_npz_sha256[("mfcc", "val-B")].startswith("04f91e3e")
    assert ev.batch_size == 256


def test_environment_record_never_raises_without_cpuinfo_nvidia_smi_or_pip(monkeypatch, tmp_path):
    def broken(cmd, **kwargs):
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        raise subprocess.CalledProcessError(1, cmd)

    def version(name):
        if name == "numba":
            raise run_module.importlib.metadata.PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr(run_module, "CPUINFO", tmp_path / "missing")
    monkeypatch.setattr(run_module.subprocess, "run", broken)
    monkeypatch.setattr(run_module.importlib.metadata, "version", version)
    record, freeze = run_module.environment_record()
    assert freeze == ""
    assert record["gpu"] == "none"
    assert record["cpu"] == {"model": "unavailable", "flags": "unavailable", "avx512": "unavailable"}
    assert record["packages"]["numba"] == "unavailable" and record["packages"]["torch"] == "1.0"
    assert {"cpu", "pip_freeze", "version:numba"} <= set(record["failures"])
    assert record["pip_freeze_sha256"] == hashlib.sha256(b"").hexdigest()


def test_environment_record_reads_cpu_flags_and_gpu(monkeypatch, tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Fake Xeon\nflags\t\t: sse avx512f avx2 avx512bw\n\n"
                       "processor\t: 1\nmodel name\t: Other\n", encoding="utf-8")

    def ok(cmd, **kwargs):
        text = "Tesla T4, 550.54\n" if cmd[0] == "nvidia-smi" else "numpy==2.5.3\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")

    monkeypatch.setattr(run_module, "CPUINFO", cpuinfo)
    monkeypatch.setattr(run_module.subprocess, "run", ok)
    record, freeze = run_module.environment_record()
    assert freeze == "numpy==2.5.3\n"
    assert record["cpu"] == {"model": "Fake Xeon", "flags": "avx2 avx512bw avx512f sse",
                             "avx512": ["avx512bw", "avx512f"]}
    assert record["gpu"] == "Tesla T4, 550.54"
    assert record["failures"] == {}


# --- Rehearsal ------------------------------------------------------------------------------------

@pytest.mark.parametrize("splits", [("test",), ("Test",), ("train",), ("val-C",), "val-A", ("val-A",),
                                    ("val-A", "test"), ("val-A", "val-B", "val-B")])
def test_rehearse_refuses_other_splits_before_reading(world, patched, tmp_path, splits):
    with pytest.raises(ValueError, match="rehearsal|not the string"):
        _rehearse(world, tmp_path / "out", splits=splits)
    assert patched["commits"] == 0 and patched["reads"] == []
    assert not (tmp_path / "out").exists()


def test_rehearse_happy_path_never_touches_a_registry(world, patched, tmp_path, monkeypatch):
    def no_registry(*args, **kwargs):
        raise AssertionError("rehearse opened a run registry")

    monkeypatch.setattr(run_module, "RunRegistry", no_registry)
    out = tmp_path / "out"
    report = _rehearse(world, out)
    assert sorted(p.name for p in out.iterdir()) == ["pip_freeze.txt", "rehearsal_report.json"]
    assert (out / "rehearsal_report.json").read_bytes() == run_module._json_text(report).encode("utf-8")
    known = report["known_answers"]
    assert all(c["expected"] == c["actual"] and c["passed"] is True for c in known.values())
    assert all(c["passed"] is True for c in report["pre_claim_checks"].values())
    assert (report["passed"], report["failed_check"], report["error"]) == (True, None, None)
    assert {"val-A.log_mel.in_memory_equals_cache", "val-B.mfcc.in_memory_equals_cache",
            "val-A.mfcc.cache_npz_sha256", "val-A.cnn-v1-seed1.eer", "val-A.baseline-v1.eer",
            "stage10.val_b_scores_sha256", "val-B.cnn-v1-seed0.raw_score_bytes", "val-B.cnn-v1-seed0.p_bytes",
            "val-B.baseline-v1.eer"} <= set(known)
    assert report["per_split"]["val-B"]["scored"] == ["baseline-v1", "cnn-v1-seed0"]  # D45: bundle seed only
    reads = set(patched["reads"])
    assert reads and all(p.startswith(("val-A/", "val-B/")) for p in reads)
    assert "registry_path" not in run_module.rehearse.__code__.co_varnames


def _altered_checkpoints(world, tmp_path, seed: int, val_a_eer: float) -> dict:
    folder = tmp_path / "checkpoints"
    shutil.copytree(world["checkpoints"], folder)
    model, mean, std, meta = load_checkpoint(world["checkpoints"] / f"seed_{seed}")
    shutil.rmtree(folder / f"seed_{seed}")
    save_checkpoint(model, mean, std, {**meta, "val_a_eer": val_a_eer}, folder / f"seed_{seed}")
    hashes = {**world["hashes"], str(seed): checkpoint_sha256(folder / f"seed_{seed}")}
    return {"checkpoints": folder, **_edited_evidence(world, tmp_path, {"stage8": _set(checkpoint_sha256=hashes)})}


def _altered_scores(world, tmp_path, *, update_evidence: bool) -> dict:
    with np.load(world["scores_path"], allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays["score"] = arrays["score"] + 1e-12
    path = tmp_path / "val_b_scores.npz"
    _write_npz(path, arrays)
    edits = {"stage10": _set(val_b_scores_sha256=_sha(path))} if update_evidence else {}
    return {"stage10_scores": path, **_edited_evidence(world, tmp_path, edits)}


REHEARSAL_MISMATCHES = {
    "in_memory_vs_cache": ("val-A.log_mel.in_memory_equals_cache", lambda w, t, s: s.update(perturb_after=0) or {}),
    "cnn_val_a_eer": ("val-A.cnn-v1-seed1.eer", lambda w, t, s: _altered_checkpoints(w, t, 1, IMPOSSIBLE_EER)),
    "baseline_val_a_eer": ("val-A.baseline-v1.eer", lambda w, t, s: _edited_evidence(w, t, {"stage7": _set(
        val_a_eer=IMPOSSIBLE_EER)})),
    "baseline_val_b_eer": ("val-B.baseline-v1.eer", lambda w, t, s: _edited_evidence(w, t, {"stage7": _set(
        val_b_eer=IMPOSSIBLE_EER)})),
    "stage10_scores": ("val-B.cnn-v1-seed0.raw_score_bytes",
                       lambda w, t, s: _altered_scores(w, t, update_evidence=True)),
    "stage10_scores_sha256": ("stage10.val_b_scores_sha256",
                              lambda w, t, s: _altered_scores(w, t, update_evidence=False)),
}


@pytest.mark.parametrize("case", sorted(REHEARSAL_MISMATCHES))
def test_rehearse_detects_mismatch(world, patched, tmp_path, case):
    key, make = REHEARSAL_MISMATCHES[case]
    overrides = make(world, tmp_path, patched)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=f"check failed: {key}:"):
        _rehearse(world, out, **overrides)
    report = json.loads((out / "rehearsal_report.json").read_text(encoding="utf-8"))
    assert (report["passed"], report["failed_check"]) == (False, key)
    assert report["known_answers"][key]["passed"] is False


def test_rehearse_mfcc_mismatch_writes_failed_report_then_raises(world, patched, tmp_path):
    patched["perturb_only"] = "mfcc"  # in-memory MFCC differs from the cache; log-Mel does not
    out = tmp_path / "out"
    key = "val-A.mfcc.in_memory_equals_cache"
    with pytest.raises(run_module.CheckFailed, match=f"check failed: {key}:"):
        _rehearse(world, out)
    report = json.loads((out / "rehearsal_report.json").read_text(encoding="utf-8"))
    assert (report["passed"], report["failed_check"]) == (False, key)
    assert report["error"].startswith("CheckFailed: check failed: val-A.mfcc.in_memory_equals_cache")
    failed = report["known_answers"][key]
    assert failed["passed"] is False and failed["expected"] != failed["actual"]
    assert all(len(failed[k]) == 64 for k in ("expected", "actual"))  # the two feature hashes
    # Every check before it is recorded and passed; nothing after it ran.
    done = {k: v for k, v in report["known_answers"].items() if k != key}
    assert set(done) == {"val-A.log_mel.cache_npz_sha256", "val-A.log_mel.in_memory_equals_cache",
                         "val-A.mfcc.cache_npz_sha256"}
    assert all(c["passed"] is True for c in (*done.values(), *report["pre_claim_checks"].values()))
    assert sorted(p.name for p in out.iterdir()) == ["pip_freeze.txt", "rehearsal_report.json"]


def test_rehearse_preflight_failure_writes_failed_report(world, patched, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(run_module.CheckFailed):
        _rehearse(world, out, cache_b=world["cache_bad"])
    report = json.loads((out / "rehearsal_report.json").read_text(encoding="utf-8"))
    key = "reproduction.log_mel.val-A.builds_identical"
    assert (report["passed"], report["failed_check"], report["code_commit"]) == (False, key, COMMIT)
    assert report["pre_claim_checks"][key]["passed"] is False
    assert report["known_answers"] == {} and patched["reads"] == []


def test_rehearse_non_check_error_writes_failed_report(world, patched, tmp_path, monkeypatch):
    def refuse():
        raise PreprocessingError("Official Stage 6 validation requires a clean Git worktree")

    monkeypatch.setattr(run_module, "code_version", refuse)
    out = tmp_path / "out"
    with pytest.raises(PreprocessingError):
        _rehearse(world, out)
    report = json.loads((out / "rehearsal_report.json").read_text(encoding="utf-8"))
    assert (report["passed"], report["failed_check"], report["code_commit"]) == (False, None, None)
    assert "clean Git worktree" in report["error"]


def test_cli_dispatches_without_a_split_argument(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(run_module, "claim", lambda **kw: seen.setdefault("claim", kw) and {})
    monkeypatch.setattr(run_module, "score", lambda **kw: seen.setdefault("score", kw) and {})
    monkeypatch.setattr(run_module, "rehearse", lambda **kw: seen.setdefault("rehearse", kw) and {})
    common = ["--checkpoints", "c", "--baseline", "b", "--risk-policy", "p", "--profile", "f",
              "--config", "g", "--manifest", "m", "--stage7-evidence", "e7", "--stage8-evidence", "e8",
              "--stage10-evidence", "e10", "--cache-a", "a", "--cache-b", "bb", "--out", "o"]
    run_module.main(["claim", *common, "--registry", "reg.jsonl", "--results-ref", RESULTS_REF])
    run_module.main(["score", "--root", "r", *common, "--registry", "reg.jsonl", "--run-id", OTHER_RUN_ID])
    run_module.main(["rehearse", "--root", "r", *common, "--stage10-scores", "s.npz"])
    assert seen["claim"]["registry_path"] == Path("reg.jsonl") and seen["claim"]["cache_b"] == Path("bb")
    assert seen["claim"]["results_ref"] == RESULTS_REF and "root" not in seen["claim"]  # claim reads no audio
    assert (seen["score"]["run_id"], seen["score"]["root"]) == (OTHER_RUN_ID, Path("r"))
    assert seen["rehearse"]["stage10_scores"] == Path("s.npz") and "splits" not in seen["rehearse"]
    for argv in (["run", "--root", "r", *common, "--registry", "reg.jsonl", "--results-ref", RESULTS_REF],
                 ["claim", *common, "--registry", "reg.jsonl", "--results-ref", RESULTS_REF, "--split", "val-A"],
                 ["claim", *common, "--registry", "reg.jsonl"],  # --results-ref is required
                 ["score", "--root", "r", *common, "--registry", "reg.jsonl"]):  # --run-id is required
        with pytest.raises(SystemExit):
            run_module.main(argv)
    assert not hasattr(run_module, "run")  # no one-step run
