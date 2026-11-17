"""Stage 11 runner for the model v2 set (PR V4b; D48 (7)): evidence binding, registry rule, IDs and
roles, the reproduction gate, two-pass inputs and scoring, claim -> score, and the rehearsal.

Synthetic only, CPU: real WAVs written by tests/_v2_fakes.py (val-A, val-B, test) and read through the
real F3 path, Stage 8-like log-Mel caches (two val-A builds and val-B), a 10-clip S, seeded untrained
v2 checkpoints (crnn and aasist, seeds 0-2) whose selection_m is computed by train_v2's own functions,
the v2 search evidence written by the real v2_search report writers, and the risk-v2 evidence
written by v2_risk_run.run on these fakes. No W&B, no real data.
Convention: label 1 = spoof (FAKE), 0 = bonafide (REAL).
"""
from __future__ import annotations

import functools
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._torch import require_torch

torch = require_torch()

import src.evaluation.stage11_run as v1  # noqa: E402
import src.evaluation.stage11_v2_run as m  # noqa: E402
import src.tuning.risk_run as stage10  # noqa: E402
import src.tuning.v2_risk_run as vr  # noqa: E402
import src.tuning.v2_search as vs  # noqa: E402
import src.tuning.v2_val_b_source as vbs  # noqa: E402
from src.evaluation.clip_scores import read_clip_scores  # noqa: E402
from src.evaluation.metrics import roc_auc  # noqa: E402
from src.evaluation.run_registry import COMPLETED, STARTED, RunRegistry  # noqa: E402
from src.evaluation.seeds import select_bundle_seed  # noqa: E402
from src.features.logmel_cache import build_logmel_cache, load_logmel_cache  # noqa: E402
from src.models.baseline import _write_npz  # noqa: E402
from src.models.v2_checkpoint import (  # noqa: E402
    INPUT_SHAPES,
    INPUTS,
    build_model,
    load_v2_checkpoint,
    save_v2_checkpoint,
    trainable_parameters,
    v2_checkpoint_sha256,
)
from src.preprocessing.features import extract_features  # noqa: E402
from src.preprocessing.validation import read_verified_audio  # noqa: E402
from src.preprocessing.waveform import preprocess_waveform  # noqa: E402
from src.risk.decisions import three_way_decisions  # noqa: E402
from src.risk.policy import RiskPolicy  # noqa: E402
from src.training.normalization import fit_bin_stats  # noqa: E402
from src.training.selection_set import load_selection_set  # noqa: E402
from src.training.train_v2 import evaluate_selection, selection_m, selection_tensors  # noqa: E402
from src.training.v2_config import load_v2_config  # noqa: E402
from tests._v2_fakes import (  # noqa: E402
    COMMIT,
    PROFILE_PATH,
    S_COUNTS,
    profile as load_test_profile,
    selection_arrays,
    sha,
    v2_config_copy,
    write_dataset,
    write_s,
)

SIZES = {"val-A": 10, "val-B": 12, "test": 8}
N_TEST = SIZES["test"]
ARCHS = ("crnn", "aasist")  # one log-Mel and the waveform candidate
RESULTS_REF = "stage11_v2/run_report.json"
OTHER_RUN_ID = "f" * 32
CLAIM_FILES = ("claim.json", "environment.json", "pip_freeze.txt")
SCORE_START_FILES = ("environment_score.json", "scoring_started.json")
V1_IDS = v1.MODEL_IDS
load_s = functools.partial(load_selection_set, expected_n_rows=10, expected_counts=S_COUNTS, recompute_rows=1)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _manifest_reader(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _test_rows(manifest):
    return manifest[manifest["split"] == "test"].sort_values("path", kind="stable")


def _independent_log_mel_sha256(root, rows, profile) -> str:
    """The v1 test log-Mel digest by hand: stacked log-Mels through v1's array_sha256."""
    stacked = np.stack([extract_features(*read_verified_audio(root, row), profile)[1] for _, row in rows.iterrows()])
    return v1.array_sha256("log_mel", stacked)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("stage11_v2")
    prof = load_test_profile()
    root, manifest, manifest_path = write_dataset(base, sizes=SIZES)
    for split in ("val-B", "test"):  # both D14 FAKE subgroups (Stage 10 diagnostics, test counts)
        fake = manifest.index[(manifest.split == split) & (manifest.label == "FAKE")]
        manifest.loc[fake[::2], "subgroup"] = "FAKE_mp3"
    manifest.to_csv(manifest_path, index=False)
    f2 = sha(manifest_path)
    caches = {}
    for name, splits in (("cache_a", ("val-A", "val-B")), ("cache_b", ("val-A",))):
        for split in splits:
            build_logmel_cache(root, manifest, prof, split, base / name / "log_mel" / split, f2_manifest_sha256=f2,
                               profile_config_sha256=sha(PROFILE_PATH), code_commit=COMMIT)
        caches[name] = base / name
    val_a_npz = json.loads((base / "cache_a" / "log_mel" / "val-A" / "val-A.json").read_text(encoding="utf-8"))[
        "npz_sha256"]
    s_dir = base / "S"
    s_sha = write_s(s_dir, selection_arrays(root, manifest, prof), prof, f2, "a" * 64, recompute_rows=1)
    v2 = v2_config_copy(base / "v2.yaml")  # batch size 8 each
    config = load_v2_config(v2)
    s_forms = {form: load_s(s_dir, prof, form=form) for form in ("logmel", "waveform")}
    mean, std = fit_bin_stats(s_forms["logmel"].features("clean"))

    val_b_rows = manifest[manifest.split == "val-B"].sort_values("path", kind="stable")

    def waveform_source():
        waves = np.stack([preprocess_waveform(read_verified_audio(root, row)[0], 16000, prof).waveform
                          for _, row in val_b_rows.iterrows()]).astype(np.float32)
        labels = val_b_rows["label"].map({"FAKE": 1, "REAL": 0}).to_numpy(dtype=np.int8)
        return waves, labels, np.array(val_b_rows["path"].tolist())

    val_b_logmel = load_logmel_cache(base / "cache_a" / "log_mel" / "val-B", "val-B", prof,
                                     expected_f2_manifest_sha256=f2)
    val_b = {"logmel": val_b_logmel[0], "waveform": waveform_source()[0]}

    def stats(kind):
        return (mean, std) if kind == "logmel" else (None, None)

    def oriented_model(arch, seed):
        # Seeded and untrained; the one-logit output is negated if val-B ranks backwards, so Platt
        # fits a > 0 for whichever seed becomes the bundle seed. Test fixture only.
        torch.manual_seed(100 * ARCHS.index(arch) + seed)
        model = build_model(arch).eval()
        kind = INPUTS[arch]
        if roc_auc(val_b_logmel[1], vr.score_clips(model, vr.model_inputs(kind, val_b[kind], *stats(kind)), 8)) < 0.5:
            out = [x for x in model.modules() if isinstance(x, torch.nn.Linear)][-1]
            with torch.no_grad():
                out.weight.neg_()
                out.bias.neg_()
        return model

    def save(arch, seed, model, folder, **changes):
        kind = INPUTS[arch]
        eers, _ = evaluate_selection(model, selection_tensors(s_forms[kind], kind, *stats(kind)),
                                     s_forms[kind].labels, 8, torch.device("cpu"))
        meta = {
            "schema_version": 1, "architecture": arch, "version": "cnn-v2-test", "input": kind,
            "input_shape": list(INPUT_SHAPES[kind]), "seed": seed, "best_epoch": 1,
            "selection_m": selection_m(eers), "code_commit": COMMIT, "environment": {"python": "3.13.16"},
            "profile_fingerprint": prof.fingerprint(), "f2_manifest_sha256": f2,
            "augmentation_config_sha256": "a" * 64, "selection_set_sha256": s_sha,
            "train_stats_npz_sha256": "b" * 64 if kind == "logmel" else None,
            "determinism": {"seed": seed}, "trainable_parameters": trainable_parameters(model),
        }
        meta.update(changes)
        save_v2_checkpoint(model, meta, folder, mean=stats(kind)[0], std=stats(kind)[1])
        return meta["selection_m"], v2_checkpoint_sha256(folder)

    def record(arch, seed, m_value, digest):
        return {"candidate": arch, "seed": seed, "trainable_parameters": trainable_parameters(build_model(arch)),
                "best_epoch": 1, "best_selection_m": m_value, "epochs_run": 1,
                "history": [{"epoch": 1, "selection_m": m_value}], "checkpoint_sha256": digest,
                "run_name": vs.run_name("cnn-v2-test", arch, seed), "peak_gpu_memory_bytes": None,
                "provenance": {"train_stats_npz_sha256": "b" * 64 if INPUTS[arch] == "logmel" else None}}

    stage8 = base / "stage8"
    _write_json(stage8 / "run_report.json", {"val_a_cache_npz_sha256": val_a_npz})
    v1_report = base / "v1_stage11" / "run_report.json"
    _write_json(v1_report, {"features": {"arrays": {
        "log_mel": _independent_log_mel_sha256(root, _test_rows(manifest), prof)}}})

    per_arch = {}
    for arch in ARCHS:
        runs = base / f"runs_{arch}"
        ms, hashes = {}, {}
        for seed in m.SEEDS:
            ms[seed], hashes[seed] = save(arch, seed, oriented_model(arch, seed),
                                          runs / vs.run_dir_name(arch, seed) / vs.CHECKPOINT_DIR)
        bundle_seed = select_bundle_seed(ms)
        search = {c: record(arch, 0, ms[0], hashes[0]) if c == arch else record(c, 0, 1.0, "e" * 64)
                  for c in config.candidate_order}
        selection = {"search": search, "winner": arch, "tied": [arch], "bundle_seed": bundle_seed,
                     "seeds": {s: search[arch] if s == 0 else record(arch, s, ms[s], hashes[s]) for s in m.SEEDS}}
        base_fields = {"code_commit": COMMIT, "v2_config_sha256": sha(v2), "augmentation_config_sha256": "a" * 64,
                       "profile_fingerprint": prof.fingerprint(), "profile_config_sha256": sha(PROFILE_PATH),
                       "f2_manifest_sha256": f2, "raw_train_cache_npz_sha256": "d" * 64,
                       "selection_set_sha256": s_sha, "val_a_cache_npz_sha256": val_a_npz}
        extra = {"version": "cnn-v2-test", "environment": {"python": "3.13.16"}, "wandb_mode": "disabled",
                 "batch_sizes": config.require_batch_sizes(), "num_workers": 0}
        search_dir = base / f"v2_search_{arch}"
        search_dir.mkdir()
        vs._write_reports(search_dir, vs._reports(config, selection, base_fields, extra))

        risk_dir = base / f"v2_risk_{arch}"
        with pytest.MonkeyPatch.context() as mp:  # v2_risk_run.run on the fakes, as in its own tests
            mp.setattr(vr, "code_version", lambda: {"git_commit": COMMIT, "git_dirty": False})
            mp.setattr(vr, "load_f2_manifest", _manifest_reader)
            mp.setattr(stage10, "F2_MANIFEST_SHA256", f2)
            mp.setattr(vr, "load_selection_set", load_s)
            mp.setattr(vr, "V2_SEARCH_EVIDENCE", search_dir)
            source = {"cache_dir": base / "cache_a" / "log_mel"} if INPUTS[arch] == "logmel" \
                else {"val_b_waveforms": waveform_source}
            vr.run(runs / vs.run_dir_name(arch, bundle_seed) / vs.CHECKPOINT_DIR, s_dir, PROFILE_PATH, v2,
                   manifest_path, risk_dir, version="risk-v2-test", **source)
        scores = risk_dir / "val_b_scores.npz"
        evidence = base / f"evidence_{arch}"  # the scores npz is passed by path (not committed, as Stage 10)
        evidence.mkdir()
        shutil.move(str(scores), str(evidence / "val_b_scores.npz"))
        per_arch[arch] = {"runs": runs, "ms": ms, "hashes": hashes, "bundle_seed": bundle_seed,
                          "search": search_dir, "risk": risk_dir, "risk_scores": evidence / "val_b_scores.npz"}
    return {"base": base, "root": root, "manifest": manifest, "manifest_path": manifest_path, "f2": f2,
            "profile": prof, "caches": caches, "s_dir": s_dir, "s_sha": s_sha, "v2": v2, "stage8": stage8,
            "v1_report": v1_report, "arch": per_arch, "save": save, "val_a_npz": val_a_npz}


def _v1_pair(path: Path) -> None:
    registry = RunRegistry(path)
    run_id = registry.claim("for", "test", V1_IDS, "8" * 40, evaluation_purpose="stage11_test")
    registry.complete(run_id, results_ref="docs/results/stage11_for_test.md")


@pytest.fixture()
def env(world, monkeypatch, tmp_path):
    """Patch the module for one architecture; returns a function arch -> state."""
    calls = []

    def setup(arch, *, registry="v1_pair"):
        a = world["arch"][arch]
        calls.append(1)
        state = {"arch": arch, "reads": [], "registry": tmp_path / f"registry_{len(calls)}.jsonl", "commits": 0, "out": None,
                 "claimed_at_first_test_read": None, "features_json_at_first_score": None, "loaded": [],
                 "perturb_pass": None, "extracts": 0}
        if registry == "v1_pair":
            _v1_pair(state["registry"])

        def code_version():
            state["commits"] += 1
            return {"git_commit": COMMIT, "git_dirty": False}

        def read(root, row):
            if row["path"].startswith("test/") and state["claimed_at_first_test_read"] is None:
                lines = state["registry"].read_bytes().splitlines() if state["registry"].exists() else []
                state["claimed_at_first_test_read"] = bool(lines) and json.loads(lines[-1])["status"] == STARTED \
                    and json.loads(lines[-1])["model_set"] == sorted(m.model_ids_v2(arch))
            state["reads"].append(row["path"])
            return read_verified_audio(root, row)

        real_score = vr.score_clips

        def score_spy(model, x, batch_size):
            if state["features_json_at_first_score"] is None and state["out"] is not None:
                state["features_json_at_first_score"] = (state["out"] / "features.json").is_file()
            return real_score(model, x, batch_size)

        def s_spy(folder, profile, *, form, **kw):
            state["loaded"].append(f"S:{form}")
            return load_s(folder, profile, form=form, **kw)

        monkeypatch.setattr(m, "code_version", code_version)
        monkeypatch.setattr(m, "load_f2_manifest", _manifest_reader)
        monkeypatch.setattr(m, "F2_MANIFEST_SHA256", world["f2"])
        monkeypatch.setattr(v1, "F2_MANIFEST_SHA256", world["f2"])  # stage11_run's cache loader
        monkeypatch.setattr(v1, "REQUIRED_PYTHON", ".".join(v1.platform.python_version_tuple()[:2]))
        monkeypatch.setattr(v1, "REQUIRED_TORCH", torch.__version__.split("+")[0])
        monkeypatch.setattr(m, "environment_record", lambda: ({"stub": True}, "pkg==1.0\n"))
        monkeypatch.setattr(m, "read_verified_audio", read)
        monkeypatch.setattr(m, "score_clips", score_spy)
        monkeypatch.setattr(m, "load_selection_set", s_spy)
        monkeypatch.setattr(m, "V2_SEARCH_EVIDENCE", a["search"])
        monkeypatch.setattr(m, "V2_RISK_EVIDENCE", a["risk"])
        monkeypatch.setattr(m, "STAGE8_EVIDENCE", world["stage8"])
        monkeypatch.setattr(m, "V1_RUN_REPORT", world["v1_report"])
        return state
    return setup


def _inputs(world, arch, **overrides) -> dict:
    inputs = {"checkpoints": world["arch"][arch]["runs"], "profile_path": PROFILE_PATH, "config_path": world["v2"],
              "manifest_path": world["manifest_path"], "selection_set": world["s_dir"]}
    if INPUTS[arch] == "logmel":
        inputs.update(cache_a=world["caches"]["cache_a"], cache_b=world["caches"]["cache_b"])
    inputs.update(overrides)
    return inputs


def _claim(world, state, out, **overrides):
    state["out"] = out
    return m.claim(root=world["root"], registry_path=state["registry"], out_dir=out, results_ref=RESULTS_REF,
                   **_inputs(world, state["arch"], **overrides))


def _score(world, state, out, run_id, **overrides):
    state["out"] = out
    return m.score(root=world["root"], registry_path=state["registry"], out_dir=out, run_id=run_id,
                   **_inputs(world, state["arch"], **overrides))


def _rehearse(world, state, out, splits=m.REHEARSAL_SPLITS, risk_scores=None, **overrides):
    state["out"] = out
    return m.rehearse(root=world["root"], risk_scores=risk_scores or world["arch"][state["arch"]]["risk_scores"],
                      out_dir=out, splits=splits, **_inputs(world, state["arch"], **overrides))


def _test_reads(state) -> list:
    return [p for p in state["reads"] if p.startswith("test/")]


# --- Evidence edits (copies; the risk-v2 binding to the search files is re-signed on request) ------

def _edit_evidence(world, state, monkeypatch, tmp_path, *, search=None, risk=None, rebind=True, stage8=None):
    """Copies of the evidence with edits {file name: function(data)}; rebind re-signs risk-v2's
    v2_search_evidence so only the edited value disagrees."""
    a = world["arch"][state["arch"]]
    s_dir, r_dir = tmp_path / "ev_search", tmp_path / "ev_risk"
    shutil.copytree(a["search"], s_dir)
    shutil.copytree(a["risk"], r_dir)
    for name, edit in (search or {}).items():
        data = json.loads((s_dir / name).read_text(encoding="utf-8"))
        edit(data)
        _write_json(s_dir / name, data)
    report = json.loads((r_dir / "run_report.json").read_text(encoding="utf-8"))
    if rebind:
        report["v2_search_evidence"] = {n: sha(s_dir / n) for n in ("run_report.json", "selection_report.json")}
    for edit in (risk or {}).values():
        edit(report)
    _write_json(r_dir / "run_report.json", report)
    monkeypatch.setattr(m, "V2_SEARCH_EVIDENCE", s_dir)
    monkeypatch.setattr(m, "V2_RISK_EVIDENCE", r_dir)
    if stage8 is not None:
        folder = tmp_path / "ev_stage8"
        _write_json(folder / "run_report.json", stage8)
        monkeypatch.setattr(m, "STAGE8_EVIDENCE", folder)
    return s_dir, r_dir


def _set(**values):
    def edit(data):
        data.update(values)
    return edit


def _other_seed(world, arch) -> int:
    """A non-bundle seed whose M can move without changing the median (bundle seed)."""
    return next(s for s in m.SEEDS if s != world["arch"][arch]["bundle_seed"])


def _nudge(world, arch, seed) -> float:
    """seed's M moved one ulp away from the median (so select_bundle_seed keeps its answer)."""
    ms = world["arch"][arch]["ms"]
    median = sorted(ms.values())[1]
    return math.nextafter(ms[seed], 2.0 if ms[seed] > median else -1.0)


def _seeds_edit(seed: int, **values):
    def edit(data):
        data["seeds"][str(seed)].update(values)
    return edit


def _bundle_not_median(world):
    """The bundle seed's own M moved to an extreme: select_bundle_seed now names another seed."""
    a = world["arch"]["crnn"]
    others = max(a["ms"][s] for s in m.SEEDS if s != a["bundle_seed"])
    return {"search": {"selection_report.json": _seeds_edit(a["bundle_seed"],
                                                            best_selection_m=1.0 if others < 1.0 else 0.0)}}


def _drop_seed_hash(world):
    name = vs.run_dir_name("crnn", _other_seed(world, "crnn"))
    return {"search": {"run_report.json": lambda d: d["checkpoint_sha256"].pop(name)}}


EVIDENCE_CASES = {  # case: (world -> edits for _edit_evidence, message)
    "selection_winner": (lambda w: {"search": {"selection_report.json": _set(selected="cnn-aug")}},
                         "disagrees on the winner"),
    "selection_bundle_seed": (lambda w: {"search": {"selection_report.json": lambda d: d.update(
        bundle_seed=(d["bundle_seed"] + 1) % 3)}}, "disagrees on the bundle seed"),
    "missing_seed_hash": (_drop_seed_hash, "checkpoint_sha256 entries for the winner"),
    "selection_seeds": (lambda w: {"search": {"selection_report.json": lambda d: d["seeds"].pop(
        str(_other_seed(w, "crnn")))}}, "selection_report seeds: expected"),
    "best_selection_m_not_a_number": (lambda w: {"search": {"selection_report.json": _seeds_edit(
        _other_seed(w, "crnn"), best_selection_m=True)}}, r"best_selection_m: expected a finite number"),
    "bundle_seed_not_median": (_bundle_not_median, "select_bundle_seed over"),
    "stage8_val_a": (lambda w: {"stage8": {"val_a_cache_npz_sha256": "a" * 64}}, "Stage 8 val_a_cache_npz_sha256"),
    "risk_architecture": (lambda w: {"risk": {"x": _set(architecture="cnn-aug")}},
                          "v2 risk run_report architecture"),
    "risk_bundle_seed": (lambda w: {"risk": {"x": lambda d: d.update(bundle_seed=(d["bundle_seed"] + 1) % 3)}},
                         "v2 risk run_report bundle_seed"),
    "risk_checkpoint": (lambda w: {"risk": {"x": _set(checkpoint_sha256="0" * 64)}},
                        "v2 risk run_report checkpoint_sha256"),
    "risk_device": (lambda w: {"risk": {"x": lambda d: d["scoring"].update(device="cuda")}}, "scoring.device"),
    "risk_input_form": (lambda w: {"risk": {"x": lambda d: d["val_b_input"].update(form="waveform")}},
                        "val_b_input.form"),
    "risk_profile": (lambda w: {"risk": {"x": _set(profile_fingerprint="5" * 64)}},
                     "v2 risk run_report profile_fingerprint"),
    "risk_f2": (lambda w: {"risk": {"x": _set(f2_manifest_sha256="7" * 64)}}, "v2 risk run_report f2_manifest_sha256"),
    "risk_search_binding": (lambda w: {"search": {"run_report.json": _set(note="edited")}, "rebind": False},
                            "v2_search_evidence"),
}


@pytest.mark.parametrize("case", sorted(EVIDENCE_CASES))
def test_evidence_disagreement_is_refused_before_the_claim(world, env, tmp_path, monkeypatch, case):
    state = env("crnn")
    make, message = EVIDENCE_CASES[case]
    edits = make(world)
    _edit_evidence(world, state, monkeypatch, tmp_path, **edits)
    before = state["registry"].read_bytes()
    with pytest.raises(ValueError, match=message) as info:
        _claim(world, state, tmp_path / "out")
    if "evidence check failed" in str(info.value):  # field, expected and actual are named
        assert "expected" in str(info.value) and "got" in str(info.value)
    assert state["reads"] == [] and state["registry"].read_bytes() == before
    assert not (tmp_path / "out").exists()


def test_risk_policy_bytes_must_match_the_risk_report(world, env, tmp_path, monkeypatch):
    state = env("crnn")
    _, r_dir = _edit_evidence(world, state, monkeypatch, tmp_path)
    (r_dir / "risk_policy.json").write_bytes((r_dir / "risk_policy.json").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="check failed: risk_policy.sha256"):
        _claim(world, state, tmp_path / "out")
    assert state["reads"] == [] and not (tmp_path / "out").exists()


def test_checkpoint_hash_and_metadata_are_checked(world, env, tmp_path, monkeypatch):
    state = env("crnn")
    seed = _other_seed(world, "crnn")
    _edit_evidence(world, state, monkeypatch, tmp_path, search={"run_report.json": lambda d: d[
        "checkpoint_sha256"].update({f"crnn_seed{seed}": "1" * 64})})
    with pytest.raises(ValueError, match=f"check failed: checkpoint.crnn_seed{seed}.sha256"):
        _claim(world, state, tmp_path / "out")
    # The evidence names a seed's M that its checkpoint metadata does not record.
    _edit_evidence(world, state, monkeypatch, tmp_path / "b",
                   search={"selection_report.json": _seeds_edit(seed, best_selection_m=_nudge(world, "crnn", seed))})
    with pytest.raises(ValueError, match=f"check failed: checkpoint.crnn_seed{seed}.selection_m"):
        _claim(world, state, tmp_path / "out")
    assert state["reads"] == [] and not (tmp_path / "out").exists()


def test_committed_evidence_paths_are_fixed():
    repo = Path(__file__).resolve().parents[2]
    assert m.V2_SEARCH_EVIDENCE == vr.V2_SEARCH_EVIDENCE == repo / "docs" / "audit_evidence" / "v2_search"
    assert m.V2_RISK_EVIDENCE == repo / "docs" / "audit_evidence" / "v2_risk"
    assert m.STAGE8_EVIDENCE == repo / "docs" / "audit_evidence" / "stage8_cnn"
    assert m.V1_RUN_REPORT == repo / "docs" / "audit_evidence" / "stage11_run" / "run_report.json"


# --- IDs and roles --------------------------------------------------------------------------------

def test_model_ids_and_roles():
    assert m.model_ids_v2("crnn") == ("crnn-v2-seed0", "crnn-v2-seed1", "crnn-v2-seed2")
    assert m.model_ids_v2("cnn-aug") == ("cnn-aug-v2-seed0", "cnn-aug-v2-seed1", "cnn-aug-v2-seed2")
    assert m.model_ids_v2("aasist")[1] == "aasist-v2-seed1"
    with pytest.raises(ValueError, match="winner must be one of"):
        m.model_ids_v2("cnn-v1")
    assert m.model_roles("aasist", 1) == {"aasist-v2-seed0": "v2_other_seed", "aasist-v2-seed1": "v2_bundle_seed",
                                          "aasist-v2-seed2": "v2_other_seed"}
    with pytest.raises(ValueError, match="bundle_seed"):
        m.model_roles("crnn", 3)
    assert len(m.ALL_V2_MODEL_IDS) == 9 and not m.ALL_V2_MODEL_IDS & set(V1_IDS)


# --- Registry rule --------------------------------------------------------------------------------

def _registry_case(path: Path, case: str) -> None:
    registry = RunRegistry(path)
    if case == "empty":
        return
    if case == "v1_started_only":
        registry.claim("for", "test", V1_IDS, "8" * 40, evaluation_purpose="stage11_test")
        return
    _v1_pair(path)
    if case == "extra_stage11_record":
        RunRegistry(path).claim("for", "test", ["other-model"], "8" * 40, evaluation_purpose="stage11_test")
    elif case == "v2_record":  # any v2 ID, any purpose
        RunRegistry(path).claim("for", "test", ["crnn-v2-seed0"], "8" * 40, evaluation_purpose="stage14_robustness")
    elif case == "v1_pair_and_stage14_v1":
        RunRegistry(path).claim("for", "test", ["cnn-v1-seed0"], "8" * 40, evaluation_purpose="stage14_robustness")


@pytest.mark.parametrize("case, message", [
    ("empty", "exactly the v1 pair"),
    ("v1_started_only", "exactly the v1 pair"),
    ("extra_stage11_record", "exactly the v1 pair"),
    ("v2_record", "already holds records with v2 model IDs"),
])
def test_registry_rule_refuses_before_the_claim(world, env, tmp_path, case, message):
    state = env("crnn", registry="none")
    _registry_case(state["registry"], case)
    before = state["registry"].read_bytes() if state["registry"].exists() else b""
    with pytest.raises(ValueError, match=message):
        _claim(world, state, tmp_path / "out")
    after = state["registry"].read_bytes() if state["registry"].exists() else b""
    assert after == before and state["reads"] == [] and not (tmp_path / "out").exists()


@pytest.mark.parametrize("case", ["v1_pair", "v1_pair_and_stage14_v1"])
def test_registry_rule_accepts_the_v1_pair(world, env, tmp_path, case):
    state = env("crnn", registry="none")
    _registry_case(state["registry"], case)
    record = _claim(world, state, tmp_path / "out")
    found = record["pre_claim_checks"]["registry.v1_pair_only"]
    assert found["passed"] is True and [r[1] for r in found["actual"]] == [STARTED, COMPLETED]
    last = RunRegistry(state["registry"]).records()[-1]
    assert (last.status, last.run_id, last.model_set) == (STARTED, record["run_id"], tuple(m.model_ids_v2("crnn")))


# --- Gates and input-kind checks ------------------------------------------------------------------

def test_input_kind_mismatch_is_refused(world, env, tmp_path):
    state = env("crnn")
    with pytest.raises(ValueError, match="cache_a and cache_b .* are required"):
        _claim(world, state, tmp_path / "out", cache_b=None)
    state = env("aasist")
    with pytest.raises(ValueError, match="must not be given"):
        _claim(world, state, tmp_path / "out", cache_a=world["caches"]["cache_a"], cache_b=world["caches"]["cache_b"])
    assert state["reads"] == [] and not (tmp_path / "out").exists()


def test_log_mel_gate_refusals(world, env, tmp_path, monkeypatch):
    state = env("crnn")
    with pytest.raises(ValueError, match="the same folder"):
        _claim(world, state, tmp_path / "out", cache_b=world["caches"]["cache_a"])
    real = m._load_cache

    def other_build_b(cache_dir, *args):
        features, labels, paths, meta = real(cache_dir, *args)
        if Path(cache_dir) == world["caches"]["cache_b"]:
            meta = {**meta, "npz_sha256": "9" * 64}
        return features, labels, paths, meta

    monkeypatch.setattr(m, "_load_cache", other_build_b)
    with pytest.raises(ValueError, match="check failed: reproduction.log_mel.val-A.builds_identical"):
        _claim(world, state, tmp_path / "out")
    monkeypatch.setattr(m, "_load_cache", real)
    _edit_evidence(world, state, monkeypatch, tmp_path, search={"run_report.json": _set(
        val_a_cache_npz_sha256="a" * 64)}, stage8={"val_a_cache_npz_sha256": "a" * 64})
    with pytest.raises(ValueError, match="check failed: reproduction.log_mel.val-A.npz_sha256"):
        _claim(world, state, tmp_path / "out")
    assert state["reads"] == [] and not (tmp_path / "out").exists()


def test_aasist_gate_compares_val_a_waveforms_with_s(world, env, tmp_path, monkeypatch):
    state = env("aasist")

    def scaled(waveform, sample_rate, profile):
        prepared, log_mel, mfcc = extract_features(waveform, sample_rate, profile)
        prepared.waveform[:] = (prepared.waveform * np.float32(1.001)).astype(np.float32)
        return prepared, log_mel, mfcc

    monkeypatch.setattr(m, "extract_features", scaled)  # the same in both passes: only S can tell
    with pytest.raises(ValueError, match="check failed: reproduction.waveform.val-A.equals_selection_clean"):
        _claim(world, state, tmp_path / "out")
    assert _test_reads(state) == [] and set(state["reads"]) == set(
        world["manifest"].loc[world["manifest"].split == "val-A", "path"])
    assert state["loaded"] == ["S:waveform"] and not (tmp_path / "out").exists()
    assert RunRegistry(state["registry"]).records()[-1].model_set == tuple(sorted(V1_IDS))


# --- Claim -> score -------------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHS)
def test_claim_then_score_happy_path(world, env, tmp_path, arch):
    state = env(arch)
    out = tmp_path / "out"
    record = _claim(world, state, out)
    assert _test_reads(state) == []  # claim reads no test audio (aasist's gate reads val-A only)
    val_a = sorted(world["manifest"].loc[world["manifest"].split == "val-A", "path"])
    assert state["reads"] == (val_a * 2 if arch == "aasist" else [])
    assert sorted(p.name for p in out.iterdir()) == sorted(CLAIM_FILES)
    assert record["model_ids"] == sorted(m.model_ids_v2(arch)) and record["architecture"] == arch
    assert record["registry_sha256"] == sha(state["registry"])
    assert record["counts"] == {"label": {"FAKE": 4, "REAL": 4},
                                "subgroup": {"FAKE_mp3": 2, "FAKE_other": 2, "REAL": 4}}
    gate = {"crnn": {"reproduction.log_mel.val-A.builds_identical", "reproduction.log_mel.val-A.npz_sha256"},
            "aasist": {"reproduction.waveform.val-A.selection_set_sha256",
                       "reproduction.waveform.val-A.equals_selection_clean"}}[arch]
    assert gate <= set(record["pre_claim_checks"]) and not any("mfcc" in k for k in record["pre_claim_checks"])
    assert all(c["passed"] is True for c in record["pre_claim_checks"].values())

    state["reads"].clear()
    report = _score(world, state, out, record["run_id"])
    assert state["claimed_at_first_test_read"] is True and state["features_json_at_first_score"] is True
    test_paths = _test_rows(world["manifest"])["path"].tolist()
    assert _test_reads(state) == test_paths * 2  # two passes, test rows only, path order
    assert sorted(p.name for p in out.iterdir()) == sorted([*CLAIM_FILES, *SCORE_START_FILES, "clip_scores",
                                                            "features.json", "run_report.json"])
    assert (out / "run_report.json").read_bytes() == m._json_text(report).encode("utf-8")
    a = world["arch"][arch]
    roles = m.model_roles(arch, a["bundle_seed"])
    assert report["model_set"] == {mid: {"role": roles[mid], "artifact_sha256": a["hashes"][k]}
                                   for k, mid in zip(m.SEEDS, m.model_ids_v2(arch))}
    assert (report["architecture"], report["input"], report["bundle_seed"]) == (arch, INPUTS[arch], a["bundle_seed"])
    assert report["scoring"] == {"device": "cpu", "batch_size": 8, "passes": 2}
    assert report["features_json_sha256"] == sha(out / "features.json")
    assert "metrics" not in json.dumps(report)

    # Feature hashes, computed by hand: the v1 log-Mel digest (equal to the v1 run report's here),
    # and for aasist also the risk-v2 content hash of the F3 waveforms.
    rows = _test_rows(world["manifest"])
    log_mel = _independent_log_mel_sha256(world["root"], rows, world["profile"])
    arrays = {"log_mel": log_mel}
    if arch == "aasist":
        waves = np.stack([extract_features(*read_verified_audio(world["root"], r), world["profile"])[0].waveform
                          for _, r in rows.iterrows()])
        arrays["waveform"] = vr._sha256_arrays(waves)
    assert report["features"] == {"arrays": arrays, "input": INPUTS[arch], "n_clips": N_TEST,
                                  "features_sha256": v1.features_sha256(rows["path"].tolist(),
                                                                        rows["label"].tolist(), arrays)}
    assert report["v1_log_mel_comparison"] == {"v1_run_report_sha256": sha(world["v1_report"]),
                                               "v1_log_mel_sha256": log_mel, "v2_log_mel_sha256": log_mel,
                                               "equal": True}

    policy = RiskPolicy.from_json((a["risk"] / "risk_policy.json").read_text(encoding="utf-8"))
    for mid, role in roles.items():
        scores = read_clip_scores(out / "clip_scores", mid)
        assert (scores.metadata["model_role"], scores.metadata["run_id"]) == (role, record["run_id"])
        assert scores.metadata["model_artifact_sha256"] == report["model_set"][mid]["artifact_sha256"]
        assert list(scores.clip_id) == test_paths and list(scores.group_id) == rows["group"].tolist()
        if role == "v2_bundle_seed":
            assert scores.p.tobytes() == policy.calibration.transform(scores.raw_score).tobytes()
            assert list(scores.prediction) == three_way_decisions(scores.p, policy.threshold,
                                                                  policy.half_width).tolist()
        else:
            assert scores.p is None and scores.prediction is None
        assert report["clip_scores"]["files"][mid]["csv_sha256"] == sha(out / "clip_scores" / f"{mid}.csv")

    records = RunRegistry(state["registry"]).records()
    assert [(r.status, r.model_set) for r in records] == [
        (STARTED, tuple(sorted(V1_IDS))), (COMPLETED, tuple(sorted(V1_IDS))),
        (STARTED, tuple(m.model_ids_v2(arch))), (COMPLETED, tuple(m.model_ids_v2(arch)))]
    assert records[-1].results_ref == RESULTS_REF and records[-1].run_id == record["run_id"]

    reads = len(state["reads"])
    with pytest.raises(ValueError, match="scoring was already started"):  # a second score
        _score(world, state, out, record["run_id"])
    for name in SCORE_START_FILES:  # even without the marker, the registry refuses
        (out / name).unlink()
    with pytest.raises(ValueError, match="exactly one started v2"):
        _score(world, state, out, record["run_id"])
    with pytest.raises(ValueError, match="already holds records with v2 model IDs"):  # a second claim
        _claim(world, state, tmp_path / "out2")
    assert not [p for p in state["reads"][reads:] if p.startswith("test/")]


def test_v1_log_mel_mismatch_is_recorded_not_asserted(world, env, tmp_path, monkeypatch):
    state = env("crnn")
    other = tmp_path / "v1_report.json"
    _write_json(other, {"features": {"arrays": {"log_mel": "0" * 64}}})
    monkeypatch.setattr(m, "V1_RUN_REPORT", other)
    out = tmp_path / "out"
    report = _score(world, state, out, _claim(world, state, out)["run_id"])
    comparison = report["v1_log_mel_comparison"]
    assert comparison["equal"] is False and comparison["v1_log_mel_sha256"] == "0" * 64
    assert RunRegistry(state["registry"]).records()[-1].status == COMPLETED


def test_score_without_a_claim_folder_is_refused(world, env, tmp_path):
    state = env("crnn")
    with pytest.raises(ValueError, match="score needs the folder written by claim"):
        _score(world, state, tmp_path / "out", OTHER_RUN_ID)
    assert state["reads"] == [] and state["commits"] == 0


def test_score_with_another_run_id_is_refused_without_a_marker(world, env, tmp_path):
    state = env("crnn")
    out = tmp_path / "out"
    record = _claim(world, state, out)
    with pytest.raises(ValueError, match="claim.json is for run_id .* nothing was written"):
        _score(world, state, out, OTHER_RUN_ID)
    assert sorted(p.name for p in out.iterdir()) == sorted(CLAIM_FILES) and state["reads"] == []
    assert _score(world, state, out, record["run_id"])["run_id"] == record["run_id"]


def test_score_with_a_registry_without_the_v2_claim_is_refused(world, env, tmp_path):
    state = env("crnn")
    out = tmp_path / "out"
    record = _claim(world, state, out)
    other = tmp_path / "other.jsonl"
    _v1_pair(other)
    with pytest.raises(ValueError, match="exactly one started v2"):
        m.score(root=world["root"], registry_path=other, out_dir=out, run_id=record["run_id"],
                **_inputs(world, "crnn"))
    assert _test_reads(state) == [] and (out / "scoring_started.json").is_file()


def _assert_started_only(state, out):
    records = RunRegistry(state["registry"]).records()
    assert [r.status for r in records] == [STARTED, COMPLETED, STARTED]  # D46 (8): the v2 claim stays
    assert {p.name for p in out.iterdir()} <= {*CLAIM_FILES, *SCORE_START_FILES, "features.json"}


def test_scoring_pass_2_differs_raises_after_the_claim(world, env, tmp_path, monkeypatch):
    state = env("crnn")
    calls, real = [], vr.score_clips

    def flaky(model, x, batch_size):
        calls.append(1)
        scores = real(model, x, batch_size)
        return scores + 1e-9 if len(calls) == 2 else scores

    out = tmp_path / "out"
    record = _claim(world, state, out)
    monkeypatch.setattr(m, "score_clips", flaky)
    first = m.model_ids_v2("crnn")[0]
    with pytest.raises(ValueError, match=f"{first}: scores differ between two CPU scoring passes"):
        _score(world, state, out, record["run_id"])
    _assert_started_only(state, out)


def test_input_pass_2_differs_raises_after_the_claim(world, env, tmp_path, monkeypatch):
    state = env("crnn")
    out = tmp_path / "out"
    record = _claim(world, state, out)
    count = {"n": 0}

    def drifting(waveform, sample_rate, profile):
        count["n"] += 1
        prepared, log_mel, mfcc = extract_features(waveform, sample_rate, profile)
        return prepared, (log_mel + np.float32(1e-3) if count["n"] > N_TEST else log_mel), mfcc

    monkeypatch.setattr(m, "extract_features", drifting)
    with pytest.raises(ValueError, match="input passes differ"):
        _score(world, state, out, record["run_id"])
    assert not (out / "features.json").exists()
    _assert_started_only(state, out)


def test_score_models_compares_two_passes_byte_for_byte(world, env, tmp_path, monkeypatch):
    # score_models itself, outside a run: equal passes return float64 scores, a one-ulp drift raises.
    state = env("crnn")
    ctx = m.preflight({}, root=world["root"], **_inputs(world, "crnn"))
    x = load_logmel_cache(world["caches"]["cache_a"] / "log_mel" / "val-B", "val-B", world["profile"],
                          expected_f2_manifest_sha256=world["f2"])[0]
    scores = m.score_models(ctx, x)
    assert list(scores) == list(m.model_ids_v2("crnn")) and all(s.dtype == np.float64 for s in scores.values())
    calls = []

    def ulp(model, tensor, batch_size):
        calls.append(1)
        out = vr.score_clips(model, tensor, batch_size)
        return np.nextafter(out, np.inf) if len(calls) % 2 == 0 else out

    monkeypatch.setattr(m, "score_clips", ulp)
    with pytest.raises(ValueError, match="crnn-v2-seed0: scores differ"):
        m.score_models(ctx, x)
    assert state["reads"] == []


# --- Rehearsal ------------------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHS)
def test_rehearse_passes_with_known_answers_and_no_registry(world, env, tmp_path, monkeypatch, arch):
    state = env(arch)

    def no_registry(*args, **kwargs):
        raise AssertionError("rehearse opened a run registry")

    monkeypatch.setattr(m, "RunRegistry", no_registry)
    out = tmp_path / "out"
    report = _rehearse(world, state, out)
    assert sorted(p.name for p in out.iterdir()) == ["pip_freeze.txt", "rehearsal_report.json"]
    assert (out / "rehearsal_report.json").read_bytes() == m._json_text(report).encode("utf-8")
    assert (report["passed"], report["failed_check"], report["error"]) == (True, None, None)
    known = report["known_answers"]
    a = world["arch"][arch]
    bundle_id = m.model_ids_v2(arch)[a["bundle_seed"]]
    for seed, mid in zip(m.SEEDS, m.model_ids_v2(arch)):
        assert known[f"val-A.{mid}.selection_m"] == {"expected": a["ms"][seed], "actual": a["ms"][seed],
                                                     "passed": True}
        assert math.fsum(known[f"val-A.{mid}.selection_eers"]["actual"].values()) / 5 == a["ms"][seed]
    val_b = {"crnn": {"val-B.log_mel.cache_npz_sha256", "val-B.log_mel.in_memory_equals_cache"},
             "aasist": {"val-B.waveform.content_sha256"}}[arch]
    assert val_b | {"val-A.selection_set_sha256", "risk_v2.val_b_scores_sha256", f"val-B.{bundle_id}.raw_score_bytes",
                    f"val-B.{bundle_id}.p_bytes", "val-B.risk_v2.paths", "val-B.risk_v2.labels"} <= set(known)
    assert all(c["passed"] is True for c in (*known.values(), *report["pre_claim_checks"].values()))
    assert report["per_split"]["val-B"]["scored"] == [bundle_id]  # D45: the bundle seed only
    assert state["reads"] and all(p.startswith(("val-A/", "val-B/")) for p in state["reads"])
    assert state["loaded"] == ([] if arch == "crnn" else ["S:waveform"]) + [f"S:{INPUTS[arch]}"]


@pytest.mark.parametrize("splits", [("test",), ("Test",), ("train",), "val-A", ("val-A",), ("val-A", "test")])
def test_rehearse_refuses_other_splits_before_reading(world, env, tmp_path, splits):
    state = env("crnn")
    with pytest.raises(ValueError, match="rehearsal|not the string"):
        _rehearse(world, state, tmp_path / "out", splits=splits)
    assert state["commits"] == 0 and state["reads"] == [] and not (tmp_path / "out").exists()


def test_rehearse_m_mismatch_fails_with_a_report(world, env, tmp_path, monkeypatch):
    # A seed's checkpoint and the evidence both record an M one ulp from what its weights give on
    # S: the pre-claim checks agree, only the recomputation sees it (no tolerance, as V4a).
    state = env("crnn")
    a = world["arch"]["crnn"]
    seed = _other_seed(world, "crnn")
    nudged = _nudge(world, "crnn", seed)
    runs = tmp_path / "runs"
    shutil.copytree(a["runs"], runs)
    name = vs.run_dir_name("crnn", seed)
    model, _, _, meta = load_v2_checkpoint(runs / name / vs.CHECKPOINT_DIR)
    shutil.rmtree(runs / name / vs.CHECKPOINT_DIR)
    _, digest = world["save"]("crnn", seed, model, runs / name / vs.CHECKPOINT_DIR, selection_m=nudged)
    _edit_evidence(world, state, monkeypatch, tmp_path, search={
        "run_report.json": lambda d: d["checkpoint_sha256"].update({name: digest}),
        "selection_report.json": _seeds_edit(seed, best_selection_m=nudged)})
    out = tmp_path / "out"
    key = f"val-A.crnn-v2-seed{seed}.selection_m"
    with pytest.raises(v1.CheckFailed, match=f"check failed: {key}:"):
        _rehearse(world, state, out, checkpoints=runs)
    report = json.loads((out / "rehearsal_report.json").read_text(encoding="utf-8"))
    assert (report["passed"], report["failed_check"]) == (False, key)
    assert report["known_answers"][key] == {"expected": nudged, "actual": a["ms"][seed], "passed": False}
    assert all(c["passed"] for c in report["pre_claim_checks"].values())
    assert not any(p.startswith("val-B/") for p in state["reads"])  # stopped before val-B


def _altered_scores(world, tmp_path, arch):
    with np.load(world["arch"][arch]["risk_scores"], allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays["score"] = arrays["score"] + 1e-12
    path = tmp_path / "val_b_scores.npz"
    _write_npz(path, arrays)
    return path


@pytest.mark.parametrize("arch", ARCHS)
def test_rehearse_val_b_bytes_mismatch_fails_with_a_report(world, env, tmp_path, monkeypatch, arch):
    state = env(arch)
    path = _altered_scores(world, tmp_path, arch)
    out = tmp_path / "out"
    with pytest.raises(v1.CheckFailed, match="check failed: risk_v2.val_b_scores_sha256"):
        _rehearse(world, state, out, risk_scores=path)  # not the file risk-v2 recorded
    _edit_evidence(world, state, monkeypatch, tmp_path, risk={"x": _set(val_b_scores_sha256=sha(path))})
    key = f"val-B.{m.model_ids_v2(arch)[world['arch'][arch]['bundle_seed']]}.raw_score_bytes"
    with pytest.raises(v1.CheckFailed, match=f"check failed: {key}:"):
        _rehearse(world, state, tmp_path / "out2", risk_scores=path)
    report = json.loads((tmp_path / "out2" / "rehearsal_report.json").read_text(encoding="utf-8"))
    assert (report["passed"], report["failed_check"]) == (False, key)
    assert not any(p.startswith("test/") for p in state["reads"])


def test_rehearse_preflight_failure_writes_failed_report(world, env, tmp_path, monkeypatch):
    state = env("crnn")
    monkeypatch.setattr(v1, "REQUIRED_TORCH", "0.0.0")
    out = tmp_path / "out"
    with pytest.raises(v1.CheckFailed, match="environment.torch"):
        _rehearse(world, state, out)
    report = json.loads((out / "rehearsal_report.json").read_text(encoding="utf-8"))
    assert (report["passed"], report["failed_check"], report["code_commit"]) == (False, "environment.torch", COMMIT)
    assert report["known_answers"] == {} and state["reads"] == []


def test_no_test_path_in_rehearse():
    assert "registry_path" not in m.rehearse.__code__.co_varnames
    assert m.REHEARSAL_SPLITS == ("val-A", "val-B") and (m.GATE_SPLIT, m.FIT_SPLIT) == ("val-A", "val-B")


# --- CLI ------------------------------------------------------------------------------------------

def test_cli_dispatches_without_a_split_argument(monkeypatch):
    seen = {}
    for name in ("claim", "score", "rehearse"):
        monkeypatch.setattr(m, name, lambda _n=name, **kw: seen.setdefault(_n, kw) and {})
    common = ["--root", "r", "--checkpoints", "c", "--selection-set", "s", "--profile", "f", "--config", "g",
              "--manifest", "mf", "--out", "o"]
    commands = {"claim": ["claim", *common, "--registry", "reg.jsonl", "--results-ref", RESULTS_REF],
                "score": ["score", *common, "--cache-a", "a", "--cache-b", "b", "--registry", "reg.jsonl",
                          "--run-id", OTHER_RUN_ID],
                "rehearse": ["rehearse", *common, "--risk-scores", "v.npz"]}
    for argv in commands.values():
        m.main(argv)
    assert seen["claim"]["cache_a"] is None and seen["claim"]["results_ref"] == RESULTS_REF
    assert (seen["score"]["cache_a"], seen["score"]["run_id"]) == (Path("a"), OTHER_RUN_ID)
    assert seen["rehearse"]["risk_scores"] == Path("v.npz") and "splits" not in seen["rehearse"]
    assert all(seen[n]["selection_set"] == Path("s") and seen[n]["root"] == Path("r") for n in commands)
    for argv in (["run", *common], ["claim", *common, "--registry", "reg.jsonl"],
                 ["score", *common, "--registry", "reg.jsonl"],
                 ["claim", *common, "--registry", "reg.jsonl", "--results-ref", RESULTS_REF, "--split", "val-A"]):
        with pytest.raises(SystemExit):
            m.main(argv)
    assert not hasattr(m, "run")


# --- PR V4c: risk-v2 for aasist fitted with the F3 waveform val-B source --------------------------

def test_rehearse_passes_for_aasist_fitted_with_the_val_b_source(world, env, tmp_path, monkeypatch):
    # End to end: v2_risk_run.run for the aasist fake with val_b_waveform_source (not the test's
    # hand-built source), then rehearse against that risk-v2 run: the val-B content-hash known answer
    # and the score / p bytes against its own val_b_scores.npz.
    state = env("aasist")  # patches stage11_v2_run's reads (spy), evidence and environment
    a = world["arch"]["aasist"]
    risk_dir = tmp_path / "v2_risk"
    source = vbs.val_b_waveform_source(world["root"], world["manifest_path"], PROFILE_PATH)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(vr, "code_version", lambda: {"git_commit": COMMIT, "git_dirty": False})
        mp.setattr(vr, "load_f2_manifest", _manifest_reader)
        mp.setattr(vbs, "load_f2_manifest", _manifest_reader)
        mp.setattr(stage10, "F2_MANIFEST_SHA256", world["f2"])
        mp.setattr(vr, "load_selection_set", load_s)
        mp.setattr(vr, "V2_SEARCH_EVIDENCE", a["search"])
        fitted = vr.run(a["runs"] / vs.run_dir_name("aasist", a["bundle_seed"]) / vs.CHECKPOINT_DIR, world["s_dir"],
                        PROFILE_PATH, world["v2"], world["manifest_path"], risk_dir, version="risk-v2-test",
                        val_b_waveforms=source)
    val_b = sorted(world["manifest"].loc[world["manifest"].split == "val-B", "path"])
    assert state["reads"] == val_b * 2  # the source: val-B only, two passes
    scores = tmp_path / "scores" / "val_b_scores.npz"
    scores.parent.mkdir()
    shutil.move(str(risk_dir / "val_b_scores.npz"), str(scores))
    # Same input and scores as the fixture's hand-built source (the fixture's risk-v2 run).
    fixture = json.loads((a["risk"] / "run_report.json").read_text(encoding="utf-8"))
    assert fitted["val_b_input"] == fixture["val_b_input"]
    assert scores.read_bytes() == a["risk_scores"].read_bytes()

    monkeypatch.setattr(m, "V2_RISK_EVIDENCE", risk_dir)
    state["reads"].clear()
    report = _rehearse(world, state, tmp_path / "out", risk_scores=scores)
    assert (report["passed"], report["failed_check"], report["error"]) == (True, None, None)
    known = report["known_answers"]
    content = fitted["val_b_input"]["content_sha256"]
    assert known["val-B.waveform.content_sha256"] == {"expected": content, "actual": content, "passed": True}
    assert known["risk_v2.val_b_scores_sha256"]["actual"] == fitted["val_b_scores_sha256"] == sha(scores)
    bundle_id = m.model_ids_v2("aasist")[a["bundle_seed"]]
    for key in (f"val-B.{bundle_id}.raw_score_bytes", f"val-B.{bundle_id}.p_bytes", "val-B.risk_v2.paths",
                "val-B.risk_v2.labels"):
        assert known[key]["passed"] is True, key
    assert _test_reads(state) == []
