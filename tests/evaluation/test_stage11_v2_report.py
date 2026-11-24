"""Tests for src/evaluation/stage11_v2_report.py (v2 Stage 11 metrics from the stored files; D57 on v2).

Synthetic only: a v2 run folder written with the real writers (write_clip_scores, RunRegistry with
the v1 pair first, the v2 runner's JSON format), a small F2-format manifest, a RiskPolicy file and
v2 search / risk-v2 evidence stubs. No dataset, audio, checkpoint or model is read, except the last
tests, which need torch: the constants against the v2 runner's, the search agreement rule against
v2_risk_run.bundle_checkpoint, and the report on the v2 runner's own synthetic claim + score output.
Each test has its hand calculation or its reasoning as a comment.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.evaluation.stage11_v2_report as report_module
from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.clip_scores import read_clip_scores, write_clip_scores
from src.evaluation.metrics import binary_metrics, eer, roc_auc, three_way_metrics
from src.evaluation.run_registry import RunRegistry
from src.evaluation.seeds import seed_summary
from src.tuning.baseline_run import _json_text
from tests.evaluation.test_stage11_report import HAND, POLICY  # the v1 hand table and policy, reused
from tests.governance.test_for_test_access import ALLOWED, file_hits

if importlib.util.find_spec("torch") is not None:  # the v2 runner's tests import torch at import time
    from tests.evaluation import test_stage11_v2_run as runner_tests  # noqa: E402
    from tests.evaluation.test_stage11_v2_run import env, world  # noqa: E402, F401  (fixtures)

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_RUNNER = REPO_ROOT / "src" / "evaluation" / "stage11_v2_run.py"
COMMIT = "c" * 40
REPORT_COMMIT = "e" * 40
SPLIT = "test"  # tests/ is not scanned by the guard; the module itself never names the split
RESULTS_REF = "stage11_v2/run_report.json"
ARCH = "crnn"
BUNDLE_SEED = 1  # not 0, so the bundle model is not the first file
IDS = tuple(f"{ARCH}-v2-seed{k}" for k in (0, 1, 2))
BUNDLE_ID = IDS[BUNDLE_SEED]
SEED_HASHES = {0: "a0" * 32, 1: "b1" * 32, 2: "c2" * 32}
V1_IDS = ("baseline-v1", "cnn-v1-seed0", "cnn-v1-seed1", "cnn-v1-seed2")

# Key lists of the v2 runner's claim.json and run_report.json, checked against its source
# (test_fixture_key_lists_equal_the_v2_runners), so the fixture writes the runner's format.
CLAIM_KEYS = ["schema_version", "run_id", "code_commit", "registry_sha256", "results_ref", "dataset", "split",
              "evaluation_purpose", "architecture", "bundle_seed", "model_ids", "pre_claim_checks",
              "environment_files", "counts"]
RUN_REPORT_KEYS = ["schema_version", "code_commit", "run_id", "dataset", "split", "evaluation_purpose",
                   "architecture", "input", "model_set", "bundle_seed", "claim_sha256", "scoring_marker_sha256",
                   "features_json_sha256", "pre_scoring_checks", "environment_files", "environment_score_sha256",
                   "environment_same_as_claim", "scoring", "features", "v1_log_mel_comparison", "clip_scores",
                   "counts"]


def _rows_from(table):
    """Clip rows: clip_id, label, subgroup, group, and raw scores per v2 model ID. The bundle seed takes
    the hand table's CNN column; the other two seeds separate perfectly (REAL < 0 < FAKE, distinct)."""
    rows = []
    for k, (clip_id, subgroup, group, cnn, _baseline) in enumerate(table):
        label = "REAL" if subgroup == "REAL" else "FAKE"
        sign = -1.0 if label == "REAL" else 1.0
        rows.append({"clip_id": clip_id, "label": label, "subgroup": subgroup, "group": group,
                     IDS[0]: sign * (k + 1), BUNDLE_ID: cnn, IDS[2]: sign * (k + 1) * 2 + 0.25})
    return rows


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _counts(rows):
    # The runner's _counts(rows) format (stage11_run._counts, imported by the v2 runner).
    frame = pd.DataFrame(rows)
    return {column: {k: int(v) for k, v in sorted(frame[column].value_counts().items())}
            for column in ("label", "subgroup")}


class Run:
    """A synthetic v2 Stage 11 output folder plus every input build_report needs."""

    def __init__(self, root: Path, monkeypatch, rows, *, complete=True, sidecar=None, columns=None, p_override=None,
                 prediction_override=None, manifest_edit=None, manifest_columns=None,
                 results_ref_completed=RESULTS_REF, registry_model_ids=IDS, registry_commit=COMMIT):
        self.root, self.rows = root, rows
        sidecar, columns = sidecar or {}, columns or {}
        # F2 manifest: the run's rows plus two rows of another split.
        manifest = [{"path": r["clip_id"], "label": r["label"], "subgroup": r["subgroup"], "group": r["group"],
                     "split": SPLIT, "hash": "a" * 64} for r in rows]
        manifest += [{"path": f"REAL/train{i}.wav", "label": "REAL", "subgroup": "REAL", "group": f"t{i}",
                      "split": "train", "hash": "a" * 64} for i in range(2)]
        if manifest_edit:
            manifest_edit(manifest)
        frame = pd.DataFrame(manifest, columns=list(report_module.F2_COLUMNS))
        if manifest_columns:
            frame = frame.rename(columns=manifest_columns)
        self.manifest = root / "manifest.csv"
        _write(self.manifest, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        manifest_sha = _sha(self.manifest.read_bytes())
        monkeypatch.setattr(report_module, "APPROVED_F2_MANIFEST_SHA256", manifest_sha)

        # v2 search evidence: the winner's three seeds and seed 0 of the other candidates.
        self.search = root / "v2_search"
        hashes = {f"{ARCH}_seed{k}": SEED_HASHES[k] for k in (0, 1, 2)}
        hashes.update({f"{c}_seed0": "e" * 64 for c in ("cnn-aug", "aasist")})
        _write(self.search / "run_report.json", _json_text({
            "selected": ARCH, "bundle_seed": BUNDLE_SEED, "checkpoint_sha256": hashes}).encode("utf-8"))
        _write(self.search / "selection_report.json", _json_text({
            "selected": ARCH, "bundle_seed": BUNDLE_SEED}).encode("utf-8"))
        # risk-v2 evidence: the policy and the run report that binds it to the search and the bundle model.
        self.risk = root / "v2_risk"
        _write(self.risk / "risk_policy.json", POLICY.to_json().encode("utf-8"))
        self.risk_report = {
            "version": POLICY.version, "architecture": ARCH, "bundle_seed": BUNDLE_SEED, "checkpoint_sha256": SEED_HASHES[BUNDLE_SEED],
            "f2_manifest_sha256": manifest_sha, "risk_policy_sha256": _sha((self.risk / "risk_policy.json").read_bytes()),
            "v2_search_evidence": self.search_hashes()}
        _write(self.risk / "run_report.json", _json_text(self.risk_report).encode("utf-8"))

        # Registry as the real one will be: the v1 pair first, then this v2 claim.
        self.registry = root / "registry" / "run_registry.jsonl"
        self.registry.parent.mkdir()
        registry = RunRegistry(self.registry)
        v1_run = registry.claim("for", SPLIT, list(V1_IDS), "8" * 40, evaluation_purpose="stage11_test")
        registry.complete(v1_run, results_ref="docs/results/stage11_for_test.md")
        self.run_id = registry.claim("for", SPLIT, list(registry_model_ids), registry_commit,
                                     evaluation_purpose="stage11_test")
        if complete:
            registry.complete(self.run_id, results_ref=results_ref_completed)

        self.run_dir = root / "out"
        counts = _counts(rows)
        claim = {"schema_version": 1, "run_id": self.run_id, "code_commit": COMMIT, "registry_sha256": "0" * 64,
                 "results_ref": RESULTS_REF, "dataset": "for", "split": SPLIT, "evaluation_purpose": "stage11_test",
                 "architecture": ARCH, "bundle_seed": BUNDLE_SEED, "model_ids": sorted(IDS), "pre_claim_checks": {},
                 "environment_files": {}, "counts": counts}
        _write(self.run_dir / "claim.json", _json_text(claim).encode("utf-8"))
        model_set = {mid: {"role": "v2_bundle_seed" if k == BUNDLE_SEED else "v2_other_seed",
                           "artifact_sha256": SEED_HASHES[k]} for k, mid in enumerate(IDS)}
        self.model_set = model_set
        clip_dir = self.run_dir / "clip_scores"
        clip_dir.mkdir()
        files = {}
        for model_id, model in model_set.items():
            raw = np.array([r[model_id] for r in rows])
            outputs = {}
            if model["role"] == "v2_bundle_seed":
                decisions = POLICY.decide(raw)
                outputs = {"p": np.array([d.p for d in decisions]), "prediction": [d.prediction for d in decisions]}
                if p_override:
                    outputs["p"] = p_override(outputs["p"])
                if prediction_override:
                    outputs["prediction"] = prediction_override(outputs["prediction"])
            meta = {"evaluation_purpose": "stage11_test", "run_id": self.run_id, "model_id": model_id,
                    "model_role": model["role"], "model_artifact_sha256": model["artifact_sha256"],
                    "manifest_sha256": manifest_sha, **sidecar.get(model_id, {})}
            cols = {"clip_id": [r["clip_id"] for r in rows], "label": [r["label"] for r in rows],
                    "group_id": [r["group"] for r in rows], **columns.get(model_id, {})}
            csv_path, json_path = write_clip_scores(clip_dir, **meta, **cols, raw_score=raw, **outputs)
            files[model_id] = {"csv_sha256": _sha(csv_path.read_bytes()), "sidecar_sha256": _sha(json_path.read_bytes()),
                               "n_rows": len(rows)}
        report = {"schema_version": 1, "code_commit": COMMIT, "run_id": self.run_id, "dataset": "for", "split": SPLIT,
                  "evaluation_purpose": "stage11_test", "architecture": ARCH, "input": "logmel", "model_set": model_set,
                  "bundle_seed": BUNDLE_SEED, "claim_sha256": _sha((self.run_dir / "claim.json").read_bytes()),
                  "scoring_marker_sha256": "0" * 64, "features_json_sha256": "0" * 64, "pre_scoring_checks": {},
                  "environment_files": {}, "environment_score_sha256": "0" * 64, "environment_same_as_claim": True,
                  "scoring": {"device": "cpu", "batch_size": 4, "passes": 2}, "features": {},
                  "v1_log_mel_comparison": {}, "clip_scores": {"folder": "clip_scores", "files": files},
                  "counts": counts}
        self.write_run_report(report)

    def search_hashes(self) -> dict:
        return {n: _sha((self.search / n).read_bytes()) for n in ("run_report.json", "selection_report.json")}

    def write_run_report(self, report: dict) -> None:
        (self.run_dir / "run_report.json").write_bytes(_json_text(report).encode("utf-8"))

    def edit_json(self, path: Path, edit) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        edit(data)
        path.write_bytes(_json_text(data).encode("utf-8"))

    def edit_claim(self, edit) -> None:
        """Edit claim.json and re-sign it in run_report.json, so only the edited field disagrees."""
        self.edit_json(self.run_dir / "claim.json", edit)
        sha = _sha((self.run_dir / "claim.json").read_bytes())
        self.edit_json(self.run_dir / "run_report.json", _set(["claim_sha256"], sha))

    def edit_search(self, edit) -> None:
        """Edit both search reports and re-sign them in risk-v2's v2_search_evidence."""
        for name in ("run_report.json", "selection_report.json"):
            self.edit_json(self.search / name, edit)
        self.edit_json(self.risk / "run_report.json", _set(["v2_search_evidence"], self.search_hashes()))

    def kwargs(self) -> dict:
        return {"run_dir": self.run_dir, "registry_path": self.registry, "manifest_path": self.manifest,
                "search_evidence_dir": self.search, "risk_evidence_dir": self.risk}

    def argv(self, out: Path) -> list[str]:
        return ["--run-dir", str(self.run_dir), "--registry", str(self.registry), "--manifest", str(self.manifest),
                "--search-evidence", str(self.search), "--risk-evidence", str(self.risk), "--out", str(out)]


def _set(path_keys, value):
    def edit(data):
        target = data
        for key in path_keys[:-1]:
            target = target[key]
        target[path_keys[-1]] = value
    return edit


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setattr(report_module, "code_version", lambda: {"git_commit": REPORT_COMMIT, "git_dirty": False})
    counter = iter(range(1000))

    def factory(rows=None, **options) -> Run:
        root = tmp_path / f"run{next(counter)}"
        root.mkdir()
        return Run(root, monkeypatch, rows if rows is not None else _rows_from(HAND), **options)
    return factory


@pytest.fixture(scope="module")
def hand_report(tmp_path_factory):
    # One build of the hand case shared by the read-only tests (the bootstrap is the slow part).
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(report_module, "code_version", lambda: {"git_commit": REPORT_COMMIT, "git_dirty": False})
    run = Run(tmp_path_factory.mktemp("hand_v2"), monkeypatch, _rows_from(HAND))
    report = report_module.build_report(**run.kwargs())
    yield run, json.loads(_json_text(report))  # as written: NaN never appears, None for undefined
    monkeypatch.undo()


# --- Hand case (the v1 hand table, bundle seed 1 of a crnn v2 set) --------------------------------

def test_hand_case_score_level(hand_report):
    _, report = hand_report
    bundle = report["models"][BUNDLE_ID]["score_level"]
    # The v1 hand case's CNN column (tests/evaluation/test_stage11_report.py, same derivation):
    # all 14 (8 spoof, 6 bonafide): EER 1/4 with threshold at the EER point 0.0; ROC-AUC 39/48.
    assert bundle["overall"]["eer"] == pytest.approx(0.25)
    assert bundle["overall"]["threshold_at_eer_point_descriptive"] == pytest.approx(0.0)
    assert bundle["overall"]["roc_auc"] == pytest.approx(39 / 48)
    # REAL vs FAKE_mp3: perfect separation -> EER 0 at score 2.0, AUC 1; 6 bonafide, 4 spoof.
    assert (bundle["REAL_vs_FAKE_mp3"]["eer"], bundle["REAL_vs_FAKE_mp3"]["roc_auc"]) == (0.0, 1.0)
    assert bundle["REAL_vs_FAKE_mp3"]["threshold_at_eer_point_descriptive"] == 2.0
    assert (bundle["REAL_vs_FAKE_mp3"]["n_bonafide"], bundle["REAL_vs_FAKE_mp3"]["n_spoof"]) == (6, 4)
    # REAL vs FAKE_other: EER 1/3 (w = 2/3 between FPR 1/6 and 1/3... as in v1), AUC 15/24.
    assert bundle["REAL_vs_FAKE_other"]["eer"] == pytest.approx(1 / 3)
    assert bundle["REAL_vs_FAKE_other"]["roc_auc"] == pytest.approx(15 / 24)
    # Other seeds: every FAKE above every REAL -> EER 0, AUC 1 (score level only, D57 (1)).
    for model_id in (IDS[0], IDS[2]):
        overall = report["models"][model_id]["score_level"]["overall"]
        assert (overall["eer"], overall["roc_auc"]) == (0.0, 1.0)


def test_hand_case_decision_level(hand_report):
    _, report = hand_report
    model = report["models"][BUNDLE_ID]
    fb = model["forced_binary"]
    # Forced binary on p >= 0.5 <=> raw >= 0. REAL: 1.0 -> FP 1, TN 5. FAKE_mp3: all 4 TP.
    # FAKE_other: 3.0, 0.5 TP; -1.0, -3.5 FN -> recall 1/2. Overall (TP, FP, TN, FN) = (6, 1, 5, 2).
    assert (fb["REAL"]["fp"], fb["REAL"]["tn"]) == (1, 5) and fb["REAL"]["fpr"] == pytest.approx(1 / 6)
    assert (fb["FAKE_mp3"]["tp"], fb["FAKE_mp3"]["fn"], fb["FAKE_mp3"]["recall"]) == (4, 0, 1.0)
    assert (fb["FAKE_other"]["tp"], fb["FAKE_other"]["fn"], fb["FAKE_other"]["recall"]) == (2, 2, 0.5)
    assert (fb["overall"]["tp"], fb["overall"]["fp"], fb["overall"]["tn"], fb["overall"]["fn"]) == (6, 1, 5, 2)
    assert fb["overall"]["f1"] == pytest.approx(2 * 6 / (2 * 6 + 1 + 2))  # 2TP / (2TP + FP + FN) = 12/15
    assert fb["REAL"]["recall"] is None and fb["FAKE_mp3"]["fpr"] is None  # undefined -> null
    # Three-way, band 0.3 <= p < 0.7: bonafide (auth, syn, unc) = (4, 1, 1), spoof = (2, 5, 1).
    tw = model["three_way"]
    assert tw["overall"]["confusion"] == {"columns": ["authentic", "synthetic", "uncertain"],
                                          "bonafide": [4, 1, 1], "spoof": [2, 5, 1]}
    assert (tw["overall"]["n"], tw["overall"]["n_deferred"]) == (14, 2)
    assert tw["overall"]["coverage"] == pytest.approx(12 / 14)
    assert (tw["overall"]["deferral_rate_bonafide"], tw["overall"]["deferral_rate_spoof"]) == \
        (pytest.approx(1 / 6), pytest.approx(1 / 8))
    nd = tw["overall"]["non_deferred"]
    assert (nd["tp"], nd["fp"], nd["tn"], nd["fn"]) == (5, 1, 4, 2)
    assert tw["FAKE_other"]["confusion"]["spoof"] == [2, 1, 1]
    assert model["frozen"] == {"threshold_on_p": 0.5, "half_width": 0.2, "policy_version": "risk-synthetic"}
    # D45: the other seeds have score level only; no baseline row (D57 (4) mapped to v2).
    for model_id in (IDS[0], IDS[2]):
        assert set(report["models"][model_id]) == {"role", "score_level"}
        assert report["models"][model_id]["role"] == "v2_other_seed"
    assert sorted(report["models"]) == sorted(IDS) and "baseline-v1" not in json.dumps(report["models"])


def test_hand_case_cis_seeds_and_no_hypotheses(hand_report):
    _, report = hand_report
    ci = report["models"][BUNDLE_ID]["ci"]
    assert sorted(ci) == ["eer", "fnr", "fpr", "roc_auc"]  # D57 (7), bundle model only
    assert ci["eer"]["estimate"] == pytest.approx(0.25) and ci["eer"]["n_groups"] == 12
    assert ci["fpr"]["counts"] == {"fp": 1, "n_bonafide": 6} and ci["fnr"]["counts"] == {"fn": 2, "n_spoof": 8}
    # Seeds (index = seed): EER 0, 1/4, 0 -> mean 1/12, sample SD sqrt(((1/6)^2 + 2 (1/12)^2) / 2) = sqrt(3/144).
    seeds = report["seeds"]
    assert seeds["eer"]["values"] == {"0": 0.0, "1": pytest.approx(0.25), "2": 0.0}
    assert seeds["eer"]["mean"] == pytest.approx(1 / 12) and seeds["eer"]["sd"] == pytest.approx((3 / 144) ** 0.5)
    assert seeds["roc_auc"]["values"] == {"0": 1.0, "1": pytest.approx(39 / 48), "2": 1.0}
    assert report["hypotheses"] is None
    assert report["hypotheses_note"] == report_module.HYPOTHESES_NOTE
    assert "H8 (Stage 13) and H9 (Stage 14)" in report["hypotheses_note"]


def test_report_numbers_agree_with_direct_harness_calls(hand_report):
    run, report = hand_report
    clip_dir = run.run_dir / "clip_scores"
    scores = {m: read_clip_scores(clip_dir, m) for m in IDS}
    bundle = scores[BUNDLE_ID]
    y = np.array([1 if x == "FAKE" else 0 for x in bundle.label])
    groups = list(bundle.group_id)
    model = report["models"][BUNDLE_ID]
    assert model["score_level"]["overall"]["eer"] == eer(y, bundle.raw_score)[0]
    assert model["score_level"]["overall"]["roc_auc"] == roc_auc(y, bundle.raw_score)
    fb = binary_metrics(y, bundle.p, POLICY.threshold)
    assert model["forced_binary"]["overall"]["f1"] == fb.f1 and model["forced_binary"]["overall"]["fp"] == fb.fp
    assert model["three_way"]["overall"]["coverage"] == three_way_metrics(y, list(bundle.prediction)).coverage
    ci = bootstrap_ci(lambda i: eer(y[i], bundle.raw_score[i])[0], len(y), clusters=groups)
    assert (model["ci"]["eer"]["low"], model["ci"]["eer"]["high"], model["ci"]["eer"]["n_invalid"]) == \
        (ci.low, ci.high, ci.n_invalid)
    ci = bootstrap_ci(lambda i: binary_metrics(y[i], bundle.p[i], POLICY.threshold).fnr, len(y), clusters=groups)
    assert (model["ci"]["fnr"]["low"], model["ci"]["fnr"]["high"]) == (ci.low, ci.high)
    summary = seed_summary({s: roc_auc(y, scores[IDS[s]].raw_score) for s in (0, 1, 2)})
    assert (report["seeds"]["roc_auc"]["mean"], report["seeds"]["roc_auc"]["sd"]) == (summary.mean, summary.sd)


def test_checks_inputs_and_conventions_are_recorded(hand_report):
    run, report = hand_report
    assert report["checks"] and all(c["passed"] for c in report["checks"].values())
    for key in ("run_report.architecture", "run_report.model_set.roles", "claim.sha256", "claim.architecture",
                "registry.records", "manifest.sha256_risk_v2", "clip_scores.folder_files",
                f"clip_scores.{BUNDLE_ID}.csv_sha256", f"same_clips.{IDS[2]}.group_id", "f2.counts",
                "search.architecture", f"search.checkpoint_sha256.{IDS[0]}", "risk_v2.v2_search_evidence",
                "risk_policy.sha256", "rederive.bundle.p_bytes", f"rederive.{IDS[0]}.p_and_prediction_absent"):
        assert key in report["checks"], key
    assert not any("baseline" in k or "stage7" in k or "stage10" in k for k in report["checks"])
    registry = run.registry.read_bytes()
    assert report["inputs"]["registry"] == {"sha256": _sha(registry), "lines": 4}  # v1 pair + v2 pair
    assert report["inputs"]["v2_risk_run_report_sha256"] == _sha((run.risk / "run_report.json").read_bytes())
    assert report["report_code_commit"] == REPORT_COMMIT and report["architecture"] == ARCH
    assert report["run"]["split"] == SPLIT and report["run"]["bundle_model_id"] == BUNDLE_ID
    assert report["conventions"]["decision"] == "D57"
    assert sorted(k for k in report["conventions"] if k not in ("decision", "scope")) == \
        sorted(report_module.CONVENTIONS_V2)
    # JSON keys: v1's minus the hypothesis content, plus architecture and the note.
    assert sorted(report) == sorted(["schema_version", "report_code_commit", "architecture", "run", "inputs",
                                     "conventions", "checks", "models", "seeds", "hypotheses", "hypotheses_note"])


def test_conventions_map_d57_to_v2_with_the_same_numbering():
    import src.evaluation.stage11_report as v1_report
    # Same nine items, in the same order; (5) and (6) are marked not applicable, not dropped.
    assert [k.split("_")[0] for k in report_module.CONVENTIONS_V2] == [k.split("_")[0] for k in v1_report.CONVENTIONS]
    for key in ("5_h5", "6_h1"):
        assert report_module.CONVENTIONS_V2[key].startswith("not applicable")
    assert "v2_bundle_seed" in report_module.CONVENTIONS_V2["1_model"]
    assert "no baseline row" in report_module.CONVENTIONS_V2["4_decision_level"]


# --- Integrity checks: each tamper raises with its own check name and writes nothing ---------------

RR = "run_report.json"


def _rr(edit):
    return lambda r: r.edit_json(r.run_dir / RR, edit)


def _drop_model(model_id):
    def edit(d):
        d["model_set"].pop(model_id)
    return edit


def _v1_model_set(d):
    d["model_set"] = {m: {"role": "cnn_other_seed", "artifact_sha256": "1" * 64} for m in V1_IDS}


def _with_baseline(d):
    d["model_set"]["baseline-v1"] = {"role": "baseline", "artifact_sha256": "4" * 64}


def _baseline_file(run):
    # A baseline score file left in the v2 run's folder (not in the run report).
    write_clip_scores(run.run_dir / "clip_scores", evaluation_purpose="stage11_test", run_id=run.run_id,
                      model_id="baseline-v1", model_role="baseline", model_artifact_sha256="4" * 64,
                      manifest_sha256="5" * 64, clip_id=[r["clip_id"] for r in run.rows],
                      label=[r["label"] for r in run.rows], group_id=[r["group"] for r in run.rows],
                      raw_score=np.zeros(len(run.rows)), prediction=["authentic"] * len(run.rows))


def _risk(edit):
    return lambda r: r.edit_json(r.risk / "run_report.json", edit)


def _resigned_policy(edit):
    """Edit risk_policy.json and re-sign risk_policy_sha256 in the risk run report, so only the edit disagrees."""
    def tamper(r):
        r.edit_json(r.risk / "risk_policy.json", edit)
        r.edit_json(r.risk / "run_report.json",
                    _set(["risk_policy_sha256"], _sha((r.risk / "risk_policy.json").read_bytes())))
    return tamper


def _search_only_selection(edit):
    return lambda r: (r.edit_json(r.search / "selection_report.json", edit),
                      r.edit_json(r.risk / "run_report.json", _set(["v2_search_evidence"], r.search_hashes())))


TAMPER = {
    # 1. run_report.json
    "run_report.schema_version": _rr(_set(["schema_version"], 2)),
    "run_report.dataset": _rr(_set(["dataset"], "asvspoof")),
    "run_report.evaluation_purpose": _rr(_set(["evaluation_purpose"], "stage14_robustness")),
    "run_report.run_id": _rr(_set(["run_id"], "XYZ")),
    "run_report.code_commit": _rr(_set(["code_commit"], "abc1234")),
    "run_report.split": _rr(_set(["split"], "")),
    "run_report.architecture": _rr(_set(["architecture"], "cnn")),  # e.g. a v1 run report value
    "run_report.model_set.ids": _rr(_drop_model(IDS[2])),
    "run_report.model_set.roles": _rr(_set(["model_set", IDS[0], "role"], "v2_bundle_seed")),
    "run_report.bundle_seed": _rr(_set(["bundle_seed"], True)),
    "run_report.bundle_seed_role": _rr(_set(["bundle_seed"], 0)),
    # 2. claim.json (re-signed, so only the field disagrees; claim.sha256 is not re-signed)
    "claim.run_id": lambda r: r.edit_claim(_set(["run_id"], "0" * 32)),
    "claim.code_commit": lambda r: r.edit_claim(_set(["code_commit"], "d" * 40)),
    "claim.split": lambda r: r.edit_claim(_set(["split"], "val-B")),
    "claim.dataset": lambda r: r.edit_claim(_set(["dataset"], "asvspoof")),
    "claim.evaluation_purpose": lambda r: r.edit_claim(_set(["evaluation_purpose"], "stage14_robustness")),
    "claim.sha256": lambda r: r.edit_json(r.run_dir / "claim.json", _set(["counts"], {})),
    "claim.results_ref": lambda r: r.edit_claim(_set(["results_ref"], " ")),
    "claim.architecture": lambda r: r.edit_claim(_set(["architecture"], "aasist")),
    "claim.bundle_seed": lambda r: r.edit_claim(_set(["bundle_seed"], 2)),
    "claim.model_ids": lambda r: r.edit_claim(_set(["model_ids"], sorted(V1_IDS))),
    # 4. score files
    f"clip_scores.{IDS[0]}.read": lambda r: (r.run_dir / "clip_scores" / f"{IDS[0]}.csv").write_bytes(
        (r.run_dir / "clip_scores" / f"{IDS[0]}.csv").read_bytes().replace(b"-1.0", b"-1.5", 1)),
    f"clip_scores.{IDS[2]}.csv_sha256": _rr(_set(["clip_scores", "files", IDS[2], "csv_sha256"], "f" * 64)),
    f"clip_scores.{BUNDLE_ID}.sidecar_sha256": _rr(_set(["clip_scores", "files", BUNDLE_ID, "sidecar_sha256"],
                                                        "f" * 64)),
    f"clip_scores.{IDS[2]}.n_rows": _rr(_set(["clip_scores", "files", IDS[2], "n_rows"], 15)),
    "clip_scores.folder": _rr(_set(["clip_scores", "folder"], "x")),
    "clip_scores.models": _rr(lambda d: d["clip_scores"]["files"].pop(IDS[0])),
    "clip_scores.folder_files": _baseline_file,
    # 6. F2 join: the run report's counts disagree with the stored files
    "f2.counts": _rr(_set(["counts", "subgroup", "REAL"], 7)),
    # 7. evidence: search, risk-v2, policy
    "search.bundle_checkpoint": _search_only_selection(_set(["selected"], "aasist")),
    # another winner (its seed 0 hash exists, so the agreement rule itself passes)
    "search.architecture": lambda r: r.edit_search(lambda d: d.update(selected="cnn-aug", bundle_seed=0)),
    "search.bundle_seed": lambda r: r.edit_search(_set(["bundle_seed"], 2)),
    f"search.checkpoint_sha256.{IDS[2]}": lambda r: r.edit_search(
        lambda d: d.get("checkpoint_sha256", {}).update({f"{ARCH}_seed2": "9" * 64})),  # run report only has it
    "risk_v2.architecture": _risk(_set(["architecture"], "aasist")),
    "risk_v2.bundle_seed": _risk(_set(["bundle_seed"], 0)),
    "risk_v2.checkpoint_sha256": _risk(_set(["checkpoint_sha256"], "9" * 64)),
    "risk_v2.v2_search_evidence": _risk(_set(["v2_search_evidence", "run_report.json"], "9" * 64)),
    "risk_policy.sha256": lambda r: (r.risk / "risk_policy.json").write_bytes(
        (r.risk / "risk_policy.json").read_bytes() + b"\n"),
    "risk_v2.policy_version": _resigned_policy(_set(["version"], "risk-other")),
    # the manifest
    "manifest.sha256_risk_v2": _risk(_set(["f2_manifest_sha256"], "9" * 64)),
}

# Tampers that need the run folder written differently (options of Run).
BUILD_TAMPER = {
    "registry.records": {"complete": False},  # an incomplete run is refused
    "registry.started.key": {"registry_model_ids": tuple(f"aasist-v2-seed{k}" for k in (0, 1, 2))},
    "registry.started.code_commit": {"registry_commit": "d" * 40},
    "registry.completed.results_ref": {"results_ref_completed": "elsewhere/run_report.json"},
    "manifest.columns": {"manifest_columns": {"group": "cluster"}},
    "manifest.unique_paths": {"manifest_edit": lambda m: m.append({**m[0], "split": "train"})},
    f"clip_scores.{IDS[0]}.run_id": {"sidecar": {IDS[0]: {"run_id": "0" * 32}}},
    f"clip_scores.{IDS[2]}.model_role": {"sidecar": {IDS[2]: {"model_role": "cnn_other_seed"}}},  # same columns
    f"clip_scores.{BUNDLE_ID}.model_artifact_sha256": {"sidecar": {BUNDLE_ID: {"model_artifact_sha256": "7" * 64}}},
    f"clip_scores.{BUNDLE_ID}.manifest_sha256": {"sidecar": {BUNDLE_ID: {"manifest_sha256": "6" * 64}}},
    f"clip_scores.{IDS[0]}.evaluation_purpose": {"sidecar": {IDS[0]: {"evaluation_purpose": "stage14_robustness"}}},
    f"same_clips.{BUNDLE_ID}.group_id": {"columns": {BUNDLE_ID: {"group_id": ["g-x"] * 14}}},
    f"same_clips.{IDS[2]}.label": {"columns": {IDS[2]: {"label": ["REAL"] * 14}}},
    f"same_clips.{IDS[2]}.clip_id": {"columns": {IDS[2]: {"clip_id": [f"x/{c}" for c, *_ in HAND]}}},
    "f2.rows": {"manifest_edit": lambda m: m[0].update(split="val-A")},
    "f2.group": {"manifest_edit": lambda m: m[3].update(group="g-other")},
    "f2.n_rows": {"manifest_edit": lambda m: m.append({"path": "REAL/missing.wav", "label": "REAL", "subgroup": "REAL",
                                                       "group": "gx", "split": SPLIT, "hash": "a" * 64})},
    "rederive.bundle.p_bytes": {"p_override": lambda p: np.nextafter(p, 1.0)},
    "rederive.bundle.prediction_mismatches": {"prediction_override": lambda pred: ["uncertain"] * len(pred)},
}


def _assert_refused(run, key, tmp_path):
    out = tmp_path / f"report-{key}"
    with pytest.raises(report_module.CheckFailed) as caught:
        report_module.main(run.argv(out))
    assert caught.value.key == key
    assert f"check failed: {key}:" in str(caught.value)
    assert not out.exists()  # nothing written


@pytest.mark.parametrize("key", sorted(TAMPER))
def test_tampered_input_is_refused_with_its_check(make, tmp_path, key):
    run = make()  # the untampered folder passes every check (test_checks_inputs_and_conventions_are_recorded)
    TAMPER[key](run)
    _assert_refused(run, key, tmp_path)


@pytest.mark.parametrize("key", sorted(BUILD_TAMPER))
def test_tampered_run_folder_is_refused_with_its_check(make, tmp_path, key):
    _assert_refused(make(**BUILD_TAMPER[key]), key, tmp_path)


def test_edited_registry_is_refused(make, tmp_path):
    run = make()
    lines = run.registry.read_bytes().split(b"\n")
    lines[2] = lines[2].replace(b'"stage11_test"', b'"stage14_robustness"')  # the v2 claim; breaks the chain
    run.registry.write_bytes(b"\n".join(lines))
    _assert_refused(run, "registry.load", tmp_path)


def test_unapproved_manifest_is_refused(make, tmp_path, monkeypatch):
    run = make()
    monkeypatch.setattr(report_module, "APPROVED_F2_MANIFEST_SHA256", report_module.F2_MANIFEST_SHA256)
    _assert_refused(run, "manifest.sha256_approved", tmp_path)


def test_other_seed_with_p_or_prediction_is_refused(make, tmp_path, monkeypatch):
    # read_clip_scores already refuses p / prediction for v2_other_seed (clip_scores tests); this check
    # is the report's own record of D45, so it is exercised with a reader that lets them through.
    run = make()
    real = report_module.read_clip_scores

    def leaky(folder, model_id):
        scores = real(folder, model_id)
        if model_id == IDS[2]:
            return type(scores)(**{**vars(scores), "p": np.full(len(scores.clip_id), 0.5)})
        return scores
    monkeypatch.setattr(report_module, "read_clip_scores", leaky)
    _assert_refused(run, f"rederive.{IDS[2]}.p_and_prediction_absent", tmp_path)


def test_v1_model_set_or_baseline_is_refused(make, tmp_path):
    # The v1 model set (four v1 IDs), a v1 run report (no v2 architecture) and a baseline-v1 entry are
    # refused before any metric; a baseline file in the folder is case clip_scores.folder_files above.
    for edit, key in ((_v1_model_set, "run_report.model_set.ids"), (_with_baseline, "run_report.model_set.ids"),
                      (lambda d: d.pop("architecture"), "run_report.architecture")):
        run = make()
        run.edit_json(run.run_dir / RR, edit)
        _assert_refused(run, key, tmp_path)


def test_every_check_name_is_distinct_and_covered(hand_report):
    _, report = hand_report
    tampered = set(TAMPER) | set(BUILD_TAMPER) | {"registry.load", "manifest.sha256_approved",
                                                  f"rederive.{IDS[2]}.p_and_prediction_absent"}
    assert not set(TAMPER) & set(BUILD_TAMPER)
    # Every check recorded on the passing case is tampered at least once, up to the per-model / per-record
    # copies of the same rule (the model ID or record status in the key).
    def rule(key):
        for model_id in IDS:
            key = key.replace(model_id, "<model>")
        return key.replace(".completed.", ".<status>.").replace(".started.", ".<status>.")
    assert {rule(k) for k in report["checks"]} - {rule(k) for k in tampered} == set()


# --- Output ---------------------------------------------------------------------------------------

def test_cli_twice_gives_byte_identical_files_rendered_from_the_json(make, tmp_path, capsys):
    run = make()
    report_module.main(run.argv(tmp_path / "cli"))
    cli = (tmp_path / "cli" / "metrics_report.json", tmp_path / "cli" / "metrics_report.md")
    assert capsys.readouterr().out.split() == [str(x) for x in cli]
    direct = report_module.write_report(report_module.build_report(**run.kwargs()), tmp_path / "direct")
    for x, y in zip(cli, direct):
        assert x.read_bytes() == y.read_bytes()
    json_path, md_path = cli
    text = json_path.read_text(encoding="utf-8")
    assert text.endswith("}\n") and "NaN" not in text and "null" in text
    data = json.loads(text, parse_constant=lambda c: pytest.fail(c))
    assert data["hypotheses"] is None and data["hypotheses_note"] == report_module.HYPOTHESES_NOTE
    markdown = md_path.read_text(encoding="utf-8")
    assert markdown == report_module.render_markdown(data)
    assert "**Reported whatever the outcome (pre-registered).**" in markdown
    assert report_module.HYPOTHESES_NOTE in markdown
    assert f"Architecture `{ARCH}`" in markdown and "undefined" in markdown
    assert "baseline" not in markdown and "| Hypothesis |" not in markdown  # no baseline row, no H table
    assert markdown.count("H1") == 1 and markdown.count("H5") == 1  # only in the note
    assert f"| {BUNDLE_ID} | eer | 0.2500 |" in markdown


def test_markdown_flags_collapsed_intervals(hand_report):
    _, report = hand_report
    report = json.loads(json.dumps(report))
    report["models"][BUNDLE_ID]["ci"]["eer"].update(low=0.0, high=0.0, collapsed=True)
    assert "[0.0000, 0.0000] collapsed" in report_module.render_markdown(report)


def test_existing_out_is_refused(make, tmp_path):
    run = make()
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        report_module.main(run.argv(out))  # refused before any work
    with pytest.raises(ValueError, match="already exists"):
        report_module.write_report({"schema_version": 1}, out)
    assert list(out.iterdir()) == []


def test_dirty_worktree_is_refused(make, monkeypatch):
    run = make()

    def dirty():
        raise ValueError("Official Stage 6 validation requires a clean Git worktree")
    monkeypatch.setattr(report_module, "code_version", dirty)
    with pytest.raises(ValueError, match="clean Git worktree"):
        report_module.build_report(**run.kwargs())


# --- Module boundaries ----------------------------------------------------------------------------

def test_import_is_torch_free():
    # A fresh interpreter: importing the report module must not import torch (it never loads a model).
    code = "import sys, src.evaluation.stage11_v2_report; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    assert out.stdout.strip() == "False"


def test_v1_report_module_is_unchanged_and_reused():
    # The v2 module imports stage11_report's model-independent helpers; the v1 module's names it uses exist
    # and are the same objects (not copies).
    import src.evaluation.stage11_report as v1_report
    for name in ("CheckFailed", "_record", "_check", "_fail", "_sha256", "_read_json", "_check_claim", "_f2_join",
                 "_score_level", "_binary", "_three_way", "_model_cis", "_subgroup_rows", "_create", "_table", "_fmt"):
        assert getattr(report_module, name) is getattr(v1_report, name), name


def test_report_module_is_not_test_split_access_for_the_guard():
    # D31: the module reads the stored files only, so it needs no allow-list entry; the guard finds no
    # test-split request in it (it never names the split) and ALLOWED holds only the three exact
    # runners plus the reviewed Stage 14 official notebook.
    rel = "src/evaluation/stage11_v2_report.py"
    assert file_hits(rel, (REPO_ROOT / rel).read_text(encoding="utf-8")) == []
    assert ALLOWED == (
        "src/evaluation/stage11_run.py",
        "src/evaluation/stage11_v2_run.py",
        "src/evaluation/stage14_run.py",
        "notebooks/11_stage14_v1_for_test.ipynb",
    )


def _dict_keys(function: str, variable: str) -> list[str]:
    """Keys of a dict literal assigned in a function of the v2 runner; **x expands x's own literal keys."""
    tree = ast.parse(V2_RUNNER.read_text(encoding="utf-8"))
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)

    def literal(name):
        return next(n for n in ast.walk(func) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)).value

    keys = []
    node = literal(variable)
    for key, value in zip(node.keys, node.values):
        keys += [k.value for k in literal(value.id).keys] if key is None else [key.value]
    return keys


def test_fixture_key_lists_equal_the_v2_runners():
    assert _dict_keys("claim", "record") == CLAIM_KEYS
    assert _dict_keys("score", "report") == RUN_REPORT_KEYS


def test_model_ids_and_roles_functions():
    assert report_module.model_ids_v2("aasist") == ("aasist-v2-seed0", "aasist-v2-seed1", "aasist-v2-seed2")
    assert report_module.model_roles(ARCH, BUNDLE_SEED) == {IDS[0]: "v2_other_seed", IDS[1]: "v2_bundle_seed",
                                                            IDS[2]: "v2_other_seed"}
    with pytest.raises(ValueError, match="winner must be one of"):
        report_module.model_ids_v2("cnn")
    with pytest.raises(ValueError, match="bundle_seed must be one of"):
        report_module.model_roles(ARCH, 3)


# --- With torch: the v2 runner's definitions and output -------------------------------------------

def test_constants_equal_the_v2_runners():
    from tests._torch import require_torch
    require_torch()
    import src.evaluation.stage11_v2_run as m
    import src.tuning.risk_run as risk_run
    import src.tuning.v2_search as vs
    from src.models.v2_checkpoint import ARCHITECTURES

    for name in ("DATASET", "PURPOSE", "SEEDS", "ROLE_BUNDLE", "ROLE_OTHER", "CLIP_SCORES_DIR", "RUN_REPORT",
                 "CLAIM_JSON"):
        assert getattr(report_module, name) == getattr(m, name), name
    assert report_module.RUN_SCHEMA_VERSION == m.SCHEMA_VERSION
    assert report_module.ARCHITECTURES == ARCHITECTURES
    assert (report_module.SEARCH_RUN_REPORT, report_module.SEARCH_SELECTION_REPORT) == (vs.RUN_REPORT,
                                                                                       vs.SELECTION_REPORT)
    assert (report_module.RISK_RUN_REPORT, report_module.RISK_POLICY) == (risk_run.RUN_REPORT, risk_run.RISK_POLICY)
    for arch in ARCHITECTURES:
        assert report_module.model_ids_v2(arch) == m.model_ids_v2(arch)
        for seed in m.SEEDS:
            assert report_module.model_roles(arch, seed) == m.model_roles(arch, seed)
            assert report_module.run_dir_name(arch, seed) == vs.run_dir_name(arch, seed)
    # The fixture's counts equal the runner's own function on the same rows.
    assert _counts(_rows_from(HAND)) == m._counts(pd.DataFrame(_rows_from(HAND)))


SEARCH_CASES = {
    "valid": None,
    "winner_disagrees": ("selection_report.json", _set(["selected"], "aasist")),
    "seed_disagrees": ("selection_report.json", _set(["bundle_seed"], 2)),
    "seed_bool": ("run_report.json", _set(["bundle_seed"], True)),
    "unknown_winner": ("run_report.json", _set(["selected"], "cnn")),
    "missing_hash": ("run_report.json", lambda d: d["checkpoint_sha256"].pop(f"{ARCH}_seed{BUNDLE_SEED}")),
    "missing_key": ("run_report.json", lambda d: d.pop("selected")),
    "missing_file": ("selection_report.json", None),
}


@pytest.mark.parametrize("case", sorted(SEARCH_CASES))
def test_search_bundle_equals_v2_risk_run_bundle_checkpoint(make, case):
    # The torch-free agreement rule gives the same answer, or the same refusal, as v2_risk_run's.
    from tests._torch import require_torch
    require_torch()
    import src.tuning.v2_risk_run as vr

    run = make()
    if SEARCH_CASES[case] is not None:
        name, edit = SEARCH_CASES[case]
        if edit is None:
            (run.search / name).unlink()
        else:
            run.edit_json(run.search / name, edit)

    def outcome(function):
        try:
            return function(run.search)
        except Exception as error:  # noqa: BLE001  (compared by type and message)
            return type(error), str(error)
    expected = outcome(vr.bundle_checkpoint)
    assert outcome(report_module.search_bundle) == expected
    if case == "valid":
        assert expected == (ARCH, BUNDLE_SEED, SEED_HASHES[BUNDLE_SEED])
    else:
        assert isinstance(expected, tuple) and isinstance(expected[0], type)


@pytest.mark.parametrize("arch", ["crnn", "aasist"])
def test_report_reads_the_v2_runners_own_output(request, monkeypatch, tmp_path, arch):
    # End to end on the v2 runner's synthetic happy path (tests/evaluation/test_stage11_v2_run.py): the
    # real claim + score write the folder (registry: v1 pair, then the v2 pair), and the report must
    # accept exactly what the runner wrote, with the committed-evidence layout of the risk-v2 run.
    from tests._torch import require_torch
    require_torch()
    world = request.getfixturevalue("world")
    state = request.getfixturevalue("env")(arch)
    out = tmp_path / "runner_out"
    record = runner_tests._claim(world, state, out)
    run_report = runner_tests._score(world, state, out, record["run_id"])
    monkeypatch.setattr(report_module, "APPROVED_F2_MANIFEST_SHA256", world["f2"])
    monkeypatch.setattr(report_module, "code_version", lambda: {"git_commit": REPORT_COMMIT, "git_dirty": False})
    a = world["arch"][arch]
    argv = ["--run-dir", str(out), "--registry", str(state["registry"]), "--manifest", str(world["manifest_path"]),
            "--search-evidence", str(a["search"]), "--risk-evidence", str(a["risk"])]
    report_module.main([*argv, "--out", str(tmp_path / "report")])
    report = json.loads((tmp_path / "report" / "metrics_report.json").read_text(encoding="utf-8"))
    assert report["checks"] and all(c["passed"] for c in report["checks"].values())
    assert (report["architecture"], report["run"]["run_id"]) == (arch, run_report["run_id"])
    assert report["run"]["counts"] == run_report["counts"] and report["inputs"]["registry"]["lines"] == 4
    bundle_id = f"{arch}-v2-seed{run_report['bundle_seed']}"
    assert report["run"]["bundle_model_id"] == bundle_id
    stored = read_clip_scores(out / "clip_scores", bundle_id)
    y = np.array([1 if x == "FAKE" else 0 for x in stored.label])
    confusion = report["models"][bundle_id]["three_way"]["overall"]["confusion"]
    assert (tuple(confusion["bonafide"]), tuple(confusion["spoof"])) == three_way_metrics(y, list(stored.prediction)).confusion
    assert report["models"][bundle_id]["score_level"]["overall"]["eer"] == eer(y, stored.raw_score)[0]
    # A risk-v2 folder of another architecture is refused (the evidence binding), nothing written.
    other = "aasist" if arch == "crnn" else "crnn"
    copy = tmp_path / "other_risk"
    shutil.copytree(world["arch"][other]["risk"], copy)
    with pytest.raises(report_module.CheckFailed) as caught:
        report_module.main([*argv[:-1], str(copy), "--out", str(tmp_path / "refused")])
    assert caught.value.key == "risk_v2.architecture"
    assert not (tmp_path / "refused").exists()
