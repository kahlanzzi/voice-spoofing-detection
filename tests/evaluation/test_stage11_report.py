"""Tests for src/evaluation/stage11_report.py (Stage 11 metrics from the stored files; D57).

Synthetic only: a run folder written with the real writers (write_clip_scores, RunRegistry, the
runner's JSON format), a small F2-format manifest, a RiskPolicy file and Stage 7 / Stage 10 evidence
stubs. No dataset, audio, checkpoint or model is read. Each test has its hand calculation or its
reasoning as a comment.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.evaluation.stage11_report as report_module
from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.clip_scores import read_clip_scores, write_clip_scores
from src.evaluation.metrics import binary_metrics, eer, roc_auc, three_way_metrics
from src.evaluation.run_registry import RunRegistry
from src.evaluation.seeds import seed_summary
from src.risk.calibration import PlattCalibration
from src.risk.decisions import AUTHENTIC, SYNTHETIC
from src.risk.policy import RiskPolicy
from src.tuning.baseline_run import _json_text
from tests.governance.test_for_test_access import ALLOWED, file_hits

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "src" / "evaluation" / "stage11_run.py"
COMMIT = "c" * 40
REPORT_COMMIT = "e" * 40
SPLIT = "test"  # tests/ is not scanned by the guard; the module itself never names the split
RESULTS_REF = "stage11/run_report.json"
BUNDLE_HASH = "b" * 64
SEED_HASHES = {0: BUNDLE_HASH, 1: "1" * 64, 2: "2" * 64}

# Key lists of the runner's claim.json and run_report.json, checked against the runner's source
# (test_fixture_key_lists_equal_the_runners), so the fixture writes the runner's format.
CLAIM_KEYS = ["schema_version", "run_id", "code_commit", "registry_sha256", "results_ref", "dataset", "split",
              "evaluation_purpose", "model_ids", "pre_claim_checks", "environment_files", "counts"]
RUN_REPORT_KEYS = ["schema_version", "code_commit", "run_id", "dataset", "split", "evaluation_purpose",
                   "model_set", "bundle_seed", "claim_sha256", "scoring_marker_sha256", "pre_scoring_checks",
                   "environment_files", "environment_score_sha256", "environment_same_as_claim", "scoring",
                   "features", "clip_scores", "counts"]

# Hand case: 14 clips, 6 REAL, 4 FAKE_mp3, 4 FAKE_other; 12 D14 groups (r1/r2 and m1/m2 share one).
# Columns: clip_id, subgroup, group, bundle CNN raw (seed 0), baseline raw.
HAND = [
    ("REAL/r1.wav", "REAL", "g-r12", -3.0, 0.9),
    ("REAL/r2.wav", "REAL", "g-r12", -2.5, 0.7),
    ("REAL/r3.wav", "REAL", "g-r3", -2.0, -0.2),
    ("REAL/r4.wav", "REAL", "g-r4", -1.5, -0.4),
    ("REAL/r5.wav", "REAL", "g-r5", -0.5, -0.6),
    ("REAL/r6.wav", "REAL", "g-r6", 1.0, -0.8),
    ("FAKE/m1.wav", "FAKE_mp3", "g-m12", 2.0, 0.8),
    ("FAKE/m2.wav", "FAKE_mp3", "g-m12", 2.5, 0.6),
    ("FAKE/m3.wav", "FAKE_mp3", "g-m3", 3.5, 0.4),
    ("FAKE/m4.wav", "FAKE_mp3", "g-m4", 4.0, -0.1),
    ("FAKE/o1.wav", "FAKE_other", "g-o1", 3.0, 0.3),
    ("FAKE/o2.wav", "FAKE_other", "g-o2", 0.5, -0.3),
    ("FAKE/o3.wav", "FAKE_other", "g-o3", -1.0, -0.5),
    ("FAKE/o4.wav", "FAKE_other", "g-o4", -3.5, -0.7),
]
# Policy: p = sigmoid(raw) (a = 1, b = 0); threshold 0.5 (p >= 0.5 <=> raw >= 0); band half-width 0.2
# (uncertain iff 0.3 <= p < 0.7, i.e. about -0.847 <= raw < 0.847).
POLICY = RiskPolicy(version="risk-synthetic", calibration=PlattCalibration(a=1.0, b=0.0, n_spoof=10, n_bonafide=10),
                    threshold=0.5, half_width=0.2, target_deferral=0.1, n_spoof_fit=10, n_bonafide_fit=10,
                    deferral_on_fit=0.05)
BASELINE_THRESHOLD = 0.0


def _rows_from(table, other_seeds=None):
    """Clip rows: clip_id, label, subgroup, group, and raw scores per model ID."""
    rows = []
    for k, (clip_id, subgroup, group, cnn, base) in enumerate(table):
        label = "REAL" if subgroup == "REAL" else "FAKE"
        # Seeds 1-2: perfect separation (REAL < 0 < FAKE, distinct), so EER 0 and ROC-AUC 1.
        sign = -1.0 if label == "REAL" else 1.0
        extra = other_seeds(k, label) if other_seeds else (sign * (k + 1), sign * (k + 1) * 2 + 0.25)
        rows.append({"clip_id": clip_id, "label": label, "subgroup": subgroup, "group": group,
                     "baseline-v1": base, "cnn-v1-seed0": cnn, "cnn-v1-seed1": extra[0], "cnn-v1-seed2": extra[1]})
    return rows


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _counts(rows):
    # The runner's _counts(rows) format (checked against it in test_constants_and_formats_equal_the_runners).
    frame = pd.DataFrame(rows)
    return {column: {k: int(v) for k, v in sorted(frame[column].value_counts().items())}
            for column in ("label", "subgroup")}


class Run:
    """A synthetic Stage 11 output folder plus every input build_report needs."""

    def __init__(self, root: Path, monkeypatch, rows, *, complete=True, sidecar=None, columns=None, p_override=None,
                 prediction_override=None, manifest_edit=None, results_ref_completed=RESULTS_REF):
        self.root, self.rows = root, rows
        sidecar, columns = sidecar or {}, columns or {}
        # F2 manifest: the run's rows plus two rows of another split.
        manifest = [{"path": r["clip_id"], "label": r["label"], "subgroup": r["subgroup"], "group": r["group"],
                     "split": SPLIT, "hash": "a" * 64} for r in rows]
        manifest += [{"path": f"REAL/train{i}.wav", "label": "REAL", "subgroup": "REAL", "group": f"t{i}",
                      "split": "train", "hash": "a" * 64} for i in range(2)]
        if manifest_edit:
            manifest_edit(manifest)
        self.manifest = root / "manifest.csv"
        _write(self.manifest, pd.DataFrame(manifest, columns=list(report_module.F2_COLUMNS))
               .to_csv(index=False, lineterminator="\n").encode("utf-8"))
        manifest_sha = _sha(self.manifest.read_bytes())
        monkeypatch.setattr(report_module, "APPROVED_F2_MANIFEST_SHA256", manifest_sha)

        self.policy = root / "risk_policy.json"
        _write(self.policy, POLICY.to_json().encode("utf-8"))
        self.stage7 = root / "stage7"
        _write(self.stage7 / "run_report.json", _json_text({
            "artifact_sha256": report_module.BASELINE_ARTIFACT_SHA256, "val_b_threshold": BASELINE_THRESHOLD,
            "version": "baseline-v1"}).encode("utf-8"))
        self.stage10 = root / "stage10"
        _write(self.stage10 / "run_report.json", _json_text({
            "bundle_seed": 0, "checkpoint_sha256": BUNDLE_HASH, "f2_manifest_sha256": manifest_sha,
            "risk_policy_sha256": _sha(self.policy.read_bytes())}).encode("utf-8"))

        self.registry = root / "registry" / "run_registry.jsonl"
        self.registry.parent.mkdir()
        registry = RunRegistry(self.registry)
        model_ids = list(report_module.MODEL_IDS)
        self.run_id = registry.claim("for", SPLIT, model_ids, COMMIT, evaluation_purpose="stage11_test")
        if complete:
            registry.complete(self.run_id, results_ref=results_ref_completed)

        self.run_dir = root / "out"
        counts = _counts(rows)
        claim = {"schema_version": 1, "run_id": self.run_id, "code_commit": COMMIT, "registry_sha256": "0" * 64,
                 "results_ref": RESULTS_REF, "dataset": "for", "split": SPLIT, "evaluation_purpose": "stage11_test",
                 "model_ids": sorted(model_ids), "pre_claim_checks": {}, "environment_files": {}, "counts": counts}
        _write(self.run_dir / "claim.json", _json_text(claim).encode("utf-8"))
        model_set = {"baseline-v1": {"role": "baseline", "artifact_sha256": report_module.BASELINE_ARTIFACT_SHA256},
                     **{f"cnn-v1-seed{s}": {"role": "cnn_bundle_seed" if s == 0 else "cnn_other_seed",
                                            "artifact_sha256": SEED_HASHES[s]} for s in (0, 1, 2)}}
        self.model_set = model_set
        clip_dir = self.run_dir / "clip_scores"
        clip_dir.mkdir()
        files = {}
        for model_id, model in model_set.items():
            raw = np.array([r[model_id] for r in rows])
            outputs = {}
            if model["role"] == "cnn_bundle_seed":
                decisions = POLICY.decide(raw)
                outputs = {"p": np.array([d.p for d in decisions]), "prediction": [d.prediction for d in decisions]}
                if p_override:
                    outputs["p"] = p_override(outputs["p"])
                if prediction_override:
                    outputs["prediction"] = prediction_override(outputs["prediction"])
            elif model["role"] == "baseline":
                outputs = {"prediction": np.where(raw >= BASELINE_THRESHOLD, SYNTHETIC, AUTHENTIC).tolist()}
            meta = {"evaluation_purpose": "stage11_test", "run_id": self.run_id, "model_id": model_id,
                    "model_role": model["role"], "model_artifact_sha256": model["artifact_sha256"],
                    "manifest_sha256": manifest_sha, **sidecar.get(model_id, {})}
            cols = {"clip_id": [r["clip_id"] for r in rows], "label": [r["label"] for r in rows],
                    "group_id": [r["group"] for r in rows], **columns.get(model_id, {})}
            csv_path, json_path = write_clip_scores(clip_dir, **meta, **cols, raw_score=raw, **outputs)
            files[model_id] = {"csv_sha256": _sha(csv_path.read_bytes()), "sidecar_sha256": _sha(json_path.read_bytes()),
                               "n_rows": len(rows)}
        report = {"schema_version": 1, "code_commit": COMMIT, "run_id": self.run_id, "dataset": "for", "split": SPLIT,
                  "evaluation_purpose": "stage11_test", "model_set": model_set, "bundle_seed": 0,
                  "claim_sha256": _sha((self.run_dir / "claim.json").read_bytes()), "scoring_marker_sha256": "0" * 64,
                  "pre_scoring_checks": {}, "environment_files": {}, "environment_score_sha256": "0" * 64,
                  "environment_same_as_claim": True, "scoring": {"device": "cpu", "batch_size": 4, "passes": 2},
                  "features": {}, "clip_scores": {"folder": "clip_scores", "files": files}, "counts": counts}
        self.write_run_report(report)

    def write_run_report(self, report: dict) -> None:
        (self.run_dir / "run_report.json").write_bytes(_json_text(report).encode("utf-8"))

    def run_report(self) -> dict:
        return json.loads((self.run_dir / "run_report.json").read_text(encoding="utf-8"))

    def edit_json(self, path: Path, edit) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        edit(data)
        path.write_bytes(_json_text(data).encode("utf-8"))

    def kwargs(self) -> dict:
        return {"run_dir": self.run_dir, "registry_path": self.registry, "manifest_path": self.manifest,
                "risk_policy_path": self.policy, "stage7_dir": self.stage7, "stage10_dir": self.stage10}

    def argv(self, out: Path) -> list[str]:
        return ["--run-dir", str(self.run_dir), "--registry", str(self.registry), "--manifest", str(self.manifest),
                "--risk-policy", str(self.policy), "--stage7-evidence", str(self.stage7),
                "--stage10-evidence", str(self.stage10), "--out", str(out)]


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
    run = Run(tmp_path_factory.mktemp("hand"), monkeypatch, _rows_from(HAND))
    report = report_module.build_report(**run.kwargs())
    yield run, json.loads(_json_text(report))  # as written: NaN never appears, None for undefined
    monkeypatch.undo()


# --- Hand case ------------------------------------------------------------------------------------

def test_hand_case_score_level(hand_report):
    _, report = hand_report
    cnn = report["models"]["cnn-v1-seed0"]["score_level"]
    # Bundle CNN, all 14 (8 spoof, 6 bonafide), scores descending:
    # 4.0F 3.5F 3.0F 2.5F 2.0F | 1.0R | 0.5F | -0.5R | -1.0F -1.5R -2.0R -2.5R -3.0R -3.5F.
    # After 0.5F: FPR 1/6, FNR 2/8; after -0.5R: FPR 2/6 >= FNR 2/8 (first). gap_before = 1/4 - 1/6 =
    # 1/12, gap_after = 1/3 - 1/4 = 1/12, w = 1/2: EER = 1/6 + 1/2 * 1/6 = 1/4; threshold 0.5 - 0.5 = 0.0.
    assert cnn["overall"]["eer"] == pytest.approx(0.25)
    assert cnn["overall"]["threshold_at_eer_point_descriptive"] == pytest.approx(0.0)
    # AUC: spoofs above 6, 6, 6, 6, 6 (top five), 0.5 above 5, -1.0 above 4, -3.5 above 0 -> 39/48.
    assert cnn["overall"]["roc_auc"] == pytest.approx(39 / 48)
    # REAL vs FAKE_mp3: perfect separation -> EER 0 (first point with FNR 0, FPR 0, at score 2.0), AUC 1.
    assert cnn["REAL_vs_FAKE_mp3"]["eer"] == 0.0
    assert cnn["REAL_vs_FAKE_mp3"]["threshold_at_eer_point_descriptive"] == 2.0
    assert cnn["REAL_vs_FAKE_mp3"]["roc_auc"] == 1.0
    assert (cnn["REAL_vs_FAKE_mp3"]["n_bonafide"], cnn["REAL_vs_FAKE_mp3"]["n_spoof"]) == (6, 4)
    # REAL vs FAKE_other: 3.0F 1.0R 0.5F -0.5R -1.0F ...: after -0.5R FPR 1/3, FNR 1/2; after -1.0F
    # FPR 1/3 >= FNR 1/4 (first): w = (1/6) / (1/6 + 1/12) = 2/3, EER = 1/3; threshold -0.5 - 2/3 * 0.5.
    assert cnn["REAL_vs_FAKE_other"]["eer"] == pytest.approx(1 / 3)
    assert cnn["REAL_vs_FAKE_other"]["threshold_at_eer_point_descriptive"] == pytest.approx(-0.5 - 1 / 3)
    # AUC: 3.0 above 6, 0.5 above 5, -1.0 above 4, -3.5 above 0 -> 15/24.
    assert cnn["REAL_vs_FAKE_other"]["roc_auc"] == pytest.approx(15 / 24)
    # Baseline: .9R .8F .7R .6F .4F .3F -.1F | -.2R: before it FPR 1/3, FNR 3/8; after FPR 1/2 (first
    # FPR >= FNR). gap_before 1/24, gap_after 1/8, w = 1/4: EER = 1/3 + 1/4 * 1/6 = 3/8; threshold -0.125.
    base = report["models"]["baseline-v1"]["score_level"]["overall"]
    assert base["eer"] == pytest.approx(0.375)
    assert base["threshold_at_eer_point_descriptive"] == pytest.approx(-0.125)


def test_hand_case_decision_level(hand_report):
    _, report = hand_report
    fb = report["models"]["cnn-v1-seed0"]["forced_binary"]
    # Forced binary on p >= 0.5 <=> raw >= 0. REAL: 1.0 -> FP 1, TN 5. FAKE_mp3: all 4 TP -> recall 1.
    # FAKE_other: 3.0, 0.5 TP; -1.0, -3.5 FN -> recall 1/2.
    assert (fb["REAL"]["fp"], fb["REAL"]["tn"]) == (1, 5)
    assert fb["REAL"]["fpr"] == pytest.approx(1 / 6)
    assert (fb["FAKE_mp3"]["tp"], fb["FAKE_mp3"]["fn"], fb["FAKE_mp3"]["recall"]) == (4, 0, 1.0)
    assert (fb["FAKE_other"]["tp"], fb["FAKE_other"]["fn"], fb["FAKE_other"]["recall"]) == (2, 2, 0.5)
    assert (fb["overall"]["tp"], fb["overall"]["fp"], fb["overall"]["tn"], fb["overall"]["fn"]) == (6, 1, 5, 2)
    # Undefined rates are null: a REAL-only subset has no spoof (recall, FNR), a FAKE-only subset no bonafide (FPR).
    assert fb["REAL"]["recall"] is None and fb["REAL"]["fnr"] is None
    assert fb["FAKE_mp3"]["fpr"] is None
    # Three-way, band 0.3 <= p < 0.7. REAL p: .047 .076 .119 .182 auth, .378 (raw -0.5) unc, .731 syn
    # -> (4, 1, 1). Spoof: mp3 all syn (p >= .88); other .953 syn, .622 unc, .269 auth, .029 auth -> (2, 5, 1).
    tw = report["models"]["cnn-v1-seed0"]["three_way"]
    assert tw["overall"]["confusion"] == {"columns": ["authentic", "synthetic", "uncertain"],
                                          "bonafide": [4, 1, 1], "spoof": [2, 5, 1]}
    assert (tw["overall"]["n"], tw["overall"]["n_deferred"]) == (14, 2)
    assert tw["overall"]["coverage"] == pytest.approx(12 / 14)
    assert tw["overall"]["deferral_rate_bonafide"] == pytest.approx(1 / 6)
    assert tw["overall"]["deferral_rate_spoof"] == pytest.approx(1 / 8)
    nd = tw["overall"]["non_deferred"]
    assert (nd["tp"], nd["fp"], nd["tn"], nd["fn"]) == (5, 1, 4, 2)
    assert tw["FAKE_other"]["confusion"]["spoof"] == [2, 1, 1]
    assert tw["REAL"]["deferral_rate_spoof"] is None  # no spoof in the REAL subset
    # Baseline forced binary at 0.0: REAL .9 .7 FP; mp3 .8 .6 .4 TP, -.1 FN; other .3 TP, three FN.
    base = report["models"]["baseline-v1"]["forced_binary"]
    assert (base["overall"]["tp"], base["overall"]["fp"], base["overall"]["tn"], base["overall"]["fn"]) == (4, 2, 4, 4)
    assert "three_way" not in report["models"]["baseline-v1"]
    for model_id in ("cnn-v1-seed1", "cnn-v1-seed2"):  # D45: score level only
        assert set(report["models"][model_id]) == {"role", "score_level"}


def test_hand_case_hypotheses_and_seeds(hand_report):
    _, report = hand_report
    h1, h5 = report["hypotheses"]["H1"], report["hypotheses"]["H5"]
    # H1 d = EER(baseline) - EER(bundle CNN) = 3/8 - 1/4 = 1/8.
    assert h1["estimate"] == pytest.approx(0.125)
    assert (h1["eer_baseline"], h1["eer_bundle_cnn"]) == (pytest.approx(0.375), pytest.approx(0.25))
    # H5 d = recall(FAKE_mp3) - recall(FAKE_other) = 1 - 1/2 = 1/2, exactly.
    assert h5["estimate"] == 0.5 and h5["estimate_exact"] == "1/2"
    assert h5["ci"]["counts"] == {"FAKE_mp3": {"tp": 4, "fn": 0}, "FAKE_other": {"tp": 2, "fn": 2}}
    # 14 clips: both intervals are wide and include 0, so d > 0 but neither is supported (the rule needs
    # the CI lower bound > 0): the "d > 0 but CI includes 0" case, end to end.
    for h in (h1, h5):
        assert h["estimate"] > 0 and h["ci"]["low"] <= 0 < h["ci"]["high"]
        assert h["conditions"]["ci_low_above_0"] is False
        assert h["outcome"] == "not supported"
    # Some resamples draw no FAKE_other group (4 of 12 groups): H5 is undefined there -> counted.
    assert h5["ci"]["n_invalid"] > 0
    # Seeds: EER 1/4, 0, 0 -> mean 1/12, sample SD sqrt(((1/6)^2 + 2 (1/12)^2) / 2) = sqrt(3/144).
    seeds = report["seeds"]
    assert seeds["eer"]["values"] == {"0": pytest.approx(0.25), "1": 0.0, "2": 0.0}
    assert seeds["eer"]["mean"] == pytest.approx(1 / 12)
    assert seeds["eer"]["sd"] == pytest.approx((3 / 144) ** 0.5)
    assert seeds["roc_auc"]["values"] == {"0": pytest.approx(39 / 48), "1": 1.0, "2": 1.0}


def test_report_numbers_agree_with_direct_harness_calls(hand_report):
    # The report wraps the harness; recomputing on the stored arrays gives the same numbers.
    run, report = hand_report
    clip_dir = run.run_dir / "clip_scores"
    scores = {m: read_clip_scores(clip_dir, m) for m in report_module.MODEL_IDS}
    y = np.array([1 if x == "FAKE" else 0 for x in scores["baseline-v1"].label])
    groups = list(scores["baseline-v1"].group_id)
    bundle, base = scores["cnn-v1-seed0"], scores["baseline-v1"]
    model = report["models"]["cnn-v1-seed0"]
    assert model["score_level"]["overall"]["eer"] == eer(y, bundle.raw_score)[0]
    assert model["score_level"]["overall"]["roc_auc"] == roc_auc(y, bundle.raw_score)
    fb = binary_metrics(y, bundle.p, POLICY.threshold)
    assert model["forced_binary"]["overall"]["f1"] == fb.f1 and model["forced_binary"]["overall"]["fp"] == fb.fp
    tw = three_way_metrics(y, list(bundle.prediction))
    assert model["three_way"]["overall"]["coverage"] == tw.coverage
    ci = bootstrap_ci(lambda i: eer(y[i], bundle.raw_score[i])[0], len(y), clusters=groups)
    assert (model["ci"]["eer"]["low"], model["ci"]["eer"]["high"], model["ci"]["eer"]["n_invalid"]) == \
        (ci.low, ci.high, ci.n_invalid)
    ci = bootstrap_ci(lambda i: binary_metrics(y[i], base.raw_score[i], BASELINE_THRESHOLD).fpr, len(y), clusters=groups)
    assert report["models"]["baseline-v1"]["ci"]["fpr"]["low"] == ci.low
    h1 = bootstrap_ci(lambda i: eer(y[i], base.raw_score[i])[0] - eer(y[i], bundle.raw_score[i])[0], len(y),
                      clusters=groups)
    assert (report["hypotheses"]["H1"]["ci"]["low"], report["hypotheses"]["H1"]["ci"]["high"]) == (h1.low, h1.high)
    summary = seed_summary({s: roc_auc(y, scores[f"cnn-v1-seed{s}"].raw_score) for s in (0, 1, 2)})
    assert (report["seeds"]["roc_auc"]["mean"], report["seeds"]["roc_auc"]["sd"]) == (summary.mean, summary.sd)


def test_checks_inputs_and_conventions_are_recorded(hand_report):
    run, report = hand_report
    assert report["checks"] and all(c["passed"] for c in report["checks"].values())
    for key in ("run_report.model_set.roles", "claim.sha256", "registry.records", "manifest.sha256_stage10",
                "clip_scores.cnn-v1-seed0.csv_sha256", "same_clips.cnn-v1-seed2.group_id", "f2.counts",
                "risk_policy.sha256", "baseline.artifact_sha256.d41", "rederive.bundle.p_bytes"):
        assert key in report["checks"], key
    registry = run.registry.read_bytes()
    assert report["inputs"]["registry"] == {"sha256": _sha(registry), "lines": 2}  # started + completed
    assert report["report_code_commit"] == REPORT_COMMIT
    assert report["run"]["split"] == SPLIT and report["run"]["bundle_model_id"] == "cnn-v1-seed0"
    assert report["conventions"]["decision"] == "D57"
    assert sorted(k for k in report["conventions"] if k != "decision") == sorted(report_module.CONVENTIONS)


# --- Outcome rules, collapsed intervals -----------------------------------------------------------

def _big_rows(n_per: int, other_hit: int):
    """Many single-clip groups. CNN: REAL raw in [-6, -5), FAKE_mp3 in [2, 3), FAKE_other in [-2, -1),
    of which the first other_hit are moved to [1, 2): every FAKE ranks above every REAL, so EER 0; recall
    FAKE_mp3 = 1 and recall FAKE_other = other_hit / n_per. Baseline: alternating overlap, EER ~0.5."""
    table = []
    for i in range(n_per * 2):
        table.append((f"REAL/r{i:03d}.wav", "REAL", f"gr{i}", -6 + i / (4 * n_per), (i % 7) / 7 - 0.4))
    for i in range(n_per):
        table.append((f"FAKE/m{i:03d}.wav", "FAKE_mp3", f"gm{i}", 2 + i / (2 * n_per), (i % 5) / 5 - 0.45))
        raw = (1 + i / (2 * n_per)) if i < other_hit else (-2 + i / (2 * n_per))
        table.append((f"FAKE/o{i:03d}.wav", "FAKE_other", f"go{i}", raw, (i % 3) / 3 - 0.3))
    return _rows_from(table)


def test_supported_case_and_collapsed_intervals(make):
    # 80 clips, 80 groups. CNN EER 0 in every resample -> EER CI [0, 0], flagged collapsed. H5: recall 1
    # vs 0 -> d = 1 in every valid resample -> CI [1, 1] collapsed, d >= 0.05 and low > 0 -> supported.
    # H1: baseline EER well above 0 in every resample, CNN 0 -> d > 0 and CI low > 0 -> supported.
    run = make(_big_rows(20, other_hit=0))
    report = json.loads(_json_text(report_module.build_report(**run.kwargs())))
    ci = report["models"]["cnn-v1-seed0"]["ci"]["eer"]
    assert (ci["estimate"], ci["low"], ci["high"], ci["collapsed"]) == (0.0, 0.0, 0.0, True)
    assert ci["counts"] == {"n_spoof": 40, "n_bonafide": 40}
    h5 = report["hypotheses"]["H5"]
    assert (h5["estimate"], h5["ci"]["low"], h5["ci"]["collapsed"], h5["outcome"]) == (1.0, 1.0, True, "supported")
    h1 = report["hypotheses"]["H1"]
    assert h1["estimate"] > 0 and h1["ci"]["low"] > 0 and h1["outcome"] == "supported"
    assert report["models"]["baseline-v1"]["ci"]["eer"]["collapsed"] is False


def test_h5_below_five_points_is_not_supported(make):
    # recall FAKE_mp3 = 25/25 = 1, FAKE_other = 24/25 -> d = 1/25 = 0.04 < 0.05: not "clearly lower",
    # whatever the CI.
    run = make(_big_rows(25, other_hit=24))
    h5 = report_module.build_report(**run.kwargs())["hypotheses"]["H5"]
    assert h5["estimate_exact"] == "1/25"
    assert h5["conditions"]["d_at_least_0.05"] is False
    assert h5["outcome"] == "not supported"


def test_h5_five_points_is_compared_exactly(make):
    # recall 17/20 vs 16/20: in floats 0.85 - 0.8 = 0.04999999999999993 < 0.05, but the difference is
    # exactly 5 points, and Protocol §7 says ">= 5 points": the rule is compared on the counts.
    assert 0.85 - 0.8 < 0.05
    rows = _big_rows(20, other_hit=16)
    for r in rows:  # move 3 FAKE_mp3 clips below the threshold (raw -1.9..): recall 17/20
        if r["clip_id"] in ("FAKE/m000.wav", "FAKE/m001.wav", "FAKE/m002.wav"):
            r["cnn-v1-seed0"] = -1.9 + int(r["clip_id"][-7:-4]) / 100
    h5 = report_module.build_report(**make(rows).kwargs())["hypotheses"]["H5"]
    assert (h5["recall_FAKE_mp3"], h5["recall_FAKE_other"]) == (0.85, 0.8)
    assert h5["estimate_exact"] == "1/20"
    assert h5["conditions"]["d_at_least_0.05"] is True
    assert h5["outcome"] == ("supported" if h5["ci"]["low"] > 0 else "not supported")


# --- Integrity checks: each tamper raises with its own check name and writes nothing ---------------

def _set(path_keys, value):
    def edit(data):
        target = data
        for key in path_keys[:-1]:
            target = target[key]
        target[path_keys[-1]] = value
    return edit


TAMPER = {
    # 1. run_report.json
    "run_report.schema_version": lambda r: r.edit_json(r.run_dir / "run_report.json", _set(["schema_version"], 2)),
    "run_report.dataset": lambda r: r.edit_json(r.run_dir / "run_report.json", _set(["dataset"], "asvspoof")),
    "run_report.evaluation_purpose": lambda r: r.edit_json(r.run_dir / "run_report.json",
                                                           _set(["evaluation_purpose"], "stage14_robustness")),
    "run_report.run_id": lambda r: r.edit_json(r.run_dir / "run_report.json", _set(["run_id"], "XYZ")),
    "run_report.code_commit": lambda r: r.edit_json(r.run_dir / "run_report.json", _set(["code_commit"], "abc1234")),
    "run_report.model_set.ids": lambda r: r.edit_json(
        r.run_dir / "run_report.json", lambda d: d["model_set"].pop("cnn-v1-seed2")),
    "run_report.model_set.roles": lambda r: r.edit_json(
        r.run_dir / "run_report.json", _set(["model_set", "cnn-v1-seed1", "role"], "cnn_bundle_seed")),
    "run_report.bundle_seed": lambda r: r.edit_json(r.run_dir / "run_report.json", _set(["bundle_seed"], True)),
    "run_report.bundle_seed_role": lambda r: r.edit_json(r.run_dir / "run_report.json", _set(["bundle_seed"], 1)),
    # 2. claim.json
    "claim.split": lambda r: r.edit_json(r.run_dir / "claim.json", _set(["split"], "val-B")),
    "claim.sha256": lambda r: r.edit_json(r.run_dir / "claim.json", _set(["counts"], {})),
    # 4. score files (the CSV changed after the run: its sidecar no longer matches)
    "clip_scores.cnn-v1-seed1.read": lambda r: (r.run_dir / "clip_scores" / "cnn-v1-seed1.csv").write_bytes(
        (r.run_dir / "clip_scores" / "cnn-v1-seed1.csv").read_bytes().replace(b"-1.0", b"-1.5", 1)),
    "clip_scores.baseline-v1.csv_sha256": lambda r: r.edit_json(
        r.run_dir / "run_report.json", _set(["clip_scores", "files", "baseline-v1", "csv_sha256"], "f" * 64)),
    "clip_scores.cnn-v1-seed0.sidecar_sha256": lambda r: r.edit_json(
        r.run_dir / "run_report.json", _set(["clip_scores", "files", "cnn-v1-seed0", "sidecar_sha256"], "f" * 64)),
    "clip_scores.cnn-v1-seed2.n_rows": lambda r: r.edit_json(
        r.run_dir / "run_report.json", _set(["clip_scores", "files", "cnn-v1-seed2", "n_rows"], 15)),
    "clip_scores.folder": lambda r: r.edit_json(r.run_dir / "run_report.json", _set(["clip_scores", "folder"], "x")),
    # 6. F2 join: the run report's counts disagree with the stored files
    "f2.counts": lambda r: r.edit_json(r.run_dir / "run_report.json", _set(["counts", "subgroup", "REAL"], 7)),
    # 7. thresholds and evidence
    "risk_policy.sha256": lambda r: r.policy.write_bytes(r.policy.read_bytes() + b"\n"),
    "stage10.bundle_seed": lambda r: r.edit_json(r.stage10 / "run_report.json", _set(["bundle_seed"], 2)),
    "stage10.checkpoint_sha256": lambda r: r.edit_json(r.stage10 / "run_report.json",
                                                       _set(["checkpoint_sha256"], "9" * 64)),
    "baseline.artifact_sha256.stage7": lambda r: r.edit_json(r.stage7 / "run_report.json",
                                                             _set(["artifact_sha256"], "9" * 64)),
    "baseline.threshold": lambda r: r.edit_json(r.stage7 / "run_report.json", _set(["val_b_threshold"], "0.0")),
    # 8. a Stage 7 threshold other than the one the stored predictions used
    "rederive.baseline.prediction_mismatches": lambda r: r.edit_json(r.stage7 / "run_report.json",
                                                                     _set(["val_b_threshold"], 0.5)),
    # the manifest
    "manifest.sha256_stage10": lambda r: r.edit_json(r.stage10 / "run_report.json",
                                                     _set(["f2_manifest_sha256"], "9" * 64)),
}

# Tampers that need the run folder written differently (options of Run).
BUILD_TAMPER = {
    "registry.records": {"complete": False},  # an incomplete run is refused
    "registry.completed.results_ref": {"results_ref_completed": "elsewhere/run_report.json"},
    "clip_scores.cnn-v1-seed1.run_id": {"sidecar": {"cnn-v1-seed1": {"run_id": "0" * 32}}},
    "clip_scores.baseline-v1.model_artifact_sha256": {"sidecar": {"baseline-v1": {"model_artifact_sha256": "7" * 64}}},
    "clip_scores.cnn-v1-seed0.manifest_sha256": {"sidecar": {"cnn-v1-seed0": {"manifest_sha256": "6" * 64}}},
    "clip_scores.cnn-v1-seed0.evaluation_purpose": {"sidecar": {"cnn-v1-seed0": {"evaluation_purpose":
                                                                                  "stage14_robustness"}}},
    "same_clips.cnn-v1-seed1.group_id": {"columns": {"cnn-v1-seed1": {"group_id": ["g-x"] * 14}}},
    "same_clips.cnn-v1-seed2.label": {"columns": {"cnn-v1-seed2": {"label": ["REAL"] * 14}}},
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
    lines[0] = lines[0].replace(b'"stage11_test"', b'"stage14_robustness"')  # breaks the hash chain
    run.registry.write_bytes(b"\n".join(lines))
    _assert_refused(run, "registry.load", tmp_path)


def test_unapproved_manifest_is_refused(make, tmp_path, monkeypatch):
    run = make()
    monkeypatch.setattr(report_module, "APPROVED_F2_MANIFEST_SHA256", report_module.F2_MANIFEST_SHA256)
    _assert_refused(run, "manifest.sha256_approved", tmp_path)


def test_every_check_name_is_distinct():
    assert not set(TAMPER) & set(BUILD_TAMPER)


# --- Output ---------------------------------------------------------------------------------------

def test_cli_twice_gives_byte_identical_files_rendered_from_the_json(make, tmp_path, capsys):
    # Two runs on the same inputs, one through the CLI and one through the functions: identical bytes.
    run = make()
    report_module.main(run.argv(tmp_path / "cli"))
    cli = (tmp_path / "cli" / "metrics_report.json", tmp_path / "cli" / "metrics_report.md")
    assert capsys.readouterr().out.split() == [str(x) for x in cli]
    direct = report_module.write_report(report_module.build_report(**run.kwargs()), tmp_path / "direct")
    for x, y in zip(cli, direct):
        assert x.read_bytes() == y.read_bytes()
    json_path, md_path = cli
    text = json_path.read_text(encoding="utf-8")
    assert text.endswith("}\n") and "NaN" not in text and "null" in text  # strict JSON, undefined rates null
    assert json.loads(text, parse_constant=lambda c: pytest.fail(c)) is not None
    markdown = md_path.read_text(encoding="utf-8")
    assert markdown == report_module.render_markdown(json.loads(text))
    assert "**Reported whatever the outcome (pre-registered).**" in markdown
    assert "| H5 | 0.5000 |" in markdown and "undefined" in markdown


def test_markdown_flags_collapsed_intervals_and_invalid_resamples(hand_report):
    _, report = hand_report
    markdown = report_module.render_markdown(report)
    n_invalid = report["hypotheses"]["H5"]["ci"]["n_invalid"]
    assert n_invalid > 0 and f"| {n_invalid} |" in markdown
    report["models"]["baseline-v1"]["ci"]["eer"].update(low=0.0, high=0.0, collapsed=True)
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
    code = "import sys, src.evaluation.stage11_report; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    assert out.stdout.strip() == "False"


def test_constants_and_formats_equal_the_runners():
    from tests._torch import require_torch
    require_torch()
    import src.evaluation.stage11_run as run_module

    for name in ("DATASET", "PURPOSE", "SEEDS", "BASELINE_ID", "CNN_IDS", "MODEL_IDS", "BASELINE_ARTIFACT_SHA256",
                 "ROLE_BASELINE", "ROLE_BUNDLE", "ROLE_OTHER", "CLIP_SCORES_DIR", "RUN_REPORT", "CLAIM_JSON",
                 "STAGE_REPORT"):
        assert getattr(report_module, name) == getattr(run_module, name), name
    assert report_module.RUN_SCHEMA_VERSION == run_module.SCHEMA_VERSION
    # The fixture's counts and model set equal the runner's own functions on the same rows.
    rows = _rows_from(HAND)
    frame = pd.DataFrame(rows)
    assert _counts(rows) == run_module._counts(frame)
    models = run_module.Models(cnn={}, baseline=None, policy=POLICY, bundle_seed=0, batch_size=4,
                               artifact_sha256={"baseline-v1": report_module.BASELINE_ARTIFACT_SHA256,
                                                **{f"cnn-v1-seed{s}": SEED_HASHES[s] for s in (0, 1, 2)}})
    expected = {  # the model set literal the fixture writes
        "baseline-v1": {"role": "baseline", "artifact_sha256": report_module.BASELINE_ARTIFACT_SHA256},
                **{f"cnn-v1-seed{s}": {"role": "cnn_bundle_seed" if s == 0 else "cnn_other_seed",
                                       "artifact_sha256": SEED_HASHES[s]} for s in (0, 1, 2)}}
    assert run_module._model_set(models) == expected


def _dict_keys(function: str, variable: str) -> list[str]:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    node = next(n for n in ast.walk(func) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets))
    return [k.value for k in node.value.keys]


def test_fixture_key_lists_equal_the_runners():
    # The fixture writes claim.json and run_report.json with the runner's exact keys (read from its source).
    assert _dict_keys("claim", "record") == CLAIM_KEYS
    assert _dict_keys("score", "report") == RUN_REPORT_KEYS


def test_report_module_is_not_test_split_access_for_the_guard():
    # D31: later uses read the stored files, so the module needs no allow-list entry; the guard finds
    # no test-split request in it (it never names the split) and ALLOWED is unchanged.
    rel = "src/evaluation/stage11_report.py"
    assert file_hits(rel, (REPO_ROOT / rel).read_text(encoding="utf-8")) == []
    assert ALLOWED == ("src/evaluation/stage11_run.py",)


def test_fraction_rule_constant():
    assert report_module.H5_MIN_DIFFERENCE == Fraction(1, 20)
