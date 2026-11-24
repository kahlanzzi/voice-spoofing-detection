"""Static safety contract for the Stage 14 v1 Colab hand-off notebook."""
from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks" / "11_stage14_v1_for_test.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code():
    return "\n".join(
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_is_clean_and_defaults_to_read_only_check_mode():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    code = _code()
    assert "MODE = 'check'" in code
    assert "COMMIT = 'REPLACE_WITH_APPROVED_COMMIT'" in code
    assert "assert MODE == 'official', 'check mode: stop before the first claim'" in code


def test_exact_commit_and_d58_runtime_are_asserted_before_any_claim():
    code = _code()
    claim = code.index("RUNNER, 'claim'")
    for required in (
        "git', 'checkout', '--detach', COMMIT",
        "assert HEAD == COMMIT and STATUS == ''",
        "COLAB_RUNTIME_VERSION = '2026.10'",
        "platform.python_version() == '3.13.16'",
        "torch.__version__ == '2.11.0+cu130'",
        "torch.cuda.is_available() and 'T4' in GPU",
        "'tests/evaluation', 'tests/governance'",
        "approval['approved_commit'] == COMMIT",
        "set(approval['approvals']) == {'Donia', 'Iheb', 'Khalil', 'Malak'}",
    ):
        assert required in code
        assert code.index(required) < claim


def test_read_only_checks_cover_both_v1_model_sets_before_the_claims():
    code = _code()
    check = code.index("RUNNER, 'check'")
    first_claim = code.index("RUNNER, 'claim'")
    assert check < first_claim
    assert "for kind in ('baseline', 'cnn_bundle')" in code
    assert "Both read-only preflights passed" in code
    assert "--colab-runtime-version" in code
    assert "--stage11-run-report" in code and "run_report.json" in code
    assert "--f4-record" in code and "stage12_bundle/f4_record.json" in code
    assert "--mfcc-reference" in code and "--cache-a" in code and "--cache-b" in code
    assert "RUNNER, 'time-condition'" in code
    assert "'split': 'val-A'" in code
    assert "estimated_two_lifecycle_seconds" in code


def test_check_mode_extracts_val_a_only_and_official_adds_test():
    code = _code()
    extraction = code.index("allowed_splits = {'val-A'} if MODE == 'check'")
    first_claim = code.index("RUNNER, 'claim'")
    assert extraction < first_claim
    assert "else {'val-A', 'test'}" in code
    assert "assert set(selected) == requested" in code
    assert "row['split'] == 'test'" in code
    assert "extractall(DATA_ROOT" in code
    assert "source.extractall(DATA_ROOT, filter='data')" not in code


def test_official_flow_has_two_separate_claim_score_lifecycles():
    code = _code()
    assert code.count("RUNNER, 'claim'") == 2
    assert code.count("RUNNER, 'score'") == 2
    assert "'--model-kind', 'baseline'" in code
    assert "'--model-kind', 'cnn_bundle'" in code
    assert "BASELINE_OUT = RUN_DEST / 'baseline-v1'" in code
    assert "BUNDLE_OUT = RUN_DEST / 'bundle-v1'" in code
    assert code.count("drive.flush_and_unmount()") >= 3
    assert code.count("force_remount=True") >= 3
    assert code.count("'--approval-record', str(APPROVAL_RECORD)") == 4
    assert "assert sha256(REGISTRY) == BASELINE_CLAIM['registry_sha256']" in code
    assert "assert sha256(REGISTRY) == BUNDLE_CLAIM['registry_sha256']" in code


def test_scope_and_outputs_stay_within_the_reviewed_stage14_contract():
    text = NOTEBOOK.read_text(encoding="utf-8").lower()
    code = _code().lower()
    assert "deep-voice" not in code and "deepvoice" not in code
    assert "wandb" not in code
    assert "processed' / 'metadata' / 'stage14' / 'v1'" in code
    assert "'scoring'] == {'device': 'cpu', 'passes': 2}" in code
    assert "len(report['conditions']) == 9" in code
    assert "per-clip condition score files remain on drive" in text
    assert "do not commit per-clip score files" in text
