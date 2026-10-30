"""Run registry tests (D31: Stage 11 runs once; key amended by D41) on temporary files only (no dataset
access)."""
from __future__ import annotations

import hashlib
import json

import pytest

from src.evaluation.run_registry import EVALUATION_PURPOSES, GENESIS_SHA256, RunRecord, RunRegistry

COMMIT = "a" * 40
PURPOSE = "stage11_test"
MODELS = ("cnn_seed1", "svm_baseline")


def _lines(path):
    return path.read_bytes().splitlines(keepends=True)


def test_second_claim_same_run_raises(tmp_path):
    # Same dataset, split and model set: the second claim must be refused, whatever the order in
    # which the models are listed, because model_set is stored sorted (one run per model set).
    registry = RunRegistry(tmp_path / "runs.jsonl")
    registry.claim("for", "test", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    with pytest.raises(ValueError, match="already run"):
        registry.claim("for", "test", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    with pytest.raises(ValueError, match="new decision"):
        registry.claim("for", "test", tuple(reversed(MODELS)), "b" * 40, evaluation_purpose=PURPOSE)


def test_crashed_run_still_blocks(tmp_path):
    # Claim before run: a run that crashed after the claim has a started record and no completed
    # one. It may already have seen test results, so it still blocks a rerun.
    registry = RunRegistry(tmp_path / "runs.jsonl")
    registry.claim("for", "test", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    assert [r.status for r in registry.records()] == ["started"]
    with pytest.raises(ValueError, match="already run"):
        registry.claim("for", "test", MODELS, COMMIT, evaluation_purpose=PURPOSE)


def test_other_model_set_or_split_can_be_claimed(tmp_path):
    # The key is (dataset, split, evaluation_purpose, model_set): changing any part is a different run.
    registry = RunRegistry(tmp_path / "runs.jsonl")
    registry.claim("for", "test", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    registry.claim("for", "test", ("cnn_seed1",), COMMIT, evaluation_purpose=PURPOSE)
    registry.claim("for", "val_b", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    assert len(registry.records()) == 3


def test_complete_once(tmp_path):
    # complete() needs a started run and works once; a second complete or an unknown run_id would
    # make the record of what was run ambiguous.
    registry = RunRegistry(tmp_path / "runs.jsonl")
    run_id = registry.claim("for", "test", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    registry.complete(run_id, "results/stage11.json")
    done = registry.records()[-1]
    assert (done.status, done.run_id, done.results_ref) == ("completed", run_id, "results/stage11.json")
    assert done.model_set == tuple(sorted(MODELS)) and done.code_commit == COMMIT
    assert done.evaluation_purpose == PURPOSE  # copied from the started record (D41)
    with pytest.raises(ValueError, match="already completed"):
        registry.complete(run_id, "results/again.json")
    with pytest.raises(ValueError, match="unknown run_id"):
        registry.complete("f" * 32, "results/x.json")
    # A completed run blocks a new claim too.
    with pytest.raises(ValueError, match="already run"):
        registry.claim("for", "test", MODELS, COMMIT, evaluation_purpose=PURPOSE)


def test_append_only(tmp_path):
    # Each write only adds bytes at the end: the earlier content is a byte-exact prefix of the
    # new content, so no write ever rewrites history.
    path = tmp_path / "runs.jsonl"
    registry = RunRegistry(path)
    run_a = registry.claim("for", "test", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    before = path.read_bytes()
    registry.complete(run_a, "results/a.json")
    after_complete = path.read_bytes()
    assert after_complete.startswith(before) and len(after_complete) > len(before)
    registry.claim("for", "val_b", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    after_claim = path.read_bytes()
    assert after_claim.startswith(after_complete) and len(after_claim) > len(after_complete)
    # Lines are newline-terminated and the chain starts at the genesis value.
    lines = _lines(path)
    assert len(lines) == 3 and all(line.endswith(b"\n") for line in lines)
    assert json.loads(lines[0])["prev_sha256"] == GENESIS_SHA256
    assert json.loads(lines[1])["prev_sha256"] == hashlib.sha256(lines[0]).hexdigest()


def _three_line_registry(path):
    registry = RunRegistry(path)
    run_id = registry.claim("for", "test", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    registry.complete(run_id, "results/a.json")
    registry.claim("for", "val_b", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    return _lines(path)


@pytest.mark.parametrize("tamper", ["edit", "delete", "reorder"])
def test_tampering_detected(tmp_path, tamper):
    # The hash chain links each line to the exact bytes of the previous one, so changing a value,
    # deleting a line (e.g. to erase a claim) or swapping two lines breaks the chain on load.
    path = tmp_path / "runs.jsonl"
    lines = _three_line_registry(path)
    if tamper == "edit":
        lines[0] = lines[0].replace(b'"split":"test"', b'"split":"tesx"')
    elif tamper == "delete":
        del lines[0]
    else:
        lines[1], lines[2] = lines[2], lines[1]
    path.write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="edited"):
        RunRegistry(path)


def test_truncation_is_not_detected_by_the_chain(tmp_path):
    # Limitation (D31): each line only hashes the line before it, so dropping the last line leaves
    # a valid chain and the registry still loads. The mitigation is procedural: the file's SHA-256
    # and line count are recorded right after Stage 11, and a truncated file no longer matches.
    path = tmp_path / "runs.jsonl"
    registry = RunRegistry(path)
    run_id = registry.claim("for", "test", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    registry.complete(run_id, "results/a.json")
    recorded_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = _lines(path)
    path.write_bytes(b"".join(lines[:-1]))
    assert [r.status for r in RunRegistry(path).records()] == ["started"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() != recorded_sha256
    assert len(_lines(path)) != len(lines)


def _rewrite_first(path, change):
    # Change the first record and write it back; first line, so the chain check itself still passes
    # and only the strict parsing can reject it.
    data = json.loads(path.read_bytes().splitlines()[0])
    change(data)
    path.write_bytes((json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode())


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(extra="x"),  # unknown key
        lambda d: d.pop("timestamp"),  # missing key
        lambda d: d.update(code_commit="A" * 40),  # uppercase hex
        lambda d: d.update(code_commit="abc123"),  # short commit
        lambda d: d.update(status="running"),  # bad status
        lambda d: d.update(model_set=["svm_baseline", "cnn_seed1"]),  # not stored sorted
        lambda d: d.update(timestamp="2026-10-17T12:00:00+02:00"),  # not UTC
        lambda d: d.update(results_ref="results/x.json"),  # started with a results_ref
        lambda d: d.update(evaluation_purpose="stage12_test"),  # purpose not in the D41 set
        lambda d: d.update(evaluation_purpose=""),  # empty purpose
        lambda d: d.update(evaluation_purpose="Stage11_Test"),  # case variant
        lambda d: d.update(evaluation_purpose=None),  # not a string
        lambda d: d.pop("evaluation_purpose"),  # old D31 schema (no purpose)
    ],
)
def test_malformed_record_raises(tmp_path, change):
    # Strict parsing: anything the registry would never write is rejected, never guessed.
    path = tmp_path / "runs.jsonl"
    RunRegistry(path).claim("for", "test", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    _rewrite_first(path, change)
    with pytest.raises(ValueError, match="registry line 1"):
        RunRegistry(path)


@pytest.mark.parametrize("content", [b"not json\n", b"[1, 2]\n", b"\n", b'{"run_id": "x"}'])
def test_malformed_line_raises(tmp_path, content):
    # Not JSON, not an object, an empty line, or a last line without "\n" (truncated write).
    path = tmp_path / "runs.jsonl"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        RunRegistry(path)


def test_bad_claim_arguments_raise(tmp_path):
    # Bad values are refused at claim time, before anything is written.
    path = tmp_path / "runs.jsonl"
    registry = RunRegistry(path)
    with pytest.raises(ValueError, match="code_commit"):
        registry.claim("for", "test", MODELS, "A" * 40, evaluation_purpose=PURPOSE)
    with pytest.raises(ValueError, match="model_set"):
        registry.claim("for", "test", "cnn_seed1", COMMIT, evaluation_purpose=PURPOSE)
    with pytest.raises(ValueError, match="duplicate"):
        registry.claim("for", "test", ("cnn", "cnn"), COMMIT, evaluation_purpose=PURPOSE)
    with pytest.raises(ValueError, match="split"):
        registry.claim("for", "", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    assert not path.exists()


def test_state_lives_in_the_file(tmp_path):
    # A new RunRegistry on the same path (e.g. a restarted notebook) sees the earlier claim, so the
    # protection does not depend on keeping one object alive.
    path = tmp_path / "runs.jsonl"
    run_id = RunRegistry(path).claim("for", "test", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    fresh = RunRegistry(path)
    assert [r.run_id for r in fresh.records()] == [run_id]
    assert isinstance(fresh.records()[0], RunRecord)
    with pytest.raises(ValueError, match="already run"):
        fresh.claim("for", "test", MODELS, COMMIT, evaluation_purpose=PURPOSE)
    fresh.complete(run_id, "results/a.json")
    assert RunRegistry(path).records()[-1].status == "completed"


# --- D41: evaluation_purpose in the key -----------------------------------------------------------


def test_purposes_are_exactly_the_d41_set():
    # D41 fixes the purposes in code; this test fails if one is added without a new decision.
    assert EVALUATION_PURPOSES == frozenset({"stage11_test", "stage14_robustness"})


@pytest.mark.parametrize(
    "purpose", ["", " ", "stage13_external", "Stage11_Test", "STAGE11_TEST", " stage11_test", None, 11]
)
def test_unknown_purpose_refused_at_claim(tmp_path, purpose):
    # A free-form purpose would let a caller invent a new key and bypass the one-run guard (ADR 0010),
    # so anything outside the set is refused, exact spelling and case only, and nothing is written.
    path = tmp_path / "runs.jsonl"
    with pytest.raises(ValueError, match="new decision"):
        RunRegistry(path).claim("for", "test", MODELS, COMMIT, evaluation_purpose=purpose)
    assert not path.exists()


def test_same_model_set_under_both_purposes(tmp_path):
    # Stage 11 and Stage 14 each get one run of the same model set (D41): two allowed claims.
    registry = RunRegistry(tmp_path / "runs.jsonl")
    registry.claim("for", "test", MODELS, COMMIT, evaluation_purpose="stage11_test")
    registry.claim("for", "test", MODELS, COMMIT, evaluation_purpose="stage14_robustness")
    assert [r.evaluation_purpose for r in registry.records()] == ["stage11_test", "stage14_robustness"]


@pytest.mark.parametrize("complete_first", [False, True])
@pytest.mark.parametrize("purpose", sorted(EVALUATION_PURPOSES))
def test_second_claim_same_purpose_refused(tmp_path, purpose, complete_first):
    # One claim per key under each purpose, whether the first run is started or completed.
    registry = RunRegistry(tmp_path / "runs.jsonl")
    run_id = registry.claim("for", "test", MODELS, COMMIT, evaluation_purpose=purpose)
    if complete_first:
        registry.complete(run_id, "results/a.json")
    with pytest.raises(ValueError, match="already run"):
        registry.claim("for", "test", tuple(reversed(MODELS)), COMMIT, evaluation_purpose=purpose)


def test_purpose_is_keyword_only_and_required(tmp_path):
    # Keyword-only: a positional purpose (or a forgotten one) is a TypeError before anything is
    # written, so the purpose can never be taken from a misplaced positional argument.
    path = tmp_path / "runs.jsonl"
    registry = RunRegistry(path)
    with pytest.raises(TypeError):
        registry.claim("for", "test", MODELS, COMMIT, "stage11_test")
    with pytest.raises(TypeError):
        registry.claim("for", "test", MODELS, COMMIT)
    assert not path.exists()


def _write_chain(path, records):
    # Write records exactly as the registry does (sorted compact JSON, chained), so only the replay
    # rules can reject the file.
    prev, out = GENESIS_SHA256, b""
    for data in records:
        line = (json.dumps({**data, "prev_sha256": prev}, sort_keys=True, separators=(",", ":"))
                + "\n").encode()
        out += line
        prev = hashlib.sha256(line).hexdigest()
    path.write_bytes(out)


def test_completed_with_other_purpose_rejected_on_load(tmp_path):
    # The completed/started match check includes the purpose: a completed line that moves a run to
    # the other purpose is a file the code would never write.
    path = tmp_path / "runs.jsonl"
    RunRegistry(path).claim("for", "test", MODELS, COMMIT, evaluation_purpose="stage11_test")
    start = json.loads(path.read_bytes())
    start.pop("prev_sha256")
    done = {**start, "status": "completed", "results_ref": "results/a.json",
            "evaluation_purpose": "stage14_robustness"}
    _write_chain(path, [start, done])
    with pytest.raises(ValueError, match="does not match its started record"):
        RunRegistry(path)


def test_second_claim_same_key_rejected_on_load(tmp_path):
    # Load-time replay uses the D41 key: two started lines with the same key (different run_ids)
    # are rejected; the same lines under different purposes load.
    path = tmp_path / "runs.jsonl"
    RunRegistry(path).claim("for", "test", MODELS, COMMIT, evaluation_purpose="stage11_test")
    first = json.loads(path.read_bytes())
    first.pop("prev_sha256")
    _write_chain(path, [first, {**first, "run_id": "f" * 32, "evaluation_purpose": "stage14_robustness"}])
    assert len(RunRegistry(path).records()) == 2
    _write_chain(path, [first, {**first, "run_id": "f" * 32}])
    with pytest.raises(ValueError, match="second claim"):
        RunRegistry(path)


def test_chain_detects_purpose_edit(tmp_path):
    # Changing the purpose of an earlier line (e.g. to free the Stage 11 key) breaks the chain.
    path = tmp_path / "runs.jsonl"
    lines = _three_line_registry(path)
    lines[0] = lines[0].replace(b'"stage11_test"', b'"stage14_robustness"')
    path.write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="edited"):
        RunRegistry(path)
