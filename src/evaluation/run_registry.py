"""Append-only run registry for test-split evaluations: "Stage 11 runs once" (D31, amended by D41).

Conventions (D31, key amended by D41):
- One JSON Lines file, path given by the caller. Each line is one RunRecord as a JSON object with
  sorted keys, compact separators, UTF-8, terminated by "\\n".
- A run is identified by (dataset, split, evaluation_purpose, model_set); model_set is stored
  sorted, so the order in which the caller lists the models does not matter.
- evaluation_purpose is closed in code (D41): exactly the values in EVALUATION_PURPOSES
  ("stage11_test", "stage14_robustness"). Any other value, including "", another spelling or
  another case, is refused when claiming and when loading; a new purpose needs a new decision.
  claim() takes it as a required keyword-only argument, so it cannot be passed by position by
  mistake (e.g. swapped with the model set or the commit).
- Claim before run: claim() appends a "started" record before the evaluation runs, and complete()
  appends a "completed" record after it, copying the purpose from the started record. Any record
  for the same key, started or completed, blocks a second claim, so a crashed run still blocks a
  rerun: a second run needs a new decision.
- Hash chain: each record's prev_sha256 is the SHA-256 of the previous line's exact bytes
  (including its "\\n"); the first record has "0" * 64. Loading checks the whole chain, so editing,
  deleting or reordering an earlier line is detected.
- Writing only appends one line; the file is never rewritten. State lives in the file, not in
  memory: every call re-reads and re-checks it.

Limitation: the code protects one copy of the file. Someone can delete the file, or start another
copy, and the code cannot see it. The team-level guarantee is that the real registry is committed
through a reviewed PR right after Stage 11. The hash chain also cannot detect removal of the last
line(s): a truncated file is still a valid chain. Mitigation (procedure): right after Stage 11, the
registry's SHA-256 and line count are recorded in the project-log entry and the results PR, so a
shortened file no longer matches.

All functions validate their input and raise ValueError on anything ambiguous (AGENTS.md code
rules: fail loudly).
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path

STARTED = "started"
COMPLETED = "completed"
GENESIS_SHA256 = "0" * 64
# D41: the only evaluation purposes; adding one needs a new decision.
EVALUATION_PURPOSES = frozenset({"stage11_test", "stage14_robustness"})

_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX32 = re.compile(r"[0-9a-f]{32}")


@dataclass(frozen=True)
class RunRecord:
    """One registry line. model_set is sorted; results_ref is "" while the run is started."""

    run_id: str
    status: str
    dataset: str
    split: str
    evaluation_purpose: str
    model_set: tuple[str, ...]
    code_commit: str
    timestamp: str
    results_ref: str
    prev_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not _HEX32.fullmatch(self.run_id):
            raise ValueError(f"run_id must be 32 lowercase hex chars (uuid4 hex), got {self.run_id!r}")
        if self.status not in (STARTED, COMPLETED):
            raise ValueError(f"status must be {STARTED!r} or {COMPLETED!r}, got {self.status!r}")
        _check_text("dataset", self.dataset)
        _check_text("split", self.split)
        _check_purpose(self.evaluation_purpose)
        _check_model_set(self.model_set)
        if tuple(sorted(self.model_set)) != self.model_set:
            raise ValueError(f"model_set must be stored sorted, got {self.model_set!r}")
        _check_commit(self.code_commit)
        _check_timestamp(self.timestamp)
        if not isinstance(self.results_ref, str):
            raise ValueError(f"results_ref must be a string, got {self.results_ref!r}")
        if self.status == STARTED and self.results_ref != "":
            raise ValueError(f"results_ref must be '' for a started record, got {self.results_ref!r}")
        if self.status == COMPLETED and not self.results_ref.strip():
            raise ValueError("results_ref must be non-empty for a completed record")
        if not isinstance(self.prev_sha256, str) or not _HEX64.fullmatch(self.prev_sha256):
            raise ValueError(f"prev_sha256 must be 64 lowercase hex chars, got {self.prev_sha256!r}")

    @property
    def key(self) -> tuple[str, str, str, tuple[str, ...]]:
        """(dataset, split, evaluation_purpose, model_set): one run per key (D31, amended by D41)."""
        return (self.dataset, self.split, self.evaluation_purpose, self.model_set)


def _check_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _check_purpose(purpose: object) -> None:
    if not isinstance(purpose, str) or purpose not in EVALUATION_PURPOSES:
        raise ValueError(
            f"evaluation_purpose must be one of {sorted(EVALUATION_PURPOSES)}, got {purpose!r}; "
            "a new purpose needs a new decision (D41)"
        )


def _check_model_set(model_set: object) -> None:
    if not isinstance(model_set, tuple) or not model_set:
        raise ValueError(f"model_set must be a non-empty tuple of strings, got {model_set!r}")
    for model in model_set:
        _check_text("each model in model_set", model)
    if len(set(model_set)) != len(model_set):
        raise ValueError(f"model_set has duplicate models: {model_set!r}")


def _check_commit(code_commit: object) -> None:
    if not isinstance(code_commit, str) or not _HEX40.fullmatch(code_commit):
        raise ValueError(f"code_commit must be 40 lowercase hex chars, got {code_commit!r}")


def _check_timestamp(timestamp: object) -> None:
    if not isinstance(timestamp, str):
        raise ValueError(f"timestamp must be an ISO 8601 UTC string, got {timestamp!r}")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        raise ValueError(f"timestamp is not ISO 8601: {timestamp!r}") from None
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"timestamp must be in UTC (offset +00:00), got {timestamp!r}")


_FIELD_NAMES = frozenset(f.name for f in fields(RunRecord))


def _encode(record: RunRecord) -> bytes:
    """The exact line bytes for a record: sorted-key compact JSON, UTF-8, newline-terminated."""
    data = asdict(record)
    data["model_set"] = list(record.model_set)
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _decode(line: bytes, number: int) -> RunRecord:
    """Parse one line strictly; any problem raises ValueError naming the line number."""
    try:
        data = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"registry line {number}: not valid UTF-8 JSON ({error})") from None
    if not isinstance(data, dict):
        raise ValueError(f"registry line {number}: expected a JSON object, got {type(data).__name__}")
    keys = set(data)
    if keys != _FIELD_NAMES:
        raise ValueError(
            f"registry line {number}: unknown keys {sorted(keys - _FIELD_NAMES)}, "
            f"missing keys {sorted(_FIELD_NAMES - keys)}"
        )
    if not isinstance(data["model_set"], list):
        raise ValueError(f"registry line {number}: model_set must be a JSON list")
    data["model_set"] = tuple(data["model_set"])
    try:
        return RunRecord(**data)
    except ValueError as error:
        raise ValueError(f"registry line {number}: {error}") from None


class RunRegistry:
    """Append-only JSON Lines registry of test-split runs (D31). See the module docstring."""

    def __init__(self, path: str | PathLike[str]) -> None:
        self._path = Path(path)
        self._load()  # an invalid or edited file fails here, before any claim

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> tuple[tuple[RunRecord, ...], str]:
        """All records, checked, and the SHA-256 the next record must carry as prev_sha256."""
        if not self._path.exists():
            return (), GENESIS_SHA256
        raw = self._path.read_bytes()
        if not raw:
            return (), GENESIS_SHA256
        if not raw.endswith(b"\n"):
            raise ValueError(f"{self._path}: last line is not newline-terminated (truncated write?)")
        records: list[RunRecord] = []
        expected_prev = GENESIS_SHA256
        started: dict[str, RunRecord] = {}
        completed: set[str] = set()
        claimed_keys: set[tuple[str, str, str, tuple[str, ...]]] = set()
        for number, body in enumerate(raw[:-1].split(b"\n"), start=1):
            line = body + b"\n"
            record = _decode(body, number)
            if record.prev_sha256 != expected_prev:
                raise ValueError(
                    f"registry line {number}: prev_sha256 does not match the previous line: "
                    "registry was edited (a line was changed, deleted or reordered)"
                )
            # Replay the claim/complete rules, so a file the code would never write is rejected.
            if record.status == STARTED:
                if record.run_id in started:
                    raise ValueError(f"registry line {number}: run_id {record.run_id} started twice")
                if record.key in claimed_keys:
                    raise ValueError(f"registry line {number}: second claim for {record.key}")
                started[record.run_id] = record
                claimed_keys.add(record.key)
            else:
                start = started.get(record.run_id)
                if start is None or record.run_id in completed:
                    raise ValueError(
                        f"registry line {number}: completed record for run_id {record.run_id} "
                        "that is not started or already completed"
                    )
                if (start.key, start.code_commit) != (record.key, record.code_commit):
                    raise ValueError(
                        f"registry line {number}: completed record does not match its started record"
                    )
                completed.add(record.run_id)
            records.append(record)
            expected_prev = hashlib.sha256(line).hexdigest()
        return tuple(records), expected_prev

    def _append(self, record: RunRecord) -> None:
        # "a" mode with newline="\n": one line added at the end, no newline translation on Windows,
        # so the bytes on disk are exactly the bytes that were hashed.
        with open(self._path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(_encode(record).decode("utf-8"))

    def records(self) -> tuple[RunRecord, ...]:
        """All records in file order, after checking the file and its hash chain."""
        return self._load()[0]

    def claim(
        self, dataset: str, split: str, model_set, code_commit: str, *, evaluation_purpose: str
    ) -> str:
        """Append a "started" record and return its run_id; ValueError if this run already exists.

        evaluation_purpose is required and keyword-only; it must be in EVALUATION_PURPOSES (D41).
        """
        if isinstance(model_set, str):
            raise ValueError("model_set must be a collection of model names, not a single string")
        try:
            models = tuple(sorted(model_set))
        except TypeError:
            raise ValueError(f"model_set must be a collection of strings, got {model_set!r}") from None
        records, prev = self._load()
        record = RunRecord(
            run_id=uuid.uuid4().hex,
            status=STARTED,
            dataset=dataset,
            split=split,
            evaluation_purpose=evaluation_purpose,
            model_set=models,
            code_commit=code_commit,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            results_ref="",
            prev_sha256=prev,
        )
        for existing in records:
            if existing.key == record.key:
                raise ValueError(
                    f"already run: {record.key} has a {existing.status} record "
                    f"(run_id {existing.run_id}); a second run needs a new decision"
                )
        self._append(record)
        return record.run_id

    def complete(self, run_id: str, results_ref: str) -> None:
        """Append a "completed" record for a started, not yet completed run_id; else ValueError."""
        records, prev = self._load()
        start = next((r for r in records if r.run_id == run_id and r.status == STARTED), None)
        if start is None:
            raise ValueError(f"unknown run_id {run_id!r}: no started record")
        if any(r.run_id == run_id and r.status == COMPLETED for r in records):
            raise ValueError(f"run_id {run_id} is already completed")
        self._append(
            RunRecord(
                run_id=run_id,
                status=COMPLETED,
                dataset=start.dataset,
                split=start.split,
                evaluation_purpose=start.evaluation_purpose,
                model_set=start.model_set,
                code_commit=start.code_commit,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                results_ref=results_ref,
                prev_sha256=prev,
            )
        )
