import dataclasses
import sqlite3

import pytest

from api.db import connect
from api.repository import COLUMNS, AnalysisRecord, AnalysisRepository, HistoryQuery, StoredNote

SECTION_7 = ["request_id", "created_at", "prediction", "confidence", "risk_level", "model_version",
             "profile_version", "policy_version", "notice", "processing_ms", "is_mock"]

RECORD = AnalysisRecord(
    request_id="req_" + "a" * 32, created_at="2026-10-30T09:14:05Z", prediction="uncertain",
    confidence=0.62, risk_level="MEDIUM", model_version="mock", profile_version="preprocessing-v1",
    policy_version="risk-v1", notice="Decision support only.", processing_ms=840, is_mock=True)


@pytest.fixture
def repo(tmp_path):
    repository = AnalysisRepository(tmp_path / "api.sqlite")
    repository.create_schema()
    return repository


def test_table_has_exactly_the_section_7_columns(repo):
    conn = connect(repo.path)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(analysis_record)")]
    finally:
        conn.close()
    assert columns == SECTION_7 == list(COLUMNS)


def test_note_table_has_exactly_the_section_7_columns(repo):
    conn = connect(repo.path)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(analysis_note)")]
        foreign_keys = [(row[2], row[3], row[4]) for row in conn.execute("PRAGMA foreign_key_list(analysis_note)")]
    finally:
        conn.close()
    assert columns == ["request_id", "text", "updated_at"]
    assert foreign_keys == [("analysis_record", "request_id", "request_id")]


def insert_note(repo, request_id, text="Called back.", updated_at="2026-10-30T09:20:11Z"):
    conn = connect(repo.path)
    try:
        with conn:
            conn.execute("INSERT INTO analysis_note (request_id, text, updated_at) VALUES (?, ?, ?)",
                         (request_id, text, updated_at))
    finally:
        conn.close()


def test_create_schema_is_idempotent_and_keeps_rows(repo):
    repo.add(RECORD)
    insert_note(repo, RECORD.request_id)
    repo.create_schema()
    assert repo.count() == 1
    assert repo.get_stored(RECORD.request_id)[1] is not None


def test_note_for_an_unknown_record_is_rejected(repo):
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        insert_note(repo, "req_" + "b" * 32)


@pytest.mark.parametrize("text, updated_at", [
    ("", "2026-10-30T09:20:11Z"), ("   ", "2026-10-30T09:20:11Z"), (None, "2026-10-30T09:20:11Z"),
    ("ok", "2026-10-30 09:20:11"), ("ok", "2026-10-30T09:20:11+00:00"), ("ok", None),
])
def test_invalid_note_values_are_rejected(repo, text, updated_at):
    repo.add(RECORD)
    with pytest.raises(sqlite3.IntegrityError):
        insert_note(repo, RECORD.request_id, text, updated_at)


def test_get_stored_returns_the_record_with_its_note(repo):
    assert repo.get_stored(RECORD.request_id) is None
    repo.add(RECORD)
    assert repo.get_stored(RECORD.request_id) == (RECORD, None)
    insert_note(repo, RECORD.request_id)
    assert repo.get_stored(RECORD.request_id) == (RECORD, StoredNote("Called back.", "2026-10-30T09:20:11Z"))
    assert repo.get(RECORD.request_id) == RECORD  # the note does not change the record


def record_row(repo, request_id):
    """The raw analysis_record row with its rowid, so a delete-and-insert would show."""
    conn = connect(repo.path)
    try:
        return conn.execute("SELECT rowid, * FROM analysis_record WHERE request_id = ?", (request_id,)).fetchone()
    finally:
        conn.close()


def test_save_note_stores_then_replaces_without_changing_the_record(repo):
    repo.add(RECORD)
    before = record_row(repo, RECORD.request_id)
    first, second = StoredNote("first", "2026-10-30T09:20:11Z"), StoredNote("second", "2026-10-30T09:21:00Z")
    assert repo.save_note(RECORD.request_id, first.text, first.updated_at) == first
    assert repo.save_note(RECORD.request_id, second.text, second.updated_at) == second
    assert repo.get_stored(RECORD.request_id)[1] == second
    assert record_row(repo, RECORD.request_id) == before


def test_delete_note_reports_whether_a_note_was_removed(repo):
    repo.add(RECORD)
    before = record_row(repo, RECORD.request_id)
    repo.save_note(RECORD.request_id, "first", "2026-10-30T09:20:11Z")
    assert repo.delete_note(RECORD.request_id) is True
    assert repo.delete_note(RECORD.request_id) is False
    assert repo.get_stored(RECORD.request_id)[1] is None
    assert record_row(repo, RECORD.request_id) == before


def test_note_writes_for_an_unknown_record_return_none_and_store_nothing(repo):
    unknown = "req_" + "b" * 32
    assert repo.save_note(unknown, "x", "2026-10-30T09:20:11Z") is None
    assert repo.delete_note(unknown) is None
    conn = connect(repo.path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM analysis_note").fetchone()[0] == 0
    finally:
        conn.close()
    assert repo.count() == 0


def _rec(n, created_at, prediction="authentic"):
    return dataclasses.replace(RECORD, request_id=f"req_{n:032x}", created_at=created_at, prediction=prediction,
                               risk_level="LOW")


def test_history_orders_newest_first_then_last_stored_first(repo):
    for r in (_rec(1, "2026-10-30T09:00:00Z"), _rec(2, "2026-10-31T09:00:00Z"), _rec(3, "2026-10-30T09:00:00Z")):
        repo.add(r)
    page = repo.history(HistoryQuery(), limit=10, offset=0)
    assert [r.request_id for r, _ in page.items] == [f"req_{n:032x}" for n in (2, 3, 1)]
    assert page.total == 3 and page.counts == {"synthetic": 0, "uncertain": 0, "authentic": 3}


def test_history_filters_total_and_counts(repo):
    repo.add(_rec(1, "2026-10-29T23:59:59Z", "synthetic"))
    repo.add(_rec(2, "2026-10-30T00:00:00Z", "synthetic"))
    repo.add(_rec(3, "2026-10-30T12:00:00Z", "authentic"))
    page = repo.history(HistoryQuery(verdicts=frozenset({"synthetic"}), day_from="2026-10-30"), limit=10, offset=0)
    assert [r.request_id for r, _ in page.items] == [f"req_{2:032x}"]
    assert page.total == 1
    assert page.counts == {"synthetic": 1, "uncertain": 0, "authentic": 1}  # verdict filter ignored


def test_history_past_the_last_page_is_empty_with_the_total(repo):
    repo.add(RECORD)
    page = repo.history(HistoryQuery(), limit=10, offset=10)
    assert page.items == [] and page.total == 1


def test_add_then_get_round_trip(repo):
    repo.add(RECORD)
    assert repo.get(RECORD.request_id) == RECORD
    assert repo.get("req_" + "b" * 32) is None


def test_duplicate_request_id_is_rejected(repo):
    repo.add(RECORD)
    with pytest.raises(sqlite3.IntegrityError):
        repo.add(RECORD)
    assert repo.count() == 1


@pytest.mark.parametrize("change", [
    {"prediction": "spoof"},
    {"risk_level": "low"},
    {"confidence": 1.5},
    {"processing_ms": -1},
], ids=lambda c: next(iter(c)))
def test_invalid_values_are_rejected(repo, change):
    with pytest.raises(sqlite3.IntegrityError):
        repo.add(dataclasses.replace(RECORD, **change))
    assert repo.count() == 0


def test_probe_passes_on_a_built_schema(repo):
    repo.add(RECORD)
    repo.probe()
    assert repo.count() == 1  # read only: nothing added or removed


def test_probe_raises_on_a_missing_file_and_creates_nothing(tmp_path):
    path = tmp_path / "missing" / "api.sqlite"
    with pytest.raises(sqlite3.Error):
        AnalysisRepository(path).probe(timeout_s=0.1)
    assert not path.parent.exists()


def test_probe_raises_when_the_table_is_missing(tmp_path):
    path = tmp_path / "api.sqlite"
    connect(path).close()  # empty database, no analysis_record table
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        AnalysisRepository(path).probe(timeout_s=0.1)


def test_probe_raises_when_the_database_stays_locked(repo):
    locker = connect(repo.path)
    try:
        locker.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            repo.probe(timeout_s=0.05)
    finally:
        locker.rollback()
        locker.close()
