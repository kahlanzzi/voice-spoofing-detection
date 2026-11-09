import dataclasses
import sqlite3

import pytest

import api.repository
from api.db import connect
from api.repository import (ACCOUNT_SCHEMA, COLUMNS, SCHEMA, SESSION_SCHEMA, SIGN_IN_FAILURE_SCHEMA, AccountRepository,
                            AnalysisRecord, AnalysisRepository, DuplicateAccount, HistoryQuery, StoredNote)

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


@pytest.fixture
def author(repo):
    """Two accounts that can write notes: {username: account_id}."""
    accounts = AccountRepository(repo.path)
    return {name: accounts.create_account(name, HASH, NOW) for name in ("analyst01", "analyst02")}


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
    assert columns == ["request_id", "text", "updated_at", "updated_by_account_id"]
    assert sorted(foreign_keys) == [("account", "updated_by_account_id", "account_id"),
                                    ("analysis_record", "request_id", "request_id")]


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
    assert repo.get_stored(RECORD.request_id) == (RECORD, StoredNote("Called back.", "2026-10-30T09:20:11Z", None))
    assert repo.get(RECORD.request_id) == RECORD  # the note does not change the record


def record_row(repo, request_id):
    """The raw analysis_record row with its rowid, so a delete-and-insert would show."""
    conn = connect(repo.path)
    try:
        return conn.execute("SELECT rowid, * FROM analysis_record WHERE request_id = ?", (request_id,)).fetchone()
    finally:
        conn.close()


def test_save_note_stores_then_replaces_without_changing_the_record(repo, author):
    repo.add(RECORD)
    before = record_row(repo, RECORD.request_id)
    first = StoredNote("first", "2026-10-30T09:20:11Z", "analyst01")
    second = StoredNote("second", "2026-10-30T09:21:00Z", "analyst02")
    assert repo.save_note(RECORD.request_id, first.text, first.updated_at, author["analyst01"]) == first
    assert repo.get_stored(RECORD.request_id)[1] == first
    # Another account replaces the text and the author (§4.6).
    assert repo.save_note(RECORD.request_id, second.text, second.updated_at, author["analyst02"]) == second
    assert repo.get_stored(RECORD.request_id)[1] == second
    assert record_row(repo, RECORD.request_id) == before


def test_save_note_needs_a_known_account(repo):
    repo.add(RECORD)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        repo.save_note(RECORD.request_id, "x", "2026-10-30T09:20:11Z", 999)
    assert repo.get_stored(RECORD.request_id)[1] is None


def test_delete_note_reports_whether_a_note_was_removed(repo, author):
    repo.add(RECORD)
    before = record_row(repo, RECORD.request_id)
    repo.save_note(RECORD.request_id, "first", "2026-10-30T09:20:11Z", author["analyst01"])
    assert repo.delete_note(RECORD.request_id) is True
    assert repo.delete_note(RECORD.request_id) is False
    assert repo.get_stored(RECORD.request_id)[1] is None
    assert record_row(repo, RECORD.request_id) == before


# --- Migration: the note author on an older database file (§7, Step 7b-2) ------------------------

# analysis_note as created by the code merged in #83 and #84, before the author column.
OLD_NOTE_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_note (
    request_id TEXT NOT NULL PRIMARY KEY REFERENCES analysis_record (request_id),
    text       TEXT NOT NULL CHECK (length(trim(text)) > 0),
    updated_at TEXT NOT NULL
               CHECK (updated_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z')
)
"""
OLD_NOTES = [("req_" + "a" * 32, "Called back.\nCaller confirmed – ok ✓", "2026-10-30T09:20:11Z"),
             ("req_" + "c" * 32, "Second note.", "2026-10-30T10:00:00Z")]


def old_database(path, *ddl):
    """A database file built with the given older DDL, holding two records with notes."""
    conn = connect(path)
    try:
        with conn:
            for statement in ddl:
                conn.execute(statement)
            for request_id, text, updated_at in OLD_NOTES:
                placeholders = ", ".join("?" * len(COLUMNS))
                conn.execute(f"INSERT INTO analysis_record ({', '.join(COLUMNS)}) VALUES ({placeholders})",
                             dataclasses.astuple(dataclasses.replace(RECORD, request_id=request_id)))
                conn.execute("INSERT INTO analysis_note (request_id, text, updated_at) VALUES (?, ?, ?)",
                             (request_id, text, updated_at))
    finally:
        conn.close()


def table_dump(path, sql):
    conn = connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def test_create_schema_adds_the_note_author_to_a_7b_1_database_without_data_loss(tmp_path):
    path = tmp_path / "old.sqlite"
    old_database(path, SCHEMA, OLD_NOTE_SCHEMA, ACCOUNT_SCHEMA, SESSION_SCHEMA, SIGN_IN_FAILURE_SCHEMA)
    records_before = table_dump(path, "SELECT rowid, * FROM analysis_record ORDER BY rowid")
    repository = AnalysisRepository(path)
    repository.create_schema()
    repository.create_schema()  # safe to run twice
    columns = [row[1] for row in table_dump(path, "PRAGMA table_info(analysis_note)")]
    assert columns == ["request_id", "text", "updated_at", "updated_by_account_id"]
    assert table_dump(path, "SELECT request_id, text, updated_at, updated_by_account_id FROM analysis_note "
                            "ORDER BY request_id") == [(*note, None) for note in OLD_NOTES]
    assert table_dump(path, "SELECT rowid, * FROM analysis_record ORDER BY rowid") == records_before
    assert table_dump(path, "PRAGMA foreign_key_check") == []
    for request_id, text, updated_at in OLD_NOTES:
        assert repository.get_stored(request_id)[1] == StoredNote(text, updated_at, None)
    # An old note can then be saved again with an author.
    account_id = AccountRepository(path).create_account("analyst01", HASH, NOW)
    saved = repository.save_note(OLD_NOTES[0][0], "Replaced.", "2026-11-01T09:00:00Z", account_id)
    assert saved == StoredNote("Replaced.", "2026-11-01T09:00:00Z", "analyst01")


class FailOnAlter(sqlite3.Connection):
    def execute(self, sql, *args):
        if sql.lstrip().startswith("ALTER TABLE"):
            raise sqlite3.OperationalError("disk I/O error")
        return super().execute(sql, *args)


def test_a_failed_migration_leaves_the_file_as_it_was(tmp_path, monkeypatch):
    path = tmp_path / "older.sqlite"
    old_database(path, SCHEMA, OLD_NOTE_SCHEMA)  # before the sign-in tables too
    schema_before = table_dump(path, "SELECT type, name, sql FROM sqlite_master ORDER BY name")
    monkeypatch.setattr(api.repository, "connect", lambda p: sqlite3.connect(p, factory=FailOnAlter))
    with pytest.raises(sqlite3.OperationalError):
        AnalysisRepository(path).create_schema()
    # One transaction: the sign-in tables created before the ALTER are rolled back with it.
    assert table_dump(path, "SELECT type, name, sql FROM sqlite_master ORDER BY name") == schema_before
    assert table_dump(path, "SELECT request_id, text, updated_at FROM analysis_note ORDER BY request_id") == OLD_NOTES


def test_note_writes_for_an_unknown_record_return_none_and_store_nothing(repo, author):
    unknown = "req_" + "b" * 32
    assert repo.save_note(unknown, "x", "2026-10-30T09:20:11Z", author["analyst01"]) is None
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


def stats_record(n: int, created_at: str, prediction: str = "authentic", processing_ms: int = 10) -> AnalysisRecord:
    return dataclasses.replace(RECORD, request_id=f"req_{n:032x}", created_at=created_at, prediction=prediction,
                               risk_level={"synthetic": "HIGH", "uncertain": "MEDIUM", "authentic": "LOW"}[prediction],
                               processing_ms=processing_ms)


def test_stats_on_an_empty_store(repo):
    figures = repo.stats("2026-10-24", "2026-10-30", "2026-10-30", 6)
    assert (figures.total, figures.per_day, figures.median_processing_ms) == (0, {}, None)
    assert figures.counts == {"synthetic": 0, "uncertain": 0, "authentic": 0}
    assert (figures.awaiting_total, figures.awaiting_recent, figures.today_total) == (0, [], 0)


@pytest.mark.parametrize("values, median", [
    ([840], 840), ([30, 10, 20], 20), ([10, 20], 15), ([10, 15], 13), ([0, 1], 1), ([7, 1, 9, 3], 5),
], ids=str)
def test_stats_median_is_the_middle_or_the_mean_of_two_rounded_halves_up(repo, values, median):
    for n, ms in enumerate(values):
        repo.add(stats_record(n, "2026-10-30T09:00:00Z", processing_ms=ms))
    assert repo.stats("2026-10-30", "2026-10-30", "2026-10-30", 6).median_processing_ms == median


def test_stats_median_ignores_records_outside_the_period(repo):
    repo.add(stats_record(1, "2026-10-29T23:59:59Z", processing_ms=1000))
    repo.add(stats_record(2, "2026-10-30T00:00:00Z", processing_ms=10))
    repo.add(stats_record(3, "2026-10-31T00:00:00Z", processing_ms=1000))
    assert repo.stats("2026-10-30", "2026-10-30", "2026-10-30", 6).median_processing_ms == 10


def test_stats_per_day_counts_and_today_outside_the_period(repo):
    repo.add(stats_record(1, "2026-10-28T08:00:00Z", "synthetic"))
    repo.add(stats_record(2, "2026-10-28T09:00:00Z", "authentic"))
    repo.add(stats_record(3, "2026-10-30T09:00:00Z", "uncertain"))
    repo.add(stats_record(4, "2026-11-01T09:00:00Z", "authentic"))
    figures = repo.stats("2026-10-28", "2026-10-30", "2026-11-01", 6)
    assert figures.per_day == {"2026-10-28": {"synthetic": 1, "uncertain": 0, "authentic": 1},
                               "2026-10-30": {"synthetic": 0, "uncertain": 1, "authentic": 0}}
    assert figures.total == 3 and figures.today_total == 1


def test_stats_awaiting_note_excludes_authentic_and_noted_results(repo, author):
    repo.add(stats_record(1, "2026-10-30T08:00:00Z", "synthetic"))
    repo.add(stats_record(2, "2026-10-30T09:00:00Z", "uncertain"))
    repo.add(stats_record(3, "2026-10-30T10:00:00Z", "authentic"))
    repo.add(stats_record(4, "2026-10-30T11:00:00Z", "synthetic"))
    repo.save_note(f"req_{4:032x}", "Checked.", "2026-10-30T12:00:00Z", author["analyst01"])
    figures = repo.stats("2026-10-30", "2026-10-30", "2026-10-30", 6)
    assert figures.awaiting_total == 2
    assert [(r.request_id, note) for r, note in figures.awaiting_recent] == [(f"req_{2:032x}", None),
                                                                           (f"req_{1:032x}", None)]


# --- Sign-in tables (§4.8, §7) ---------------------------------------------------------------

NOW = "2026-11-01T09:00:00Z"
HASH = "$argon2id$v=19$m=65536,t=3,p=4$c2FsdHNhbHRzYWx0c2FsdA$aGFzaGhhc2hoYXNoaGFzaGhhc2hoYXNoaGFzaGhhc2g"
SIGN_IN_TABLES = {
    "account": ["account_id", "username", "password_hash", "disabled", "created_at", "password_changed_at"],
    "session": ["session_hash", "account_id", "created_at", "last_seen_at", "expires_at"],
    "sign_in_failure": ["username_hmac", "failures", "last_failure_at", "locked_until"],
}


@pytest.fixture
def accounts(repo):
    return AccountRepository(repo.path)


def add_session(repo, account_id, n):
    conn = connect(repo.path)
    try:
        with conn:
            conn.execute("INSERT INTO session VALUES (?, ?, ?, ?, ?)",
                         (f"{n:064x}", account_id, NOW, NOW, "2026-11-01T21:00:00Z"))
    finally:
        conn.close()


def sessions_of(repo, account_id):
    conn = connect(repo.path)
    try:
        return conn.execute("SELECT COUNT(*) FROM session WHERE account_id = ?", (account_id,)).fetchone()[0]
    finally:
        conn.close()


@pytest.mark.parametrize("table", SIGN_IN_TABLES)
def test_sign_in_tables_have_exactly_the_section_7_columns(repo, table):
    conn = connect(repo.path)
    try:
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()
    assert columns == SIGN_IN_TABLES[table]


def test_no_sign_in_table_has_an_ip_user_agent_or_clear_secret_column(repo):
    conn = connect(repo.path)
    try:
        for table in SIGN_IN_TABLES:
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            assert not columns & {"ip", "client_ip", "ip_address", "user_agent", "password", "session_id"}
    finally:
        conn.close()


def test_create_schema_twice_keeps_accounts(repo, accounts):
    accounts.create_account("analyst01", HASH, NOW)
    repo.create_schema()
    assert accounts.get_account("analyst01").username == "analyst01"


def test_create_and_get_account(accounts):
    account_id = accounts.create_account("analyst01", HASH, NOW)
    account = accounts.get_account("analyst01")
    assert account.account_id == account_id
    assert (account.password_hash, account.disabled, account.created_at, account.password_changed_at) == \
        (HASH, False, NOW, NOW)
    assert accounts.get_account("nobody") is None


def test_duplicate_username_is_refused(accounts):
    accounts.create_account("analyst01", HASH, NOW)
    with pytest.raises(DuplicateAccount):
        accounts.create_account("analyst01", HASH, NOW)


@pytest.mark.parametrize("username, password_hash, now", [
    ("Analyst01", HASH, NOW),                       # not lower case
    ("ab", HASH, NOW),                              # too short
    ("x" * 65, HASH, NOW),                          # too long
    ("analyst01", "clear-text-password", NOW),      # not an argon2id hash
    ("analyst01", "$argon2i$v=19$m=65536", NOW),    # argon2i, not argon2id
    ("analyst01", HASH, "2026-11-01 09:00:00"),     # not UTC ISO with Z
])
def test_invalid_account_values_are_refused(accounts, username, password_hash, now):
    with pytest.raises(sqlite3.IntegrityError):
        accounts.create_account(username, password_hash, now)


def test_set_password_ends_all_sessions_of_that_account_only(repo, accounts):
    mine = accounts.create_account("analyst01", HASH, NOW)
    other = accounts.create_account("analyst02", HASH, NOW)
    add_session(repo, mine, 1)
    add_session(repo, mine, 2)
    add_session(repo, other, 3)
    new_hash = HASH.replace("c2FsdH", "bmV3c2")
    assert accounts.set_password("analyst01", new_hash, "2026-11-02T10:00:00Z") == 2
    account = accounts.get_account("analyst01")
    assert (account.password_hash, account.password_changed_at, account.created_at) == \
        (new_hash, "2026-11-02T10:00:00Z", NOW)
    assert sessions_of(repo, mine) == 0 and sessions_of(repo, other) == 1
    assert accounts.set_password("nobody", HASH, NOW) is None


def test_disable_ends_all_sessions_and_enable_keeps_none(repo, accounts):
    mine = accounts.create_account("analyst01", HASH, NOW)
    other = accounts.create_account("analyst02", HASH, NOW)
    add_session(repo, mine, 1)
    add_session(repo, other, 2)
    assert accounts.set_disabled("analyst01", True) == 1
    assert accounts.get_account("analyst01").disabled is True
    assert sessions_of(repo, mine) == 0 and sessions_of(repo, other) == 1
    assert accounts.set_disabled("analyst01", False) == 0
    assert accounts.get_account("analyst01").disabled is False
    assert sessions_of(repo, mine) == 0
    assert accounts.set_disabled("nobody", True) is None


def test_session_needs_a_known_account_and_a_hex_hash(repo, accounts):
    with pytest.raises(sqlite3.IntegrityError):
        add_session(repo, 999, 1)  # foreign key
    account_id = accounts.create_account("analyst01", HASH, NOW)
    conn = connect(repo.path)
    try:
        with pytest.raises(sqlite3.IntegrityError), conn:
            conn.execute("INSERT INTO session VALUES (?, ?, ?, ?, ?)", ("Z" * 64, account_id, NOW, NOW, NOW))
    finally:
        conn.close()


def test_clear_failures(repo, accounts):
    digest = "ab" * 32
    conn = connect(repo.path)
    try:
        with conn:
            conn.execute("INSERT INTO sign_in_failure VALUES (?, 5, ?, ?)", (digest, NOW, "2026-11-01T09:15:00Z"))
    finally:
        conn.close()
    assert accounts.clear_failures(digest) is True
    assert accounts.clear_failures(digest) is False


DIGEST = "ef" * 32
LATER = "2026-11-01T09:01:00Z"


def failure_row(repo):
    conn = connect(repo.path)
    try:
        return conn.execute("SELECT failures, last_failure_at, locked_until FROM sign_in_failure").fetchall()
    finally:
        conn.close()


def test_record_failure_counts_then_locks_at_the_threshold(repo, accounts):
    for n in (1, 2):
        assert accounts.record_failure(DIGEST, NOW, "2026-11-01T08:45:00Z", 3, "2026-11-01T09:15:00Z") == n
    assert not accounts.is_locked(DIGEST, NOW)
    assert accounts.record_failure(DIGEST, NOW, "2026-11-01T08:45:00Z", 3, "2026-11-01T09:15:00Z") == 3
    assert failure_row(repo) == [(3, NOW, "2026-11-01T09:15:00Z")]
    assert accounts.is_locked(DIGEST, LATER)
    assert not accounts.is_locked(DIGEST, "2026-11-01T09:15:00Z")
    assert not accounts.is_locked("00" * 32, NOW)


def test_record_failure_keeps_a_running_lock(repo, accounts):
    for _ in range(3):
        accounts.record_failure(DIGEST, NOW, "2026-11-01T08:45:00Z", 3, "2026-11-01T09:15:00Z")
    assert accounts.record_failure(DIGEST, LATER, "2026-11-01T08:46:00Z", 3, "2026-11-01T09:16:00Z") == 3
    assert failure_row(repo) == [(3, NOW, "2026-11-01T09:15:00Z")]


def test_record_failure_restarts_after_a_lock_ends_or_an_old_failure(repo, accounts):
    for _ in range(3):
        accounts.record_failure(DIGEST, NOW, "2026-11-01T08:45:00Z", 3, "2026-11-01T09:15:00Z")
    assert accounts.record_failure(DIGEST, "2026-11-01T09:20:00Z", "2026-11-01T09:05:00Z", 3,
                                   "2026-11-01T09:35:00Z") == 1
    assert failure_row(repo) == [(1, "2026-11-01T09:20:00Z", None)]
    assert accounts.record_failure(DIGEST, "2026-11-01T10:00:00Z", "2026-11-01T09:45:00Z", 3,
                                   "2026-11-01T10:15:00Z") == 1


def test_rehash_password_replaces_only_the_expected_hash_and_keeps_sessions(repo, accounts):
    account_id = accounts.create_account("analyst01", HASH, "2026-11-01T08:00:00Z")
    add_session(repo, account_id, 1)
    new_hash = HASH.replace("c2FsdH", "bmV3c2")
    assert not accounts.rehash_password(account_id, new_hash, HASH)
    assert accounts.rehash_password(account_id, HASH, new_hash)
    account = accounts.get_account("analyst01")
    assert (account.password_hash, account.password_changed_at) == (new_hash, "2026-11-01T08:00:00Z")
    assert sessions_of(repo, account_id) == 1


def test_start_session_replaces_the_old_one_and_deletes_ended_ones(repo, accounts):
    account_id = accounts.create_account("analyst01", HASH, NOW)
    accounts.start_session("01" * 32, account_id, NOW, "2026-11-01T21:00:00Z", "2026-11-01T08:30:00Z")
    accounts.start_session("02" * 32, account_id, NOW, "2026-11-01T21:00:00Z", "2026-11-01T08:30:00Z")
    accounts.start_session("03" * 32, account_id, LATER, "2026-11-01T21:01:00Z", "2026-11-01T08:31:00Z",
                           old_session_hash="01" * 32)
    assert sessions_of(repo, account_id) == 2
    # Two hours later, 02 and 03 are idle (last seen before the cutoff) and are deleted.
    accounts.start_session("04" * 32, account_id, "2026-11-01T11:00:00Z", "2026-11-01T23:00:00Z",
                           "2026-11-01T10:30:00Z")
    assert sessions_of(repo, account_id) == 1


def test_touch_session_updates_or_deletes(repo, accounts):
    account_id = accounts.create_account("analyst01", HASH, NOW)
    accounts.start_session("01" * 32, account_id, NOW, "2026-11-01T21:00:00Z", "2026-11-01T08:30:00Z")
    session = accounts.touch_session("01" * 32, LATER, "2026-11-01T08:31:00Z")
    assert (session.account_id, session.username, session.created_at, session.expires_at) == \
        (account_id, "analyst01", NOW, "2026-11-01T21:00:00Z")
    assert accounts.touch_session("02" * 32, LATER, "2026-11-01T08:31:00Z") is None
    # Idle: last seen at LATER, cutoff after it.
    assert accounts.touch_session("01" * 32, "2026-11-01T09:40:00Z", "2026-11-01T09:10:00Z") is None
    assert sessions_of(repo, account_id) == 0
    accounts.start_session("01" * 32, account_id, NOW, "2026-11-01T21:00:00Z", "2026-11-01T08:30:00Z")
    assert accounts.touch_session("01" * 32, "2026-11-01T21:00:00Z", "2026-11-01T20:30:00Z") is None  # absolute


def test_end_session(repo, accounts):
    account_id = accounts.create_account("analyst01", HASH, NOW)
    accounts.start_session("01" * 32, account_id, NOW, "2026-11-01T21:00:00Z", "2026-11-01T08:30:00Z")
    assert accounts.end_session("01" * 32)
    assert not accounts.end_session("01" * 32)
