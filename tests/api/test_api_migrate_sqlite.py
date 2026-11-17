"""api/migrate_sqlite.py: the one-time import of the old SQLite store (D53 (4), SDD v2.0 §5.4).

The source files are built here with the SQLite DDL the API used before PostgreSQL (copied from
api/repository.py at main 4b8f8f5), with and without the later columns and tables."""
from __future__ import annotations

import dataclasses
import hashlib
import sqlite3

import pytest

from api import admin, migrate_sqlite
from api.repository import AccountRepository, AnalysisRepository, HistoryQuery, StoredNote
from tests.api._pg import query

_UTC = "GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'"
_HEX64 = "length({0}) = 64 AND {0} NOT GLOB '*[^0-9a-f]*'"
OLD_RECORD = """CREATE TABLE analysis_record (
    request_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
    prediction TEXT NOT NULL CHECK (prediction IN ('synthetic', 'uncertain', 'authentic')),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    risk_level TEXT NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    model_version TEXT NOT NULL, profile_version TEXT NOT NULL, policy_version TEXT NOT NULL,
    notice TEXT NOT NULL, processing_ms INTEGER NOT NULL CHECK (processing_ms >= 0),
    is_mock INTEGER NOT NULL CHECK (is_mock IN (0, 1)))"""
OLD_NOTE_WITH_AUTHOR = f"""CREATE TABLE analysis_note (
    request_id TEXT NOT NULL PRIMARY KEY REFERENCES analysis_record (request_id),
    text TEXT NOT NULL CHECK (length(trim(text)) > 0), updated_at TEXT NOT NULL CHECK (updated_at {_UTC}),
    updated_by_account_id INTEGER REFERENCES account (account_id))"""
OLD_NOTE_WITHOUT_AUTHOR = f"""CREATE TABLE analysis_note (
    request_id TEXT NOT NULL PRIMARY KEY REFERENCES analysis_record (request_id),
    text TEXT NOT NULL CHECK (length(trim(text)) > 0), updated_at TEXT NOT NULL CHECK (updated_at {_UTC}))"""
OLD_ACCOUNT = f"""CREATE TABLE account (
    account_id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE CHECK (username = lower(username) AND length(username) BETWEEN 3 AND 64),
    password_hash TEXT NOT NULL CHECK (password_hash LIKE '$argon2id$%'),
    disabled INTEGER NOT NULL CHECK (disabled IN (0, 1)),
    created_at TEXT NOT NULL CHECK (created_at {_UTC}), password_changed_at TEXT NOT NULL CHECK (password_changed_at {_UTC}))"""
OLD_SESSION = f"""CREATE TABLE session (
    session_hash TEXT PRIMARY KEY CHECK ({_HEX64.format("session_hash")}),
    account_id INTEGER NOT NULL REFERENCES account (account_id),
    created_at TEXT NOT NULL CHECK (created_at {_UTC}), last_seen_at TEXT NOT NULL CHECK (last_seen_at {_UTC}),
    expires_at TEXT NOT NULL CHECK (expires_at {_UTC}))"""
OLD_FAILURE = f"""CREATE TABLE sign_in_failure (
    username_hmac TEXT PRIMARY KEY CHECK ({_HEX64.format("username_hmac")}),
    failures INTEGER NOT NULL CHECK (failures >= 0), last_failure_at TEXT NOT NULL CHECK (last_failure_at {_UTC}),
    locked_until TEXT CHECK (locked_until IS NULL OR locked_until {_UTC}))"""

HASH = "$argon2id$v=19$m=65536,t=3,p=4$c2FsdHNhbHRzYWx0c2FsdA$aGFzaGhhc2hoYXNoaGFzaGhhc2hoYXNoaGFzaGhhc2g"
NOTE_TEXT = "Called back.\nCaller confirmed – ok ✓ 50% _x_"
T = "2026-10-30T09:00:00Z"


def record(n: int, created_at: str = T, prediction: str = "uncertain", confidence: float = 0.6234567890123456):
    risk = {"synthetic": "HIGH", "uncertain": "MEDIUM", "authentic": "LOW"}[prediction]
    return (f"req_{n:032x}", created_at, prediction, confidence, risk, "mock", "preprocessing-v1", "risk-v1",
            "Decision support only.", 100 + n, 1)


def build(path, *, with_author=True, with_sign_in=True, records=(3, 1, 2)):
    """An old store: records inserted in the given order, all in the same second (rowid breaks the tie)."""
    conn = sqlite3.connect(path)
    with conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(OLD_RECORD)
        if with_sign_in:
            for ddl in (OLD_ACCOUNT, OLD_SESSION, OLD_FAILURE):
                conn.execute(ddl)
        conn.execute(OLD_NOTE_WITH_AUTHOR if with_author and with_sign_in else OLD_NOTE_WITHOUT_AUTHOR)
        for n in records:
            conn.execute("INSERT INTO analysis_record VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", record(n))
        if with_sign_in:
            conn.execute("INSERT INTO account VALUES (7, 'analyst01', ?, 0, ?, ?)", (HASH, T, T))
            conn.execute("INSERT INTO account VALUES (9, 'analyst02', ?, 1, ?, ?)", (HASH, T, T))
            conn.execute("INSERT INTO session VALUES (?, 7, ?, ?, '2026-10-30T21:00:00Z')", ("ab" * 32, T, T))
            conn.execute("INSERT INTO sign_in_failure VALUES (?, 2, ?, NULL)", ("cd" * 32, T))
        if with_author and with_sign_in:
            conn.execute("INSERT INTO analysis_note VALUES (?, ?, ?, 9)", (f"req_{1:032x}", NOTE_TEXT, T))
        else:
            conn.execute("INSERT INTO analysis_note VALUES (?, ?, ?)", (f"req_{1:032x}", NOTE_TEXT, T))
    conn.close()
    return path


@pytest.fixture
def target(config, empty_db, monkeypatch):
    """The import points at an empty database (no tables); the API's schema is applied by the tool."""
    name = query(empty_db, "SELECT current_database()")[0][0]
    monkeypatch.setattr(migrate_sqlite, "load_api_config", lambda: dataclasses.replace(config, database_name=name))
    return empty_db


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_imports_every_table_and_verifies_it(tmp_path, target, capsys):
    source = build(tmp_path / "api.sqlite")
    before = sha(source)
    assert migrate_sqlite.main([str(source)]) == 0
    out = capsys.readouterr().out
    assert "Imported and verified: 8 rows in 5 tables" in out
    for table, rows in (("account", 2), ("analysis_record", 3), ("analysis_note", 1), ("session", 1),
                        ("sign_in_failure", 1)):
        assert f"{table}" in out and f" {rows} rows  sha256 " in out
    assert sha(source) == before  # the SQLite file is never written
    repo, accounts = AnalysisRepository(target), AccountRepository(target)
    stored = repo.get_stored(f"req_{1:032x}")
    assert stored[0].confidence == 0.6234567890123456 and stored[0].is_mock is True
    assert stored[1] == StoredNote(NOTE_TEXT, T, "analyst02")
    assert accounts.get_account("analyst02").disabled is True
    assert accounts.touch_session("ab" * 32, T, "2026-10-30T08:00:00Z").username == "analyst01"
    # Same second: SQLite rowid order (3, 1, 2) is kept, last stored first.
    page = repo.history(HistoryQuery(), limit=10, offset=0)
    assert [r.request_id for r, _ in page.items] == [f"req_{n:032x}" for n in (2, 1, 3)]
    # IDs are kept and the identity continues after the highest one.
    assert accounts.create_account("analyst03", HASH, T) == 10


def test_a_file_from_before_the_note_author_imports_a_null_author(tmp_path, target):
    source = build(tmp_path / "old.sqlite", with_author=False)
    assert migrate_sqlite.main([str(source)]) == 0
    assert AnalysisRepository(target).get_stored(f"req_{1:032x}")[1] == StoredNote(NOTE_TEXT, T, None)


def test_a_file_from_before_sign_in_imports_records_and_notes_only(tmp_path, target, capsys):
    source = build(tmp_path / "older.sqlite", with_sign_in=False)
    assert migrate_sqlite.main([str(source)]) == 0
    assert "Imported and verified: 4 rows" in capsys.readouterr().out
    assert query(target, "SELECT COUNT(*) FROM account") == [(0,)]


def test_a_non_empty_store_is_refused_and_left_as_it_was(tmp_path, target, capsys):
    source = build(tmp_path / "api.sqlite")
    assert migrate_sqlite.main([str(source)]) == 0
    capsys.readouterr()
    assert migrate_sqlite.main([str(source)]) == 1
    assert "is not empty (8 rows). Nothing was changed." in capsys.readouterr().err
    assert query(target, "SELECT COUNT(*) FROM analysis_record") == [(3,)]


def test_a_failed_verification_rolls_everything_back(tmp_path, target, monkeypatch, capsys):
    source = build(tmp_path / "api.sqlite")
    real = migrate_sqlite.digest
    calls = []

    def tampered(columns, rows):
        calls.append(1)
        return real(columns, rows) if len(calls) % 2 else "0" * 64  # every PostgreSQL side differs
    monkeypatch.setattr(migrate_sqlite, "digest", tampered)
    assert migrate_sqlite.main([str(source)]) == 1
    assert "Verification failed for account" in capsys.readouterr().err
    for table in migrate_sqlite.TABLES:
        assert query(target, f"SELECT COUNT(*) FROM {table}") == [(0,)]


@pytest.mark.parametrize("content", [None, b"not a database" * 100], ids=["missing", "not sqlite"])
def test_a_missing_or_unreadable_file_is_refused(tmp_path, target, capsys, content):
    path = tmp_path / "api.sqlite"
    if content is not None:
        path.write_bytes(content)
    assert migrate_sqlite.main([str(path)]) == 1
    assert "Nothing was changed." in capsys.readouterr().err
    assert query(target, "SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'public'") == [(0,)]


def test_the_output_never_holds_a_row_value(tmp_path, target, capsys):
    source = build(tmp_path / "api.sqlite")
    migrate_sqlite.main([str(source)])
    migrate_sqlite.main([str(source)])  # refused: the error line too
    out = capsys.readouterr()
    for value in (NOTE_TEXT, HASH, "analyst01", "ab" * 32, "cd" * 32, f"req_{1:032x}"):
        assert value not in out.out + out.err


def test_without_the_app_password_the_import_is_refused(tmp_path, target, monkeypatch, capsys):
    monkeypatch.delenv("VSD_DB_APP_PASSWORD")
    assert migrate_sqlite.main([str(build(tmp_path / "api.sqlite"))]) == 1
    assert "VSD_DB_APP_PASSWORD" in capsys.readouterr().err


def test_a_missing_argument_is_a_usage_error():
    with pytest.raises(SystemExit) as caught:
        migrate_sqlite.main([])
    assert caught.value.code == 2


def test_the_admin_and_the_import_share_one_way_to_open_the_database():
    assert migrate_sqlite.open_database is admin.open_database
