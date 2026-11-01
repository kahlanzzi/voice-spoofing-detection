"""All SQL of the API (docs/api_design.md §7): the analysis_record and analysis_note tables, and
the sign-in tables account, session and sign_in_failure (§4.8).
probe() is the read-only check behind GET /api/status; get_stored() and history() serve §4.5 and
§4.4; save_note() and delete_note() serve §4.6; stats() serves §4.7. Notes live in their own table
and never change analysis_record (FR-11). AccountRepository serves the admin command (api/admin.py), sign-in
(api/routers/session.py) and the session check (api/session_guard.py).

Never stored: audio, the original file name, the client IP address, passwords in clear (only the
argon2id hash), session IDs in clear (only their SHA-256), the typed username of a failed sign-in
(only its HMAC). No table has a column for them.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import astuple, dataclass, fields
from pathlib import Path

from api.db import connect, connect_readonly

PROBE_TIMEOUT_S = 1.0  # internal probe setting, not a §9 contract value

SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_record (
    request_id      TEXT    PRIMARY KEY,
    created_at      TEXT    NOT NULL,
    prediction      TEXT    NOT NULL CHECK (prediction IN ('synthetic', 'uncertain', 'authentic')),
    confidence      REAL    NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    risk_level      TEXT    NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    model_version   TEXT    NOT NULL,
    profile_version TEXT    NOT NULL,
    policy_version  TEXT    NOT NULL,
    notice          TEXT    NOT NULL,
    processing_ms   INTEGER NOT NULL CHECK (processing_ms >= 0),
    is_mock         INTEGER NOT NULL CHECK (is_mock IN (0, 1))
)
"""

# note_max_chars is config (§9), so the API enforces it; a CHECK here would need a migration per change.
NOTE_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_note (
    request_id TEXT NOT NULL PRIMARY KEY REFERENCES analysis_record (request_id),
    text       TEXT NOT NULL CHECK (length(trim(text)) > 0),
    updated_at TEXT NOT NULL
               CHECK (updated_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z')
)
"""

_UTC = "GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'"
_HEX64 = "length({0}) = 64 AND {0} NOT GLOB '*[^0-9a-f]*'"

# Usernames: rules in api/auth.py; the CHECK keeps the stored form lower case and in length.
ACCOUNT_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS account (
    account_id          INTEGER PRIMARY KEY,
    username            TEXT    NOT NULL UNIQUE
                        CHECK (username = lower(username) AND length(username) BETWEEN 3 AND 64),
    password_hash       TEXT    NOT NULL CHECK (password_hash LIKE '$argon2id$%'),
    disabled            INTEGER NOT NULL CHECK (disabled IN (0, 1)),
    created_at          TEXT    NOT NULL CHECK (created_at {_UTC}),
    password_changed_at TEXT    NOT NULL CHECK (password_changed_at {_UTC})
)
"""

# Written by sign-in (Step 7b-1b); no IP or user-agent column (§4.8 privacy).
SESSION_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS session (
    session_hash TEXT    PRIMARY KEY CHECK ({_HEX64.format("session_hash")}),
    account_id   INTEGER NOT NULL REFERENCES account (account_id),
    created_at   TEXT    NOT NULL CHECK (created_at {_UTC}),
    last_seen_at TEXT    NOT NULL CHECK (last_seen_at {_UTC}),
    expires_at   TEXT    NOT NULL CHECK (expires_at {_UTC})
)
"""

SIGN_IN_FAILURE_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS sign_in_failure (
    username_hmac   TEXT    PRIMARY KEY CHECK ({_HEX64.format("username_hmac")}),
    failures        INTEGER NOT NULL CHECK (failures >= 0),
    last_failure_at TEXT    NOT NULL CHECK (last_failure_at {_UTC}),
    locked_until    TEXT             CHECK (locked_until IS NULL OR locked_until {_UTC})
)
"""

VERDICTS = ("synthetic", "uncertain", "authentic")


@dataclass(frozen=True)
class AnalysisRecord:
    """One analysis_record row; field order = column order."""
    request_id: str
    created_at: str  # UTC, ISO 8601 with Z
    prediction: str
    confidence: float
    risk_level: str
    model_version: str
    profile_version: str
    policy_version: str
    notice: str
    processing_ms: int
    is_mock: bool


COLUMNS = tuple(f.name for f in fields(AnalysisRecord))


@dataclass(frozen=True)
class StoredNote:
    """One analysis_note row, without its key."""
    text: str
    updated_at: str  # UTC, ISO 8601 with Z


@dataclass(frozen=True)
class HistoryQuery:
    """The §4.4 filters, already validated. An empty verdicts means all verdicts; days are
    YYYY-MM-DD in UTC (open decision 6); q is matched as a plain substring, ignoring case."""
    verdicts: frozenset[str] = frozenset()
    day_from: str | None = None
    day_to: str | None = None
    q: str | None = None


@dataclass(frozen=True)
class HistoryPage:
    items: list[tuple[AnalysisRecord, StoredNote | None]]
    total: int  # matching all filters
    counts: dict[str, int]  # per verdict, matching day_from, day_to and q only


@dataclass(frozen=True)
class StatsFigures:
    """The §4.7 figures of one period, read in one snapshot. Every figure but today_total covers
    the period only; mock and real records are both counted, as in history (§4.7)."""
    total: int
    counts: dict[str, int]  # per verdict, same as HistoryPage.counts for the same days
    per_day: dict[str, dict[str, int]]  # YYYY-MM-DD -> per-verdict counts, days with results only
    median_processing_ms: int | None  # None when total is 0
    awaiting_total: int  # synthetic or uncertain with no note
    awaiting_recent: list[tuple[AnalysisRecord, None]]  # newest first, as in history
    today_total: int  # all verdicts on the given day, whatever the period


AWAITING_VERDICTS = frozenset({"synthetic", "uncertain"})


# Newest first; the same second keeps store order, last stored first (rowid grows with each insert).
_ORDER = "ORDER BY r.created_at DESC, r.rowid DESC"
_SELECT_STORED = (f"SELECT {', '.join('r.' + c for c in COLUMNS)}, n.text, n.updated_at "
                  "FROM analysis_record AS r LEFT JOIN analysis_note AS n ON n.request_id = r.request_id")


def _stored_from_row(row) -> tuple[AnalysisRecord, StoredNote | None]:
    *rest, is_mock, text, updated_at = row
    note = None if text is None else StoredNote(text, updated_at)
    return AnalysisRecord(*rest, is_mock=bool(is_mock)), note


def _where(query: HistoryQuery, with_verdicts: bool) -> tuple[str, list]:
    """WHERE clause from fixed fragments; every value goes through a ? parameter."""
    clauses, params = [], []
    if query.day_from is not None:
        clauses.append("substr(r.created_at, 1, 10) >= ?")
        params.append(query.day_from)
    if query.day_to is not None:
        clauses.append("substr(r.created_at, 1, 10) <= ?")
        params.append(query.day_to)
    if query.q:
        # instr, not LIKE: % and _ in q are plain characters. Request IDs are lower case (§2).
        clauses.append("instr(r.request_id, ?) > 0")
        params.append(query.q.lower())
    if with_verdicts and query.verdicts:
        verdicts = sorted(query.verdicts)
        clauses.append(f"r.prediction IN ({', '.join('?' for _ in verdicts)})")
        params.extend(verdicts)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


class AnalysisRepository:
    def __init__(self, path: Path):
        self.path = path

    def create_schema(self) -> None:
        """Create the tables if they are missing; safe to run at every start."""
        with closing(connect(self.path)) as conn, conn:
            conn.execute(SCHEMA)
            conn.execute(NOTE_SCHEMA)
            conn.execute(ACCOUNT_SCHEMA)
            conn.execute(SESSION_SCHEMA)
            conn.execute(SIGN_IN_FAILURE_SCHEMA)

    def add(self, record: AnalysisRecord) -> None:
        """Insert one record; raises sqlite3.IntegrityError on a duplicate ID or an invalid value."""
        placeholders = ", ".join("?" for _ in COLUMNS)
        with closing(connect(self.path)) as conn, conn:
            conn.execute(f"INSERT INTO analysis_record ({', '.join(COLUMNS)}) VALUES ({placeholders})",
                         astuple(record))

    def get(self, request_id: str) -> AnalysisRecord | None:
        with closing(connect(self.path)) as conn:
            row = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM analysis_record WHERE request_id = ?",
                               (request_id,)).fetchone()
        if row is None:
            return None
        *rest, is_mock = row
        return AnalysisRecord(*rest, is_mock=bool(is_mock))

    def get_stored(self, request_id: str) -> tuple[AnalysisRecord, StoredNote | None] | None:
        """One record with its note (None if it has none); None if the ID is unknown."""
        with closing(connect(self.path)) as conn:
            row = conn.execute(f"{_SELECT_STORED} WHERE r.request_id = ?", (request_id,)).fetchone()
        return None if row is None else _stored_from_row(row)

    def history(self, query: HistoryQuery, limit: int, offset: int) -> HistoryPage:
        """One page of records with their notes, newest first, plus total and per-verdict counts.
        A page past the last one has no items; its query is skipped (no huge OFFSET is sent)."""
        where_all, params_all = _where(query, with_verdicts=True)
        where_counts, params_counts = _where(query, with_verdicts=False)
        with closing(connect(self.path)) as conn, conn:  # one transaction: one consistent snapshot
            conn.execute("BEGIN")
            total = conn.execute(f"SELECT COUNT(*) FROM analysis_record AS r{where_all}", params_all).fetchone()[0]
            counts = dict.fromkeys(VERDICTS, 0)
            counts.update(conn.execute(f"SELECT r.prediction, COUNT(*) FROM analysis_record AS r{where_counts} "
                                       "GROUP BY r.prediction", params_counts).fetchall())
            rows = []
            if offset < total:
                rows = conn.execute(f"{_SELECT_STORED}{where_all} {_ORDER} LIMIT ? OFFSET ?",
                                    [*params_all, limit, offset]).fetchall()
        return HistoryPage([_stored_from_row(row) for row in rows], total, counts)

    def stats(self, day_from: str, day_to: str, today: str, recent_limit: int) -> StatsFigures:
        """The §4.7 figures for the inclusive UTC days day_from..day_to. The period filter is the one
        history() uses, so total and counts equal history's for the same days."""
        period = HistoryQuery(day_from=day_from, day_to=day_to)
        where, params = _where(period, with_verdicts=False)
        awaiting_where, awaiting_params = _where(HistoryQuery(verdicts=AWAITING_VERDICTS, day_from=day_from,
                                                              day_to=day_to), with_verdicts=True)
        awaiting_where += (" AND " if awaiting_where else " WHERE ") + "n.request_id IS NULL"
        today_where, today_params = _where(HistoryQuery(day_from=today, day_to=today), with_verdicts=False)
        joined = "FROM analysis_record AS r LEFT JOIN analysis_note AS n ON n.request_id = r.request_id"
        with closing(connect(self.path)) as conn, conn:  # one transaction: one consistent snapshot
            conn.execute("BEGIN")
            counts = dict.fromkeys(VERDICTS, 0)
            per_day: dict[str, dict[str, int]] = {}
            for day, prediction, n in conn.execute(
                    f"SELECT substr(r.created_at, 1, 10), r.prediction, COUNT(*) FROM analysis_record AS r{where} "
                    "GROUP BY substr(r.created_at, 1, 10), r.prediction", params):
                per_day.setdefault(day, dict.fromkeys(VERDICTS, 0))[prediction] = n
                counts[prediction] += n
            total = sum(counts.values())
            median = None
            if total:
                # The middle value, or the two middle values for an even total; mean rounded halves up.
                middle = [row[0] for row in conn.execute(
                    f"SELECT r.processing_ms FROM analysis_record AS r{where} ORDER BY r.processing_ms "
                    "LIMIT ? OFFSET ?", [*params, 2 - total % 2, (total - 1) // 2])]
                median = (sum(middle) * 2 // len(middle) + 1) // 2
            awaiting_total = conn.execute(f"SELECT COUNT(*) {joined}{awaiting_where}", awaiting_params).fetchone()[0]
            recent = conn.execute(f"{_SELECT_STORED}{awaiting_where} {_ORDER} LIMIT ?",
                                  [*awaiting_params, recent_limit]).fetchall()
            today_total = conn.execute(f"SELECT COUNT(*) FROM analysis_record AS r{today_where}",
                                       today_params).fetchone()[0]
        return StatsFigures(total, counts, per_day, median, awaiting_total,
                            [_stored_from_row(row) for row in recent], today_total)

    def save_note(self, request_id: str, text: str, updated_at: str) -> StoredNote | None:
        """Store or replace the note of one record; None if the record is unknown. Checked in the same
        transaction as the write; analysis_record is only read."""
        with closing(connect(self.path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM analysis_record WHERE request_id = ?", (request_id,)).fetchone() is None:
                return None
            conn.execute("INSERT INTO analysis_note (request_id, text, updated_at) VALUES (?, ?, ?) "
                         "ON CONFLICT (request_id) DO UPDATE SET text = excluded.text, "
                         "updated_at = excluded.updated_at",
                         (request_id, text, updated_at))
        return StoredNote(text, updated_at)

    def delete_note(self, request_id: str) -> bool | None:
        """Remove the note of one record: True if one was removed, False if it had none, None if the
        record is unknown. analysis_record is only read."""
        with closing(connect(self.path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM analysis_record WHERE request_id = ?", (request_id,)).fetchone() is None:
                return None
            return conn.execute("DELETE FROM analysis_note WHERE request_id = ?", (request_id,)).rowcount > 0

    def probe(self, timeout_s: float = PROBE_TIMEOUT_S) -> None:
        """Cheap read-only check that the table can be read; never writes or creates anything.
        Raises if the file is missing, is not a database, has no analysis_record table, or stays
        locked past timeout_s.

        Known limit: a database that is readable but not writable (disk full, read-only mount,
        permissions) passes, so GET /api/status can say ready while POST /api/analyses answers 500.
        It stays read-only because a write probe would take write locks on every status call; a
        write-capability check is part of open decision 7."""
        with closing(connect_readonly(self.path, timeout_s)) as conn:
            conn.execute("SELECT 1 FROM analysis_record LIMIT 1").fetchall()

    def count(self) -> int:
        with closing(connect(self.path)) as conn:
            return conn.execute("SELECT COUNT(*) FROM analysis_record").fetchone()[0]


class DuplicateAccount(Exception):
    """create_account() on a username that already exists."""


@dataclass(frozen=True)
class Account:
    """One account row."""
    account_id: int
    username: str
    password_hash: str
    disabled: bool
    created_at: str  # UTC, ISO 8601 with Z
    password_changed_at: str


ACCOUNT_COLUMNS = tuple(f.name for f in fields(Account))


@dataclass(frozen=True)
class ActiveSession:
    """A valid session, as the guard hands it to the routes. The session ID itself is never kept."""
    session_hash: str
    account_id: int
    username: str
    created_at: str  # UTC, ISO 8601 with Z: the sign-in time
    expires_at: str  # absolute end


class AccountRepository:
    """The sign-in tables (§4.8, §7). Usernames are given in their stored, lower-case form."""

    def __init__(self, path: Path):
        self.path = path

    def create_account(self, username: str, password_hash: str, now: str) -> int:
        """Insert an enabled account and return its ID; raise DuplicateAccount if the username exists."""
        with closing(connect(self.path)) as conn, conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO account (username, password_hash, disabled, created_at, password_changed_at) "
                    "VALUES (?, ?, 0, ?, ?)", (username, password_hash, now, now))
            except sqlite3.IntegrityError:
                if conn.execute("SELECT 1 FROM account WHERE username = ?", (username,)).fetchone():
                    raise DuplicateAccount(username) from None
                raise
            return cursor.lastrowid

    def get_account(self, username: str) -> Account | None:
        with closing(connect(self.path)) as conn:
            row = conn.execute(f"SELECT {', '.join(ACCOUNT_COLUMNS)} FROM account WHERE username = ?",
                               (username,)).fetchone()
        if row is None:
            return None
        *rest, disabled, created_at, changed_at = row
        return Account(*rest, disabled=bool(disabled), created_at=created_at, password_changed_at=changed_at)

    def set_password(self, username: str, password_hash: str, now: str) -> int | None:
        """Replace the hash and end all of the account's sessions in one transaction (§4.8). Return the
        number of sessions ended, or None if the account is unknown."""
        with closing(connect(self.path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT account_id FROM account WHERE username = ?", (username,)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE account SET password_hash = ?, password_changed_at = ? WHERE account_id = ?",
                         (password_hash, now, row[0]))
            return conn.execute("DELETE FROM session WHERE account_id = ?", (row[0],)).rowcount

    def set_disabled(self, username: str, disabled: bool) -> int | None:
        """Disable or enable an account. Disabling ends all of its sessions in the same transaction
        (§4.8). Return the number of sessions ended (0 when enabling), or None if the account is unknown."""
        with closing(connect(self.path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT account_id FROM account WHERE username = ?", (username,)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE account SET disabled = ? WHERE account_id = ?", (int(disabled), row[0]))
            if not disabled:
                return 0
            return conn.execute("DELETE FROM session WHERE account_id = ?", (row[0],)).rowcount

    def clear_failures(self, username_hmac: str) -> bool:
        """Reset the failure count and lock of one typed username (admin unlock). True if one was held."""
        with closing(connect(self.path)) as conn, conn:
            return conn.execute("DELETE FROM sign_in_failure WHERE username_hmac = ?",
                                (username_hmac,)).rowcount > 0

    def is_locked(self, username_hmac: str, now: str) -> bool:
        """True while a lock on this typed username has not ended (§4.8 lockout)."""
        with closing(connect(self.path)) as conn:
            row = conn.execute("SELECT locked_until FROM sign_in_failure WHERE username_hmac = ?",
                               (username_hmac,)).fetchone()
        return row is not None and row[0] is not None and row[0] > now

    def record_failure(self, username_hmac: str, now: str, stale_before: str, threshold: int,
                       locked_until: str) -> int:
        """Count one failed sign-in and return the count. It restarts at 1 when a lock has ended or the
        last failure is older than stale_before; at threshold the username is locked until locked_until.
        A lock still running is left as it is."""
        with closing(connect(self.path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT failures, last_failure_at, locked_until FROM sign_in_failure "
                               "WHERE username_hmac = ?", (username_hmac,)).fetchone()
            if row is not None and row[2] is not None and row[2] > now:
                return row[0]
            restart = row is None or row[2] is not None or row[1] < stale_before
            failures = 1 if restart else row[0] + 1
            conn.execute("INSERT INTO sign_in_failure (username_hmac, failures, last_failure_at, locked_until) "
                         "VALUES (?, ?, ?, ?) ON CONFLICT (username_hmac) DO UPDATE SET "
                         "failures = excluded.failures, last_failure_at = excluded.last_failure_at, "
                         "locked_until = excluded.locked_until",
                         (username_hmac, failures, now, locked_until if failures >= threshold else None))
        return failures

    def rehash_password(self, account_id: int, old_hash: str, new_hash: str) -> bool:
        """Store a hash with the current parameters (§4.8 rehash). Only if the stored hash is still
        old_hash; sessions and password_changed_at are kept, unlike set_password(). True if replaced."""
        with closing(connect(self.path)) as conn, conn:
            return conn.execute("UPDATE account SET password_hash = ? WHERE account_id = ? AND password_hash = ?",
                                (new_hash, account_id, old_hash)).rowcount > 0

    def start_session(self, session_hash: str, account_id: int, now: str, expires_at: str, idle_before: str,
                      old_session_hash: str | None = None) -> None:
        """Create a session in one transaction, after deleting the session of the cookie sent with the
        sign-in (no session fixation) and every ended session (§4.8)."""
        with closing(connect(self.path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if old_session_hash is not None:
                conn.execute("DELETE FROM session WHERE session_hash = ?", (old_session_hash,))
            conn.execute("DELETE FROM session WHERE expires_at <= ? OR last_seen_at < ?", (now, idle_before))
            conn.execute("INSERT INTO session (session_hash, account_id, created_at, last_seen_at, expires_at) "
                         "VALUES (?, ?, ?, ?, ?)", (session_hash, account_id, now, now, expires_at))

    def touch_session(self, session_hash: str, now: str, idle_before: str) -> ActiveSession | None:
        """The session with this hash, its last_seen_at set to now; None if unknown or ended. An ended
        session (expires_at reached, or unused since before idle_before) is deleted."""
        with closing(connect(self.path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT s.account_id, a.username, s.created_at, s.last_seen_at, s.expires_at "
                               "FROM session AS s JOIN account AS a ON a.account_id = s.account_id "
                               "WHERE s.session_hash = ?", (session_hash,)).fetchone()
            if row is None:
                return None
            account_id, username, created_at, last_seen_at, expires_at = row
            if expires_at <= now or last_seen_at < idle_before:
                conn.execute("DELETE FROM session WHERE session_hash = ?", (session_hash,))
                return None
            conn.execute("UPDATE session SET last_seen_at = ? WHERE session_hash = ?", (now, session_hash))
        return ActiveSession(session_hash, account_id, username, created_at, expires_at)

    def end_session(self, session_hash: str) -> bool:
        """Delete one session (sign-out). True if it existed."""
        with closing(connect(self.path)) as conn, conn:
            return conn.execute("DELETE FROM session WHERE session_hash = ?", (session_hash,)).rowcount > 0
