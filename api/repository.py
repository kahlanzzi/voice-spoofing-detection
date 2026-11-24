"""All SQL of the API (docs/api_design.md §7): the analysis_record and analysis_note tables, and
the sign-in tables account, session and sign_in_failure (§4.8).
probe() is the read-only check behind GET /api/status; get_stored() and history() serve §4.5 and
§4.4; save_note() and delete_note() serve §4.6; stats() serves §4.7. Notes live in their own table
and never change analysis_record (FR-11). AccountRepository serves the admin command (api/admin.py), sign-in
(api/routers/session.py) and the session check (api/session_guard.py).

The store is PostgreSQL (D53 (4)); the tables are defined in api/schema/NNNN_*.sql and applied by
create_schema(); connections and the driver-error mapping are in api/db.py.

Never stored: audio, the original file name, the client IP address, passwords in clear (only the
argon2id hash), session IDs in clear (only their SHA-256), the typed username of a failed sign-in
(only its HMAC). No table has a column for them.
"""
from __future__ import annotations

import re
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import psycopg

from api.db import Database

PROBE_TIMEOUT_S = 1.0  # internal probe setting, not a §9 contract value

# Schema files: api/schema/NNNN_name.sql, applied in order once each and recorded in schema_migration.
SCHEMA_DIR = Path(__file__).with_name("schema")
SCHEMA_FILE = re.compile(r"(\d{4})_[a-z0-9_]+\.sql")
MIGRATION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migration (
    version    INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""
TABLES = ("analysis_record", "account", "analysis_note", "session", "sign_in_failure")

# Transaction-level advisory locks (pg_advisory_xact_lock(namespace, key)): released at commit or rollback.
LOCK_SCHEMA = 5301  # key 0: one schema change at a time (API start and admin commands can race)
LOCK_SIGN_IN_FAILURE = 5302  # key hashtext(username_hmac): one failure count update at a time per username


class SchemaError(Exception):
    """The schema files or the database's applied versions are not what this code expects."""


def schema_files(directory: Path | None = None) -> list[tuple[int, str, str]]:
    """(version, file name, SQL) of every schema file in directory (default SCHEMA_DIR), by version;
    SchemaError on a stray file, a duplicate version or a gap (versions must be 1, 2, 3, ...)."""
    found = []
    for path in sorted((directory or SCHEMA_DIR).iterdir()):
        if path.suffix != ".sql":
            continue
        match = SCHEMA_FILE.fullmatch(path.name)
        if match is None:
            raise SchemaError(f"schema file name must be NNNN_name.sql: {path.name}")
        found.append((int(match.group(1)), path.name, path.read_text(encoding="utf-8")))
    versions = [version for version, _, _ in found]
    if versions != list(range(1, len(found) + 1)):
        raise SchemaError(f"schema versions must be 1..n with no gap or duplicate, got {versions}")
    return found


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
    """One analysis_note row, without its key; the author as a username."""
    text: str
    updated_at: str  # UTC, ISO 8601 with Z
    updated_by: str | None  # None for notes saved before sign-in existed


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


# Newest first; the same second keeps store order, last stored first (seq grows with each insert).
_ORDER = "ORDER BY r.created_at DESC, r.seq DESC"
_SELECT_STORED = (f"SELECT {', '.join('r.' + c for c in COLUMNS)}, n.text, n.updated_at, u.username "
                  "FROM analysis_record AS r LEFT JOIN analysis_note AS n ON n.request_id = r.request_id "
                  "LEFT JOIN account AS u ON u.account_id = n.updated_by_account_id")


def _stored_from_row(row) -> tuple[AnalysisRecord, StoredNote | None]:
    *rest, is_mock, text, updated_at, updated_by = row
    note = None if text is None else StoredNote(text, updated_at, updated_by)
    return AnalysisRecord(*rest, is_mock=bool(is_mock)), note


def _where(query: HistoryQuery, with_verdicts: bool) -> tuple[str, list]:
    """WHERE clause from fixed fragments; every value goes through a %s parameter."""
    clauses, params = [], []
    if query.day_from is not None:
        clauses.append("substr(r.created_at, 1, 10) >= %s")
        params.append(query.day_from)
    if query.day_to is not None:
        clauses.append("substr(r.created_at, 1, 10) <= %s")
        params.append(query.day_to)
    if query.q:
        # strpos, not LIKE: % and _ in q are plain characters. Request IDs are lower case (§2).
        clauses.append("strpos(r.request_id, %s) > 0")
        params.append(query.q.lower())
    if with_verdicts and query.verdicts:
        clauses.append("r.prediction = ANY(%s)")
        params.append(sorted(query.verdicts))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


# The docs/setup.md section on a separate store. Never the one that empties a store: that deletes the History.
REAL_STACK_SECTION = "Real model stack (vsd-real)"


class MixedStoreError(Exception):
    """The store holds results of the other kind (mock or real): the API must not start."""


class AnalysisRepository:
    def __init__(self, db: Database):
        self.db = db

    def create_schema(self) -> None:
        """Apply every schema file not yet recorded in schema_migration, in one transaction under an
        advisory lock: a failure leaves the database as it was, and two starts at once apply each file
        once. Safe to run at every start; no row is rewritten. SchemaError if the database holds a
        version this code does not know (a newer API wrote it)."""
        files = schema_files()
        with self.db.transaction() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s, 0)", (LOCK_SCHEMA,))
            conn.execute(MIGRATION_TABLE)
            applied = {version for (version,) in conn.execute("SELECT version FROM schema_migration")}
            unknown = sorted(applied - {version for version, _, _ in files})
            if unknown:
                raise SchemaError(f"the database has schema versions {unknown} that this API does not know")
            for version, name, sql in files:
                if version not in applied:
                    conn.execute(sql)
                    conn.execute("INSERT INTO schema_migration (version, name) VALUES (%s, %s)", (version, name))

    def schema_versions(self) -> list[int]:
        with self.db.transaction(read_only=True) as conn:
            return [version for (version,) in conn.execute("SELECT version FROM schema_migration ORDER BY version")]

    def add(self, record: AnalysisRecord) -> None:
        """Insert one record; raises IntegrityViolation on a duplicate ID or an invalid value."""
        placeholders = ", ".join("%s" for _ in COLUMNS)
        with self.db.transaction() as conn:
            conn.execute(f"INSERT INTO analysis_record ({', '.join(COLUMNS)}) VALUES ({placeholders})",
                         astuple(record))

    def require_one_kind(self, mock: bool) -> None:
        """Raise MixedStoreError if any record is of the other kind than this analyzer's (§4.7: the figures
        never mix mock and real results). Checked at start, before the API serves."""
        with self.db.transaction(read_only=True) as conn:
            (other,) = conn.execute("SELECT EXISTS (SELECT 1 FROM analysis_record WHERE is_mock <> %s)",
                                    (mock,)).fetchone()
        if other:
            ours, theirs = ("mock", "real") if mock else ("real", "mock")
            raise MixedStoreError(f"the analyzer gives {ours} results but the store holds {theirs} results; run the "
                                  f"{ours} analyzer on a store of its own (docs/setup.md, \"{REAL_STACK_SECTION}\")")

    def get(self, request_id: str) -> AnalysisRecord | None:
        with self.db.transaction(read_only=True) as conn:
            row = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM analysis_record WHERE request_id = %s",
                               (request_id,)).fetchone()
        if row is None:
            return None
        *rest, is_mock = row
        return AnalysisRecord(*rest, is_mock=bool(is_mock))

    def get_stored(self, request_id: str) -> tuple[AnalysisRecord, StoredNote | None] | None:
        """One record with its note (None if it has none); None if the ID is unknown."""
        with self.db.transaction(read_only=True) as conn:
            row = conn.execute(f"{_SELECT_STORED} WHERE r.request_id = %s", (request_id,)).fetchone()
        return None if row is None else _stored_from_row(row)

    def history(self, query: HistoryQuery, limit: int, offset: int) -> HistoryPage:
        """One page of records with their notes, newest first, plus total and per-verdict counts.
        A page past the last one has no items; its query is skipped (no huge OFFSET is sent)."""
        where_all, params_all = _where(query, with_verdicts=True)
        where_counts, params_counts = _where(query, with_verdicts=False)
        # One REPEATABLE READ transaction: every query sees the same snapshot.
        with self.db.transaction("REPEATABLE READ", read_only=True) as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM analysis_record AS r{where_all}", params_all).fetchone()[0]
            counts = dict.fromkeys(VERDICTS, 0)
            counts.update(conn.execute(f"SELECT r.prediction, COUNT(*) FROM analysis_record AS r{where_counts} "
                                       "GROUP BY r.prediction", params_counts).fetchall())
            rows = []
            if offset < total:
                rows = conn.execute(f"{_SELECT_STORED}{where_all} {_ORDER} LIMIT %s OFFSET %s",
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
        # One REPEATABLE READ transaction: every query sees the same snapshot.
        with self.db.transaction("REPEATABLE READ", read_only=True) as conn:
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
                    "LIMIT %s OFFSET %s", [*params, 2 - total % 2, (total - 1) // 2])]
                median = (sum(middle) * 2 // len(middle) + 1) // 2
            awaiting_total = conn.execute(f"SELECT COUNT(*) {joined}{awaiting_where}", awaiting_params).fetchone()[0]
            recent = conn.execute(f"{_SELECT_STORED}{awaiting_where} {_ORDER} LIMIT %s",
                                  [*awaiting_params, recent_limit]).fetchall()
            today_total = conn.execute(f"SELECT COUNT(*) FROM analysis_record AS r{today_where}",
                                       today_params).fetchone()[0]
        return StatsFigures(total, counts, per_day, median, awaiting_total,
                            [_stored_from_row(row) for row in recent], today_total)

    def save_note(self, request_id: str, text: str, updated_at: str, account_id: int) -> StoredNote | None:
        """Store or replace the note of one record, with account_id as its author (a replace also replaces
        the author); None if the record is unknown. Checked in the same transaction as the write;
        analysis_record is only read (records are never deleted, so the check cannot go stale)."""
        with self.db.transaction() as conn:
            if conn.execute("SELECT 1 FROM analysis_record WHERE request_id = %s", (request_id,)).fetchone() is None:
                return None
            conn.execute("INSERT INTO analysis_note (request_id, text, updated_at, updated_by_account_id) "
                         "VALUES (%s, %s, %s, %s) ON CONFLICT (request_id) DO UPDATE SET text = excluded.text, "
                         "updated_at = excluded.updated_at, updated_by_account_id = excluded.updated_by_account_id",
                         (request_id, text, updated_at, account_id))
            username = conn.execute("SELECT username FROM account WHERE account_id = %s", (account_id,)).fetchone()[0]
        return StoredNote(text, updated_at, username)

    def delete_note(self, request_id: str) -> bool | None:
        """Remove the note of one record: True if one was removed, False if it had none, None if the
        record is unknown. analysis_record is only read."""
        with self.db.transaction() as conn:
            if conn.execute("SELECT 1 FROM analysis_record WHERE request_id = %s", (request_id,)).fetchone() is None:
                return None
            return conn.execute("DELETE FROM analysis_note WHERE request_id = %s", (request_id,)).rowcount > 0

    def probe(self, timeout_s: float = PROBE_TIMEOUT_S) -> None:
        """Cheap read-only check that the table can be read; never writes or creates anything.
        Raises StoreError if the pool is not open, no connection is free or reachable within
        timeout_s, the table is missing, or the read takes longer than timeout_s.

        Known limit: a database that is readable but not writable (disk full, a read-only role)
        passes, so GET /api/status can say ready while POST /api/analyses answers 500. It stays
        read-only because a write probe would write on every status call; a write-capability check is
        part of open decision 7."""
        with self.db.transaction(read_only=True, timeout_s=timeout_s) as conn:
            conn.execute(f"SET LOCAL statement_timeout = {max(1, int(timeout_s * 1000))}")
            conn.execute("SELECT 1 FROM analysis_record LIMIT 1").fetchall()

    def count(self) -> int:
        with self.db.transaction(read_only=True) as conn:
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

    def __init__(self, db: Database):
        self.db = db

    def create_account(self, username: str, password_hash: str, now: str) -> int:
        """Insert an enabled account and return its ID; raise DuplicateAccount if the username exists."""
        with self.db.transaction() as conn:
            try:
                return conn.execute(
                    "INSERT INTO account (username, password_hash, disabled, created_at, password_changed_at) "
                    "VALUES (%s, %s, false, %s, %s) RETURNING account_id",
                    (username, password_hash, now, now)).fetchone()[0]
            except psycopg.errors.UniqueViolation as exc:
                if exc.diag.constraint_name == "account_username_key":
                    raise DuplicateAccount(username) from None
                raise

    def get_account(self, username: str) -> Account | None:
        with self.db.transaction(read_only=True) as conn:
            row = conn.execute(f"SELECT {', '.join(ACCOUNT_COLUMNS)} FROM account WHERE username = %s",
                               (username,)).fetchone()
        if row is None:
            return None
        *rest, disabled, created_at, changed_at = row
        return Account(*rest, disabled=bool(disabled), created_at=created_at, password_changed_at=changed_at)

    def set_password(self, username: str, password_hash: str, now: str) -> int | None:
        """Replace the hash and end all of the account's sessions in one transaction (§4.8). Return the
        number of sessions ended, or None if the account is unknown."""
        with self.db.transaction() as conn:
            row = conn.execute("SELECT account_id FROM account WHERE username = %s FOR UPDATE",
                               (username,)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE account SET password_hash = %s, password_changed_at = %s WHERE account_id = %s",
                         (password_hash, now, row[0]))
            return conn.execute("DELETE FROM session WHERE account_id = %s", (row[0],)).rowcount

    def set_disabled(self, username: str, disabled: bool) -> int | None:
        """Disable or enable an account. Disabling ends all of its sessions in the same transaction
        (§4.8). Return the number of sessions ended (0 when enabling), or None if the account is unknown."""
        with self.db.transaction() as conn:
            row = conn.execute("SELECT account_id FROM account WHERE username = %s FOR UPDATE",
                               (username,)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE account SET disabled = %s WHERE account_id = %s", (bool(disabled), row[0]))
            if not disabled:
                return 0
            return conn.execute("DELETE FROM session WHERE account_id = %s", (row[0],)).rowcount

    def clear_failures(self, username_hmac: str) -> bool:
        """Reset the failure count and lock of one typed username (admin unlock). True if one was held."""
        with self.db.transaction() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s, hashtext(%s))", (LOCK_SIGN_IN_FAILURE, username_hmac))
            return conn.execute("DELETE FROM sign_in_failure WHERE username_hmac = %s",
                                (username_hmac,)).rowcount > 0

    def is_locked(self, username_hmac: str, now: str) -> bool:
        """True while a lock on this typed username has not ended (§4.8 lockout)."""
        with self.db.transaction(read_only=True) as conn:
            row = conn.execute("SELECT locked_until FROM sign_in_failure WHERE username_hmac = %s",
                               (username_hmac,)).fetchone()
        return row is not None and row[0] is not None and row[0] > now

    def record_failure(self, username_hmac: str, now: str, stale_before: str, threshold: int,
                       locked_until: str) -> int:
        """Count one failed sign-in and return the count. It restarts at 1 when a lock has ended or the
        last failure is older than stale_before; at threshold the username is locked until locked_until.
        A lock still running is left as it is. An advisory lock per username makes the read and the
        write one step, so two failures at once count as two (the row may not exist yet, so a row lock
        cannot do this)."""
        with self.db.transaction() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s, hashtext(%s))", (LOCK_SIGN_IN_FAILURE, username_hmac))
            row = conn.execute("SELECT failures, last_failure_at, locked_until FROM sign_in_failure "
                               "WHERE username_hmac = %s", (username_hmac,)).fetchone()
            if row is not None and row[2] is not None and row[2] > now:
                return row[0]
            restart = row is None or row[2] is not None or row[1] < stale_before
            failures = 1 if restart else row[0] + 1
            conn.execute("INSERT INTO sign_in_failure (username_hmac, failures, last_failure_at, locked_until) "
                         "VALUES (%s, %s, %s, %s) ON CONFLICT (username_hmac) DO UPDATE SET "
                         "failures = excluded.failures, last_failure_at = excluded.last_failure_at, "
                         "locked_until = excluded.locked_until",
                         (username_hmac, failures, now, locked_until if failures >= threshold else None))
        return failures

    def rehash_password(self, account_id: int, old_hash: str, new_hash: str) -> bool:
        """Store a hash with the current parameters (§4.8 rehash). Only if the stored hash is still
        old_hash; sessions and password_changed_at are kept, unlike set_password(). True if replaced."""
        with self.db.transaction() as conn:
            return conn.execute("UPDATE account SET password_hash = %s WHERE account_id = %s AND password_hash = %s",
                                (new_hash, account_id, old_hash)).rowcount > 0

    def start_session(self, session_hash: str, account_id: int, now: str, expires_at: str, idle_before: str,
                      old_session_hash: str | None = None) -> None:
        """Create a session in one transaction, after deleting the session of the cookie sent with the
        sign-in (no session fixation) and every ended session (§4.8)."""
        with self.db.transaction() as conn:
            if old_session_hash is not None:
                conn.execute("DELETE FROM session WHERE session_hash = %s", (old_session_hash,))
            conn.execute("DELETE FROM session WHERE expires_at <= %s OR last_seen_at < %s", (now, idle_before))
            conn.execute("INSERT INTO session (session_hash, account_id, created_at, last_seen_at, expires_at) "
                         "VALUES (%s, %s, %s, %s, %s)", (session_hash, account_id, now, now, expires_at))

    def touch_session(self, session_hash: str, now: str, idle_before: str) -> ActiveSession | None:
        """The session with this hash, its last_seen_at set to now; None if unknown or ended. An ended
        session (expires_at reached, or unused since before idle_before) is deleted. The row is locked
        from the read to the write."""
        with self.db.transaction() as conn:
            row = conn.execute("SELECT s.account_id, a.username, s.created_at, s.last_seen_at, s.expires_at "
                               "FROM session AS s JOIN account AS a ON a.account_id = s.account_id "
                               "WHERE s.session_hash = %s FOR UPDATE OF s", (session_hash,)).fetchone()
            if row is None:
                return None
            account_id, username, created_at, last_seen_at, expires_at = row
            if expires_at <= now or last_seen_at < idle_before:
                conn.execute("DELETE FROM session WHERE session_hash = %s", (session_hash,))
                return None
            conn.execute("UPDATE session SET last_seen_at = %s WHERE session_hash = %s", (now, session_hash))
        return ActiveSession(session_hash, account_id, username, created_at, expires_at)

    def end_session(self, session_hash: str) -> bool:
        """Delete one session (sign-out). True if it existed."""
        with self.db.transaction() as conn:
            return conn.execute("DELETE FROM session WHERE session_hash = %s", (session_hash,)).rowcount > 0
