"""All SQL of the API (docs/api_design.md §7): the analysis_record and analysis_note tables.
probe() is the read-only check behind GET /api/status; get_stored() and history() serve §4.5 and
§4.4; save_note() and delete_note() serve §4.6; stats() serves §4.7. Notes live in their own table
and never change analysis_record (FR-11).

Never stored: audio, the original file name, the client IP address. No table has a column for them.
"""
from __future__ import annotations

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
