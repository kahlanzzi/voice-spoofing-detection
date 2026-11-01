"""All SQL of the API (docs/api_design.md §7). Only analysis_record for now; analysis_note comes
with the notes endpoint. probe() is the read-only check behind GET /api/status.

Never stored: audio, the original file name, the client IP address. The table has no column for them.
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


class AnalysisRepository:
    def __init__(self, path: Path):
        self.path = path

    def create_schema(self) -> None:
        """Create the tables if they are missing; safe to run at every start."""
        with closing(connect(self.path)) as conn, conn:
            conn.execute(SCHEMA)

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
