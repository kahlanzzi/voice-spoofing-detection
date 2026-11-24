"""One-time import of the old SQLite store into the PostgreSQL store (D53 (4); SDD v2.0 §5.4).

    docker compose exec api python -m api.migrate_sqlite /app/var/api.sqlite

The SQLite file is opened read-only (the vsd-var volume is mounted read-only too) and never changed, so
it stays the rollback copy. The target is the API's own database (config and VSD_DB_APP_PASSWORD, as
the admin command), with the schema applied; it must be empty, or nothing is done. Refused when the config
selects the bundle analyzer: the old store holds mock results, and a store never holds both kinds.

All five tables are copied in one transaction: account IDs are kept (and the identity continues after
the highest one), records are inserted in their SQLite rowid order (so seq keeps the same-second order
of the history), and notes from a file older than the note author keep a null author. Before the commit,
every table is checked on both sides: the row count and a SHA-256 of its rows in key order must match,
or the transaction is rolled back. Only table names, counts and hashes are printed: never a row.

Exit codes: 0 imported and verified, 1 refused or failed (one line on stderr, nothing changed),
2 usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from api.admin import Refused, open_database
from api.config import load_api_config
from api.db import STORE_ERRORS, Database, connect_readonly
from api.repository import COLUMNS

SOURCE_TIMEOUT_S = 5.0
# table -> (columns in a fixed order, key that orders the rows for the hash). Parents before children.
TABLES = {
    "account": (("account_id", "username", "password_hash", "disabled", "created_at", "password_changed_at"),
                "account_id"),
    "analysis_record": (COLUMNS, "request_id"),
    "analysis_note": (("request_id", "text", "updated_at", "updated_by_account_id"), "request_id"),
    "session": (("session_hash", "account_id", "created_at", "last_seen_at", "expires_at"), "session_hash"),
    "sign_in_failure": (("username_hmac", "failures", "last_failure_at", "locked_until"), "username_hmac"),
}
BOOLEAN_COLUMNS = frozenset({"is_mock", "disabled"})  # INTEGER 0/1 in SQLite, BOOLEAN in PostgreSQL
NOTE_AUTHOR = "updated_by_account_id"  # missing in files made before sign-in: imported as null


def canonical(columns: tuple[str, ...], rows: list[tuple]) -> list[list]:
    """Rows as comparable JSON values: booleans as 0/1, everything else as stored."""
    flags = [name in BOOLEAN_COLUMNS for name in columns]
    return [[int(bool(value)) if flag and value is not None else value for value, flag in zip(row, flags)]
            for row in rows]


def digest(columns: tuple[str, ...], rows: list[tuple]) -> str:
    """SHA-256 of the canonical rows; the rows must already be in key order."""
    text = json.dumps(canonical(columns, rows), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_source(path: Path) -> dict[str, tuple[list[tuple], list[tuple]]]:
    """table -> (rows in insert order, rows in key order). A table the old file lacks has no rows."""
    if not path.is_file():
        raise Refused(f"No SQLite file at {path}. Nothing was changed.")
    try:
        conn = connect_readonly(path, SOURCE_TIMEOUT_S)
        try:
            present = {name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            data = {}
            for table, (columns, key) in TABLES.items():
                if table not in present:
                    data[table] = ([], [])
                    continue
                have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                select = ", ".join(c if c in have or c != NOTE_AUTHOR else f"NULL AS {c}" for c in columns)
                missing = [c for c in columns if c not in have and c != NOTE_AUTHOR]
                if missing:
                    raise Refused(f"{table} in {path.name} lacks columns {missing}. Nothing was changed.")
                in_order = conn.execute(f"SELECT {select} FROM {table} ORDER BY rowid").fetchall()
                by_key = conn.execute(f"SELECT {select} FROM {table} ORDER BY {key}").fetchall()
                data[table] = (in_order, by_key)
            return data
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        raise Refused(f"{path.name} is not a readable SQLite store ({type(exc).__name__}). "
                      "Nothing was changed.") from None


def import_store(source: dict[str, tuple[list[tuple], list[tuple]]], database: Database) -> list[str]:
    """Copy and verify in one transaction; return the report lines. Refused if the target is not empty
    or a check fails (the transaction is then rolled back)."""
    report = []
    with database.transaction() as conn:
        held = sum(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in TABLES)
        if held:
            raise Refused(f"The PostgreSQL store is not empty ({held} rows). Nothing was changed.")
        for table, (columns, _) in TABLES.items():
            rows, _ = source[table]
            if not rows:
                continue
            values = canonical(columns, rows)
            for i, name in enumerate(columns):
                if name in BOOLEAN_COLUMNS:
                    for row in values:
                        row[i] = None if row[i] is None else bool(row[i])
            overriding = " OVERRIDING SYSTEM VALUE" if table == "account" else ""
            placeholders = ", ".join("%s" for _ in columns)
            with conn.cursor() as cursor:
                cursor.executemany(f"INSERT INTO {table} ({', '.join(columns)}){overriding} "
                                   f"VALUES ({placeholders})", values)
        conn.execute("SELECT setval(pg_get_serial_sequence('account', 'account_id'), "
                     "COALESCE(MAX(account_id), 0) + 1, false) FROM account")
        for table, (columns, key) in TABLES.items():
            _, by_key = source[table]
            target = conn.execute(f"SELECT {', '.join(columns)} FROM {table} ORDER BY {key}").fetchall()
            expected, found = digest(columns, by_key), digest(columns, target)
            if len(target) != len(by_key) or found != expected:
                raise Refused(f"Verification failed for {table}: {len(by_key)} rows / {expected[:12]} in SQLite, "
                              f"{len(target)} rows / {found[:12]} in PostgreSQL. Nothing was changed.")
            report.append(f"{table:<16} {len(target):>6} rows  sha256 {found}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m api.migrate_sqlite",
                                     description="Import the old SQLite store into an empty PostgreSQL store once.")
    parser.add_argument("sqlite_file", type=Path, help="the old store, e.g. /app/var/api.sqlite")
    args = parser.parse_args(argv)
    try:
        config = load_api_config()
        if config.analyzer != "mock":
            raise Refused(f"The old store holds mock results and the config selects the {config.analyzer} analyzer. "
                          "Nothing was changed.")
        source = read_source(args.sqlite_file)
        database = open_database(config)
        try:
            report = import_store(source, database)
        except STORE_ERRORS as exc:
            raise Refused(f"The import failed ({exc}). Nothing was changed.") from None
        finally:
            database.close()
    except Refused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("\n".join(report + [f"Imported and verified: {sum(len(rows) for rows, _ in source.values())} rows "
                              f"in {len(TABLES)} tables; the SQLite file was not changed."]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
