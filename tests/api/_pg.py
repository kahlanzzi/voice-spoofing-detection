"""Small helpers for API tests that read or write the PostgreSQL test database directly, as the app's
role (vsd_app), outside the repository under test."""
from __future__ import annotations

from api.db import Database


def query(db: Database, sql: str, params=()) -> list[tuple]:
    """Run one statement in its own transaction and return its rows ([] if it returns none)."""
    with db.transaction() as conn:
        cursor = conn.execute(sql, params)
        return cursor.fetchall() if cursor.description is not None else []


def columns(db: Database, table: str) -> list[str]:
    """The table's column names, in order."""
    return [name for (name,) in query(db, "SELECT column_name FROM information_schema.columns "
                                          "WHERE table_schema = 'public' AND table_name = %s "
                                          "ORDER BY ordinal_position", (table,))]


def foreign_keys(db: Database, table: str) -> list[tuple[str, str, str]]:
    """(referenced table, column, referenced column) of every foreign key of the table."""
    return sorted(query(db, """
        SELECT ref.relname, att.attname, ref_att.attname
        FROM pg_constraint AS c
        JOIN pg_class AS tbl ON tbl.oid = c.conrelid
        JOIN pg_class AS ref ON ref.oid = c.confrelid
        JOIN pg_attribute AS att ON att.attrelid = c.conrelid AND att.attnum = c.conkey[1]
        JOIN pg_attribute AS ref_att ON ref_att.attrelid = c.confrelid AND ref_att.attnum = c.confkey[1]
        WHERE c.contype = 'f' AND tbl.relname = %s""", (table,)))


def tables(db: Database) -> list[str]:
    return [name for (name,) in query(db, "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
                                          "ORDER BY tablename")]


def config_query(config, sql: str, params=()) -> list[tuple]:
    """query() on the database an ApiConfig points at, through a short-lived Database as vsd_app."""
    from tests.api.conftest import TEST_APP_PASSWORD
    database = Database.from_config(config)
    database.open(TEST_APP_PASSWORD)
    try:
        return query(database, sql, params)
    finally:
        database.close()
