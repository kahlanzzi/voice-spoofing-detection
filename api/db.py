"""PostgreSQL connections for the API store (docs/api_design.md §7, D53 (4)). The SQL lives in
api/repository.py.

One connection pool per process, opened by the app lifespan (or by the admin command) and closed at
shutdown. The API connects as the non-superuser role named in the config (vsd_app); its password
comes only from VSD_DB_APP_PASSWORD. The superuser password (VSD_DB_PASSWORD) belongs to the db
container's first start and is never read here.

Driver errors leave this module as StoreError, whose message is the driver's exception class name
only: a driver message can quote row values (a note, a password hash) in its DETAIL line, and the
API logs exceptions (SDD v2.0 §5.1: driver exceptions become service errors at the adapter boundary).

connect_readonly() opens a SQLite file read-only. It serves only the one-time import of the old
SQLite store (api/migrate_sqlite.py).
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg_pool import ConnectionPool

APP_PASSWORD_ENV = "VSD_DB_APP_PASSWORD"
APPLICATION_NAME = "vsd-api"
ISOLATION_LEVELS = ("READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE")


class StoreError(Exception):
    """The store cannot be read or written. The message is the driver's exception class name only."""


class IntegrityViolation(StoreError):
    """A key, foreign key or CHECK refused the write."""


# The store cannot be read or written: what the sign-in routes and the session check answer with 503.
STORE_ERRORS = (StoreError, OSError)


def app_password_from_env() -> str:
    """The password of the API's database role; ValueError if missing. The message never holds it."""
    value = os.environ.get(APP_PASSWORD_ENV, "")
    if not value:
        raise ValueError(f"{APP_PASSWORD_ENV} is not set: the API cannot connect to its database "
                         "(see .env.example and docs/setup.md)")
    return value


def _store_error(exc: psycopg.Error) -> StoreError:
    kind = IntegrityViolation if isinstance(exc, psycopg.errors.IntegrityError) else StoreError
    return kind(type(exc).__name__)


class Database:
    """A pool of connections to one PostgreSQL database. Build it at import time, open() it later:
    building never connects."""

    def __init__(self, host: str, port: int, name: str, user: str, *, pool_max: int = 4,
                 timeout_s: float = 5.0):
        self.conninfo = make_conninfo(host=host, port=port, dbname=name, user=user,
                                      application_name=APPLICATION_NAME,
                                      connect_timeout=max(1, round(timeout_s)))
        self.pool_max = pool_max
        self.timeout_s = timeout_s
        self._pool: ConnectionPool | None = None

    @classmethod
    def from_config(cls, config) -> "Database":
        return cls(config.database_host, config.database_port, config.database_name, config.database_user,
                   pool_max=config.database_pool_max, timeout_s=config.database_timeout_s)

    @property
    def is_open(self) -> bool:
        return self._pool is not None

    def open(self, password: str | None = None) -> None:
        """Open the pool and wait for its first connection; StoreError if the server cannot be reached
        within timeout_s. The password is kept only inside the pool's connection string."""
        if self._pool is not None:
            return
        conninfo = make_conninfo(self.conninfo, password=password) if password else self.conninfo
        # check: a connection broken by a server restart is replaced before use, not handed out.
        pool = ConnectionPool(conninfo, min_size=1, max_size=self.pool_max, timeout=self.timeout_s,
                              open=False, check=ConnectionPool.check_connection, name=APPLICATION_NAME)
        try:
            pool.open(wait=True, timeout=self.timeout_s)
        except psycopg.Error as exc:
            pool.close()
            raise _store_error(exc) from None
        self._pool = pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    @contextmanager
    def transaction(self, isolation: str | None = None, read_only: bool = False,
                    timeout_s: float | None = None) -> Iterator[psycopg.Connection]:
        """One transaction on a pooled connection: committed if the block ends normally, rolled back
        if it raises. isolation is one of ISOLATION_LEVELS (default READ COMMITTED). timeout_s bounds
        the wait for a free connection (default: the pool's). Driver errors become StoreError."""
        if self._pool is None:
            raise StoreError("DatabaseNotOpen")
        if isolation is not None and isolation not in ISOLATION_LEVELS:
            raise ValueError(f"unknown isolation level {isolation!r}")
        try:
            with self._pool.connection(timeout=timeout_s) as conn:
                if isolation is not None or read_only:
                    # Fixed fragments only: never a value from a request.
                    mode = ", ".join(filter(None, [f"ISOLATION LEVEL {isolation}" if isolation else "",
                                                   "READ ONLY" if read_only else ""]))
                    conn.execute(f"SET TRANSACTION {mode}")
                yield conn
        except psycopg.Error as exc:
            raise _store_error(exc) from None


def connect_readonly(path: Path, timeout_s: float) -> sqlite3.Connection:
    """Open an existing SQLite database file read-only (mode=ro): never creates the file or its
    folder, and refuses writes. timeout_s bounds the wait on a locked database."""
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout_s)
