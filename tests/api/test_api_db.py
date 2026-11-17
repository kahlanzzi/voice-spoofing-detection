import sqlite3
import traceback

import pytest

from api.db import (APP_PASSWORD_ENV, STORE_ERRORS, Database, IntegrityViolation, StoreError, app_password_from_env,
                    connect_readonly)
from api.repository import AccountRepository, AnalysisRepository
from tests.api._pg import query

HASH = "$argon2id$v=19$m=65536,t=3,p=4$c2FsdHNhbHRzYWx0c2FsdA$aGFzaGhhc2hoYXNoaGFzaGhhc2hoYXNoaGFzaGhhc2g"


# --- the API's role (C4: no superuser for the API) ------------------------------------------------

def test_the_app_connects_as_vsd_app_which_is_not_a_superuser(db):
    assert query(db, "SELECT current_user") == [("vsd_app",)]
    assert query(db, "SELECT rolsuper, rolcreatedb, rolcreaterole FROM pg_roles WHERE rolname = current_user") == \
        [(False, False, False)]


def test_the_app_owns_its_tables(db):
    owners = query(db, "SELECT DISTINCT tableowner FROM pg_tables WHERE schemaname = 'public'")
    assert owners == [("vsd_app",)]


# --- Database --------------------------------------------------------------------------------------

def test_building_a_database_never_connects():
    database = Database("host.invalid", 5432, "vsd", "vsd_app")
    assert not database.is_open
    with pytest.raises(StoreError, match="DatabaseNotOpen"):
        with database.transaction():
            pass


def test_the_password_is_never_in_the_connection_string_kept_on_the_object(db):
    assert "password" not in db.conninfo


def test_a_transaction_commits_or_rolls_back(db):
    with db.transaction() as conn:
        conn.execute("CREATE TABLE scratch (x INTEGER)")
    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            conn.execute("INSERT INTO scratch VALUES (1)")
            raise RuntimeError("rolled back")
    assert query(db, "SELECT COUNT(*) FROM scratch") == [(0,)]


def test_isolation_and_read_only_are_applied(db):
    with db.transaction("REPEATABLE READ", read_only=True) as conn:
        assert conn.execute("SHOW transaction_isolation").fetchone() == ("repeatable read",)
        assert conn.execute("SHOW transaction_read_only").fetchone() == ("on",)
    with pytest.raises(ValueError):
        with db.transaction("READ UNCOMMITTED; DROP TABLE account"):
            pass


def test_driver_errors_leave_as_store_errors_with_the_class_name_only(db):
    # A CHECK violation's DETAIL quotes the whole row (here a password hash and a secret username):
    # neither the message nor a logged traceback may hold it.
    AnalysisRepository(db)  # schema is there (template)
    accounts = AccountRepository(db)
    username = "-".join(["secret", "user", "7731"])  # built at run time: the traceback quotes source lines
    with pytest.raises(IntegrityViolation) as caught:
        accounts.create_account(username, HASH, "not-a-time")
    exc = caught.value
    assert str(exc) == "CheckViolation"
    assert exc.__cause__ is None and exc.__suppress_context__  # the driver error is not chained
    text = "".join(traceback.format_exception(exc))
    assert username not in text and HASH not in text
    assert isinstance(exc, STORE_ERRORS)


def test_the_app_password_comes_only_from_its_variable(monkeypatch):
    monkeypatch.setenv(APP_PASSWORD_ENV, "from-env")
    monkeypatch.setenv("VSD_DB_PASSWORD", "superuser-password")
    assert app_password_from_env() == "from-env"
    monkeypatch.delenv(APP_PASSWORD_ENV)
    with pytest.raises(ValueError, match=APP_PASSWORD_ENV) as caught:
        app_password_from_env()
    assert "superuser-password" not in str(caught.value)


# --- the old SQLite store, read-only (the one-time import) -----------------------------------------

def test_connect_readonly_does_not_create_a_missing_file_or_folder(tmp_path):
    path = tmp_path / "missing" / "api.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        connect_readonly(path, timeout_s=0.1).execute("SELECT 1").fetchall()
    assert not path.parent.exists()


def test_connect_readonly_refuses_writes(tmp_path):
    path = tmp_path / "api.sqlite"
    sqlite3.connect(path).close()
    conn = connect_readonly(path, timeout_s=0.1)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("CREATE TABLE t (x INTEGER)")
    finally:
        conn.close()
