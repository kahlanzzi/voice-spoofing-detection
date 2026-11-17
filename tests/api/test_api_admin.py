import dataclasses
import logging
import secrets

import pytest

from api import admin
from api.auth import HMAC_KEY_ENV, password_hasher, username_hmac
from api.db import APP_PASSWORD_ENV, IntegrityViolation
from api.repository import TABLES, AccountRepository
from tests.api._pg import query

PASSWORD = "correct horse battery staple"
OTHER_PASSWORD = "another long passphrase here"


@pytest.fixture
def admin_db(config, db, monkeypatch):
    """The admin command pointed at this test's database (as vsd_app); the test reads it through db."""
    monkeypatch.setattr(admin, "load_api_config", lambda: config)
    monkeypatch.delenv(HMAC_KEY_ENV, raising=False)
    return db


@pytest.fixture
def typed(monkeypatch):
    """Answers for the hidden prompts, in order; fails if the command asks for more."""
    answers = []

    def fake_getpass(prompt=""):
        return answers.pop(0)

    monkeypatch.setattr(admin.getpass, "getpass", fake_getpass)
    return answers


def dump(db) -> str:
    """Every row of every table, as text."""
    return "\n".join(repr(row) for table in TABLES for row in query(db, f"SELECT * FROM {table}"))


def add_session(db, account_id, n):
    query(db, "INSERT INTO session VALUES (%s, %s, %s, %s, %s)", (f"{n:064x}", account_id,
          "2026-11-01T09:00:00Z", "2026-11-01T09:00:00Z", "2026-11-01T21:00:00Z"))


def create(typed, username="analyst01", password=PASSWORD):
    typed[:] = [password, password]
    return admin.main(["create-account", username])


def test_create_account_stores_a_lower_case_username_and_an_argon2id_hash(admin_db, typed, capsys):
    assert create(typed, "Analyst01") == 0
    assert "Account analyst01 created." in capsys.readouterr().out
    account = AccountRepository(admin_db).get_account("analyst01")
    assert account.disabled is False
    assert account.password_hash.startswith("$argon2id$v=19$m=65536,t=3,p=4$")
    assert password_hasher().verify(account.password_hash, PASSWORD)
    assert typed == []  # asked exactly twice


def test_create_account_refuses_a_duplicate(admin_db, typed, capsys):
    create(typed)
    capsys.readouterr()
    typed[:] = [OTHER_PASSWORD, OTHER_PASSWORD]
    assert admin.main(["create-account", "ANALYST01"]) == 1
    assert "already exists" in capsys.readouterr().err
    assert password_hasher().verify(AccountRepository(admin_db).get_account("analyst01").password_hash, PASSWORD)


@pytest.mark.parametrize("answers, reason", [
    ([PASSWORD, PASSWORD + "x"], "differ"),
    (["short pass", "short pass"], "at least 15"),
    (["a" * 129, "a" * 129], "at most 128"),
    (["analyst01 is my password", "analyst01 is my password"], "username"),
    (["my voice is my password", "my voice is my password"], "service name"),
], ids=["mismatch", "too short", "too long", "username", "service word"])
def test_create_account_refuses_a_bad_password_and_stores_nothing(admin_db, typed, capsys, answers, reason):
    typed[:] = answers
    assert admin.main(["create-account", "analyst01"]) == 1
    err = capsys.readouterr().err
    assert reason in err and "Nothing was changed." in err
    assert answers[0] not in err
    assert AccountRepository(admin_db).get_account("analyst01") is None


def test_create_account_refuses_an_invalid_username_before_asking(admin_db, typed, capsys):
    assert admin.main(["create-account", "a b"]) == 1
    assert "3 to 64 characters" in capsys.readouterr().err


def test_set_password_replaces_the_hash_and_ends_all_sessions(admin_db, typed, capsys):
    create(typed)
    account_id = AccountRepository(admin_db).get_account("analyst01").account_id
    add_session(admin_db, account_id, 1)
    add_session(admin_db, account_id, 2)
    typed[:] = [OTHER_PASSWORD, OTHER_PASSWORD]
    assert admin.main(["set-password", "analyst01"]) == 0
    assert "2 session(s) ended" in capsys.readouterr().out
    stored = AccountRepository(admin_db).get_account("analyst01").password_hash
    assert password_hasher().verify(stored, OTHER_PASSWORD)
    assert query(admin_db, "SELECT COUNT(*) FROM session") == [(0,)]


def test_set_password_refuses_a_bad_password_and_keeps_the_old_one(admin_db, typed, capsys):
    create(typed)
    typed[:] = ["short", "short"]
    assert admin.main(["set-password", "analyst01"]) == 1
    assert password_hasher().verify(AccountRepository(admin_db).get_account("analyst01").password_hash, PASSWORD)


@pytest.mark.parametrize("command", ["set-password", "disable", "enable"])
def test_commands_on_an_unknown_account_are_refused(admin_db, typed, capsys, command):
    typed[:] = [PASSWORD, PASSWORD]
    assert admin.main([command, "nobody"]) == 1
    assert "No account nobody." in capsys.readouterr().err


def test_disable_ends_all_sessions_and_enable_restores_the_account(admin_db, typed, capsys):
    create(typed)
    account_id = AccountRepository(admin_db).get_account("analyst01").account_id
    add_session(admin_db, account_id, 1)
    assert admin.main(["disable", "analyst01"]) == 0
    assert "disabled; 1 session(s) ended" in capsys.readouterr().out
    assert AccountRepository(admin_db).get_account("analyst01").disabled is True
    assert admin.main(["enable", "analyst01"]) == 0
    assert "enabled" in capsys.readouterr().out
    assert AccountRepository(admin_db).get_account("analyst01").disabled is False


def add_failures(db, digest):
    query(db, "INSERT INTO sign_in_failure VALUES (%s, 5, %s, %s)",
          (digest, "2026-11-01T09:00:00Z", "2026-11-01T09:15:00Z"))


@pytest.mark.parametrize("typed_name", ["Analyst01", "not an account"])
def test_unlock_resets_the_count_of_any_typed_username(admin_db, monkeypatch, capsys, typed_name):
    key = secrets.token_bytes(32)
    monkeypatch.setenv(HMAC_KEY_ENV, key.hex())
    admin.AnalysisRepository(admin_db).create_schema()
    add_failures(admin_db, username_hmac(key, typed_name.lower()))
    assert admin.main(["unlock", typed_name]) == 0
    assert "unlocked" in capsys.readouterr().out
    assert admin.main(["unlock", typed_name]) == 0
    assert "had no failed sign-ins" in capsys.readouterr().out


def test_unlock_without_the_key_is_refused(admin_db, capsys):
    assert admin.main(["unlock", "analyst01"]) == 1
    assert HMAC_KEY_ENV in capsys.readouterr().err


def test_missing_username_is_a_usage_error(admin_db):
    with pytest.raises(SystemExit) as caught:
        admin.main(["create-account"])
    assert caught.value.code == 2


def test_the_password_never_reaches_the_output_the_log_or_the_database(admin_db, typed, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    create(typed)
    typed[:] = [OTHER_PASSWORD, OTHER_PASSWORD]
    admin.main(["set-password", "analyst01"])
    typed[:] = ["voice voice voice voice", "voice voice voice voice"]
    admin.main(["set-password", "analyst01"])
    out = capsys.readouterr()
    database = dump(admin_db)
    for secret in (PASSWORD, OTHER_PASSWORD, "voice voice voice voice"):
        assert secret not in out.out + out.err
        assert secret not in caplog.text
        assert secret not in database
    with pytest.raises(IntegrityViolation):  # the CHECK also refuses a clear password
        AccountRepository(admin_db).create_account("analyst02", PASSWORD, "2026-11-01T09:00:00Z")


def test_without_the_app_password_the_command_is_refused(admin_db, typed, monkeypatch, capsys):
    monkeypatch.delenv(APP_PASSWORD_ENV)
    typed[:] = [PASSWORD, PASSWORD]
    assert admin.main(["create-account", "analyst01"]) == 1
    assert APP_PASSWORD_ENV in capsys.readouterr().err
    assert AccountRepository(admin_db).get_account("analyst01") is None


def test_an_unreachable_database_is_refused_without_a_traceback(config, monkeypatch, typed, capsys):
    unreachable = dataclasses.replace(config, database_name="no_such_database", database_timeout_s=1)
    monkeypatch.setattr(admin, "load_api_config", lambda: unreachable)
    typed[:] = [PASSWORD, PASSWORD]
    assert admin.main(["create-account", "analyst01"]) == 1
    err = capsys.readouterr().err
    assert "database is not reachable" in err and "Nothing was changed." in err
    assert PASSWORD not in err
