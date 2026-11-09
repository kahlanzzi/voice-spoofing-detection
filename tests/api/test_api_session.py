import logging
import secrets
import sqlite3
from contextlib import closing

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from api.config import resolve_path
from api.main import create_app
from api.repository import AccountRepository
from api.session_guard import COOKIE, session_hash
from tests.api.conftest import fast_hasher

PASSWORD = "correct horse battery staple"
USERNAME = "analyst01"


@pytest.fixture
def app(config):
    return create_app(config)


@pytest.fixture
def https(app):
    # https base URL: httpx keeps the Secure cookie (§4.8 Cookie).
    with TestClient(app, base_url="https://testserver") as test_client:
        yield test_client


@pytest.fixture
def origin(config):
    return {"Origin": config.allowed_origins[0]}


@pytest.fixture
def account(https, app):
    return app.state.accounts.create_account(USERNAME, fast_hasher().hash(PASSWORD), "2026-11-01T08:00:00Z")


def sign_in(client, origin, username=USERNAME, password=PASSWORD):
    return client.post("/api/session", json={"username": username, "password": password}, headers=origin)


def rows(config, sql, params=()):
    with closing(sqlite3.connect(resolve_path(config.database_path))) as conn, conn:
        return conn.execute(sql, params).fetchall()


def failures(config):
    return rows(config, "SELECT failures, locked_until FROM sign_in_failure")


class SpyHasher(PasswordHasher):
    """The fast hasher, recording every stored hash given to verify."""

    def __init__(self):
        super().__init__(time_cost=1, memory_cost=8, parallelism=1)
        self.verified = []

    def verify(self, hash, password):
        self.verified.append(hash)
        return super().verify(hash, password)


def test_sign_in_returns_the_session_and_a_section_4_8_cookie(https, account, origin, config):
    response = sign_in(https, origin)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"username", "signed_in_at", "expires_at"}
    assert body["username"] == USERNAME
    cookie = response.headers["set-cookie"]
    session_id = cookie.split(";")[0].removeprefix(f"{COOKIE}=")
    assert len(session_id) == 43
    attributes = {part.strip().lower() for part in cookie.split(";")[1:]}
    assert attributes == {"httponly", "secure", "samesite=strict", "path=/api"}
    stored = rows(config, "SELECT session_hash, account_id FROM session")
    assert stored == [(session_hash(session_id), account)]


def test_username_is_compared_in_lower_case(https, account, origin):
    assert sign_in(https, origin, username="Analyst01").json()["username"] == USERNAME


def test_get_session_answers_the_same_shape_and_delete_signs_out(https, account, origin, config):
    signed_in = sign_in(https, origin).json()
    assert https.get("/api/session").json() == signed_in
    response = https.delete("/api/session", headers=origin)
    assert response.status_code == 204
    assert response.content == b""
    assert f'{COOKIE}=""' in response.headers["set-cookie"]
    assert rows(config, "SELECT * FROM session") == []
    assert https.get("/api/session").status_code == 401


def test_unknown_username_and_wrong_password_give_byte_identical_answers(https, account, origin):
    unknown = sign_in(https, origin, username="nobody")
    wrong = sign_in(https, origin, password=PASSWORD + "x")
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json()["error"] == {"code": "invalid_credentials",
                                       "message": "The username or password is not correct.", "request_id": None}
    assert unknown.content == wrong.content
    drop = {"x-request-id"}
    assert ({k: v for k, v in unknown.headers.items() if k not in drop}
            == {k: v for k, v in wrong.headers.items() if k not in drop})


def test_unknown_username_is_verified_against_the_dummy_hash(config, origin):
    spy = SpyHasher()
    app = create_app(config, hasher=spy)
    with TestClient(app, base_url="https://testserver") as client:
        assert sign_in(client, origin, username="nobody").status_code == 401
    assert spy.verified == [app.state.dummy_hash]


def test_a_locked_username_is_refused_without_checking_the_password(config, origin):
    spy = SpyHasher()
    app = create_app(config, hasher=spy)
    with TestClient(app, base_url="https://testserver") as client:
        app.state.accounts.create_account(USERNAME, spy.hash(PASSWORD), "2026-11-01T08:00:00Z")
        for _ in range(config.lockout_threshold):
            assert sign_in(client, origin, password="wrong password here").status_code == 401
        spy.verified.clear()
        response = sign_in(client, origin)  # the right password
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "account_locked"
    assert spy.verified == []


def test_a_password_typed_as_username_is_never_stored_logged_or_returned(https, account, origin, config, caplog):
    canary = "canary-" + secrets.token_hex(16)
    with caplog.at_level(logging.DEBUG):
        response = sign_in(https, origin, username=canary, password=canary)
    assert response.status_code == 401
    assert canary not in response.text
    assert canary not in caplog.text
    tables = [name for (name,) in rows(config, "SELECT name FROM sqlite_master WHERE type = 'table'")]
    for table in tables:
        assert canary not in repr(rows(config, f"SELECT * FROM {table}"))
    assert len(failures(config)) == 1


def test_logs_hold_no_username_and_only_the_account_id_on_success(https, account, origin, caplog):
    with caplog.at_level(logging.INFO, logger="api"):
        sign_in(https, origin, password="wrong password here")
        sign_in(https, origin)
    assert USERNAME not in caplog.text
    assert PASSWORD not in caplog.text
    assert f"account_id={account}" in caplog.text


def test_lockout_after_the_threshold_even_for_an_unknown_username(https, origin, config):
    for _ in range(config.lockout_threshold):
        assert sign_in(https, origin, username="nobody").status_code == 401
    assert sign_in(https, origin, username="NOBODY").status_code == 429


def test_success_resets_the_failure_count(https, account, origin, config):
    for _ in range(config.lockout_threshold - 1):
        sign_in(https, origin, password="wrong password here")
    assert sign_in(https, origin).status_code == 200
    assert failures(config) == []
    for _ in range(config.lockout_threshold - 1):
        sign_in(https, origin, password="wrong password here")
    assert sign_in(https, origin).status_code == 200


def test_the_count_restarts_when_the_lock_ends_or_the_last_failure_is_old(https, account, origin, config):
    for _ in range(config.lockout_threshold):
        sign_in(https, origin, password="wrong password here")
    rows(config, "UPDATE sign_in_failure SET locked_until = '2000-01-01T00:00:00Z'")
    assert sign_in(https, origin, password="wrong password here").status_code == 401
    assert failures(config) == [(1, None)]
    rows(config, "UPDATE sign_in_failure SET failures = ?, last_failure_at = '2000-01-01T00:00:00Z'",
         (config.lockout_threshold - 1,))
    sign_in(https, origin, password="wrong password here")
    assert failures(config) == [(1, None)]


def test_a_disabled_account_is_403_and_neither_counts_nor_resets_failures(https, account, origin, config, app):
    sign_in(https, origin, password="wrong password here")
    app.state.accounts.set_disabled(USERNAME, True)
    response = sign_in(https, origin)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "access_denied"
    assert "set-cookie" not in response.headers
    assert failures(config) == [(1, None)]
    assert rows(config, "SELECT * FROM session") == []


def test_a_disabled_account_with_a_wrong_password_is_401(https, account, origin, app):
    app.state.accounts.set_disabled(USERNAME, True)
    assert sign_in(https, origin, password="wrong password here").status_code == 401


def test_sign_in_replaces_the_session_sent_with_it(https, account, origin, config):
    sign_in(https, origin)
    first = rows(config, "SELECT session_hash FROM session")
    sign_in(https, origin)  # the client sends the first cookie
    second = rows(config, "SELECT session_hash FROM session")
    assert len(first) == len(second) == 1
    assert first != second


def test_sign_in_deletes_ended_sessions(https, account, origin, config):
    sign_in(https, origin)
    rows(config, "UPDATE session SET expires_at = '2000-01-01T00:00:00Z'")
    https.cookies.clear()
    sign_in(https, origin)
    assert len(rows(config, "SELECT * FROM session")) == 1


def test_an_old_hash_is_rehashed_at_sign_in_without_ending_sessions(https, origin, config, app):
    old = PasswordHasher(time_cost=2, memory_cost=16, parallelism=1).hash(PASSWORD)
    app.state.accounts.create_account(USERNAME, old, "2026-11-01T08:00:00Z")
    sign_in(https, origin)
    https.cookies.clear()
    sign_in(https, origin)
    account = app.state.accounts.get_account(USERNAME)
    assert account.password_hash != old
    assert not app.state.hasher.check_needs_rehash(account.password_hash)
    assert account.password_changed_at == "2026-11-01T08:00:00Z"
    assert len(rows(config, "SELECT * FROM session")) == 2


def test_the_real_hasher_gives_the_dummy_hash_and_rehash_the_section_4_8_parameters(real_hasher, config, origin):
    app = create_app(config)
    with TestClient(app, base_url="https://testserver") as client:
        assert app.state.dummy_hash.startswith("$argon2id$v=19$m=65536,t=3,p=4$")
        app.state.accounts.create_account(USERNAME, fast_hasher().hash(PASSWORD), "2026-11-01T08:00:00Z")
        assert sign_in(client, origin).status_code == 200
    assert app.state.accounts.get_account(USERNAME).password_hash.startswith("$argon2id$v=19$m=65536,t=3,p=4$")


@pytest.mark.parametrize("body", [
    {"username": USERNAME}, {"password": PASSWORD}, {"username": "", "password": PASSWORD},
    {"username": USERNAME, "password": ""}, {"username": 1, "password": PASSWORD},
    {"username": USERNAME, "password": PASSWORD, "remember": True}, [USERNAME, PASSWORD]],
    ids=["no password", "no username", "empty username", "empty password", "number", "extra field", "list"])
def test_a_bad_body_is_400_without_echoing_it(https, account, origin, body):
    response = https.post("/api/session", json=body, headers=origin)
    assert response.status_code == 400
    assert response.json()["error"] == {"code": "invalid_request", "message": "The request is not valid.",
                                        "request_id": None}
    assert PASSWORD not in response.text


def test_a_password_over_the_limit_or_a_non_json_body_is_400(https, account, origin, config):
    assert sign_in(https, origin, password="x" * (config.password_max_chars + 1)).status_code == 400
    response = https.post("/api/session", content=b"username=a&password=b", headers=origin)
    assert response.status_code == 400


def test_a_body_over_4_kib_is_400_by_length_or_while_streaming(https, account, origin):
    big = b'{"username": "analyst01", "password": "' + b"x" * 4096 + b'"}'
    headers = {**origin, "Content-Type": "application/json"}
    assert https.post("/api/session", content=big, headers=headers).status_code == 400
    streamed = https.post("/api/session", content=iter([big]), headers=headers)
    assert streamed.status_code == 400


def test_a_store_error_is_503_sign_in_unavailable(https, account, origin, monkeypatch):
    def broken(*args):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(AccountRepository, "is_locked", broken)
    response = sign_in(https, origin)
    assert response.status_code == 503
    assert response.json()["error"] == {"code": "sign_in_unavailable",
                                        "message": "The sign-in service is unavailable.", "request_id": None}


def test_a_store_error_at_sign_out_is_503(https, account, origin, monkeypatch):
    sign_in(https, origin)
    def broken(*args):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(AccountRepository, "end_session", broken)
    assert https.delete("/api/session", headers=origin).status_code == 503
