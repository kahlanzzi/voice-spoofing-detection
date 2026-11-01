import re
import sqlite3
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from api.main import create_app
from api.repository import AccountRepository
from api.session_guard import (COOKIE, NEEDS_ORIGIN, NEEDS_SESSION, iso, new_session_id, session_hash, session_id_of,
                               utc_now)
from tests.api.conftest import audio_bytes, fast_hasher
from tests.api.test_api_session import PASSWORD, USERNAME, rows, sign_in


@pytest.fixture
def https(config):
    with TestClient(create_app(config), base_url="https://testserver") as test_client:
        test_client.app.state.accounts.create_account(USERNAME, fast_hasher().hash(PASSWORD), "2026-11-01T08:00:00Z")
        yield test_client


@pytest.fixture
def origin(config):
    return {"Origin": config.allowed_origins[0]}


def test_session_ids_are_43_url_safe_characters_and_stored_as_sha_256():
    session_id = new_session_id()
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", session_id)
    assert new_session_id() != session_id
    assert re.fullmatch(r"[0-9a-f]{64}", session_hash(session_id))


def test_session_id_is_read_from_the_cookie_header():
    assert session_id_of(Headers({"cookie": f"other=1; {COOKIE}=abc"})) == "abc"
    assert session_id_of(Headers({})) is None


def test_the_route_table_covers_the_session_routes_only():
    assert {path for _, path in NEEDS_ORIGIN | NEEDS_SESSION} == {"/api/session"}


@pytest.mark.parametrize("headers", [{}, {"Origin": "https://evil.example"}, {"Origin": "null"}],
                         ids=["missing", "other", "null"])
@pytest.mark.parametrize("method", ["POST", "DELETE"])
def test_origin_is_checked_first_on_post_and_delete(https, method, headers):
    response = https.request(method, "/api/session", json={"username": USERNAME, "password": PASSWORD},
                             headers=headers)
    assert response.status_code == 403
    assert response.json()["error"] == {"code": "origin_not_allowed",
                                        "message": "The request origin is not allowed.", "request_id": None}
    assert response.headers["x-request-id"].startswith("req_")


def test_no_cookie_or_an_unknown_one_is_401_and_clears_the_cookie(https, origin):
    for response in (https.get("/api/session"), https.delete("/api/session", headers=origin)):
        assert response.status_code == 401
        assert response.json()["error"] == {"code": "session_expired", "message": "No valid session.",
                                            "request_id": None}
        assert f'{COOKIE}=""' in response.headers["set-cookie"]
    https.cookies.set(COOKIE, new_session_id(), domain="testserver.local", path="/api")
    assert https.get("/api/session").status_code == 401


def test_a_valid_session_updates_last_seen_at(https, origin, config):
    sign_in(https, origin)
    a_minute_ago = iso(utc_now() - timedelta(minutes=1))
    rows(config, "UPDATE session SET last_seen_at = ?", (a_minute_ago,))
    assert https.get("/api/session").status_code == 200
    assert rows(config, "SELECT last_seen_at FROM session")[0][0] > a_minute_ago


@pytest.mark.parametrize("column", ["last_seen_at", "expires_at"], ids=["idle", "absolute"])
def test_an_ended_session_is_401_and_its_row_is_deleted(https, origin, config, column):
    sign_in(https, origin)
    rows(config, f"UPDATE session SET {column} = '2000-01-01T00:00:00Z'")
    assert https.get("/api/session").status_code == 401
    assert rows(config, "SELECT * FROM session") == []


def test_a_store_error_in_the_session_check_is_503(https, origin, monkeypatch):
    sign_in(https, origin)
    def broken(*args):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(AccountRepository, "touch_session", broken)
    response = https.get("/api/session")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "sign_in_unavailable"


def test_existing_routes_still_need_no_cookie_and_no_origin(client):
    # Step 7b-2 changes this; until then every earlier route answers as before.
    for path in ("/api/status", "/api/limits", "/api/analyses", "/api/stats"):
        assert client.get(path).status_code == 200
    upload = client.post("/api/analyses", files={"file": ("a.wav", audio_bytes(), "audio/wav")})
    assert upload.status_code == 201
    request_id = upload.json()["request_id"]
    assert client.put(f"/api/analyses/{request_id}/note", json={"text": "checked"}).status_code == 200
