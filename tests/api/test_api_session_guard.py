import asyncio
import json
import re
import sqlite3
from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from api.main import create_app
from api.repository import AccountRepository
from api.session_guard import (COOKIE, iso, needs_origin, needs_session, new_session_id, session_hash, session_id_of,
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


REQUEST_ID = "req_" + "a" * 32
NOTE = f"/api/analyses/{REQUEST_ID}/note"
EXPIRED = {"code": "session_expired", "message": "No valid session.", "request_id": None}
ORIGIN_REFUSED = {"code": "origin_not_allowed", "message": "The request origin is not allowed.", "request_id": None}
INTERNAL = {"code": "internal_error", "message": "An unexpected error occurred.", "request_id": None}

# Every route under /api but POST /session and GET /health (§4.8), plus an unknown path and a wrong method.
PROTECTED = [("GET", "/api/status"), ("GET", "/api/limits"), ("GET", "/api/analyses"), ("POST", "/api/analyses"),
             ("GET", f"/api/analyses/{REQUEST_ID}"), ("PUT", NOTE), ("GET", "/api/stats"), ("GET", "/api/session"),
             ("DELETE", "/api/session"), ("GET", "/api/does-not-exist"), ("POST", "/api/status")]
# The routes that change data (§4.8 CSRF).
CHANGES_DATA = [("POST", "/api/analyses"), ("PUT", NOTE), ("POST", "/api/session"), ("DELETE", "/api/session")]


@pytest.mark.parametrize("method, path", PROTECTED)
def test_needs_session_covers_every_api_route_but_two(method, path):
    assert needs_session(method, path)


@pytest.mark.parametrize("method, path", [("POST", "/api/session"), ("GET", "/api/health"), ("GET", "/status"),
                                          ("GET", "/docs")])
def test_needs_session_leaves_sign_in_health_and_paths_outside_api_open(method, path):
    assert not needs_session(method, path)


def test_needs_origin_covers_exactly_the_routes_that_change_data():
    assert all(needs_origin(method, path) for method, path in CHANGES_DATA)
    assert not any(needs_origin(method, path) for method, path in PROTECTED if (method, path) not in CHANGES_DATA)
    assert needs_origin("PUT", "/api/analyses/not-an-id/note")  # checked before the route's 404
    assert not needs_origin("GET", "/api/health")


@pytest.mark.parametrize("method, path", PROTECTED)
def test_every_protected_route_is_401_without_a_cookie(anon, config, method, path):
    headers = {"Origin": config.allowed_origins[0]}  # so the session check is the one that answers
    response = anon.request(method, path, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"error": EXPIRED}
    assert f'{COOKIE}=""' in response.headers["set-cookie"]


def test_health_and_sign_in_need_no_cookie(anon, config):
    assert anon.get("/api/health").json() == {"ok": True}
    response = anon.post("/api/session", json={"username": "nobody", "password": "x" * 20},
                         headers={"Origin": config.allowed_origins[0]})
    assert response.json()["error"]["code"] == "invalid_credentials"


@pytest.mark.parametrize("origin_header", [None, "https://evil.example"], ids=["missing", "other"])
def test_upload_and_note_without_an_allowed_origin_are_403_and_change_nothing(client, config, origin_header):
    first = client.post("/api/analyses", files={"file": ("a.wav", audio_bytes(), "audio/wav")})
    request_id = first.json()["request_id"]
    if origin_header is None:
        del client.headers["Origin"]
    else:
        client.headers["Origin"] = origin_header
    upload = client.post("/api/analyses", files={"file": ("a.wav", audio_bytes(), "audio/wav")})
    note = client.put(f"/api/analyses/{request_id}/note", json={"text": "checked"})
    for response in (upload, note):
        assert response.status_code == 403
        assert response.json() == {"error": ORIGIN_REFUSED}
    assert client.app.state.repository.count() == 1
    assert client.get(f"/api/analyses/{request_id}").json()["note"] is None


def test_origin_is_checked_before_the_session(anon):
    for response in (anon.post("/api/analyses", files={"file": ("a.wav", audio_bytes(), "audio/wav")}),
                     anon.put(NOTE, json={"text": "checked"})):
        assert response.status_code == 403
        assert response.json() == {"error": ORIGIN_REFUSED}


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


@pytest.mark.parametrize("method, path", [("GET", "/api/status"), ("GET", "/api/stats"), ("POST", "/api/analyses"),
                                          ("PUT", NOTE)])
def test_a_store_error_in_the_session_check_is_500_on_other_routes(client, monkeypatch, method, path):
    def broken(*args):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(AccountRepository, "touch_session", broken)
    response = client.request(method, path)
    assert response.status_code == 500
    assert response.json() == {"error": INTERNAL}


def test_signed_in_every_route_answers_as_before(client):
    for path in ("/api/status", "/api/limits", "/api/analyses", "/api/stats", "/api/session"):
        assert client.get(path).status_code == 200
    upload = client.post("/api/analyses", files={"file": ("a.wav", audio_bytes(), "audio/wav")})
    assert upload.status_code == 201
    request_id = upload.json()["request_id"]
    assert client.put(f"/api/analyses/{request_id}/note", json={"text": "checked"}).status_code == 200
    assert client.get("/api/does-not-exist").status_code == 404
    assert client.post("/api/status").status_code == 405


# --- Refused before any request body is read (§4.8) -------------------------------------------

def call_asgi(app, method, path, headers, body):
    """Send one request straight to the ASGI app. Return the status, the JSON body and the number of
    times the app asked for the request body: a refusal before the body is read asks 0 times."""
    reads, sent = [], []

    async def receive():
        reads.append(1)
        if len(reads) == 1:
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "https",
             "server": ("testserver", 443), "client": ("testclient", 50000), "root_path": "", "path": path,
             "raw_path": path.encode(), "query_string": b"",
             "headers": [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()]}
    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    content = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, json.loads(content) if content else None, len(reads)


def upload_request():
    """A real multipart upload: headers and body bytes."""
    request = httpx.Request("POST", "https://testserver/api/analyses",
                            files={"file": ("a.wav", audio_bytes(), "audio/wav")})
    return dict(request.headers), request.read()


def note_request(request_id):
    body = json.dumps({"text": "checked"}).encode()
    return f"/api/analyses/{request_id}/note", {"content-type": "application/json",
                                                 "content-length": str(len(body))}, body


@pytest.mark.parametrize("case", ["no cookie", "no origin", "unknown cookie", "expired cookie"])
def test_upload_and_note_are_refused_without_reading_the_body(https, origin, config, case):
    sign_in(https, origin)
    session_id = https.cookies.get(COOKIE)
    headers = {"cookie": f"{COOKIE}={session_id}", **{k.lower(): v for k, v in origin.items()}}
    if case == "no cookie":
        del headers["cookie"]
    elif case == "no origin":
        del headers["origin"]
    elif case == "unknown cookie":
        headers["cookie"] = f"{COOKIE}={new_session_id()}"
    else:
        rows(config, "UPDATE session SET expires_at = '2000-01-01T00:00:00Z'")
    expected = (403, {"error": ORIGIN_REFUSED}) if case == "no origin" else (401, {"error": EXPIRED})

    upload_headers, upload_body = upload_request()
    status, body, reads = call_asgi(https.app, "POST", "/api/analyses", {**upload_headers, **headers}, upload_body)
    assert (status, body, reads) == (*expected, 0)  # request_id null: refused before the upload gets one

    path, note_headers, note_body = note_request(REQUEST_ID)
    status, body, reads = call_asgi(https.app, "PUT", path, {**note_headers, **headers}, note_body)
    assert (status, body, reads) == (*expected, 0)
    assert https.app.state.repository.count() == 0


def test_with_a_session_and_origin_the_body_is_read(https, origin):
    # Control for the test above: the same probe does see the body being read once the checks pass.
    sign_in(https, origin)
    headers = {"cookie": f"{COOKIE}={https.cookies.get(COOKIE)}", "origin": origin["Origin"]}
    upload_headers, upload_body = upload_request()
    status, body, reads = call_asgi(https.app, "POST", "/api/analyses", {**upload_headers, **headers}, upload_body)
    assert status == 201 and reads >= 1
    path, note_headers, note_body = note_request(body["request_id"])
    status, body, reads = call_asgi(https.app, "PUT", path, {**note_headers, **headers}, note_body)
    assert status == 200 and reads >= 1 and body["note"]["updated_by"] == USERNAME
