import json
import logging
import re
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.repository import ACCOUNT_SCHEMA, SCHEMA, SESSION_SCHEMA, SIGN_IN_FAILURE_SCHEMA
from tests.api.conftest import ANALYST, HTTPS, audio_bytes, signed_in, signed_in_client
from tests.api.test_api_repository import OLD_NOTE_SCHEMA, OLD_NOTES, old_database

SECRET = "note-secret-text-5519"
INVALID = {"error": {"code": "invalid_request", "message": "The request is not valid.", "request_id": None}}
NOT_FOUND = {"error": {"code": "not_found", "message": "The requested resource was not found.",
                       "request_id": None}}
UTC_SECOND = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
JSON = {"content-type": "application/json"}


def upload(client):
    response = client.post("/api/analyses", files={"file": ("call.wav", audio_bytes(), "audio/wav")})
    assert response.status_code == 201
    return response.json()["request_id"]


def put(client, request_id, text):
    return client.put(f"/api/analyses/{request_id}/note", json={"text": text})


def put_raw(client, request_id, content, headers=JSON):
    return client.put(f"/api/analyses/{request_id}/note", content=content, headers=headers)


def rows(config, sql, *params):
    conn = sqlite3.connect(config.database_path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def note_rows(config):
    return rows(config, "SELECT request_id, text FROM analysis_note")


def record_row(config, request_id):
    """The raw analysis_record row with its rowid, so a delete-and-insert would show."""
    return rows(config, "SELECT rowid, * FROM analysis_record WHERE request_id = ?", request_id)


# --- save, replace, remove ---------------------------------------------------------------------

def test_save_returns_the_note_and_details_show_it(client):
    request_id = upload(client)
    response = put(client, request_id, "Called back on the known number.")
    assert response.status_code == 200
    note = response.json()["note"]
    assert set(response.json()) == {"note"} and set(note) == {"text", "updated_at", "updated_by"}
    assert note["text"] == "Called back on the known number."
    assert UTC_SECOND.fullmatch(note["updated_at"])
    assert note["updated_by"] == ANALYST
    assert client.get(f"/api/analyses/{request_id}").json()["note"] == note
    assert client.get("/api/analyses").json()["items"][0]["note"] == note


def test_a_second_put_replaces_the_note(client, config):
    request_id = upload(client)
    put(client, request_id, "first")
    assert put(client, request_id, "second").json()["note"]["text"] == "second"
    assert note_rows(config) == [(request_id, "second")]


def author_rows(config):
    return rows(config, "SELECT n.text, a.username FROM analysis_note AS n "
                        "LEFT JOIN account AS a ON a.account_id = n.updated_by_account_id")


def test_another_account_replaces_the_text_and_the_author(client, config):
    request_id = upload(client)
    put(client, request_id, "first")
    other = signed_in(TestClient(client.app, base_url=HTTPS), "analyst02")  # a second analyst, same service
    note = put(other, request_id, "second").json()["note"]
    assert (note["text"], note["updated_by"]) == ("second", "analyst02")
    assert client.get(f"/api/analyses/{request_id}").json()["note"] == note
    assert author_rows(config) == [("second", "analyst02")]


def test_removing_a_note_keeps_no_author(client, config):
    request_id = upload(client)
    put(client, request_id, "first")
    assert put(client, request_id, " ").json() == {"note": None}
    assert author_rows(config) == []
    assert client.get(f"/api/analyses/{request_id}").json()["note"] is None
    assert put(client, request_id, "again").json()["note"]["updated_by"] == ANALYST


def test_a_note_saved_before_sign_in_has_a_null_author(client, config):
    request_id = upload(client)
    conn = sqlite3.connect(config.database_path)
    try:
        with conn:
            conn.execute("INSERT INTO analysis_note (request_id, text, updated_at) VALUES (?, ?, ?)",
                         (request_id, "old", "2026-10-30T09:20:11Z"))
    finally:
        conn.close()
    expected = {"text": "old", "updated_at": "2026-10-30T09:20:11Z", "updated_by": None}
    assert client.get(f"/api/analyses/{request_id}").json()["note"] == expected
    assert client.get("/api/analyses").json()["items"][0]["note"] == expected


def test_a_database_from_before_the_author_column_is_migrated_at_start(config):
    old_database(Path(config.database_path), SCHEMA, OLD_NOTE_SCHEMA, ACCOUNT_SCHEMA, SESSION_SCHEMA,
                 SIGN_IN_FAILURE_SCHEMA)
    with signed_in_client(create_app(config)) as client:  # the lifespan runs the migration
        request_id, text, updated_at = OLD_NOTES[0]
        details = client.get(f"/api/analyses/{request_id}").json()
        assert details["note"] == {"text": text, "updated_at": updated_at, "updated_by": None}
        assert put(client, request_id, "Replaced.").json()["note"]["updated_by"] == ANALYST


def test_json_with_a_charset_parameter_is_accepted(client):
    request_id = upload(client)
    response = put_raw(client, request_id, json.dumps({"text": "ok"}),
                       {"content-type": "application/json; charset=utf-8"})
    assert response.status_code == 200
    assert response.json()["note"]["text"] == "ok"


def test_surrounding_whitespace_is_stripped_and_not_counted(client, config):
    request_id = upload(client)
    text = "x" * config.note_max_chars
    response = put(client, request_id, f" \n\t {text} \r\n ")
    assert response.status_code == 200
    assert response.json()["note"]["text"] == text
    assert note_rows(config) == [(request_id, text)]


@pytest.mark.parametrize("char", ["x", "é", "😀"])  # 1 code point each; 😀 is two UTF-16 units
def test_exactly_note_max_chars_is_accepted(client, config, char):
    request_id = upload(client)
    text = char * config.note_max_chars
    assert put(client, request_id, text).json()["note"]["text"] == text


def test_one_char_over_is_400_and_stores_nothing(client, config):
    request_id = upload(client)
    response = put(client, request_id, "x" * (config.note_max_chars + 1))
    assert response.status_code == 400
    assert response.json() == INVALID
    assert note_rows(config) == []


def test_one_char_over_keeps_the_existing_note(client, config):
    request_id = upload(client)
    put(client, request_id, "kept")
    assert put(client, request_id, "x" * (config.note_max_chars + 1)).status_code == 400
    assert note_rows(config) == [(request_id, "kept")]


@pytest.mark.parametrize("blank", ["", "   ", " \n\t "])
def test_whitespace_only_removes_the_note(client, config, blank):
    request_id = upload(client)
    put(client, request_id, "to be removed")
    response = put(client, request_id, blank)
    assert response.status_code == 200
    assert response.json() == {"note": None}
    assert note_rows(config) == []
    assert client.get(f"/api/analyses/{request_id}").json()["note"] is None


def test_removing_an_absent_note_answers_null(client, config):
    request_id = upload(client)
    response = put(client, request_id, "  ")
    assert response.status_code == 200
    assert response.json() == {"note": None}
    assert note_rows(config) == []


def test_the_analysis_record_row_never_changes(client, config):
    request_id = upload(client)
    before = record_row(config, request_id)
    assert len(before) == 1
    put(client, request_id, "first")
    assert record_row(config, request_id) == before
    put(client, request_id, "replaced")
    assert record_row(config, request_id) == before
    put(client, request_id, "   ")
    assert record_row(config, request_id) == before
    put(client, request_id, "   ")  # removing an absent note
    assert record_row(config, request_id) == before
    assert client.app.state.repository.count() == 1


# --- errors ------------------------------------------------------------------------------------

@pytest.mark.parametrize("content, headers", [
    ("{}", JSON),                                         # missing text
    ('{"text": 5}', JSON),                                # not a string
    ('{"text": null}', JSON),
    ('["text"]', JSON),
    ('{"text": "ok", "author": "me"}', JSON),             # extra field
    ('{"text": "ok"', JSON),                              # not valid JSON
    ("", JSON),                                           # empty body
    ('{"text": "\\ud800"}', JSON),                        # lone surrogate: cannot be stored as UTF-8
    ('{"text": "ok"}', {}),                               # no Content-Type
    ('{"text": "ok"}', {"content-type": "text/plain"}),
    ("text=ok", {"content-type": "application/x-www-form-urlencoded"}),
])
def test_bad_bodies_are_400_and_store_nothing(client, config, content, headers):
    request_id = upload(client)
    response = put_raw(client, request_id, content, headers)
    assert response.status_code == 400
    assert response.json() == INVALID
    assert note_rows(config) == []


def test_unknown_id_is_404_and_stores_nothing(client, config):
    upload(client)
    unknown = "req_" + "0" * 32
    response = put(client, unknown, "x")
    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    assert note_rows(config) == []
    assert record_row(config, unknown) == []


@pytest.mark.parametrize("request_id", ["req_" + "A" * 32, "req_123", "nope"])
def test_malformed_id_is_404(client, request_id):
    response = put(client, request_id, "x")
    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_content_length_over_the_note_limit_is_a_section_6_400(client, config):
    request_id = upload(client)
    response = put_raw(client, request_id, '{"text": "ok"}', {**JSON, "content-length": "99999"})
    assert response.status_code == 400
    assert response.json() == INVALID
    assert note_rows(config) == []


def test_streamed_body_over_the_note_limit_is_400(client, config):
    request_id = upload(client)

    def chunks():  # no Content-Length: the route counts the bytes
        yield b'{"text": "'
        yield b" " * 30000
        yield b'ok"}'

    response = put_raw(client, request_id, chunks())
    assert response.status_code == 400
    assert response.json() == INVALID
    assert note_rows(config) == []


# --- privacy: the note text never reaches an error or a log -------------------------------------

def test_note_text_is_never_logged_on_success(client, caplog):
    caplog.set_level(logging.DEBUG)
    request_id = upload(client)
    assert put(client, request_id, SECRET).status_code == 200
    assert SECRET not in caplog.text


@pytest.mark.parametrize("content, headers", [
    (json.dumps({"text": SECRET + "x" * 2000}), JSON),                    # too long by more than 1
    (json.dumps({"text": SECRET, "x": 1}), JSON),                         # extra field
    (f"text={SECRET}", {"content-type": "text/plain"}),                   # not JSON
    ('{"text": "' + SECRET, JSON),                                        # invalid JSON
    (json.dumps({"text": SECRET + " " * 30000}), JSON),                   # over the body limit
    (json.dumps({"text": 5, SECRET: SECRET}), JSON),                      # pydantic would echo it
])
def test_no_400_echoes_the_note_text(client, caplog, content, headers):
    caplog.set_level(logging.DEBUG)
    request_id = upload(client)
    response = put_raw(client, request_id, content, headers)
    assert response.status_code == 400
    assert response.json() == INVALID
    assert SECRET not in response.text
    assert SECRET not in caplog.text


def test_exactly_one_char_over_does_not_echo_the_note_text(client, config, caplog):
    caplog.set_level(logging.DEBUG)
    request_id = upload(client)
    text = SECRET + "x" * (config.note_max_chars + 1 - len(SECRET))
    response = put(client, request_id, text)
    assert response.status_code == 400
    assert SECRET not in response.text and SECRET not in caplog.text


@pytest.mark.parametrize("method", ["save_note", "delete_note"])
def test_storage_failure_is_500_and_logs_only_the_error_class(client, config, caplog, monkeypatch, method):
    caplog.set_level(logging.DEBUG)
    request_id = upload(client)

    def failing(*args):
        raise RuntimeError(SECRET)  # a message that carries the text must still not be logged

    monkeypatch.setattr(client.app.state.repository, method, failing)
    response = put(client, request_id, SECRET if method == "save_note" else "  ")
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error", "message": "An unexpected error occurred.",
                                         "request_id": None}}
    assert SECRET not in response.text
    assert "note not saved" in caplog.text and "RuntimeError" in caplog.text
    assert SECRET not in caplog.text
    assert note_rows(config) == []
