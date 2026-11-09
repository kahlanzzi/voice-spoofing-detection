import asyncio

import pytest

from api.intake import (MULTIPART_OVERHEAD_BYTES, ContentLengthGuard, _Undecodable, body_limit, decoded_length,
                        note_body_limit)
from tests.api.conftest import audio_bytes


def test_body_limit_adds_the_multipart_allowance(config):
    assert body_limit(config) == config.max_size_bytes + MULTIPART_OVERHEAD_BYTES == config.max_size_bytes + 65536


def test_note_body_limit_fits_the_longest_note_fully_escaped(config):
    assert note_body_limit(config) == 12 * config.note_max_chars + 1024
    assert note_body_limit(config) == 25024  # note_max_chars 2000 in configs/api.yaml


@pytest.mark.parametrize("fmt, ext", [("WAV", "wav"), ("FLAC", "flac")])
def test_decoded_length_counts_decoded_frames(tmp_path, fmt, ext):
    path = tmp_path / f"a.{ext}"
    path.write_bytes(audio_bytes(seconds=1.5, rate=16000, fmt=fmt))
    assert decoded_length(path) == (24000, 16000)


def test_decoded_length_rejects_random_bytes(tmp_path):
    path = tmp_path / "noise.wav"
    path.write_bytes(bytes(range(256)) * 10)
    with pytest.raises(_Undecodable):
        decoded_length(path)


def run_guard(headers, method="POST", path="/api/analyses"):
    """Call the guard as ASGI; return (status of the guard's answer or None, whether the app ran)."""
    reached = []
    sent = []

    async def app(scope, receive, send):
        reached.append(True)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": method, "path": path, "headers": headers,
             "state": {"request_id": "req_" + "c" * 32}}
    asyncio.run(ContentLengthGuard(app, max_body_size=100, max_note_body_size=50)(scope, receive, send))
    status = sent[0]["status"] if sent else None
    return status, bool(reached), b"".join(m.get("body", b"") for m in sent)


def test_guard_rejects_a_content_length_above_the_limit_with_the_request_id():
    status, reached, body = run_guard([(b"content-length", b"101")])
    assert (status, reached) == (413, False)
    assert b'"file_too_large"' in body and b"req_" + b"c" * 32 in body


def test_guard_rejects_a_malformed_content_length():
    status, reached, body = run_guard([(b"content-length", b"abc")])
    assert (status, reached) == (400, False)
    assert b'"invalid_request"' in body


NOTE = "/api/analyses/req_" + "d" * 32 + "/note"


@pytest.mark.parametrize("value", [b"51", b"abc", b"-1", b"\xb2", b"99999999999999999999"])
def test_guard_answers_a_note_body_over_its_limit_with_a_section_6_400(value):
    status, reached, body = run_guard([(b"content-length", value)], "PUT", NOTE)
    assert (status, reached) == (400, False)
    assert body == (b'{"error":{"code":"invalid_request","message":"The request is not valid.",'
                    b'"request_id":null}}')


@pytest.mark.parametrize("value", [b"4097", b"abc", b"-1"])
def test_guard_answers_a_sign_in_body_over_4_kib_with_a_section_6_400(value):
    status, reached, body = run_guard([(b"content-length", value)], "POST", "/api/session")
    assert (status, reached) == (400, False)
    assert b'"invalid_request"' in body and b'"request_id":null' in body


@pytest.mark.parametrize("headers, method, path", [
    ([(b"content-length", b"100")], "POST", "/api/analyses"),   # at the limit
    ([], "POST", "/api/analyses"),                              # no Content-Length: layer 2's job
    ([(b"content-length", b"999")], "POST", "/api/status"),     # other path
    ([(b"content-length", b"999")], "GET", "/api/analyses"),    # other method
    ([(b"content-length", b"50")], "PUT", NOTE),                # note at its limit
    ([], "PUT", NOTE),                                          # note without Content-Length: the route counts
    ([(b"content-length", b"999")], "GET", NOTE),               # note path, other method
    ([(b"content-length", b"999")], "PUT", NOTE + "/x"),        # not the note path
    ([(b"content-length", b"4096")], "POST", "/api/session"),   # sign-in at its limit
    ([], "POST", "/api/session"),                               # sign-in without Content-Length: the route counts
])
def test_guard_lets_other_requests_through(headers, method, path):
    assert run_guard(headers, method, path)[:2] == (None, True)
