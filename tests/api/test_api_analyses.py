import dataclasses
import logging
import os
import re
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from api.analyzer import AnalysisOutcome, AnalyzerUnavailable, MockAnalyzer
from api.intake import MULTIPART_OVERHEAD_BYTES
from api.main import create_app
from tests.api.conftest import HTTPS, audio_bytes, signed_in

URL = "/api/analyses"
REQUEST_ID = re.compile(r"req_[0-9a-f]{32}")
SECTION_11 = {"prediction", "confidence", "risk_level", "model_version", "profile_version",
              "policy_version", "request_id", "notice"}
VERDICT_FIELDS = {"prediction", "confidence", "risk_level"}
SECRET_NAME = "caller-secret-name-7731"


def upload(client, content, name="call.wav"):
    return client.post(URL, files={"file": (name, content, "audio/wav")})


def records(client):
    return client.app.state.repository.count()


def temp_files(config):
    return os.listdir(config.temp_dir)


def assert_upload_error(client, config, response, status, code):
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["code"] == code
    assert REQUEST_ID.fullmatch(body["error"]["request_id"])
    assert body["error"]["request_id"] == response.headers["x-request-id"]
    assert not VERDICT_FIELDS & set(body["error"])
    assert "location" not in response.headers
    assert records(client) == 0
    assert temp_files(config) == []


@pytest.fixture
def app_with(config):
    """A started, signed-in client for a changed config and/or another analyzer."""
    clients = []

    def make(analyzer=None, **changes):
        cfg = dataclasses.replace(config, **changes)
        client = TestClient(create_app(cfg, analyzer=analyzer), base_url=HTTPS)
        client.__enter__()
        clients.append(client)
        signed_in(client)
        return client, cfg

    yield make
    for client in clients:
        client.__exit__(None, None, None)


# --- success ---------------------------------------------------------------------------------

@pytest.mark.parametrize("fmt, name", [("WAV", "call.wav"), ("FLAC", "call.flac"), ("WAV", "CALL.WAV")])
def test_success_returns_201_with_exactly_the_section_11_fields(client, config, fmt, name):
    response = upload(client, audio_bytes(fmt=fmt), name)
    assert response.status_code == 201
    body = response.json()
    assert set(body) == SECTION_11
    request_id = body["request_id"]
    assert REQUEST_ID.fullmatch(request_id)
    assert response.headers["x-request-id"] == request_id
    assert response.headers["location"] == f"/api/analyses/{request_id}"
    assert body["notice"] == config.notice_text
    assert body["model_version"] == "mock"
    assert temp_files(config) == []


def test_stored_row_matches_the_response(client):
    body = upload(client, audio_bytes()).json()
    stored = client.app.state.repository.get(body["request_id"])
    assert records(client) == 1
    assert {k: getattr(stored, k) for k in SECTION_11} == body
    assert stored.is_mock is True
    assert isinstance(stored.processing_ms, int) and stored.processing_ms >= 0
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stored.created_at)


def test_same_file_twice_gives_the_same_outcome(client):
    content = audio_bytes(seed=3)
    first, second = upload(client, content).json(), upload(client, content, "other.wav").json()
    assert first["request_id"] != second["request_id"]
    assert {k: first[k] for k in VERDICT_FIELDS} == {k: second[k] for k in VERDICT_FIELDS}
    assert records(client) == 2


def test_file_of_exactly_max_size_bytes_is_accepted(app_with):
    content = audio_bytes(seconds=0.1)
    client, _ = app_with(max_size_bytes=len(content))
    assert upload(client, content).status_code == 201


def test_recording_of_exactly_max_duration_is_accepted(app_with):
    client, _ = app_with(max_duration_s=1)
    assert upload(client, audio_bytes(seconds=1.0)).status_code == 201


# --- upload errors -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["call.ogg", "call", "call.wav.txt", ".wav"])
def test_wrong_extension_is_415(client, config, name):
    assert_upload_error(client, config, upload(client, audio_bytes(), name), 415, "unsupported_format")


def test_oversized_by_content_length_is_413(app_with):
    client, cfg = app_with(max_size_bytes=1000)
    response = upload(client, b"\0" * (1000 + MULTIPART_OVERHEAD_BYTES + 1))
    assert_upload_error(client, cfg, response, 413, "file_too_large")


def test_oversized_body_without_content_length_is_413(app_with):
    # The excess sits in an extra text field next to a small valid file: only the body limit (layer 2)
    # can refuse this request; the file-byte count (layer 3) would accept it.
    content = audio_bytes(seconds=0.1)
    client, cfg = app_with(max_size_bytes=len(content))
    boundary = "testboundary"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="pad"\r\n\r\n'.encode()
            + b"x" * (len(content) + MULTIPART_OVERHEAD_BYTES + 1)
            + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="call.wav"\r\n\r\n'.encode()
            + content + f"\r\n--{boundary}--\r\n".encode())
    response = client.post(URL, content=iter([body[:5000], body[5000:]]),
                           headers={"content-type": f"multipart/form-data; boundary={boundary}"})
    assert "content-length" not in {k.lower() for k in response.request.headers}
    assert_upload_error(client, cfg, response, 413, "file_too_large")


def test_oversized_by_actual_bytes_is_413(app_with):
    content = audio_bytes(seconds=0.1)
    client, cfg = app_with(max_size_bytes=len(content) - 1)  # within the framing allowance
    assert_upload_error(client, cfg, upload(client, content), 413, "file_too_large")


def test_zero_byte_file_is_422_empty_file(client, config):
    assert_upload_error(client, config, upload(client, b""), 422, "empty_file")


def test_zero_frame_wav_is_422_empty_file(client, config):
    content = audio_bytes(seconds=0)
    assert len(content) > 0
    assert_upload_error(client, config, upload(client, content), 422, "empty_file")


def test_random_bytes_named_wav_is_422_corrupt_audio(client, config):
    assert_upload_error(client, config, upload(client, os.urandom(4000)), 422, "corrupt_audio")


def test_too_long_recording_is_422(app_with):
    client, cfg = app_with(max_duration_s=1)
    assert_upload_error(client, cfg, upload(client, audio_bytes(seconds=1.01)), 422, "recording_too_long")


@pytest.mark.parametrize("kwargs", [
    {},                                                          # no body at all
    {"files": {"other": ("call.wav", b"abc", "audio/wav")}},     # wrong part name
    {"data": {"file": "not a file"}},                            # a field, not a file
    {"files": [("file", ("a.wav", b"a", "audio/wav")), ("file", ("b.wav", b"b", "audio/wav"))]},
], ids=["empty", "wrong-name", "field", "two-files"])
def test_missing_file_part_is_400_invalid_request(client, config, kwargs):
    assert_upload_error(client, config, client.post(URL, **kwargs), 400, "invalid_request")


# --- analyzer and storage errors ---------------------------------------------------------------

class FakeAnalyzer(MockAnalyzer):
    def __init__(self, ready=True, error=None, outcome=None):
        super().__init__(0.1)
        self.ready, self.error, self.outcome = ready, error, outcome

    def analyze(self, path):
        if self.error is not None:
            raise self.error
        return self.outcome or super().analyze(path)

    def info(self):
        return dataclasses.replace(super().info(), ready=self.ready)


class BlockingAnalyzer(MockAnalyzer):
    def __init__(self):
        super().__init__(0.1)
        self.release, self.finished = threading.Event(), threading.Event()

    def analyze(self, path):
        self.release.wait(10)
        try:
            return super().analyze(path)
        finally:
            self.finished.set()


def test_timeout_is_504_and_nothing_is_stored(app_with):
    analyzer = BlockingAnalyzer()
    client, cfg = app_with(analyzer, analysis_timeout_s=1)
    response = upload(client, audio_bytes())
    analyzer.release.set()
    assert analyzer.finished.wait(10)
    assert_upload_error(client, cfg, response, 504, "analysis_timeout")


@pytest.mark.parametrize("analyzer", [FakeAnalyzer(ready=False), FakeAnalyzer(error=AnalyzerUnavailable())],
                         ids=["not-ready", "refuses"])
def test_unavailable_analyzer_is_503(app_with, analyzer):
    client, cfg = app_with(analyzer)
    assert_upload_error(client, cfg, upload(client, audio_bytes()), 503, "analysis_unavailable")


def test_analyzer_exception_is_500(app_with):
    client, cfg = app_with(FakeAnalyzer(error=RuntimeError("model exploded at /secret/path.py")))
    response = upload(client, audio_bytes())
    assert_upload_error(client, cfg, response, 500, "internal_error")
    assert "secret" not in response.text


def test_invalid_outcome_is_500(app_with):
    bad = AnalysisOutcome("synthetic", 1.7, "HIGH", "mock", "preprocessing-v1", "risk-v1")
    client, cfg = app_with(FakeAnalyzer(outcome=bad))
    assert_upload_error(client, cfg, upload(client, audio_bytes()), 500, "internal_error")


def test_failed_store_is_500_without_a_verdict(client, config):
    real = client.app.state.repository

    class Failing:
        def add(self, record):
            raise sqlite3.OperationalError("disk I/O error")

    client.app.state.repository = Failing()
    response = upload(client, audio_bytes())
    client.app.state.repository = real
    assert_upload_error(client, config, response, 500, "internal_error")


# --- privacy -----------------------------------------------------------------------------------

def test_original_file_name_is_never_stored_or_logged(client, config, caplog):
    caplog.set_level(logging.DEBUG)
    assert upload(client, audio_bytes(), f"{SECRET_NAME}.wav").status_code == 201
    assert upload(client, os.urandom(4000), f"{SECRET_NAME}.wav").status_code == 422
    client.app.state.analyzer = FakeAnalyzer(error=RuntimeError("boom"))
    assert upload(client, audio_bytes(), f"{SECRET_NAME}.wav").status_code == 500
    db_dir = os.path.dirname(config.database_path)
    for name in os.listdir(db_dir):  # the database plus any journal file
        with open(os.path.join(db_dir, name), "rb") as f:
            assert SECRET_NAME.encode() not in f.read()
    assert caplog.records  # the 500 was logged
    assert SECRET_NAME not in caplog.text
    assert temp_files(config) == []
