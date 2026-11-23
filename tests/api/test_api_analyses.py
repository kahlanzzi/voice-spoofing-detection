import dataclasses
import functools
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.analyzer import AnalysisOutcome, AnalyzerUnavailable
from api.db import StoreError
from api.intake import MULTIPART_OVERHEAD_BYTES
from api.main import create_app
from api.repository import TABLES
from tests.api._analyzers import FakeAnalyzer, ScriptedAnalyzer, started
from tests.api._pg import config_query
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

    def make(analyzer_factory=None, **changes):
        cfg = dataclasses.replace(config, **changes)
        client = TestClient(create_app(cfg, analyzer_factory=analyzer_factory), base_url=HTTPS)
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

@pytest.fixture
def control(tmp_path) -> Path:
    """The control folder of tests/api/_analyzers.py: mode, release and the build/started markers."""
    folder = tmp_path / "control"
    folder.mkdir()
    return folder


def scripted(control: Path):
    return functools.partial(ScriptedAnalyzer, str(control))


def until(predicate, limit_s: float = 30) -> None:
    deadline = time.monotonic() + limit_s
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.02)


def test_timeout_is_504_and_nothing_is_stored(app_with, control):
    (control / "mode").write_text("hang")
    client, cfg = app_with(scripted(control), analysis_timeout_s=1)
    assert_upload_error(client, cfg, upload(client, audio_bytes()), 504, "analysis_timeout")


@pytest.mark.parametrize("factory", [functools.partial(FakeAnalyzer, ready=False),
                                     functools.partial(FakeAnalyzer, error=AnalyzerUnavailable())],
                         ids=["not-ready", "refuses"])
def test_unavailable_analyzer_is_503(app_with, factory):
    client, cfg = app_with(factory)
    response = upload(client, audio_bytes())
    assert_upload_error(client, cfg, response, 503, "analysis_unavailable")
    assert "retry-after" not in response.headers  # only a full pool says when to retry


def test_analyzer_exception_is_500(app_with):
    client, cfg = app_with(functools.partial(FakeAnalyzer, error=RuntimeError("model exploded at /secret/path.py")))
    response = upload(client, audio_bytes())
    assert_upload_error(client, cfg, response, 500, "internal_error")
    assert "secret" not in response.text


def test_invalid_outcome_is_500(app_with):
    bad = AnalysisOutcome("synthetic", 1.7, "HIGH", "mock", "preprocessing-v1", "risk-v1")
    client, cfg = app_with(functools.partial(FakeAnalyzer, outcome=bad))
    assert_upload_error(client, cfg, upload(client, audio_bytes()), 500, "internal_error")


def test_failed_store_is_500_without_a_verdict(client, config):
    real = client.app.state.repository

    class Failing:
        def add(self, record):
            raise StoreError("OperationalError")

    client.app.state.repository = Failing()
    response = upload(client, audio_bytes())
    client.app.state.repository = real
    assert_upload_error(client, config, response, 500, "internal_error")


# --- privacy -----------------------------------------------------------------------------------

def test_original_file_name_is_never_stored_or_logged(app_with, control, caplog):
    caplog.set_level(logging.DEBUG)
    client, config = app_with(scripted(control))
    assert upload(client, audio_bytes(), f"{SECRET_NAME}.wav").status_code == 201
    assert upload(client, os.urandom(4000), f"{SECRET_NAME}.wav").status_code == 422
    (control / "mode").write_text("error")
    assert upload(client, audio_bytes(), f"{SECRET_NAME}.wav").status_code == 500
    for table in TABLES:  # every row of every table
        assert SECRET_NAME not in repr(config_query(config, f"SELECT * FROM {table}"))
    assert caplog.records  # the 500 was logged
    assert SECRET_NAME not in caplog.text
    assert temp_files(config) == []


# --- analysis pool (D53 (2)): 503 with Retry-After when full, and every slot release path ----------

def in_threads(client, count):
    """Start count uploads at once, each in its own thread; return the threads and their responses."""
    responses = [None] * count

    def send(i):
        responses[i] = upload(client, audio_bytes())

    threads = [threading.Thread(target=send, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    return threads, responses


def test_full_pool_answers_503_with_retry_after_and_code(app_with, control):
    (control / "mode").write_text("wait")
    client, cfg = app_with(scripted(control), analysis_workers=2)
    threads, responses = in_threads(client, 2)
    until(lambda: len(started(control)) == 2)  # both workers hold a job
    response = upload(client, audio_bytes())
    assert response.status_code == 503
    assert response.headers["retry-after"] == str(cfg.analysis_retry_after_s) == "30"
    body = response.json()
    assert body == {"error": {"code": "analysis_unavailable", "message": "The analysis service is busy; try again later.",
                              "request_id": response.headers["x-request-id"]}}
    assert REQUEST_ID.fullmatch(body["error"]["request_id"])
    assert records(client) == 0
    assert len(temp_files(cfg)) == 2  # the two held uploads only: the refused body was never written
    (control / "release").touch()
    for thread in threads:
        thread.join(30)
    assert [r.status_code for r in responses] == [201, 201]


def test_full_pool_then_a_freed_slot_serves_the_next_request(app_with, control):
    (control / "mode").write_text("wait")
    client, cfg = app_with(scripted(control), analysis_workers=1)
    threads, responses = in_threads(client, 1)
    until(lambda: len(started(control)) == 1)
    assert upload(client, audio_bytes()).status_code == 503
    (control / "mode").unlink()
    (control / "release").touch()
    threads[0].join(30)
    assert responses[0].status_code == 201
    assert upload(client, audio_bytes()).status_code == 201
    assert client.app.state.pool.counts() == {"idle": 1, "held": 0, "restarting": 0}


def test_intake_check_failure_releases_slot(client, config):
    assert config.analysis_workers == 1
    assert upload(client, audio_bytes(), "call.ogg").status_code == 415
    assert upload(client, b"", "call.wav").status_code == 422
    assert upload(client, audio_bytes()).status_code == 201
    assert client.app.state.pool.counts() == {"idle": 1, "held": 0, "restarting": 0}


@pytest.mark.parametrize("mode, status", [("error", 500), ("unavailable", 503)])
def test_analyzer_error_or_unavailable_releases_slot(app_with, control, mode, status):
    (control / "mode").write_text(mode)
    client, cfg = app_with(scripted(control), analysis_workers=1)
    pids = client.app.state.pool.worker_pids()
    assert upload(client, audio_bytes()).status_code == status
    (control / "mode").unlink()
    assert upload(client, audio_bytes()).status_code == 201
    assert client.app.state.pool.worker_pids() == pids  # the worker survived and was reused


def test_timeout_releases_slot_and_next_request_succeeds(app_with, control):
    (control / "mode").write_text("hang")
    client, cfg = app_with(scripted(control), analysis_workers=1, analysis_timeout_s=1)
    assert_upload_error(client, cfg, upload(client, audio_bytes()), 504, "analysis_timeout")
    (marker,) = started(control)
    hung = int(marker.name.split("_")[1])
    # Right after the 504: the temp folder is empty, and the hung worker is already reaped, so no process
    # still holds the unlinked temp file open (a listing alone cannot show that).
    assert temp_files(cfg) == []
    if sys.platform == "linux":  # /proc: the container and CI
        assert not Path(f"/proc/{hung}").exists()
    (control / "mode").unlink()
    until(lambda: client.app.state.pool.counts()["idle"] == 1)
    assert upload(client, audio_bytes()).status_code == 201
    assert hung not in client.app.state.pool.worker_pids()


def test_worker_death_mid_job_is_500_and_the_worker_is_replaced(app_with, control):
    (control / "mode").write_text("exit")
    client, cfg = app_with(scripted(control), analysis_workers=1)
    assert_upload_error(client, cfg, upload(client, audio_bytes()), 500, "internal_error")
    (control / "mode").unlink()
    until(lambda: client.app.state.pool.counts()["idle"] == 1)
    assert upload(client, audio_bytes()).status_code == 201
