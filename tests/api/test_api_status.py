import dataclasses
import re
from pathlib import Path

import pytest

from api.analyzer import AnalyzerInfo
from api.main import create_app
from api.repository import AnalysisRepository
from api.routers.status import derive_state
from tests.api.conftest import signed_in_client


def test_status_returns_the_section_4_1_shape(client, config):
    response = client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"state", "checked_at", "checks", "versions", "bundle_manifest_sha256",
                         "deferral_on_fit", "mock"}
    assert body["state"] == "ready"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", body["checked_at"])
    assert body["checks"] == {"analysis": "available", "history_and_stats": "available"}
    assert body["versions"] == {"model_version": "mock", "profile_version": "preprocessing-v1",
                                "policy_version": "risk-v1"}
    assert body["bundle_manifest_sha256"] is None
    assert body["deferral_on_fit"] == config.mock_deferral_on_fit
    assert body["mock"] is True


def test_deferral_on_fit_comes_from_config(config):
    with signed_in_client(create_app(dataclasses.replace(config, mock_deferral_on_fit=0.25))) as client:
        assert client.get("/api/status").json()["deferral_on_fit"] == 0.25


class InfoOnly:
    def __init__(self, info):
        self._info = info

    def info(self):
        return self._info


BUNDLE_INFO = AnalyzerInfo(ready=True, model_version="cnn-v3", profile_version="preprocessing-v1",
                           policy_version="risk-v1", deferral_on_fit=0.12, bundle_manifest_sha256="ab" * 32,
                           mock=False)


def status_of(config, analyzer=None, probe_path=None):
    """GET /api/status, signed in. The session lives in the started database; with probe_path, the
    history_and_stats check reads that file instead, so a broken history store can be tested while the
    session check still works (§4.8: a session check that cannot read the store is a 500)."""
    app = create_app(config, analyzer=analyzer)
    with signed_in_client(app) as client:
        if probe_path is not None:
            app.state.repository = AnalysisRepository(Path(probe_path))
        return client.get("/api/status")


def test_status_values_come_from_the_analyzer_info(config):
    body = status_of(config, InfoOnly(BUNDLE_INFO)).json()
    assert body["state"] == "ready"
    assert body["checks"]["analysis"] == "available"
    assert body["versions"] == {"model_version": "cnn-v3", "profile_version": "preprocessing-v1",
                                "policy_version": "risk-v1"}
    assert body["bundle_manifest_sha256"] == "ab" * 32
    assert body["deferral_on_fit"] == 0.12
    assert body["mock"] is False


def test_status_is_unavailable_when_the_analyzer_is_not_ready(config):
    info = dataclasses.replace(BUNDLE_INFO, ready=False)
    response = status_of(config, InfoOnly(info))
    assert response.status_code == 200
    assert response.json()["state"] == "unavailable"
    assert response.json()["checks"] == {"analysis": "unavailable", "history_and_stats": "available"}


def test_status_is_only_under_the_api_prefix(client):
    assert client.get("/status").status_code == 404


def assert_database_unavailable(response):
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "unavailable"
    assert body["checks"] == {"analysis": "available", "history_and_stats": "unavailable"}
    assert body["versions"]["model_version"] == "mock"  # still the full §4.1 body


def test_status_reports_a_missing_database_and_does_not_create_it(config, tmp_path):
    db_path = tmp_path / "missing" / "api.sqlite"  # never made
    assert_database_unavailable(status_of(config, probe_path=db_path))
    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_status_reports_a_file_that_is_not_a_database(config, tmp_path):
    garbage = tmp_path / "garbage.sqlite"
    garbage.write_bytes(b"not a database" * 100)
    assert_database_unavailable(status_of(config, probe_path=garbage))
    assert garbage.read_bytes() == b"not a database" * 100  # the probe never writes


def test_status_reports_an_unreadable_database_after_start(config, tmp_path):
    history = AnalysisRepository(tmp_path / "history.sqlite")
    history.create_schema()
    app = create_app(config)
    with signed_in_client(app) as client:
        app.state.repository = history
        assert client.get("/api/status").json()["checks"]["history_and_stats"] == "available"
        history.path.unlink()  # database gone while the service runs
        assert_database_unavailable(client.get("/api/status"))


def test_status_is_500_when_the_session_store_is_gone(config):
    # One SQLite file holds the sessions too: the session check fails first, before the status body (§4.8).
    app = create_app(config)
    with signed_in_client(app) as client:
        app.state.repository.path.unlink()
        response = client.get("/api/status")
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error", "message": "An unexpected error occurred.",
                                         "request_id": None}}


def test_status_is_unavailable_when_both_checks_fail(config, tmp_path):
    info = dataclasses.replace(BUNDLE_INFO, ready=False)
    body = status_of(config, InfoOnly(info), probe_path=tmp_path / "missing.sqlite").json()
    assert body["state"] == "unavailable"
    assert body["checks"] == {"analysis": "unavailable", "history_and_stats": "unavailable"}


@pytest.mark.parametrize("analysis, history_and_stats, state", [
    ("available", "available", "ready"),
    ("unavailable", "available", "unavailable"),
    ("unavailable", "unavailable", "unavailable"),
    ("available", "unavailable", "unavailable"),  # records are stored before the 201
])
def test_derive_state_provisional_rule(analysis, history_and_stats, state):
    assert derive_state(analysis, history_and_stats) == state
