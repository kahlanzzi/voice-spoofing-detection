import dataclasses
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.analyzer import AnalyzerInfo
from api.main import create_app
from api.routers.status import derive_state


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
    client = TestClient(create_app(dataclasses.replace(config, mock_deferral_on_fit=0.25)))
    assert client.get("/api/status").json()["deferral_on_fit"] == 0.25


class InfoOnly:
    def __init__(self, info):
        self._info = info

    def info(self):
        return self._info


BUNDLE_INFO = AnalyzerInfo(ready=True, model_version="cnn-v3", profile_version="preprocessing-v1",
                           policy_version="risk-v1", deferral_on_fit=0.12, bundle_manifest_sha256="ab" * 32,
                           mock=False)


def status_of(config, analyzer=None, lifespan=True):
    """GET /api/status; lifespan=True builds the database first, as at a real start."""
    app = create_app(config, analyzer=analyzer)
    if not lifespan:
        return TestClient(app).get("/api/status")
    with TestClient(app) as client:
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


def test_status_reports_a_missing_database_and_does_not_create_it(config):
    response = status_of(config, lifespan=False)  # no lifespan: the database file was never made
    assert_database_unavailable(response)
    db_path = Path(config.database_path)
    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_status_reports_a_file_that_is_not_a_database(config, tmp_path):
    garbage = tmp_path / "garbage.sqlite"
    garbage.write_bytes(b"not a database" * 100)
    response = status_of(dataclasses.replace(config, database_path=str(garbage)), lifespan=False)
    assert_database_unavailable(response)
    assert garbage.read_bytes() == b"not a database" * 100  # the probe never writes


def test_status_reports_an_unreadable_database_after_start(config):
    app = create_app(config)
    with TestClient(app) as client:
        app.state.repository.path.unlink()  # database gone while the service runs
        assert_database_unavailable(client.get("/api/status"))


def test_status_is_unavailable_when_both_checks_fail(config):
    info = dataclasses.replace(BUNDLE_INFO, ready=False)
    body = status_of(config, InfoOnly(info), lifespan=False).json()
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
