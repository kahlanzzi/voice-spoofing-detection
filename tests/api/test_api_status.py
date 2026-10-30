import dataclasses
import re

from fastapi.testclient import TestClient

from api.main import create_app


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


def test_status_is_only_under_the_api_prefix(client):
    assert client.get("/status").status_code == 404
