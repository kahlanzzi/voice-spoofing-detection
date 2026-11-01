import re

import pytest

from api.main import create_app
from api.middleware import new_request_id
from tests.api.conftest import signed_in_client

REQUEST_ID = re.compile(r"req_[0-9a-f]{32}")


def test_new_request_id_format_and_uniqueness():
    ids = {new_request_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(REQUEST_ID.fullmatch(i) for i in ids)


@pytest.fixture
def client_with_failures(config):
    app = create_app(config)

    @app.get("/api/_boom")
    def boom():
        raise RuntimeError("boom")

    @app.get("/api/_typed")
    def typed(page: int):
        return {"page": page}

    with signed_in_client(app, raise_server_exceptions=False) as client:
        yield client


@pytest.mark.parametrize("method, path, status", [
    ("GET", "/api/status", 200),
    ("GET", "/api/does-not-exist", 404),
    ("POST", "/api/status", 405),
    ("GET", "/api/_typed?page=abc", 400),
    ("GET", "/api/_boom", 500),
])
def test_every_response_has_a_request_id(client_with_failures, method, path, status):
    response = client_with_failures.request(method, path)
    assert response.status_code == status
    assert REQUEST_ID.fullmatch(response.headers["x-request-id"])


def test_each_request_gets_a_new_id(client):
    first = client.get("/api/status").headers["x-request-id"]
    second = client.get("/api/status").headers["x-request-id"]
    assert first != second


def test_client_request_id_is_ignored(client):
    sent = "req_" + "0" * 32
    response = client.get("/api/status", headers={"X-Request-ID": sent})
    assert response.headers["x-request-id"] != sent
    assert response.headers.get_list("x-request-id") == [response.headers["x-request-id"]]
