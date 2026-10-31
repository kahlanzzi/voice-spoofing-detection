import pytest
from fastapi.testclient import TestClient

from api.errors import ApiError
from api.main import create_app

VERDICT_FIELDS = {"prediction", "confidence", "risk_level"}
SECRET = "secret-internal-detail /home/someone/api/model.py"


def assert_error_shape(response, status, code):
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] is None
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    assert not VERDICT_FIELDS & set(body["error"])
    return body


@pytest.fixture
def test_app(config):
    """The real app plus routes that fail on purpose."""
    app = create_app(config)

    @app.get("/api/_boom")
    def boom():
        raise RuntimeError(SECRET)

    @app.get("/api/_typed")
    def typed(page: int):
        return {"page": page}

    @app.get("/api/_api_error")
    def api_error():
        raise ApiError(503, "analysis_unavailable", "The analysis service is not available.",
                       request_id="req_" + "a" * 32)

    return app


def test_unknown_route_is_404_not_found(client):
    assert_error_shape(client.get("/api/does-not-exist"), 404, "not_found")


def test_wrong_method_is_405_invalid_request(client):
    response = client.post("/api/status")
    assert_error_shape(response, 405, "invalid_request")
    assert response.headers["allow"] == "GET"


def test_unhandled_exception_is_500_without_traceback(test_app):
    client = TestClient(test_app, raise_server_exceptions=False)
    response = client.get("/api/_boom")
    assert_error_shape(response, 500, "internal_error")
    for leak in ("Traceback", "RuntimeError", "secret-internal-detail", "/home/", ".py"):
        assert leak not in response.text


def test_invalid_query_param_is_400_invalid_request(test_app):
    client = TestClient(test_app)
    assert_error_shape(client.get("/api/_typed?page=abc"), 400, "invalid_request")
    assert_error_shape(client.get("/api/_typed"), 400, "invalid_request")
    assert client.get("/api/_typed?page=2").json() == {"page": 2}


def test_api_error_keeps_its_status_code_and_request_id(test_app):
    response = TestClient(test_app).get("/api/_api_error")
    assert response.status_code == 503
    assert response.json() == {"error": {"code": "analysis_unavailable",
                                         "message": "The analysis service is not available.",
                                         "request_id": "req_" + "a" * 32}}
