import pytest

from tests.api.conftest import audio_bytes

SECRET_NAME = "caller-secret-name-7731"
NOT_FOUND = {"error": {"code": "not_found", "message": "The requested resource was not found.",
                       "request_id": None}}


def upload(client, name="call.wav"):
    response = client.post("/api/analyses", files={"file": (name, audio_bytes(), "audio/wav")})
    assert response.status_code == 201
    return response


def test_location_from_post_resolves_to_the_stored_result(client):
    posted = upload(client)
    response = client.get(posted.headers["location"])
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"analysis", "created_at", "note"}
    assert body["analysis"] == posted.json()
    assert body["note"] is None
    stored = client.app.state.repository.get(posted.json()["request_id"])
    assert body["created_at"] == stored.created_at
    assert body["created_at"].endswith("Z")


def test_details_has_no_file_name_ip_or_stored_only_fields(client):
    response = client.get(upload(client, f"{SECRET_NAME}.wav").headers["location"])
    for absent in (SECRET_NAME, "testclient", "processing_ms", "is_mock"):
        assert absent not in response.text


@pytest.mark.parametrize("request_id", [
    "req_" + "0" * 32,          # right form, never stored
    "req_" + "A" * 32,          # upper case: not the §2 form
    "req_123", "nope", "req_" + "0" * 33,
])
def test_unknown_or_malformed_id_is_404_not_found(client, request_id):
    upload(client)
    response = client.get(f"/api/analyses/{request_id}")
    assert response.status_code == 404
    assert response.json() == NOT_FOUND
