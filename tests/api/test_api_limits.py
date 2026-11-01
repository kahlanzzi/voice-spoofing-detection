import yaml

from api.config import DEFAULT_PATH, ENV_VAR, load_api_config
from api.main import create_app
from tests.api.conftest import signed_in_client

SECTION_4_2 = {"formats", "max_size_bytes", "max_duration_s", "audio_deletion"}


def test_limits_equal_the_repo_config(client, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    repo = load_api_config(DEFAULT_PATH)
    response = client.get("/api/limits")
    assert response.status_code == 200
    assert response.json() == {"formats": list(repo.formats), "max_size_bytes": repo.max_size_bytes,
                               "max_duration_s": repo.max_duration_s, "audio_deletion": repo.audio_deletion}


def test_limits_follow_the_config_named_by_the_env_var(tmp_path, monkeypatch):
    cfg = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    cfg.update(formats=["wav"], max_size_bytes=1_048_576, max_duration_s=10,
               audio_deletion="Deleted at once", database_path=str(tmp_path / "api.sqlite"),
               temp_dir=str(tmp_path / "uploads"))
    path = tmp_path / "api.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(path))
    with signed_in_client(create_app()) as client:
        body = client.get("/api/limits").json()
    assert body == {"formats": ["wav"], "max_size_bytes": 1_048_576, "max_duration_s": 10,
                    "audio_deletion": "Deleted at once"}


def test_limits_have_exactly_the_section_4_2_fields(client):
    # No IP, file name or other request data: only the four config values.
    assert set(client.get("/api/limits").json()) == SECTION_4_2


def test_limits_are_only_under_the_api_prefix(client):
    assert client.get("/limits").status_code == 404


def test_limits_reject_other_methods_with_the_error_shape(client):
    response = client.post("/api/limits")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "invalid_request"
