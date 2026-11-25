import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from api.config import DEFAULT_PATH, ENV_VAR, REPO_ROOT, ApiConfig, config_path, load_api_config, resolve_path

SECTION_9 = {
    "formats": ("wav", "flac", "mp3"),
    "max_size_bytes": 20_971_520,
    "max_duration_s": 60,
    "audio_deletion": "Deleted when processing finishes",
    "analysis_timeout_s": 30,
    "analysis_workers": 2,
    "analysis_retry_after_s": 30,
    "analysis_worker_start_timeout_s": 20,
    "page_size_max": 50,
    "note_max_chars": 2000,
    "stats_time_zone": "UTC",
    "notice_text": "Decision support only. This result estimates whether the recording is synthetic. "
                   "It does not prove fraud and does not verify identity.",
    "analyzer": "mock",
    "bundle_dir": "/app/bundle/bundle-v1",
    "bundle_manifest_sha256": "edf4a205e12fc948a68575ebaa929145481aaa535d34e3353ef9535f92840efc",
    "mock_deferral_on_fit": 0.0993,
    "temp_dir": None,
    "database_host": "db",
    "database_port": 5432,
    "database_name": "vsd",
    "database_user": "vsd_app",
    "database_pool_max": 4,
    "database_timeout_s": 5,
    "session_idle_minutes": 30,
    "session_absolute_hours": 12,
    "lockout_threshold": 5,
    "lockout_minutes": 15,
    "password_min_chars": 15,
    "password_max_chars": 128,
    "allowed_origins": ("http://localhost:8001", "http://127.0.0.1:8001"),
}


def write_config(tmp_path: Path, **changes) -> Path:
    cfg = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is ...:
            del cfg[key]
        else:
            cfg[key] = value
    path = tmp_path / "api.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_repo_config_has_the_section_9_values(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert load_api_config() == ApiConfig(**SECTION_9)


def test_env_var_points_to_another_file(tmp_path, monkeypatch):
    path = write_config(tmp_path, page_size_max=7)
    monkeypatch.setenv(ENV_VAR, str(path))
    assert config_path() == path
    assert load_api_config().page_size_max == 7


def test_default_path_without_env_var(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert config_path() == DEFAULT_PATH


@pytest.mark.parametrize("changes", [
    {"page_size_max": ...},               # missing key
    {"extra_key": 1},                     # unknown key
    {"max_size_bytes": "20MB"},           # wrong type
    {"max_duration_s": True},             # bool is not an int here
    {"analysis_timeout_s": 30.5},         # float is not an int
    {"mock_deferral_on_fit": 1},          # int is not a float
    {"mock_deferral_on_fit": 1.5},        # outside [0, 1]
    {"note_max_chars": 0},                # must be positive
    {"formats": []},                      # empty
    {"formats": ["WAV"]},                 # not lower case
    {"notice_text": "  "},                # empty notice
    {"temp_dir": 5},                      # neither str nor null
    {"database_path": "var/api.sqlite"},  # the SQLite key is gone (D53 (4)): unknown key
    {"database_host": None},              # required string
    {"database_host": " "},               # empty host
    {"database_port": 0},                 # must be positive
    {"database_port": 70000},             # above 65535
    {"database_port": "5432"},            # not an int
    {"database_name": "Vsd"},             # not a lower-case PostgreSQL name
    {"database_user": "vsd_app; DROP"},   # not a plain name
    {"database_user": "postgres" * 10},   # longer than 63
    {"database_pool_max": 0},             # must be positive
    {"database_timeout_s": 2.5},          # float is not an int
    {"session_idle_minutes": ...},        # sign-in keys are required (§9)
    {"lockout_threshold": 0},             # must be positive
    {"session_absolute_hours": 12.0},     # float is not an int
    {"password_min_chars": True},         # bool is not an int
    {"password_min_chars": 129},          # above password_max_chars
    {"allowed_origins": []},              # empty
    {"allowed_origins": "http://localhost:8001"},   # not a list
    {"allowed_origins": ["http://localhost:8001/"]},  # trailing slash
    {"allowed_origins": ["http://localhost:8001/api"]},  # a path
    {"allowed_origins": ["localhost:8001"]},  # no scheme
    {"allowed_origins": [8001]},          # not a string
], ids=lambda c: next(iter(c)) + "=" + repr(next(iter(c.values()))))
def test_invalid_config_raises(tmp_path, changes):
    with pytest.raises(ValueError):
        load_api_config(write_config(tmp_path, **changes))


def test_unreadable_or_non_mapping_config_raises(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        load_api_config(tmp_path / "missing.yaml")
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_api_config(path)


def test_resolve_path_is_relative_to_the_repo_root(tmp_path):
    assert resolve_path("var/uploads") == REPO_ROOT / "var" / "uploads"
    assert resolve_path(str(tmp_path / "uploads")) == tmp_path / "uploads"


def test_the_config_holds_no_database_password():
    text = DEFAULT_PATH.read_text(encoding="utf-8")
    assert "password:" not in text.replace("password_min_chars:", "").replace("password_max_chars:", "")
    assert "VSD_DB_APP_PASSWORD" in text  # it says where the password comes from instead


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2", None])
def test_config_rejects_fewer_than_one_worker(tmp_path, workers):
    with pytest.raises(ValueError, match="analysis_workers"):
        load_api_config(write_config(tmp_path, analysis_workers=workers))


def test_config_accepts_one_worker(tmp_path):
    assert load_api_config(write_config(tmp_path, analysis_workers=1)).analysis_workers == 1


@pytest.mark.parametrize("retry_after", [0, 31, 2.5, True])
def test_retry_after_must_be_positive_and_not_exceed_timeout(tmp_path, retry_after):
    with pytest.raises(ValueError, match="analysis_retry_after_s"):
        load_api_config(write_config(tmp_path, analysis_retry_after_s=retry_after))


@pytest.mark.parametrize("start_timeout", [0, -5, 60.0, None])
def test_start_timeout_must_be_positive(tmp_path, start_timeout):
    with pytest.raises(ValueError, match="analysis_worker_start_timeout_s"):
        load_api_config(write_config(tmp_path, analysis_worker_start_timeout_s=start_timeout))


# --- analyzer selection and the bundle pin ---------------------------------------------------------

F4_RECORD = REPO_ROOT / "docs" / "audit_evidence" / "stage12_bundle" / "f4_record.json"


def test_bundle_config_differs_only_in_analyzer():
    default, bundle = load_api_config(DEFAULT_PATH), load_api_config(DEFAULT_PATH.with_name("api.bundle.yaml"))
    assert (default.analyzer, bundle.analyzer) == ("mock", "bundle")
    assert dataclasses.replace(bundle, analyzer="mock") == default


def test_the_pin_is_f4():
    recorded = json.loads(F4_RECORD.read_text(encoding="utf-8"))["manifest_sha256"]
    assert load_api_config(DEFAULT_PATH).bundle_manifest_sha256 == recorded


@pytest.mark.parametrize("changes", [
    {"analyzer": "Bundle"}, {"analyzer": ""}, {"analyzer": None}, {"analyzer": ...},
    {"bundle_dir": "bundle/bundle-v1"}, {"bundle_dir": ""}, {"bundle_dir": None}, {"bundle_dir": ...},
    {"bundle_manifest_sha256": "edf4a205"}, {"bundle_manifest_sha256": "E" * 64},
    {"bundle_manifest_sha256": "a" * 64 + "\n"}, {"bundle_manifest_sha256": None}, {"bundle_manifest_sha256": ...},
])
def test_invalid_analyzer_settings_are_refused(tmp_path, changes):
    with pytest.raises(ValueError):
        load_api_config(write_config(tmp_path, **changes))
