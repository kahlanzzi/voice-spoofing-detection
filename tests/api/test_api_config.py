from pathlib import Path

import pytest
import yaml

from api.config import DEFAULT_PATH, ENV_VAR, ApiConfig, config_path, load_api_config

SECTION_9 = {
    "formats": ("wav", "flac", "mp3"),
    "max_size_bytes": 20_971_520,
    "max_duration_s": 60,
    "audio_deletion": "Deleted when processing finishes",
    "analysis_timeout_s": 30,
    "page_size_max": 50,
    "note_max_chars": 2000,
    "stats_time_zone": "UTC",
    "notice_text": "Decision support only. This result estimates whether the recording is synthetic. "
                   "It does not prove fraud and does not verify identity.",
    "mock_deferral_on_fit": 0.0993,
    "temp_dir": None,
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
