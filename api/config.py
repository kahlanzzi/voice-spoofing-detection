"""configs/api.yaml: the Stage 17 API settings (docs/api_design.md §9), validated.

The loader refuses a missing key, an unknown key or a value of the wrong type (True for 1, 30.5
for 30), so a typo in the file stops the API at start instead of changing its behaviour.
VSD_API_CONFIG, when set, points to another file (tests, deployments).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ENV_VAR = "VSD_API_CONFIG"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO_ROOT / "configs" / "api.yaml"

# §9 key -> accepted type (a tuple accepts several; NoneType allows null).
TYPES = {
    "formats": list,
    "max_size_bytes": int,
    "max_duration_s": int,
    "audio_deletion": str,
    "analysis_timeout_s": int,
    "page_size_max": int,
    "note_max_chars": int,
    "stats_time_zone": str,
    "notice_text": str,
    "mock_deferral_on_fit": float,
    "temp_dir": (str, type(None)),
    "database_path": str,
    "session_idle_minutes": int,
    "session_absolute_hours": int,
    "lockout_threshold": int,
    "lockout_minutes": int,
    "password_min_chars": int,
    "password_max_chars": int,
    "allowed_origins": list,
}
POSITIVE = ("max_size_bytes", "max_duration_s", "analysis_timeout_s", "page_size_max", "note_max_chars",
            "session_idle_minutes", "session_absolute_hours", "lockout_threshold", "lockout_minutes",
            "password_min_chars", "password_max_chars")
# An Origin header value: scheme://host[:port], no path, no trailing slash (§4.8 CSRF).
ORIGIN = re.compile(r"https?://[A-Za-z0-9.-]+(:[0-9]{1,5})?")


@dataclass(frozen=True)
class ApiConfig:
    formats: tuple[str, ...]
    max_size_bytes: int
    max_duration_s: int
    audio_deletion: str
    analysis_timeout_s: int
    page_size_max: int
    note_max_chars: int
    stats_time_zone: str
    notice_text: str
    mock_deferral_on_fit: float
    temp_dir: str | None
    database_path: str
    session_idle_minutes: int
    session_absolute_hours: int
    lockout_threshold: int
    lockout_minutes: int
    password_min_chars: int
    password_max_chars: int
    allowed_origins: tuple[str, ...]


def config_path() -> Path:
    """The config file to load: $VSD_API_CONFIG if set, else configs/api.yaml in this repo."""
    override = os.environ.get(ENV_VAR)
    return Path(override) if override else DEFAULT_PATH


def _type_ok(value, expected) -> bool:
    if isinstance(value, bool):  # bool is an int subclass; never accept it for a number
        return False
    return isinstance(value, expected)


def load_api_config(path=None) -> ApiConfig:
    """Read the API config with yaml.safe_load; strict keys and types; raise ValueError if invalid."""
    path = Path(path) if path is not None else config_path()
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read API config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must hold a YAML mapping")
    missing = sorted(set(TYPES) - set(cfg))
    extra = sorted(set(cfg) - set(TYPES))
    if missing or extra:
        raise ValueError(f"{path}: missing keys {missing}, unknown keys {extra}")
    for key, expected in TYPES.items():
        if not _type_ok(cfg[key], expected):
            raise ValueError(f"{path}: {key} has the wrong type: {cfg[key]!r}")
    for key in POSITIVE:
        if cfg[key] <= 0:
            raise ValueError(f"{path}: {key} must be > 0, got {cfg[key]!r}")
    formats = cfg["formats"]
    if not formats or not all(isinstance(f, str) and f and f == f.lower() for f in formats):
        raise ValueError(f"{path}: formats must be a non-empty list of lower-case extensions, got {formats!r}")
    if not 0.0 <= cfg["mock_deferral_on_fit"] <= 1.0:
        raise ValueError(f"{path}: mock_deferral_on_fit must be in [0, 1], got {cfg['mock_deferral_on_fit']!r}")
    if not cfg["notice_text"].strip():
        raise ValueError(f"{path}: notice_text must not be empty")
    if not cfg["database_path"].strip():
        raise ValueError(f"{path}: database_path must not be empty")
    if cfg["password_min_chars"] > cfg["password_max_chars"]:
        raise ValueError(f"{path}: password_min_chars must not exceed password_max_chars")
    origins = cfg["allowed_origins"]
    if not origins or not all(isinstance(o, str) and ORIGIN.fullmatch(o) for o in origins):
        raise ValueError(f"{path}: allowed_origins must be a non-empty list of scheme://host[:port], got {origins!r}")
    return ApiConfig(**{**cfg, "formats": tuple(formats), "allowed_origins": tuple(origins)})


def resolve_path(value: str) -> Path:
    """A path from config: absolute as given, relative to the repo root otherwise (not the cwd)."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path
