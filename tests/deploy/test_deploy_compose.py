"""docker-compose.yml (Stage 17 Steps 8a, 8b): secret, storage and exposure of the api and web services.

The HMAC key comes only from the environment (.env through Compose interpolation), never from a file in the
repo; the api publishes no port (only nginx may front it, §4.8); the store is on a named volume; temp uploads
are on a tmpfs large enough for the declared number of concurrent uploads at the configs/api.yaml limit.
nginx (web) is pinned by digest, reads deploy/nginx.conf read-only, publishes the port in allowed_origins and
buffers request bodies on a tmpfs.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from api.auth import HMAC_KEY_ENV
from api.config import load_api_config
from api.intake import body_limit

REPO = Path(__file__).resolve().parents[2]
TEXT = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
COMPOSE = yaml.safe_load(TEXT)
API = COMPOSE["services"]["api"]
WEB = COMPOSE["services"]["web"]
UNITS = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}


def tmpfs_bytes(entry: str) -> int:
    """The size= option of a short-syntax tmpfs entry, in bytes."""
    match = re.search(r"size=(\d+)([kmg]?)(?:,|$)", entry)
    assert match, f"tmpfs entry without size: {entry!r}"
    return int(match.group(1)) * UNITS[match.group(2)]


def test_hmac_key_comes_only_from_the_environment_and_is_required():
    assert API["environment"] == {HMAC_KEY_ENV: API["environment"][HMAC_KEY_ENV]}
    assert re.fullmatch(r"\$\{" + HMAC_KEY_ENV + r":\?[^}]+\}", API["environment"][HMAC_KEY_ENV])
    assert "env_file" not in API  # .env also holds the Kaggle, HF and W&B tokens


def test_no_literal_hex_secret_in_the_file():
    # Image digests are the only long hex allowed.
    assert not re.search(r"[0-9a-fA-F]{32,}", re.sub(r"@sha256:[0-9a-f]{64}", "", TEXT))


def test_api_publishes_no_port():
    assert "ports" not in API
    assert API["expose"] == ["8000"]


def test_store_is_on_a_named_volume_at_the_database_folder():
    assert "vsd-var:/app/var" in API["volumes"]
    assert "vsd-var" in COMPOSE["volumes"]
    assert load_api_config().database_path.startswith("var/")  # resolves to /app/var/... in the image


def test_tmpfs_holds_the_declared_concurrent_uploads_at_the_config_limit():
    (entry,) = [t for t in API["tmpfs"] if t.split(":", 1)[0] == "/tmp"]
    assert load_api_config().temp_dir is None  # so uploads go to the system temp folder, /tmp
    uploads = COMPOSE["x-vsd-concurrent-uploads"]
    assert isinstance(uploads, int) and uploads >= 1
    # Two copies per upload: Starlette's spooled multipart part and the API's temp file (api/intake.py).
    assert tmpfs_bytes(entry) >= 2 * body_limit(load_api_config()) * uploads


def test_tmpfs_size_parser():
    assert tmpfs_bytes("/tmp:size=192m,mode=1777") == 192 * 1024 ** 2
    assert tmpfs_bytes("/tmp:mode=1777,size=4096") == 4096


def test_api_container_is_hardened():
    assert API["read_only"] is True
    assert API["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in API["security_opt"]


def test_web_image_is_pinned_by_tag_and_digest():
    assert re.fullmatch(r"nginx:\d+\.\d+\.\d+-alpine@sha256:[0-9a-f]{64}", WEB["image"]), WEB["image"]
    assert "build" not in WEB


def test_web_mounts_the_repo_config_read_only_and_starts_after_a_healthy_api():
    assert WEB["volumes"] == ["./deploy/nginx.conf:/etc/nginx/nginx.conf:ro"]
    assert WEB["depends_on"] == {"api": {"condition": "service_healthy"}}
    assert "env_file" not in WEB and HMAC_KEY_ENV not in WEB.get("environment", {})


def test_web_publishes_the_port_of_allowed_origins():
    assert WEB["ports"] == ["8001:80"]
    published = int(WEB["ports"][0].split(":", 1)[0])
    assert {urlsplit(o).port for o in load_api_config().allowed_origins} == {published}


def test_web_tmpfs_holds_the_declared_concurrent_uploads_at_the_nginx_limit():
    (entry,) = [t for t in WEB["tmpfs"] if t.split(":", 1)[0] == "/var/cache/nginx"]
    conf = (REPO / "deploy" / "nginx.conf").read_text(encoding="utf-8")
    match = re.search(r"^\s*client_max_body_size\s+(\d+)([kmg]?);", conf, re.M)
    nginx_limit = int(match.group(1)) * UNITS[match.group(2)]
    assert tmpfs_bytes(entry) >= nginx_limit * COMPOSE["x-vsd-concurrent-uploads"]
