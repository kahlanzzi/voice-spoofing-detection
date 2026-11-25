"""docker-compose.yml (Stage 17 Steps 8a, 8b; P2a, D53 (4)): secrets, storage and exposure of the db, api and
web services.

The HMAC key and the database passwords come only from the environment (.env through Compose interpolation),
never from a file in the repo; the api receives its own role's password only, never the superuser's; the api
and db publish no port (only nginx may front the api, §4.8); the store is PostgreSQL on a named volume, its
image pinned by digest, its log kept free of row values; temp uploads are on a tmpfs large enough for K
(analysis_workers) uploads at the configs/api.yaml limit (D53 (2), (3)).
nginx (web) is built from deploy/web.Dockerfile (pins: tests/deploy/test_deploy_web_image.py) with the app and
deploy/nginx.conf inside, publishes the port in allowed_origins and buffers request bodies on a tmpfs.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from api.auth import HMAC_KEY_ENV
from api.config import load_api_config
from api.db import APP_PASSWORD_ENV
from api.intake import body_limit

REPO = Path(__file__).resolve().parents[2]
TEXT = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
COMPOSE = yaml.safe_load(TEXT)
API = COMPOSE["services"]["api"]
WEB = COMPOSE["services"]["web"]
DB = COMPOSE["services"]["db"]
DB_TEST = COMPOSE["services"]["db-test"]
SUPERUSER_PASSWORD_ENV = "VSD_DB_PASSWORD"
INIT_SCRIPT = (REPO / "deploy" / "postgres-init" / "10-app-role.sh").read_text(encoding="utf-8")
UNITS = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}


def tmpfs_bytes(entry: str) -> int:
    """The size= option of a short-syntax tmpfs entry, in bytes."""
    match = re.search(r"size=(\d+)([kmg]?)(?:,|$)", entry)
    assert match, f"tmpfs entry without size: {entry!r}"
    return int(match.group(1)) * UNITS[match.group(2)]


def required(variable: str, value: str) -> bool:
    """value is ${variable:?message}: interpolated from the environment, no default, no start without it."""
    return re.fullmatch(r"\$\{" + variable + r":\?[^}]+\}", value) is not None


def test_api_secrets_come_only_from_the_environment_and_are_required():
    assert set(API["environment"]) == {HMAC_KEY_ENV, APP_PASSWORD_ENV}
    assert required(HMAC_KEY_ENV, API["environment"][HMAC_KEY_ENV])
    assert required(APP_PASSWORD_ENV, API["environment"][APP_PASSWORD_ENV])
    assert "env_file" not in API  # .env also holds the Kaggle, HF and W&B tokens


def test_the_api_never_receives_the_superuser_password():
    assert SUPERUSER_PASSWORD_ENV not in API["environment"]
    assert SUPERUSER_PASSWORD_ENV not in TEXT.split("\n  api:", 1)[1].split("\n  web:", 1)[0]
    assert load_api_config().database_user == "vsd_app"


def test_db_passwords_are_required_with_no_default():
    env = DB["environment"]
    assert required(SUPERUSER_PASSWORD_ENV, env["POSTGRES_PASSWORD"])
    assert required(APP_PASSWORD_ENV, env[APP_PASSWORD_ENV])
    assert env["POSTGRES_USER"] == "postgres" and env["POSTGRES_DB"] == load_api_config().database_name
    assert "env_file" not in DB
    for name in (SUPERUSER_PASSWORD_ENV, APP_PASSWORD_ENV):
        assert not re.search(r"\$\{" + name + r":?-", TEXT)  # never a ${VAR:-default}


def test_no_literal_hex_secret_in_the_file():
    # Image digests are the only long hex allowed.
    assert not re.search(r"[0-9a-fA-F]{32,}", re.sub(r"@sha256:[0-9a-f]{64}", "", TEXT))


def test_api_publishes_no_port():
    assert "ports" not in API
    assert API["expose"] == ["8000"]


def test_store_is_postgresql_on_a_named_volume_pinned_by_digest():
    assert re.fullmatch(r"postgres:18\.[0-9]+-trixie@sha256:[0-9a-f]{64}", DB["image"])
    assert DB_TEST["image"] == DB["image"]
    assert "vsd-pg:/var/lib/postgresql" in DB["volumes"]
    assert "vsd-pg" in COMPOSE["volumes"]
    assert "--locale=C" in DB["environment"]["POSTGRES_INITDB_ARGS"]
    assert "--locale=C" in DB_TEST["environment"]["POSTGRES_INITDB_ARGS"]
    assert load_api_config().database_host == "db"


def test_the_old_sqlite_store_is_mounted_read_only_for_the_import():
    assert API["volumes"] == ["vsd-var:/app/var:ro"]
    assert "vsd-var" in COMPOSE["volumes"]


def test_db_publishes_no_port_and_the_test_db_only_on_loopback():
    assert "ports" not in DB and "expose" not in DB
    assert DB_TEST["profiles"] == ["test"]
    assert DB_TEST["ports"] == ["127.0.0.1:55432:5432"]
    assert DB_TEST["tmpfs"] == ["/var/lib/postgresql"]  # no volume: never holds real data
    assert "volumes" not in DB_TEST


def test_api_starts_after_a_healthy_db_whose_check_proves_the_init_is_done():
    assert API["depends_on"] == {"db": {"condition": "service_healthy"}}
    # Over TCP: the image's init server listens on the socket only, so on a new volume this passes after the
    # init (which creates vsd_app). pg_isready does not log in.
    assert DB["healthcheck"]["test"] == ["CMD", "pg_isready", "-h", "127.0.0.1", "-U", "vsd_app", "-d", "vsd"]


def test_init_script_creates_a_non_superuser_app_role_from_its_own_variable():
    assert "./deploy/postgres-init:/docker-entrypoint-initdb.d:ro" in DB["volumes"]
    assert re.search(r'^: "\$\{VSD_DB_APP_PASSWORD:\?[^}]+\}"$', INIT_SCRIPT, re.M)
    assert "set -euo pipefail" in INIT_SCRIPT and "ON_ERROR_STOP=1" in INIT_SCRIPT
    assert re.search(r"CREATE ROLE vsd_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD :'app_pw';",
                     INIT_SCRIPT)
    assert "ALTER DATABASE vsd OWNER TO vsd_app;" in INIT_SCRIPT
    assert "<<'SQL'" in INIT_SCRIPT  # quoted heredoc: the shell never splices the password into the SQL
    assert SUPERUSER_PASSWORD_ENV not in INIT_SCRIPT
    assert (REPO / "deploy" / "postgres-init" / "10-app-role.sh").stat().st_mode & 0o111  # executable


# Each flag keeps row values (notes, hashes) and client addresses out of the PostgreSQL log.
PRIVATE_LOG_FLAGS = {
    "log_statement": "none", "log_min_error_statement": "panic", "log_error_verbosity": "terse",
    "log_parameter_max_length": "0", "log_parameter_max_length_on_error": "0",
    "log_min_duration_statement": "-1", "log_connections": "off", "log_disconnections": "off",
    "log_hostname": "off", "log_line_prefix": "%m [%p] ",
}


def test_db_log_keeps_row_values_and_client_addresses_out():
    command = DB["command"]
    assert command[0] == "postgres"
    flags = dict(value.split("=", 1) for flag, value in zip(command[1::2], command[2::2]) if flag == "-c")
    assert len(command) == 1 + 2 * len(flags)
    assert flags == PRIVATE_LOG_FLAGS
    assert "%h" not in flags["log_line_prefix"] and "%r" not in flags["log_line_prefix"]


def test_db_container_is_hardened():
    assert DB["read_only"] is True
    assert DB["cap_drop"] == ["ALL"]
    assert set(DB["cap_add"]) == {"CHOWN", "DAC_OVERRIDE", "FOWNER", "SETGID", "SETUID"}
    assert "no-new-privileges:true" in DB["security_opt"]


def test_api_tmpfs_holds_k_uploads_at_the_config_limit():
    (entry,) = [t for t in API["tmpfs"] if t.split(":", 1)[0] == "/tmp"]
    config = load_api_config()
    assert config.temp_dir is None  # so uploads go to the system temp folder, /tmp
    # The API takes a worker before it reads the body, so at most K uploads are in /tmp at once (api/pool.py).
    # Two copies per upload: Starlette's spooled multipart part and the API's temp file (api/intake.py).
    assert tmpfs_bytes(entry) >= 2 * body_limit(config) * config.analysis_workers


def test_api_tmpfs_is_the_32_mib_round_up_for_k():
    (entry,) = [t for t in API["tmpfs"] if t.split(":", 1)[0] == "/tmp"]
    config = load_api_config()
    needed = 2 * body_limit(config) * config.analysis_workers
    assert tmpfs_bytes(entry) == -(-needed // (32 * 1024 ** 2)) * 32 * 1024 ** 2  # not oversized either


def test_concurrent_uploads_key_is_gone():
    # Replaced by analysis_workers, which the pool enforces (D53 (3), Entry 125).
    assert "x-vsd-concurrent-uploads" not in COMPOSE
    assert "x-vsd-concurrent-uploads" not in TEXT


def test_tmpfs_size_parser():
    assert tmpfs_bytes("/tmp:size=192m,mode=1777") == 192 * 1024 ** 2
    assert tmpfs_bytes("/tmp:mode=1777,size=4096") == 4096


def test_api_container_is_hardened():
    assert API["read_only"] is True
    assert API["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in API["security_opt"]


def test_web_is_built_from_its_dockerfile_with_the_config_inside():
    assert WEB["build"] == {"context": ".", "dockerfile": "deploy/web.Dockerfile"}
    assert "image" not in WEB  # the nginx pin lives in the Dockerfile
    assert "volumes" not in WEB  # a mount would mask the image's nginx.conf


def test_web_starts_after_a_healthy_api_without_secrets():
    assert WEB["depends_on"] == {"api": {"condition": "service_healthy"}}
    assert "env_file" not in WEB
    for name in (HMAC_KEY_ENV, APP_PASSWORD_ENV, SUPERUSER_PASSWORD_ENV):
        assert name not in WEB.get("environment", {})


def test_web_publishes_the_port_of_allowed_origins():
    assert WEB["ports"] == ["8001:80"]
    published = int(WEB["ports"][0].split(":", 1)[0])
    assert {urlsplit(o).port for o in load_api_config().allowed_origins} == {published}


def test_web_tmpfs_holds_two_k_bodies_at_the_nginx_limit():
    (entry,) = [t for t in WEB["tmpfs"] if t.split(":", 1)[0] == "/var/cache/nginx"]
    conf = (REPO / "deploy" / "nginx.conf").read_text(encoding="utf-8")
    match = re.search(r"^\s*client_max_body_size\s+(\d+)([kmg]?);", conf, re.M)
    nginx_limit = int(match.group(1)) * UNITS[match.group(2)]
    assert tmpfs_bytes(entry) >= nginx_limit * 2 * load_api_config().analysis_workers

# --- the numba cache and the real model (P1) -------------------------------------------------------

def test_numba_cache_is_its_own_writable_tmpfs():
    """In the base file, so the mock stack and CI's image check have it too; owned by the image's user, not /tmp."""
    dockerfile = (REPO / "deploy" / "api.Dockerfile").read_text(encoding="utf-8")
    (folder,) = re.findall(r"^ENV NUMBA_CACHE_DIR=(\S+)$", dockerfile, re.M)
    (uid,) = re.findall(r"useradd --uid (\d+)", dockerfile)
    assert folder != "/tmp" and not folder.startswith("/tmp/")
    (entry,) = [t for t in API["tmpfs"] if t.split(":", 1)[0] == folder]
    options = entry.split(":", 1)[1].split(",")
    assert tmpfs_bytes(entry) > 0
    assert f"uid={uid}" in options and f"gid={uid}" in options and "mode=0700" in options


def test_bundle_override_mounts_read_only_and_selects_the_bundle_config():
    override = yaml.safe_load((REPO / "docker-compose.bundle.yml").read_text(encoding="utf-8"))
    assert set(override) == {"services"} and set(override["services"]) == {"api"}
    api = override["services"]["api"]
    assert set(api) == {"environment", "volumes"}
    assert api["environment"] == {"VSD_API_CONFIG": "/app/configs/api.bundle.yaml"}
    (mount,) = api["volumes"]
    assert required("VSD_BUNDLE_DIR", mount["source"])
    assert mount["type"] == "bind" and mount["read_only"] is True
    assert mount["bind"] == {"create_host_path": False}
    assert mount["target"] == load_api_config(REPO / "configs" / "api.bundle.yaml").bundle_dir


def test_the_base_stack_serves_the_mock():
    assert "VSD_API_CONFIG" not in TEXT and "VSD_BUNDLE_DIR" not in TEXT
    assert load_api_config(REPO / "configs" / "api.yaml").analyzer == "mock"
