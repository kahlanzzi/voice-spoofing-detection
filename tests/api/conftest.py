"""Shared fixtures for the API tests: an app built from configs/api.yaml, with the temp folder moved into
tmp_path and the database moved to a fresh PostgreSQL database per test, so tests never write into the
repo or share rows. Audio is generated here, never committed.

The database tests need a PostgreSQL server: VSD_TEST_DATABASE_URL (default: the Compose db-test service,
postgresql://postgres@127.0.0.1:55432/postgres; docs/setup.md "Tests and CI"). That URL is a superuser
used only to set up: it creates the API's role vsd_app (not a superuser) and one template database owned
by vsd_app with the schema applied as vsd_app; each test gets its own copy of the template, and the app
under test always connects as vsd_app. Without a server the database tests are skipped and the run
ends with a WARNING line; with REQUIRE_POSTGRES_TESTS=1 (CI) they fail instead.

Every test gets a fresh random VSD_SIGNIN_HMAC_KEY (no key is written anywhere) and, unless it asks
for real_hasher, a fast argon2id hasher in place of the §4.8 one (about 80 ms per hash here).

Every route under /api but POST /session and GET /health needs a session (§4.8), so `client` is signed
in and sends an allowed Origin on every request; `anon` is the same client before sign-in. Tests that
build their own app sign in with signed_in()."""
from __future__ import annotations

import dataclasses
import io
import os
import secrets
import uuid
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import psycopg
import pytest
import soundfile as sf
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

import api.main
from api.auth import HMAC_KEY_ENV, password_hasher
from api.config import load_api_config
from api.db import APP_PASSWORD_ENV, Database
from api.main import create_app
from api.repository import AnalysisRepository

HTTPS = "https://testserver"  # httpx keeps the Secure cookie only over https (§4.8 Cookie)
ANALYST = "analyst01"
ANALYST_PASSWORD = "correct horse battery staple"

TEST_URL_ENV = "VSD_TEST_DATABASE_URL"
DEFAULT_TEST_URL = "postgresql://postgres@127.0.0.1:55432/postgres"
REQUIRE_ENV = "REQUIRE_POSTGRES_TESTS"
DB_SKIP_REASON = "no PostgreSQL server"
APP_ROLE = "vsd_app"
# The test server trusts local connections, so this is never checked; the API still requires the variable.
TEST_APP_PASSWORD = "test-only-not-a-secret"
_server_version: list[str] = []  # filled once the server is reached: shown in the run's summary
_summary: dict[str, int] = {}


def test_database_url() -> str:
    return os.environ.get(TEST_URL_ENV) or DEFAULT_TEST_URL


test_database_url.__test__ = False  # a helper, not a test


def admin_connect(dbname: str | None = None) -> psycopg.Connection:
    """An autocommit superuser connection to the test server (setup only, never the app)."""
    kwargs = {"dbname": dbname} if dbname else {}
    return psycopg.connect(test_database_url(), autocommit=True, connect_timeout=3, **kwargs)


def pytest_terminal_summary(terminalreporter):
    """Count the database tests skipped for want of a server; the line itself is printed last."""
    skipped = [r for r in terminalreporter.stats.get("skipped", [])
               if isinstance(r.longrepr, tuple) and DB_SKIP_REASON in str(r.longrepr[2])]
    _summary["db_skipped"] = len(skipped)
    if _server_version:
        terminalreporter.write_line(f"database tests ran on {_server_version[0]}")


def pytest_unconfigure(config):
    """After pytest's own result line: the WARNING is the last line of the run (docs/setup.md)."""
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None and _summary.get("db_skipped"):
        reporter.write_line(f"WARNING: {_summary['db_skipped']} database tests skipped: {DB_SKIP_REASON} "
                            f"(set {REQUIRE_ENV}=1 to fail instead)", yellow=True, bold=True)


def audio_bytes(seconds: float = 2.0, rate: int = 16000, fmt: str = "WAV", seed: int = 0) -> bytes:
    """A mono 16-bit recording of low noise, encoded in memory."""
    samples = np.random.default_rng(seed).uniform(-0.1, 0.1, int(round(seconds * rate)))
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format=fmt, subtype="PCM_16")
    return buffer.getvalue()


def fast_hasher() -> PasswordHasher:
    """argon2id at the lowest cost: tests only, never the §4.8 parameters."""
    return PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)


def signed_in(test_client: TestClient, username: str = ANALYST) -> TestClient:
    """Create an enabled account unless it exists, sign it in on test_client (which must have run the
    lifespan and use HTTPS) and send the first allowed Origin on every later request."""
    accounts = test_client.app.state.accounts
    if accounts.get_account(username) is None:
        accounts.create_account(username, fast_hasher().hash(ANALYST_PASSWORD), "2026-11-01T08:00:00Z")
    app = test_client.app
    test_client.headers["Origin"] = app.state.config.allowed_origins[0]
    response = test_client.post("/api/session", json={"username": username, "password": ANALYST_PASSWORD})
    assert response.status_code == 200, response.text
    return test_client


@contextmanager
def signed_in_client(app, username: str = ANALYST, **kwargs) -> Iterator[TestClient]:
    """A TestClient of app, lifespan run, signed in as username."""
    with TestClient(app, base_url=HTTPS, **kwargs) as test_client:
        yield signed_in(test_client, username)


@pytest.fixture(autouse=True)
def signin_key(monkeypatch):
    monkeypatch.setenv(HMAC_KEY_ENV, secrets.token_hex(32))


@pytest.fixture(autouse=True)
def _fast_app_hasher(monkeypatch):
    monkeypatch.setattr(api.main, "password_hasher", fast_hasher)


@pytest.fixture
def real_hasher(monkeypatch):
    """Build apps with the real §4.8 hasher again."""
    monkeypatch.setattr(api.main, "password_hasher", password_hasher)


@pytest.fixture(autouse=True)
def app_db_password(monkeypatch):
    monkeypatch.setenv(APP_PASSWORD_ENV, TEST_APP_PASSWORD)


@pytest.fixture(scope="session")
def pg_template() -> Iterator[str]:
    """Reach the test server, create the vsd_app role if missing and a template database owned by it with
    the schema applied as vsd_app. Skip (or fail with REQUIRE_POSTGRES_TESTS=1) if the server is unreachable."""
    try:
        admin = admin_connect()
    except psycopg.OperationalError as exc:
        reason = f"{DB_SKIP_REASON} at {test_database_url()} ({type(exc).__name__})"
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(f"{reason}; {REQUIRE_ENV}=1 makes this a failure", pytrace=False)
        pytest.skip(f"{reason}; set {REQUIRE_ENV}=1 to make this a failure")
    template = f"vsd_tpl_{uuid.uuid4().hex[:12]}"
    with admin:
        _server_version.append(admin.execute("SELECT version()").fetchone()[0].split(" on ")[0])
        if admin.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (APP_ROLE,)).fetchone() is None:
            admin.execute(f"CREATE ROLE {APP_ROLE} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE")
        admin.execute(f"CREATE DATABASE {template} OWNER {APP_ROLE}")
    database = app_database(template)
    database.open(TEST_APP_PASSWORD)
    try:
        AnalysisRepository(database).create_schema()
    finally:
        database.close()
    yield template
    with admin_connect() as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {template} WITH (FORCE)")


def app_database(name: str, timeout_s: int = 5) -> Database:
    """A Database on the test server, as the app's role (not opened)."""
    info = psycopg.conninfo.conninfo_to_dict(test_database_url())
    return Database(info.get("host", "127.0.0.1"), int(info.get("port", 5432)), name, APP_ROLE, timeout_s=timeout_s)


@pytest.fixture
def pg_database(pg_template) -> Iterator[str]:
    """A fresh database for one test: a copy of the template (schema applied, no rows), owned by vsd_app."""
    name = f"t_{uuid.uuid4().hex}"
    with admin_connect() as admin:
        admin.execute(f"CREATE DATABASE {name} TEMPLATE {pg_template} OWNER {APP_ROLE}")
    yield name
    with admin_connect() as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


@pytest.fixture
def empty_db(pg_template) -> Iterator[Database]:
    """An open Database on a new database with no tables at all (not from the template), owned by vsd_app:
    for the schema runner tests."""
    name = f"t_{uuid.uuid4().hex}"
    with admin_connect() as admin:
        admin.execute(f"CREATE DATABASE {name} OWNER {APP_ROLE}")
    database = app_database(name)
    database.open(TEST_APP_PASSWORD)
    yield database
    database.close()
    with admin_connect() as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


@pytest.fixture
def db(pg_database) -> Iterator[Database]:
    """An open Database on this test's database, as vsd_app."""
    database = app_database(pg_database)
    database.open(TEST_APP_PASSWORD)
    yield database
    database.close()


@pytest.fixture
def config(tmp_path, pg_database):
    info = psycopg.conninfo.conninfo_to_dict(test_database_url())
    return dataclasses.replace(load_api_config(), database_host=info.get("host", "127.0.0.1"),
                               database_port=int(info.get("port", 5432)), database_name=pg_database,
                               database_user=APP_ROLE, temp_dir=str(tmp_path / "uploads"))


@pytest.fixture
def client(config):
    with signed_in_client(create_app(config)) as test_client:  # runs the lifespan: tables and temp folder
        yield test_client


@pytest.fixture
def anon(config):
    """No session and no Origin header."""
    with TestClient(create_app(config), base_url=HTTPS) as test_client:
        yield test_client
