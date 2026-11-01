"""Shared fixtures for the API tests: an app built from configs/api.yaml, with the database and the
temp folder moved into tmp_path so tests never write into the repo. Audio is generated here, never
committed.

Every test gets a fresh random VSD_SIGNIN_HMAC_KEY (no key is written anywhere) and, unless it asks
for real_hasher, a fast argon2id hasher in place of the §4.8 one (about 80 ms per hash here).

Every route under /api but POST /session and GET /health needs a session (§4.8), so `client` is signed
in and sends an allowed Origin on every request; `anon` is the same client before sign-in. Tests that
build their own app sign in with signed_in()."""
from __future__ import annotations

import dataclasses
import io
import secrets
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pytest
import soundfile as sf
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

import api.main
from api.auth import HMAC_KEY_ENV, password_hasher
from api.config import load_api_config
from api.main import create_app

HTTPS = "https://testserver"  # httpx keeps the Secure cookie only over https (§4.8 Cookie)
ANALYST = "analyst01"
ANALYST_PASSWORD = "correct horse battery staple"

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


@pytest.fixture
def config(tmp_path):
    return dataclasses.replace(load_api_config(), database_path=str(tmp_path / "db" / "api.sqlite"),
                               temp_dir=str(tmp_path / "uploads"))


@pytest.fixture
def client(config):
    with signed_in_client(create_app(config)) as test_client:  # runs the lifespan: tables and temp folder
        yield test_client


@pytest.fixture
def anon(config):
    """No session and no Origin header."""
    with TestClient(create_app(config), base_url=HTTPS) as test_client:
        yield test_client
