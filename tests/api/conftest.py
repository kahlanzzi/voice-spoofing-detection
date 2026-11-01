"""Shared fixtures for the API tests: an app built from configs/api.yaml, with the database and the
temp folder moved into tmp_path so tests never write into the repo. Audio is generated here, never
committed.

Every test gets a fresh random VSD_SIGNIN_HMAC_KEY (no key is written anywhere) and, unless it asks
for real_hasher, a fast argon2id hasher in place of the §4.8 one (about 80 ms per hash here)."""
from __future__ import annotations

import dataclasses
import io
import secrets

import numpy as np
import pytest
import soundfile as sf
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

import api.main
from api.auth import HMAC_KEY_ENV, password_hasher
from api.config import load_api_config
from api.main import create_app


def audio_bytes(seconds: float = 2.0, rate: int = 16000, fmt: str = "WAV", seed: int = 0) -> bytes:
    """A mono 16-bit recording of low noise, encoded in memory."""
    samples = np.random.default_rng(seed).uniform(-0.1, 0.1, int(round(seconds * rate)))
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format=fmt, subtype="PCM_16")
    return buffer.getvalue()


def fast_hasher() -> PasswordHasher:
    """argon2id at the lowest cost: tests only, never the §4.8 parameters."""
    return PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)


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
    with TestClient(create_app(config)) as test_client:  # runs the lifespan: table and temp folder
        yield test_client
