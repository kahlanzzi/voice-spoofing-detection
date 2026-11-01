"""Shared fixtures for the API tests: an app built from configs/api.yaml, with the database and the
temp folder moved into tmp_path so tests never write into the repo. Audio is generated here, never
committed."""
from __future__ import annotations

import dataclasses
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from api.config import load_api_config
from api.main import create_app


def audio_bytes(seconds: float = 2.0, rate: int = 16000, fmt: str = "WAV", seed: int = 0) -> bytes:
    """A mono 16-bit recording of low noise, encoded in memory."""
    samples = np.random.default_rng(seed).uniform(-0.1, 0.1, int(round(seconds * rate)))
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format=fmt, subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture
def config(tmp_path):
    return dataclasses.replace(load_api_config(), database_path=str(tmp_path / "db" / "api.sqlite"),
                               temp_dir=str(tmp_path / "uploads"))


@pytest.fixture
def client(config):
    with TestClient(create_app(config)) as test_client:  # runs the lifespan: table and temp folder
        yield test_client
