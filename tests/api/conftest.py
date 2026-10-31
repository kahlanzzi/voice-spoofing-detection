"""Shared fixtures for the API tests: an app built from the real configs/api.yaml."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.config import load_api_config
from api.main import create_app


@pytest.fixture
def config():
    return load_api_config()


@pytest.fixture
def client(config):
    return TestClient(create_app(config))
