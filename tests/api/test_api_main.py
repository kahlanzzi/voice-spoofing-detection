import dataclasses

import pytest
from fastapi.testclient import TestClient

from api.auth import HMAC_KEY_ENV
from api.main import create_app


def test_create_app_refuses_a_time_zone_other_than_utc(config):
    # History days are UTC days with no conversion until open decision 6.
    with pytest.raises(ValueError, match="stats_time_zone"):
        create_app(dataclasses.replace(config, stats_time_zone="Africa/Tunis"))


def test_create_app_accepts_utc(config):
    assert config.stats_time_zone == "UTC"
    create_app(config)


@pytest.mark.parametrize("value", [None, "abcd"], ids=["missing", "too short"])
def test_the_api_refuses_to_start_without_a_valid_signin_key(config, monkeypatch, value):
    if value is None:
        monkeypatch.delenv(HMAC_KEY_ENV)
    else:
        monkeypatch.setenv(HMAC_KEY_ENV, value)
    with pytest.raises(ValueError, match=HMAC_KEY_ENV):
        with TestClient(create_app(config)):
            pass


def test_middleware_order_is_request_id_session_content_length_body_limit(config):
    names = [m.cls.__name__ for m in create_app(config).user_middleware]
    assert names == ["RequestIdMiddleware", "SessionGuard", "ContentLengthGuard", "RequestBodyLimitMiddleware"]
