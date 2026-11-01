import dataclasses

import pytest

from api.main import create_app


def test_create_app_refuses_a_time_zone_other_than_utc(config):
    # History days are UTC days with no conversion until open decision 6.
    with pytest.raises(ValueError, match="stats_time_zone"):
        create_app(dataclasses.replace(config, stats_time_zone="Africa/Tunis"))


def test_create_app_accepts_utc(config):
    assert config.stats_time_zone == "UTC"
    create_app(config)
