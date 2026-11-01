"""configs/api_logging.yaml (Stage 17 Step 8a): the api logger writes INFO, uvicorn writes no access line.

Applied with logging.config.dictConfig as uvicorn --log-config does, then the touched loggers are restored
so other tests' caplog checks are not affected.
"""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((REPO / "configs" / "api_logging.yaml").read_text(encoding="utf-8"))
NAMES = ("api", "uvicorn.error", "uvicorn.access")


@pytest.fixture
def applied():
    saved = {n: (logging.getLogger(n).level, list(logging.getLogger(n).handlers), logging.getLogger(n).propagate)
             for n in NAMES}
    logging.config.dictConfig(CONFIG)
    yield
    for name, (level, handlers, propagate) in saved.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers[:] = handlers
        logger.propagate = propagate


def test_api_logger_writes_info(applied):
    logger = logging.getLogger("api")
    assert logger.isEnabledFor(logging.INFO)
    assert logger.handlers


def test_access_logger_writes_nothing(applied):
    logger = logging.getLogger("uvicorn.access")
    own = [h for h in logger.handlers if not type(h).__module__.startswith("_pytest")]  # pytest adds its capture
    assert not own and not logger.propagate
    assert not logger.isEnabledFor(logging.ERROR)


def test_existing_loggers_are_kept():
    assert CONFIG["disable_existing_loggers"] is False
