"""A store holds mock results or real ones, never both (§4.7): checked at every start, in both directions.
The analyzers are fakes; the real one's info().mock is tested in test_api_bundle_analyzer.py."""
from __future__ import annotations

import functools
import re
import sys

import pytest
from fastapi.testclient import TestClient

from api.config import REPO_ROOT
from api.main import create_app
from api.pool import NEEDS_SELECTOR_LOOP
from api.repository import REAL_STACK_SECTION, AnalysisRecord, MixedStoreError
from tests.api._analyzers import FakeAnalyzer
from tests.api.conftest import HTTPS, audio_bytes, signed_in_client

pytestmark = [pytest.mark.own_pool, pytest.mark.skipif(sys.platform == "win32", reason=NEEDS_SELECTOR_LOOP)]


def app_for(config, mock: bool):
    return create_app(config, analyzer_factory=functools.partial(FakeAnalyzer, info_overrides={"mock": mock}))


def record(n: int, mock: bool) -> AnalysisRecord:
    return AnalysisRecord(request_id=f"req_{n:032x}", created_at="2026-11-24T10:00:00Z", prediction="authentic",
                          confidence=0.9, risk_level="LOW", model_version="m", profile_version="p",
                          policy_version="r", notice="Decision support only.", processing_ms=5, is_mock=mock)


def analyse(config, mock: bool) -> None:
    with signed_in_client(app_for(config, mock)) as client:
        response = client.post("/api/analyses", files={"file": ("call.wav", audio_bytes(), "audio/wav")})
        assert response.status_code == 201, response.text


@pytest.mark.parametrize("stored, served", [(True, False), (False, True)], ids=["mock-store", "real-store"])
def test_store_with_the_other_kind_of_row_refuses_to_start(config, stored, served):
    analyse(config, stored)
    app = app_for(config, served)
    words = {True: "mock", False: "real"}
    with pytest.raises(MixedStoreError, match=f"gives {words[served]} results but the store holds {words[stored]}") as raised:
        with TestClient(app, base_url=HTTPS):
            pass
    message = str(raised.value)
    assert f"run the {words[served]} analyzer on a store of its own" in message
    assert "vsd-real" in message and "Empty the database" not in message  # switch stores, never wipe one
    assert app.state.pool.worker_pids() == set()  # no worker left
    assert not app.state.database.is_open


@pytest.mark.parametrize("mock", [True, False])
def test_the_same_kind_and_an_empty_store_start(config, mock):
    with TestClient(app_for(config, mock), base_url=HTTPS):  # empty store
        pass
    analyse(config, mock)
    with TestClient(app_for(config, mock), base_url=HTTPS) as client:
        assert client.get("/api/health").status_code == 200


def test_one_row_of_the_other_kind_among_many_refuses_to_start(config):
    with TestClient(app_for(config, False), base_url=HTTPS) as client:
        for n in range(5):
            client.app.state.repository.add(record(n, mock=False))
        client.app.state.repository.add(record(9, mock=True))
    with pytest.raises(MixedStoreError):
        with TestClient(app_for(config, False), base_url=HTTPS):
            pass


def test_the_section_named_in_the_message_is_a_heading_of_setup_md():
    text = (REPO_ROOT / "docs" / "setup.md").read_text(encoding="utf-8")
    headings = re.findall(r"^#{2,} (.+?)\s*$", text, flags=re.MULTILINE)
    assert headings.count(REAL_STACK_SECTION) == 1
