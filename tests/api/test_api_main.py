import dataclasses
import functools
import multiprocessing
import re
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.analyzer import MockAnalyzer
from api.auth import HMAC_KEY_ENV
from api.bundle_analyzer import BundleAnalyzer
from api.main import analyzer_factory_for, create_app
from api.pool import NEEDS_SELECTOR_LOOP, PoolStartError
from tests.api._analyzers import ScriptedAnalyzer, started
from tests.api.conftest import audio_bytes, signed_in_client


def test_create_app_refuses_a_time_zone_other_than_utc(config):
    # History days are UTC days with no conversion until open decision 6.
    with pytest.raises(ValueError, match="stats_time_zone"):
        create_app(dataclasses.replace(config, stats_time_zone="Africa/Tunis"))


def test_create_app_accepts_utc(config):
    assert config.stats_time_zone == "UTC"
    create_app(config)


def assert_builds_but_skips_at_the_lifespan(config, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    before = set(multiprocessing.active_children())
    try:
        client = TestClient(create_app(config))
    except pytest.skip.Exception:  # would otherwise count as a skip, not a failure
        pytest.fail("building an app must not be skipped")
    with pytest.raises(pytest.skip.Exception, match=re.escape(NEEDS_SELECTOR_LOOP)):
        with client:
            pass
    assert set(multiprocessing.active_children()) <= before


def test_windows_skips_where_the_pool_starts_not_where_the_app_is_built(config, monkeypatch):
    assert_builds_but_skips_at_the_lifespan(config, monkeypatch)


@pytest.mark.own_pool
def test_windows_skip_also_covers_apps_with_their_own_pool(config, monkeypatch):
    assert_builds_but_skips_at_the_lifespan(config, monkeypatch)


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


# --- analyzer selection ---------------------------------------------------------------------------

@pytest.mark.own_pool
def test_the_config_selects_the_analyzer(config):
    mock = create_app(config).state.pool.factory
    assert (mock.func, mock.args, mock.keywords) == (MockAnalyzer, (config.mock_deferral_on_fit,), {})
    bundle = create_app(dataclasses.replace(config, analyzer="bundle")).state.pool.factory
    assert (bundle.func, bundle.keywords) == (BundleAnalyzer, {})
    assert bundle.args == (config.bundle_dir, config.bundle_manifest_sha256)


def test_an_unknown_analyzer_is_never_the_mock(config):
    with pytest.raises(ValueError, match="unknown analyzer"):
        analyzer_factory_for(dataclasses.replace(config, analyzer="other"))


# --- analysis pool lifecycle (D53 (2)) ------------------------------------------------------------

def test_app_refuses_to_start_when_a_worker_hook_fails(config, tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    failing = functools.partial(ScriptedAnalyzer, str(control), fail_from_build=1)
    with pytest.raises(PoolStartError, match="start-up hook failed"):
        with TestClient(create_app(config, analyzer_factory=failing)):
            pass
    if sys.platform == "linux":  # /proc: the container and CI
        pids = [int(p.read_text()) for p in control.glob("build_*")]
        assert pids and not [pid for pid in pids if Path(f"/proc/{pid}").exists()]


@pytest.mark.own_pool
def test_create_app_starts_no_process(config):
    app = create_app(config)
    assert app.state.pool.worker_pids() == set()


@pytest.mark.own_pool
def test_lifespan_starts_k_workers_and_stops_them(config):
    with TestClient(create_app(dataclasses.replace(config, analysis_workers=2))) as client:
        workers = [*client.app.state.pool._idle]
        assert len(workers) == 2 and all(w.process.is_alive() for w in workers)
    assert all(w.closed and w.exitcode is not None for w in workers)
    if sys.platform == "linux":  # /proc: the container and CI
        assert not [w.pid for w in workers if Path(f"/proc/{w.pid}").exists()]


def test_status_stays_available_while_pool_is_full(config, tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    (control / "mode").write_text("wait")
    app = create_app(config, analyzer_factory=functools.partial(ScriptedAnalyzer, str(control)))
    with signed_in_client(app) as client:
        responses = []
        thread = threading.Thread(target=lambda: responses.append(
            client.post("/api/analyses", files={"file": ("a.wav", audio_bytes(), "audio/wav")})))
        thread.start()
        deadline = time.monotonic() + 30
        while not started(control):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert client.post("/api/analyses", files={"file": ("b.wav", audio_bytes(), "audio/wav")}).status_code == 503
        body = client.get("/api/status").json()
        assert body["state"] == "ready" and body["checks"]["analysis"] == "available"
        assert client.get("/api/health").json() == {"ok": True}
        (control / "release").touch()
        thread.join(30)
        assert responses[0].status_code == 201
