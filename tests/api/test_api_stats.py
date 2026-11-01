import dataclasses
import logging
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api.analyzer import MockAnalyzer
from api.main import create_app
from api.repository import AnalysisRecord
from tests.api.conftest import audio_bytes

URL = "/api/stats"
NOW = "2026-10-30T09:14:05Z"
ZERO_COUNTS = {"synthetic": 0, "uncertain": 0, "authentic": 0}
INVALID = {"error": {"code": "invalid_request", "message": "The request is not valid.", "request_id": None}}
KEYS = {"from", "to", "time_zone", "generated_at", "today", "total", "counts", "uncertain_share",
        "median_processing_ms", "awaiting_note", "days", "monitoring_marker"}


def record(n: int, created_at: str, prediction: str = "authentic", processing_ms: int = 12,
           is_mock: bool = True) -> AnalysisRecord:
    return AnalysisRecord(
        request_id=f"req_{n:032x}", created_at=created_at, prediction=prediction, confidence=0.9,
        risk_level={"synthetic": "HIGH", "uncertain": "MEDIUM", "authentic": "LOW"}[prediction],
        model_version="mock", profile_version="preprocessing-v1", policy_version="risk-v1",
        notice="Decision support only.", processing_ms=processing_ms, is_mock=is_mock)


def store(client, *records):
    for r in records:
        client.app.state.repository.add(r)


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr("api.routers.stats.utc_now_iso", lambda: NOW)


def stats(client, **params):
    response = client.get(URL, params=params)
    assert response.status_code == 200
    return response.json()


def day(date, synthetic=0, uncertain=0, authentic=0):
    return {"date": date, "synthetic": synthetic, "uncertain": uncertain, "authentic": authentic,
            "total": synthetic + uncertain + authentic}


def test_empty_store_default_period_is_the_last_7_days_including_today(client, config):
    body = stats(client)
    assert set(body) == KEYS
    assert body == {
        "from": "2026-10-24", "to": "2026-10-30", "time_zone": "UTC", "generated_at": NOW,
        "today": {"date": "2026-10-30", "total": 0}, "total": 0, "counts": ZERO_COUNTS,
        "uncertain_share": None, "median_processing_ms": None, "awaiting_note": {"total": 0, "recent": []},
        "days": [day(f"2026-10-{d}") for d in range(24, 31)],
        "monitoring_marker": {"deferral_on_fit": config.mock_deferral_on_fit, "policy_version": "risk-v1",
                              "mock": True},
    }


def test_empty_store_explicit_period(client):
    body = stats(client, **{"from": "2026-09-01", "to": "2026-09-03"})
    assert (body["from"], body["to"], body["total"]) == ("2026-09-01", "2026-09-03", 0)
    assert body["days"] == [day("2026-09-01"), day("2026-09-02"), day("2026-09-03")]


def test_single_day(client):
    store(client, record(1, "2026-10-29T08:00:00Z", "synthetic", 100), record(2, "2026-10-29T09:00:00Z", "authentic", 300))
    body = stats(client, **{"from": "2026-10-29", "to": "2026-10-29"})
    assert body["days"] == [day("2026-10-29", synthetic=1, authentic=1)]
    assert (body["total"], body["uncertain_share"], body["median_processing_ms"]) == (2, 0.0, 200)


def test_multi_day_period_lists_empty_days_as_zeros_oldest_first(client):
    store(client, record(1, "2026-10-25T08:00:00Z", "synthetic"), record(2, "2026-10-28T08:00:00Z", "uncertain"),
          record(3, "2026-10-28T09:00:00Z", "uncertain"))
    body = stats(client, **{"from": "2026-10-25", "to": "2026-10-29"})
    assert body["days"] == [day("2026-10-25", synthetic=1), day("2026-10-26"), day("2026-10-27"),
                            day("2026-10-28", uncertain=2), day("2026-10-29")]


def test_verdict_mix_counts_and_uncertain_share_rounded_to_4_places(client):
    store(client, record(1, "2026-10-30T08:00:00Z", "synthetic"), record(2, "2026-10-30T08:01:00Z", "uncertain"),
          record(3, "2026-10-30T08:02:00Z", "authentic"))
    body = stats(client)
    assert body["counts"] == {"synthetic": 1, "uncertain": 1, "authentic": 1}
    assert body["uncertain_share"] == 0.3333
    assert body["days"][-1] == day("2026-10-30", 1, 1, 1)


def test_period_bounds_are_inclusive_utc_days(client):
    store(client, record(1, "2026-10-26T23:59:59Z"), record(2, "2026-10-27T00:00:00Z"),
          record(3, "2026-10-28T23:59:59Z"), record(4, "2026-10-29T00:00:00Z"))
    body = stats(client, **{"from": "2026-10-27", "to": "2026-10-28"})
    assert body["total"] == 2
    assert body["days"] == [day("2026-10-27", authentic=1), day("2026-10-28", authentic=1)]


def test_only_from_runs_to_today_and_only_to_starts_6_days_before(client):
    body = stats(client, **{"from": "2026-10-01"})
    assert (body["from"], body["to"], len(body["days"])) == ("2026-10-01", "2026-10-30", 30)
    body = stats(client, to="2026-10-10")
    assert (body["from"], body["to"], len(body["days"])) == ("2026-10-04", "2026-10-10", 7)


def test_today_is_the_current_day_whatever_the_period(client):
    store(client, record(1, "2026-10-30T01:00:00Z", "synthetic"), record(2, "2026-10-30T02:00:00Z"),
          record(3, "2026-10-01T02:00:00Z"))
    body = stats(client, **{"from": "2026-10-01", "to": "2026-10-01"})
    assert body["today"] == {"date": "2026-10-30", "total": 2}
    assert body["total"] == 1


def test_a_future_period_is_allowed_and_empty(client):
    body = stats(client, **{"from": "2026-12-01", "to": "2026-12-02"})
    assert body["total"] == 0 and len(body["days"]) == 2


def test_awaiting_note_six_newest_synthetic_or_uncertain_without_note_in_the_period(client):
    store(client, record(1, "2026-10-20T08:00:00Z", "synthetic"))  # outside the period
    store(client, *[record(10 + n, f"2026-10-29T08:0{n}:00Z", "uncertain" if n % 2 else "synthetic") for n in range(8)])
    store(client, record(30, "2026-10-29T08:07:00Z", "synthetic"))  # same second as n=17, stored later
    store(client, record(40, "2026-10-30T08:00:00Z", "authentic"), record(41, "2026-10-30T09:00:00Z", "synthetic"))
    client.put(f"/api/analyses/req_{41:032x}/note", json={"text": "Called back."})
    body = stats(client)
    awaiting = body["awaiting_note"]
    assert awaiting["total"] == 9
    assert [i["analysis"]["request_id"] for i in awaiting["recent"]] == \
        [f"req_{n:032x}" for n in (30, 17, 16, 15, 14, 13)]
    assert all(i["note"] is None and set(i) == {"analysis", "created_at", "note"} for i in awaiting["recent"])


def test_monitoring_marker_comes_from_the_analyzer(config):
    class RealLike(MockAnalyzer):
        def info(self):
            return dataclasses.replace(super().info(), policy_version="risk-test", mock=False)

    with TestClient(create_app(config, analyzer=RealLike(0.1234))) as client:
        assert stats(client)["monitoring_marker"] == {"deferral_on_fit": 0.1234, "policy_version": "risk-test",
                                                      "mock": False}
    with TestClient(create_app(config, analyzer=MockAnalyzer(0.5))) as client:
        assert stats(client)["monitoring_marker"]["deferral_on_fit"] == 0.5


def test_generated_at_and_today_use_one_real_utc_clock(client, monkeypatch):
    monkeypatch.undo()
    body = stats(client)
    assert body["generated_at"].endswith("Z") and len(body["generated_at"]) == 20
    assert body["today"]["date"] == body["generated_at"][:10] == body["to"]


@pytest.mark.parametrize("params", [
    {"from": "20261030"}, {"from": "2026-02-30"}, {"to": "2026-1-5"}, {"to": "٢٠٢٦-١٠-٣٠"},
    {"from": ""}, {"to": "abc"}, {"from": "true"}, {"to": "12345"}, {"from": "2026-10-30T00:00:00Z"},
    {"from": "2026-11-02", "to": "2026-11-01"},
    {"from": "2025-10-29", "to": "2026-10-30"},  # 367 days
    {"from": "2026-10-31"},  # after today, no to
    {"to": "0001-01-03"},  # would start before 0001-01-01
    {"from": "0001-01-01"},  # runs to today: far over 366 days
    {"from": "0001-01-01", "to": "9999-12-31"},
], ids=str)
def test_bad_queries_are_400_invalid_request(client, params):
    response = client.get(URL, params=params)
    assert response.status_code == 400
    assert response.json() == INVALID


def test_366_days_are_accepted(client):
    body = stats(client, **{"from": "2025-10-30", "to": "2026-10-30"})
    assert len(body["days"]) == 366


def test_unknown_query_parameters_are_ignored(client):
    store(client, record(1, "2026-10-30T08:00:00Z"))
    assert stats(client, verdict="spoof", page=0)["total"] == 1


@pytest.mark.parametrize("bounds", [("2026-10-28", "2026-10-28"), ("2026-10-27", "2026-10-30"),
                                    ("2026-10-01", "2026-10-31"), ("2026-11-01", "2026-11-05")], ids=str)
def test_stats_total_and_counts_equal_history_for_the_same_period(client, bounds):
    store(client, record(1, "2026-10-27T23:59:59Z", "synthetic"), record(2, "2026-10-28T00:00:00Z", "uncertain"),
          record(3, "2026-10-28T12:00:00Z", "authentic"), record(4, "2026-10-30T23:59:59Z", "synthetic"),
          record(5, "2026-10-31T00:00:00Z", "authentic"), record(6, "2026-10-29T08:00:00Z", "uncertain", is_mock=False))
    params = {"from": bounds[0], "to": bounds[1]}
    body = stats(client, **params)
    history = client.get("/api/analyses", params=params).json()
    assert (body["total"], body["counts"]) == (history["total"], history["counts"])
    assert sum(d["total"] for d in body["days"]) == body["total"]


def test_mock_and_real_results_are_both_counted(client):
    store(client, record(1, "2026-10-30T08:00:00Z", is_mock=True), record(2, "2026-10-30T09:00:00Z", is_mock=False))
    assert stats(client)["total"] == 2
    assert client.get("/api/analyses").json()["total"] == 2


def test_a_failed_read_is_500_internal_error_with_null_request_id(config):
    app = create_app(config)

    def broken(*args, **kwargs):
        raise RuntimeError("disk on fire")

    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.repository.stats = broken
        response = client.get(URL)
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error", "message": "An unexpected error occurred.",
                                         "request_id": None}}
    assert "disk on fire" not in response.text


def test_stats_have_no_file_name_ip_note_text_or_stored_only_item_fields(client, caplog):
    caplog.set_level(logging.DEBUG)
    secret_name, secret_note = "caller-secret-name-7731", "secret-note-text-5519"
    for seed in range(6):  # the mock picks verdicts from the file hash: some will await a note
        response = client.post("/api/analyses",
                               files={"file": (f"{secret_name}.wav", audio_bytes(seed=seed), "audio/wav")})
        assert response.status_code == 201
    first = client.get("/api/analyses").json()["items"][0]["analysis"]["request_id"]
    assert client.put(f"/api/analyses/{first}/note", json={"text": secret_note}).status_code == 200
    caplog.clear()
    real_today = datetime.now(timezone.utc).date()  # POST stores with the real clock; the stats clock is fixed
    around = {"from": (real_today - timedelta(days=1)).isoformat(), "to": (real_today + timedelta(days=1)).isoformat()}
    responses = [client.get(URL), client.get(URL, params=around)]
    assert responses[1].json()["total"] == 6
    for response in responses:
        assert response.status_code == 200
        for absent in (secret_name, secret_note, "testclient", "is_mock"):
            assert absent not in response.text
        for item in response.json()["awaiting_note"]["recent"]:
            assert set(item) == {"analysis", "created_at", "note"}
            assert "processing_ms" not in item["analysis"]
    for absent in (secret_name, secret_note, "testclient"):
        assert absent not in caplog.text
