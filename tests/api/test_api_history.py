import dataclasses

import pytest

from api.main import create_app
from api.repository import AnalysisRecord
from tests.api.conftest import audio_bytes, signed_in_client

URL = "/api/analyses"
SECTION_11 = {"prediction", "confidence", "risk_level", "model_version", "profile_version",
              "policy_version", "request_id", "notice"}
ZERO_COUNTS = {"synthetic": 0, "uncertain": 0, "authentic": 0}


def record(n: int, created_at: str, prediction: str = "authentic") -> AnalysisRecord:
    return AnalysisRecord(
        request_id=f"req_{n:032x}", created_at=created_at, prediction=prediction, confidence=0.9,
        risk_level={"synthetic": "HIGH", "uncertain": "MEDIUM", "authentic": "LOW"}[prediction],
        model_version="mock", profile_version="preprocessing-v1", policy_version="risk-v1",
        notice="Decision support only.", processing_ms=12, is_mock=True)


def store(client, *records):
    for r in records:
        client.app.state.repository.add(r)


def ids(response):
    return [item["analysis"]["request_id"] for item in response.json()["items"]]


@pytest.fixture
def five(client):
    """Five records on three days; n=5 is the newest."""
    rows = [record(1, "2026-10-28T08:00:00Z", "synthetic"), record(2, "2026-10-29T08:00:00Z", "uncertain"),
            record(3, "2026-10-29T09:00:00Z", "authentic"), record(4, "2026-10-30T23:59:59Z", "synthetic"),
            record(5, "2026-10-31T00:00:00Z", "authentic")]
    store(client, *rows)
    return rows


def test_empty_history(client):
    response = client.get(URL)
    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "page": 1, "page_size": 10, "counts": ZERO_COUNTS}


def test_items_are_stored_analyses_with_the_whole_section_11_object(client, five):
    item = client.get(URL).json()["items"][0]
    assert set(item) == {"analysis", "created_at", "note"}
    assert set(item["analysis"]) == SECTION_11
    assert item == {"analysis": {k: v for k, v in dataclasses.asdict(five[4]).items() if k in SECTION_11},
                    "created_at": "2026-10-31T00:00:00Z", "note": None}


def test_newest_first(client, five):
    assert ids(client.get(URL)) == [r.request_id for r in reversed(five)]


def test_same_second_keeps_store_order_last_stored_first(client):
    store(client, record(9, "2026-10-30T09:00:00Z"), record(3, "2026-10-30T09:00:00Z"),
          record(7, "2026-10-30T09:00:00Z"))
    assert ids(client.get(URL)) == [f"req_{n:032x}" for n in (7, 3, 9)]


def test_paging_including_last_and_past_last_page(client, five):
    newest_first = [r.request_id for r in reversed(five)]
    for page, expected in [(1, newest_first[0:2]), (2, newest_first[2:4]), (3, newest_first[4:]), (4, [])]:
        body = client.get(URL, params={"page": page, "page_size": 2}).json()
        assert [i["analysis"]["request_id"] for i in body["items"]] == expected
        assert (body["total"], body["page"], body["page_size"]) == (5, page, 2)


def test_a_huge_page_is_empty_not_an_error(client, five):
    response = client.get(URL, params={"page": 10**30})
    assert response.status_code == 200
    assert response.json()["items"] == [] and response.json()["total"] == 5


def test_page_size_max_is_accepted(client, config, five):
    response = client.get(URL, params={"page_size": config.page_size_max})
    assert response.status_code == 200 and response.json()["page_size"] == config.page_size_max


def test_default_page_size_is_lowered_to_a_smaller_page_size_max(config):
    with signed_in_client(create_app(dataclasses.replace(config, page_size_max=3))) as client:
        store(client, *[record(n, f"2026-10-30T09:00:0{n}Z") for n in range(1, 6)])
        body = client.get(URL).json()
    assert body["page_size"] == 3 and len(body["items"]) == 3


def test_verdict_filter_single_repeated_and_duplicate(client, five):
    assert ids(client.get(URL, params={"verdict": "synthetic"})) == [five[3].request_id, five[0].request_id]
    both = client.get(URL, params=[("verdict", "synthetic"), ("verdict", "uncertain")]).json()
    assert both["total"] == 3
    dup = client.get(URL, params=[("verdict", "uncertain"), ("verdict", "uncertain")]).json()
    assert dup["total"] == 1


def test_counts_ignore_the_verdict_filter_but_follow_days_and_q(client, five):
    body = client.get(URL, params={"verdict": "uncertain"}).json()
    assert body["total"] == 1
    assert body["counts"] == {"synthetic": 2, "uncertain": 1, "authentic": 2}
    body = client.get(URL, params={"verdict": "uncertain", "from": "2026-10-29", "to": "2026-10-29"}).json()
    assert body["counts"] == {"synthetic": 0, "uncertain": 1, "authentic": 1}
    body = client.get(URL, params={"q": f"{1:032x}"[-4:]}).json()  # ...0001: only record 1
    assert body["counts"] == {"synthetic": 1, "uncertain": 0, "authentic": 0}


def test_day_range_is_inclusive_utc_days(client, five):
    assert ids(client.get(URL, params={"from": "2026-10-29", "to": "2026-10-30"})) == \
        [five[3].request_id, five[2].request_id, five[1].request_id]
    assert ids(client.get(URL, params={"from": "2026-10-31"})) == [five[4].request_id]
    assert ids(client.get(URL, params={"to": "2026-10-28"})) == [five[0].request_id]
    assert client.get(URL, params={"from": "2026-11-01"}).json()["total"] == 0


def test_q_is_case_insensitive_substring_of_the_request_id(client):
    store(client, record(0xABCDEF, "2026-10-30T09:00:00Z"), record(0x123456, "2026-10-30T09:00:01Z"))
    assert ids(client.get(URL, params={"q": "CdE"})) == [f"req_{0xABCDEF:032x}"]
    assert client.get(URL, params={"q": "req_"}).json()["total"] == 2
    assert client.get(URL, params={"q": ""}).json()["total"] == 2  # empty q: no filter


@pytest.mark.parametrize("q", ["%", "_", "req%", "zz"])
def test_q_wildcards_are_plain_characters(client, five, q):
    response = client.get(URL, params={"q": q})
    assert response.status_code == 200 and response.json()["total"] == (5 if q == "_" else 0)


@pytest.mark.parametrize("params", [
    {"page": 0}, {"page": -1}, {"page": "abc"},
    {"page_size": 0}, {"page_size": 51}, {"page_size": "abc"},
    {"verdict": "spoof"}, {"verdict": "Synthetic"},
    {"from": "20261030"}, {"from": "2026-02-30"}, {"to": "2026-1-5"}, {"to": "٢٠٢٦-١٠-٣٠"},
    {"from": "2026-11-02", "to": "2026-11-01"},
], ids=str)
def test_bad_queries_are_400_invalid_request(client, params):
    response = client.get(URL, params=params)
    assert response.status_code == 400
    assert response.json() == {"error": {"code": "invalid_request", "message": "The request is not valid.",
                                         "request_id": None}}


def test_unknown_query_parameters_are_ignored(client, five):
    response = client.get(URL, params={"sort": "oldest"})
    assert response.status_code == 200 and response.json()["total"] == 5


def test_history_has_no_file_name_ip_or_stored_only_fields(client):
    secret = "caller-secret-name-7731"
    assert client.post(URL, files={"file": (f"{secret}.wav", audio_bytes(), "audio/wav")}).status_code == 201
    response = client.get(URL)
    assert response.json()["total"] == 1
    for absent in (secret, "testclient", "processing_ms", "is_mock"):
        assert absent not in response.text
