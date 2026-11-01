import dataclasses
import sqlite3

import pytest

from api.db import connect
from api.repository import COLUMNS, AnalysisRecord, AnalysisRepository

SECTION_7 = ["request_id", "created_at", "prediction", "confidence", "risk_level", "model_version",
             "profile_version", "policy_version", "notice", "processing_ms", "is_mock"]

RECORD = AnalysisRecord(
    request_id="req_" + "a" * 32, created_at="2026-10-30T09:14:05Z", prediction="uncertain",
    confidence=0.62, risk_level="MEDIUM", model_version="mock", profile_version="preprocessing-v1",
    policy_version="risk-v1", notice="Decision support only.", processing_ms=840, is_mock=True)


@pytest.fixture
def repo(tmp_path):
    repository = AnalysisRepository(tmp_path / "api.sqlite")
    repository.create_schema()
    return repository


def test_table_has_exactly_the_section_7_columns(repo):
    conn = connect(repo.path)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(analysis_record)")]
    finally:
        conn.close()
    assert columns == SECTION_7 == list(COLUMNS)


def test_create_schema_is_idempotent_and_keeps_rows(repo):
    repo.add(RECORD)
    repo.create_schema()
    assert repo.count() == 1


def test_add_then_get_round_trip(repo):
    repo.add(RECORD)
    assert repo.get(RECORD.request_id) == RECORD
    assert repo.get("req_" + "b" * 32) is None


def test_duplicate_request_id_is_rejected(repo):
    repo.add(RECORD)
    with pytest.raises(sqlite3.IntegrityError):
        repo.add(RECORD)
    assert repo.count() == 1


@pytest.mark.parametrize("change", [
    {"prediction": "spoof"},
    {"risk_level": "low"},
    {"confidence": 1.5},
    {"processing_ms": -1},
], ids=lambda c: next(iter(c)))
def test_invalid_values_are_rejected(repo, change):
    with pytest.raises(sqlite3.IntegrityError):
        repo.add(dataclasses.replace(RECORD, **change))
    assert repo.count() == 0
