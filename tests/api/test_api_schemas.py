import pytest
from pydantic import ValidationError

from api.schemas import AnalysisResponse, Error, NoteRequest, NoteResponse, StoredAnalysis

VALID = {
    "prediction": "uncertain",
    "confidence": 0.62,
    "risk_level": "MEDIUM",
    "model_version": "mock",
    "profile_version": "preprocessing-v1",
    "policy_version": "risk-v1",
    "request_id": "req_8fa98c02e4b14d0c9a1f3b7d6e5c2a10",
    "notice": "Decision support only.",
}


def test_valid_analysis_response_has_exactly_the_8_fields():
    assert AnalysisResponse(**VALID).model_dump() == VALID
    assert len(AnalysisResponse.model_fields) == 8


def test_ninth_field_is_rejected():
    with pytest.raises(ValidationError):
        AnalysisResponse(**VALID, processing_ms=840)


@pytest.mark.parametrize("field", sorted(VALID))
def test_missing_field_is_rejected(field):
    data = {k: v for k, v in VALID.items() if k != field}
    with pytest.raises(ValidationError):
        AnalysisResponse(**data)


@pytest.mark.parametrize("field, value", [
    ("prediction", "spoof"),
    ("risk_level", "low"),
    ("confidence", 1.5),
    ("confidence", -0.1),
    ("request_id", "req_8fa98c02"),
    ("request_id", "req_8FA98C02E4B14D0C9A1F3B7D6E5C2A10"),
])
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        AnalysisResponse(**{**VALID, field: value})


def test_stored_analysis_keeps_the_section_11_object_whole():
    stored = StoredAnalysis(analysis=VALID, created_at="2026-10-30T09:14:05Z", note=None)
    assert stored.model_dump() == {"analysis": VALID, "created_at": "2026-10-30T09:14:05Z", "note": None}
    with pytest.raises(ValidationError):
        StoredAnalysis(analysis=VALID, created_at="2026-10-30T09:14:05Z", note=None, processing_ms=1)


def test_error_shape_allows_null_request_id_and_nothing_else():
    body = {"error": {"code": "not_found", "message": "Not found.", "request_id": None}}
    assert Error(**body).model_dump() == body
    with pytest.raises(ValidationError):
        Error(error={**body["error"], "prediction": "synthetic"})


def test_note_request_is_exactly_one_string_text():
    assert NoteRequest.model_validate_json('{"text": "ok"}').text == "ok"


@pytest.mark.parametrize("body", ['{}', '{"text": 5}', '{"text": null}', '{"text": ["a"]}',
                                  '{"text": "ok", "author": "x"}', '"ok"', 'not json'])
def test_note_request_refuses_anything_else(body):
    with pytest.raises(ValidationError):
        NoteRequest.model_validate_json(body)


def test_note_response_allows_a_null_note():
    assert NoteResponse(note=None).model_dump() == {"note": None}
