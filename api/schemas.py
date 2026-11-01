"""Request and response shapes (docs/api_design.md §4.1, §4.2, §4.4, §4.6, §4.7, §4.8, §5, §6). Every model
forbids extra fields.

AnalysisResponse is exactly the 8 Protocol §11 fields; the web app mirrors it (ADR 0008 §3), so a
field is added or removed only through a contract change.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr

REQUEST_ID_PATTERN = r"^req_[0-9a-f]{32}$"

Verdict = Literal["synthetic", "uncertain", "authentic"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalysisResponse(_Strict):
    prediction: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    model_version: str
    profile_version: str
    policy_version: str
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    notice: str


class Note(_Strict):
    text: str
    updated_at: str  # UTC, ISO 8601 with Z
    updated_by: str | None  # the author's username; None for notes saved before sign-in existed


class NoteRequest(_Strict):
    text: StrictStr  # §4.6 body; trimming and the length check are done by the route


class NoteResponse(_Strict):
    note: Note | None


class StoredAnalysis(_Strict):
    analysis: AnalysisResponse
    created_at: str  # UTC, ISO 8601 with Z
    note: Note | None


class VerdictCounts(_Strict):
    synthetic: int
    uncertain: int
    authentic: int


class HistoryResponse(_Strict):
    items: list[StoredAnalysis]
    total: int
    page: int
    page_size: int
    counts: VerdictCounts


class ErrorDetail(_Strict):
    code: str
    message: str
    request_id: str | None


class Error(_Strict):
    error: ErrorDetail


class StatusChecks(_Strict):
    analysis: Literal["available", "unavailable"]
    history_and_stats: Literal["available", "unavailable"]


class StatusVersions(_Strict):
    model_version: str
    profile_version: str
    policy_version: str


class StatusResponse(_Strict):
    state: Literal["ready", "degraded", "unavailable"]
    checked_at: str  # UTC, ISO 8601 with Z
    checks: StatusChecks
    versions: StatusVersions
    bundle_manifest_sha256: str | None
    deferral_on_fit: float
    mock: bool


class StatsToday(_Strict):
    date: str  # YYYY-MM-DD in stats_time_zone
    total: int


class StatsDay(_Strict):
    date: str  # YYYY-MM-DD in stats_time_zone
    synthetic: int
    uncertain: int
    authentic: int
    total: int


class AwaitingNote(_Strict):
    total: int
    recent: list[StoredAnalysis]


class MonitoringMarker(_Strict):
    deferral_on_fit: float
    policy_version: str
    mock: bool


class StatsResponse(_Strict):
    from_: str = Field(alias="from")  # "from" is a Python keyword
    to: str
    time_zone: str
    generated_at: str  # UTC, ISO 8601 with Z
    today: StatsToday
    total: int
    counts: VerdictCounts
    uncertain_share: float | None
    median_processing_ms: int | None
    awaiting_note: AwaitingNote
    days: list[StatsDay]
    monitoring_marker: MonitoringMarker


class LimitsResponse(_Strict):
    formats: list[str]
    max_size_bytes: int
    max_duration_s: int
    audio_deletion: str


class SignInRequest(_Strict):
    # §4.8 body; the password length limit is config, checked by the route. Never trimmed.
    username: StrictStr = Field(min_length=1)
    password: StrictStr = Field(min_length=1)


class SessionResponse(_Strict):
    username: str
    signed_in_at: str
    expires_at: str


class HealthResponse(_Strict):
    ok: Literal[True]
