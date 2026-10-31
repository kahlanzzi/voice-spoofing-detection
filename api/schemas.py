"""Response shapes (docs/api_design.md §4.1, §5, §6). Every model forbids extra fields.

AnalysisResponse is exactly the 8 Protocol §11 fields; the web app mirrors it (ADR 0008 §3), so a
field is added or removed only through a contract change.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

REQUEST_ID_PATTERN = r"^req_[0-9a-f]{32}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalysisResponse(_Strict):
    prediction: Literal["synthetic", "uncertain", "authentic"]
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


class StoredAnalysis(_Strict):
    analysis: AnalysisResponse
    created_at: str  # UTC, ISO 8601 with Z
    note: Note | None


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
