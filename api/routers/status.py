"""GET /api/status (docs/api_design.md §4.1).

No analyzer yet: the versions, manifest hash and mock flag are fixed mock values here, and
deferral_on_fit is the labelled mock value from config. The analyzer's info() replaces all of
them (§8); the response shape does not change.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from api.schemas import StatusChecks, StatusResponse, StatusVersions

router = APIRouter()

MOCK_VERSIONS = StatusVersions(model_version="mock", profile_version="preprocessing-v1", policy_version="risk-v1")


def utc_now_iso() -> str:
    """Current UTC time as ISO 8601 with Z, to the second (§2)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@router.get("/status", response_model=StatusResponse)
def get_status(request: Request) -> StatusResponse:
    config = request.app.state.config
    return StatusResponse(
        state="ready",
        checked_at=utc_now_iso(),
        checks=StatusChecks(analysis="available", history_and_stats="available"),
        versions=MOCK_VERSIONS,
        bundle_manifest_sha256=None,
        deferral_on_fit=config.mock_deferral_on_fit,
        mock=True,
    )
