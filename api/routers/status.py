"""GET /api/status (docs/api_design.md §4.1).

versions, bundle_manifest_sha256, deferral_on_fit and mock come from the analyzer's info() (§8),
never from constants here. state is unavailable when the analyzer is not ready. History and
statistics are always reported available until open decision 7 defines "degraded".
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from api.schemas import StatusChecks, StatusResponse, StatusVersions

router = APIRouter()


def utc_now_iso() -> str:
    """Current UTC time as ISO 8601 with Z, to the second (§2)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@router.get("/status", response_model=StatusResponse)
def get_status(request: Request) -> StatusResponse:
    info = request.app.state.analyzer.info()
    return StatusResponse(
        state="ready" if info.ready else "unavailable",
        checked_at=utc_now_iso(),
        checks=StatusChecks(analysis="available" if info.ready else "unavailable", history_and_stats="available"),
        versions=StatusVersions(model_version=info.model_version, profile_version=info.profile_version,
                                policy_version=info.policy_version),
        bundle_manifest_sha256=info.bundle_manifest_sha256,
        deferral_on_fit=info.deferral_on_fit,
        mock=info.mock,
    )
