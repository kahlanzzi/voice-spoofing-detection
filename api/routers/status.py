"""GET /api/status (docs/api_design.md §4.1).

versions, bundle_manifest_sha256, deferral_on_fit and mock come from the analyzer's info() (§8),
never from constants here. checks.analysis is info().ready; checks.history_and_stats is a read-only
probe of the database (AnalysisRepository.probe). A probe failure is reported in the body, never as
an error: the endpoint answers 200 with state unavailable (§4.1). state comes from derive_state().
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Request

from api.repository import AnalysisRepository
from api.schemas import StatusChecks, StatusResponse, StatusVersions

log = logging.getLogger("api")

router = APIRouter()

Check = Literal["available", "unavailable"]
State = Literal["ready", "degraded", "unavailable"]


def utc_now_iso() -> str:
    """Current UTC time as ISO 8601 with Z, to the second (§2)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def history_and_stats_check(repository: AnalysisRepository) -> Check:
    """Probe the database read-only; never raises. Logs only the exception class (no path, no IP)."""
    try:
        repository.probe()
    except Exception as exc:
        log.warning("status probe: database unavailable (%s)", type(exc).__name__)
        return "unavailable"
    return "available"


def derive_state(analysis: Check, history_and_stats: Check) -> State:
    """PROVISIONAL: "degraded" is open decision 7 (api_design §4.1, HANDOFF §11 item 7).

    ready when both checks are available; unavailable when analysis is unavailable (§4.1). Also
    unavailable when only the database is: records are stored before the 201, so a database
    failure makes every analysis fail too. degraded is not produced until the team defines it.

    Known limit: the probe is read-only, so a database that is readable but not writable passes
    and this says ready while POST /api/analyses answers 500 (see AnalysisRepository.probe)."""
    if analysis == "available" and history_and_stats == "available":
        return "ready"
    return "unavailable"


@router.get("/status", response_model=StatusResponse)
def get_status(request: Request) -> StatusResponse:
    info = request.app.state.analyzer.info()
    analysis: Check = "available" if info.ready else "unavailable"
    history_and_stats = history_and_stats_check(request.app.state.repository)
    return StatusResponse(
        state=derive_state(analysis, history_and_stats),
        checked_at=utc_now_iso(),
        checks=StatusChecks(analysis=analysis, history_and_stats=history_and_stats),
        versions=StatusVersions(model_version=info.model_version, profile_version=info.profile_version,
                                policy_version=info.policy_version),
        bundle_manifest_sha256=info.bundle_manifest_sha256,
        deferral_on_fit=info.deferral_on_fit,
        mock=info.mock,
    )
