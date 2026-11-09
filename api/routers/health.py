"""GET /api/health (docs/api_design.md §4.8): liveness for the container health check.

Always {"ok": true} while the process is up. No session, no Origin, no database, no analyzer, no
version or count, no log line. Readiness stays in GET /api/status.
"""
from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(ok=True)
