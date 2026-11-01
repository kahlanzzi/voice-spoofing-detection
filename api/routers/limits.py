"""GET /api/limits (docs/api_design.md §4.2).

All four values come from configs/api.yaml (§9), never from constants here. The browser runs its
pre-upload checks with these values only (NFR-08); the API repeats the same checks on upload.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas import LimitsResponse

router = APIRouter()


@router.get("/limits", response_model=LimitsResponse)
def get_limits(request: Request) -> LimitsResponse:
    config = request.app.state.config
    return LimitsResponse(formats=list(config.formats), max_size_bytes=config.max_size_bytes,
                          max_duration_s=config.max_duration_s, audio_deletion=config.audio_deletion)
