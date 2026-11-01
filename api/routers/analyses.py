"""POST /api/analyses (docs/api_design.md §4.3).

The request ID is assigned first (by the middleware), so every error carries it. A verdict is
returned only after its record is stored; on any failure nothing is stored and the error has no
prediction, confidence or risk_level (NFR-06). processing_ms runs from the handler's start (the
handler reads the body itself) to the result being ready.
"""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from api.analyzer import AnalyzerUnavailable
from api.errors import INTERNAL, ApiError
from api.intake import received_upload
from api.middleware import request_id_of
from api.repository import AnalysisRecord
from api.routers.status import utc_now_iso
from api.schemas import AnalysisResponse

log = logging.getLogger("api")

router = APIRouter()

UNAVAILABLE = (503, "analysis_unavailable", "The analysis service is not available.")
TIMEOUT = (504, "analysis_timeout", "The analysis did not finish in time.")


@router.post("/analyses", status_code=201, response_model=AnalysisResponse)
async def create_analysis(request: Request) -> JSONResponse:
    started = time.perf_counter()
    request_id = request_id_of(request)
    config = request.app.state.config
    analyzer = request.app.state.analyzer
    internal = ApiError(500, *INTERNAL, request_id=request_id)

    async with received_upload(request, config, request_id) as path:
        info = analyzer.info()
        if not info.ready:
            raise ApiError(*UNAVAILABLE, request_id=request_id)
        try:
            # The thread cannot be stopped: on timeout we stop waiting and nothing stores its result.
            outcome = await asyncio.wait_for(asyncio.to_thread(analyzer.analyze, path), config.analysis_timeout_s)
        except TimeoutError:
            raise ApiError(*TIMEOUT, request_id=request_id) from None
        except AnalyzerUnavailable:
            raise ApiError(*UNAVAILABLE, request_id=request_id) from None
        except Exception:
            log.exception("analyzer failed, request_id=%s", request_id)
            raise internal from None

    try:
        response = AnalysisResponse(
            prediction=outcome.prediction,
            confidence=outcome.confidence,
            risk_level=outcome.risk_level,
            model_version=outcome.model_version,
            profile_version=outcome.profile_version,
            policy_version=outcome.policy_version,
            request_id=request_id,
            notice=config.notice_text,
        )
    except ValueError:
        log.exception("analyzer outcome is not a valid §11 response, request_id=%s", request_id)
        raise internal from None
    processing_ms = round((time.perf_counter() - started) * 1000)

    record = AnalysisRecord(**response.model_dump(), created_at=utc_now_iso(),
                            processing_ms=processing_ms, is_mock=info.mock)
    try:
        await run_in_threadpool(request.app.state.repository.add, record)
    except Exception:
        log.exception("analysis record not stored, request_id=%s", request_id)
        raise internal from None

    return JSONResponse(response.model_dump(), status_code=201,
                        headers={"Location": f"/api/analyses/{request_id}"})
