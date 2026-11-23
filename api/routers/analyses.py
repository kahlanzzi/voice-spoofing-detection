"""POST /api/analyses (§4.3), GET /api/analyses (§4.4) and GET /api/analyses/{request_id} (§4.5)
of docs/api_design.md.

POST:
The request ID is assigned first (by the middleware), so every error carries it. Then one worker of the
analysis pool is taken (D53 (2)), before the body is read: at most K uploads are in the temp folder at
once, which is what the tmpfs sizing relies on. If every worker is busy the answer is 503
analysis_unavailable with Retry-After, and the body is drained unread (no temp file). A verdict is
returned only after its record is stored; on any failure nothing is stored and the error has no
prediction, confidence or risk_level (NFR-06). processing_ms runs from the handler's start (the
handler reads the body itself) to the result being ready.

GET: stored results as StoredAnalysis (§5). processing_ms and is_mock are stored but never returned.
Days in from/to are UTC days (open decision 6; create_app refuses another stats_time_zone). Bad
queries answer 400 invalid_request; unknown or malformed IDs 404 not_found; both with request_id
null (§6).
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from api.analyzer import AnalyzerUnavailable
from api.errors import HTTP_ERRORS, INTERNAL, ApiError
from api.intake import received_upload
from api.middleware import request_id_of
from api.pool import AnalyzerFailed, PoolBusy, PoolUnavailable, WorkerLost
from api.repository import AnalysisRecord, HistoryQuery, StoredNote
from api.routers.status import utc_now_iso
from api.schemas import (REQUEST_ID_PATTERN, AnalysisResponse, HistoryResponse, Note, StoredAnalysis,
                         Verdict, VerdictCounts)

log = logging.getLogger("api")

router = APIRouter()

UNAVAILABLE = (503, "analysis_unavailable", "The analysis service is not available.")
BUSY = (503, "analysis_unavailable", "The analysis service is busy; try again later.")
TIMEOUT = (504, "analysis_timeout", "The analysis did not finish in time.")

DEFAULT_PAGE_SIZE = 10  # §4.4; lowered to page_size_max if that is smaller
DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
REQUEST_ID = re.compile(REQUEST_ID_PATTERN)


def invalid_request() -> ApiError:
    return ApiError(400, *HTTP_ERRORS[400])


def parse_day(value: str | None) -> date | None:
    """A §4.4 day: exactly YYYY-MM-DD and a real date (fromisoformat alone also takes 20261030)."""
    if value is None:
        return None
    if not DAY.fullmatch(value):
        raise invalid_request()
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise invalid_request() from None


def to_stored(record: AnalysisRecord, note: StoredNote | None) -> StoredAnalysis:
    """The §5 StoredAnalysis: the §11 object rebuilt whole from the record, beside created_at and note."""
    analysis = AnalysisResponse(**{name: getattr(record, name) for name in AnalysisResponse.model_fields})
    return StoredAnalysis(analysis=analysis, created_at=record.created_at,
                          note=None if note is None else Note(text=note.text, updated_at=note.updated_at,
                                                              updated_by=note.updated_by))


async def drain(request: Request) -> None:
    """Read and discard a refused upload's body, in memory chunk by chunk (no temp file), so the connection
    ends cleanly and nginx relays this answer. Errors while reading (client gone, body limit) are ignored:
    the refusal stands."""
    try:
        async for _ in request.stream():
            pass
    except Exception:
        pass


@router.post("/analyses", status_code=201, response_model=AnalysisResponse)
async def create_analysis(request: Request) -> JSONResponse:
    started = time.perf_counter()
    request_id = request_id_of(request)
    config = request.app.state.config
    pool = request.app.state.pool
    internal = ApiError(500, *INTERNAL, request_id=request_id)

    try:
        async with pool.slot() as job:
            async with received_upload(request, config, request_id) as path:
                try:
                    outcome = await pool.run(job, path, config.analysis_timeout_s)
                except TimeoutError:  # the worker was already killed and reaped: nothing holds the file
                    raise ApiError(*TIMEOUT, request_id=request_id) from None
                except AnalyzerUnavailable:
                    raise ApiError(*UNAVAILABLE, request_id=request_id) from None
                except AnalyzerFailed as exc:
                    log.error("analyzer failed, request_id=%s\n%s", request_id, exc.worker_traceback)
                    raise internal from None
                except WorkerLost:
                    log.error("analysis worker lost during the job, request_id=%s", request_id)
                    raise internal from None
    except PoolBusy:
        await drain(request)
        log.warning("analysis refused: all %d workers busy, request_id=%s", pool.size, request_id)
        raise ApiError(*BUSY, request_id=request_id,
                       headers={"Retry-After": str(config.analysis_retry_after_s)}) from None
    except PoolUnavailable:
        await drain(request)
        raise ApiError(*UNAVAILABLE, request_id=request_id) from None

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
                            processing_ms=processing_ms, is_mock=pool.info().mock)
    try:
        await run_in_threadpool(request.app.state.repository.add, record)
    except Exception:
        log.exception("analysis record not stored, request_id=%s", request_id)
        raise internal from None

    return JSONResponse(response.model_dump(), status_code=201,
                        headers={"Location": f"/api/analyses/{request_id}"})


@router.get("/analyses", response_model=HistoryResponse)
def list_analyses(
    request: Request,
    verdict: Annotated[list[Verdict] | None, Query()] = None,
    day_from: Annotated[str | None, Query(alias="from")] = None,
    to: str | None = None,
    q: str | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int | None, Query(ge=1)] = None,
) -> HistoryResponse:
    config = request.app.state.config
    if page_size is None:
        page_size = min(DEFAULT_PAGE_SIZE, config.page_size_max)
    elif page_size > config.page_size_max:
        raise invalid_request()
    first, last = parse_day(day_from), parse_day(to)
    if first is not None and last is not None and first > last:
        raise invalid_request()
    query = HistoryQuery(verdicts=frozenset(verdict or ()),
                         day_from=None if first is None else first.isoformat(),
                         day_to=None if last is None else last.isoformat(), q=q or None)
    result = request.app.state.repository.history(query, limit=page_size, offset=(page - 1) * page_size)
    return HistoryResponse(items=[to_stored(record, note) for record, note in result.items],
                           total=result.total, page=page, page_size=page_size,
                           counts=VerdictCounts(**result.counts))


@router.get("/analyses/{request_id}", response_model=StoredAnalysis)
def get_analysis(request: Request, request_id: str) -> StoredAnalysis:
    not_found = ApiError(404, *HTTP_ERRORS[404])
    if not REQUEST_ID.fullmatch(request_id):  # not the §2 form: cannot exist, no lookup
        raise not_found
    stored = request.app.state.repository.get_stored(request_id)
    if stored is None:
        raise not_found
    return to_stored(*stored)
