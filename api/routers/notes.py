"""PUT /api/analyses/{request_id}/note (docs/api_design.md §4.6).

Save, replace or remove the analyst note, with the signed-in account as its author (the session the
guard put in the request state). The note lives in analysis_note and never changes the
analysis record (FR-11). The body is read and parsed here, not by a FastAPI body parameter, so the
size limit holds while streaming and no error can echo the input. The note text is never logged and
never appears in an error: every 400 is the fixed §6 invalid_request, with request_id null (§6).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from api.errors import HTTP_ERRORS, INTERNAL, ApiError
from api.intake import note_body_limit
from api.middleware import request_id_of
from api.repository import ActiveSession
from api.routers.analyses import REQUEST_ID, invalid_request
from api.routers.status import utc_now_iso
from api.schemas import Note, NoteRequest, NoteResponse
from api.session_guard import STATE_KEY

log = logging.getLogger("api")

router = APIRouter()

JSON_MEDIA_TYPE = "application/json"


async def read_body(request: Request, limit: int) -> bytes:
    """The whole body, refused with 400 as soon as it passes limit bytes."""
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise invalid_request()
    return bytes(body)


@router.put("/analyses/{request_id}/note", response_model=NoteResponse)
async def put_note(request: Request, request_id: str) -> NoteResponse:
    if not REQUEST_ID.fullmatch(request_id):  # not the §2 form: cannot exist, no lookup
        raise ApiError(404, *HTTP_ERRORS[404])
    media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if media_type != JSON_MEDIA_TYPE:
        raise invalid_request()
    config = request.app.state.config
    body = await read_body(request, note_body_limit(config))
    try:
        text = NoteRequest.model_validate_json(body).text.strip()
    except ValidationError:  # its text holds the input: dropped, never logged
        raise invalid_request() from None
    if len(text) > config.note_max_chars:
        raise invalid_request()

    session: ActiveSession = getattr(request.state, STATE_KEY)
    repository = request.app.state.repository
    try:
        if text:
            stored = await run_in_threadpool(repository.save_note, request_id, text, utc_now_iso(),
                                             session.account_id)
            found = stored is not None
        else:
            stored = None
            found = await run_in_threadpool(repository.delete_note, request_id) is not None
    except Exception as exc:
        # Class name only: an exception message could carry the note text.
        log.error("note not saved, request_id=%s, error=%s", request_id_of(request), type(exc).__name__)
        raise ApiError(500, *INTERNAL) from None
    if not found:
        raise ApiError(404, *HTTP_ERRORS[404])
    return NoteResponse(note=None if stored is None else Note(text=stored.text, updated_at=stored.updated_at,
                                                              updated_by=stored.updated_by))
