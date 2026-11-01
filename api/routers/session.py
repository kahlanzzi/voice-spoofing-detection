"""POST, GET and DELETE /api/session (docs/api_design.md §4.8): sign in, who am I, sign out.

SessionGuard (api/session_guard.py) has already checked Origin, and for GET and DELETE the session.
Sign-in checks run in the §4.8 order: locked username (429, the password is not checked), unknown
username or wrong password (401, same body; an unknown username is verified against the dummy hash
so both take the same time), disabled account (403; failures neither counted nor reset), success.

Never logged and never in a response: the password, the typed username of a failed sign-in, the
session ID or its hash. The body is read here, not by a FastAPI body parameter, so the 4 KiB limit
holds while streaming and no validation error can echo the input.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from argon2.exceptions import VerifyMismatchError
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from api.auth import username_hmac
from api.errors import ApiError
from api.intake import SESSION_BODY_LIMIT
from api.repository import ActiveSession, AccountRepository
from api.routers.analyses import invalid_request
from api.routers.notes import JSON_MEDIA_TYPE, read_body
from api.schemas import SessionResponse, SignInRequest
from api.session_guard import (ERRORS, STATE_KEY, STORE_ERRORS, clear_session_cookie, idle_before, iso,
                               new_session_id, session_hash, session_id_of, set_session_cookie, utc_now)

log = logging.getLogger("api")

router = APIRouter()


def sign_in_error(code: str) -> ApiError:
    status, message = ERRORS[code]
    return ApiError(status, code, message)


def _verified(hasher, stored_hash: str, password: str) -> bool:
    try:
        return hasher.verify(stored_hash, password)
    except VerifyMismatchError:
        return False


def sign_in(state, typed_username: str, password: str, old_session_id: str | None) -> tuple[str, SessionResponse]:
    """The §4.8 sign-in checks. Return the new session ID and the response; raise ApiError."""
    config = state.config
    accounts: AccountRepository = state.accounts
    key = username_hmac(state.signin_key, typed_username)
    now = utc_now()
    try:
        if accounts.is_locked(key, iso(now)):
            log.info("sign-in refused: username locked")
            raise sign_in_error("account_locked")
        account = accounts.get_account(typed_username.lower())
        stored_hash = account.password_hash if account is not None else state.dummy_hash
        if not _verified(state.hasher, stored_hash, password) or account is None:
            accounts.record_failure(key, iso(now), iso(now - timedelta(minutes=config.lockout_minutes)),
                                    config.lockout_threshold, iso(now + timedelta(minutes=config.lockout_minutes)))
            log.info("sign-in failed")
            raise sign_in_error("invalid_credentials")
        if account.disabled:
            log.info("sign-in refused: account disabled, account_id=%s", account.account_id)
            raise sign_in_error("access_denied")
        accounts.clear_failures(key)
        if state.hasher.check_needs_rehash(account.password_hash):
            accounts.rehash_password(account.account_id, account.password_hash, state.hasher.hash(password))
        session_id = new_session_id()
        expires_at = iso(now + timedelta(hours=config.session_absolute_hours))
        accounts.start_session(session_hash(session_id), account.account_id, iso(now), expires_at,
                               idle_before(now, config.session_idle_minutes),
                               session_hash(old_session_id) if old_session_id else None)
    except STORE_ERRORS as exc:
        log.error("sign-in store error, error=%s", type(exc).__name__)
        raise sign_in_error("sign_in_unavailable") from None
    log.info("sign-in succeeded, account_id=%s", account.account_id)
    return session_id, SessionResponse(username=account.username, signed_in_at=iso(now), expires_at=expires_at)


@router.post("/session", response_model=SessionResponse)
async def post_session(request: Request) -> JSONResponse:
    media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if media_type != JSON_MEDIA_TYPE:
        raise invalid_request()
    body = await read_body(request, SESSION_BODY_LIMIT)
    try:
        credentials = SignInRequest.model_validate_json(body)
    except ValidationError:  # its text holds the input: dropped, never logged
        raise invalid_request() from None
    if len(credentials.password) > request.app.state.config.password_max_chars:
        raise invalid_request()
    session_id, answer = await run_in_threadpool(sign_in, request.app.state, credentials.username,
                                                 credentials.password, session_id_of(request.headers))
    response = JSONResponse(answer.model_dump())
    set_session_cookie(response, session_id)
    return response


@router.get("/session", response_model=SessionResponse)
def get_session(request: Request) -> SessionResponse:
    session: ActiveSession = getattr(request.state, STATE_KEY)
    return SessionResponse(username=session.username, signed_in_at=session.created_at, expires_at=session.expires_at)


@router.delete("/session", status_code=204)
async def delete_session(request: Request) -> Response:
    session: ActiveSession = getattr(request.state, STATE_KEY)
    try:
        await run_in_threadpool(request.app.state.accounts.end_session, session.session_hash)
    except STORE_ERRORS as exc:
        log.error("sign-out store error, error=%s", type(exc).__name__)
        raise sign_in_error("sign_in_unavailable") from None
    response = Response(status_code=204)
    clear_session_cookie(response)
    return response
