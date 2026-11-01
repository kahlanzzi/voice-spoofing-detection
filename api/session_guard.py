"""The Origin and session checks (docs/api_design.md §4.8), in middleware: they run before the route
and before any request body is read. Origin is checked first, then the session.

The route table covers the /session routes only (Step 7b-1b); Step 7b-2 extends it to every route
under /api except POST /session and GET /health. A refusal uses the §6 shape with request_id null.

The session ID travels only in the vsd_session cookie; the database keeps its SHA-256 (§7). The ID,
its hash and the Cookie header are never logged.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.requests import cookie_parser
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from api.errors import HTTP_ERRORS, INTERNAL, error_body
from api.repository import AccountRepository

log = logging.getLogger("api")

COOKIE = "vsd_session"
COOKIE_PATH = "/api"
SESSION_PATH = "/api/session"
STATE_KEY = "session"
# (method, path) pairs. Step 7b-2 widens these to every route that changes data / needs a session.
NEEDS_ORIGIN = frozenset({("POST", SESSION_PATH), ("DELETE", SESSION_PATH)})
NEEDS_SESSION = frozenset({("GET", SESSION_PATH), ("HEAD", SESSION_PATH), ("DELETE", SESSION_PATH)})
# The account store cannot be read or written.
STORE_ERRORS = (sqlite3.Error, OSError)

# §6 code -> (HTTP status, fixed message).
ERRORS = {
    "origin_not_allowed": (403, "The request origin is not allowed."),
    "session_expired": (401, HTTP_ERRORS[401][1]),
    "sign_in_unavailable": (503, "The sign-in service is unavailable."),
    "invalid_credentials": (401, "The username or password is not correct."),
    "access_denied": (403, "This account is not allowed to sign in."),
    "account_locked": (429, "This username is temporarily locked."),
}


def utc_now() -> datetime:
    """Current UTC time, to the second (§2)."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def idle_before(now: datetime, idle_minutes: int) -> str:
    """A session last seen before this time has passed the idle limit."""
    return iso(now - timedelta(minutes=idle_minutes))


def new_session_id() -> str:
    """32 random bytes, URL-safe: 43 characters (§4.8 Cookie)."""
    return secrets.token_urlsafe(32)


def session_hash(session_id: str) -> str:
    """SHA-256 hex of a session ID: the only form stored."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def session_id_of(headers: Headers) -> str | None:
    """The vsd_session cookie value, or None."""
    return cookie_parser(headers.get("cookie", "")).get(COOKIE) or None


def set_session_cookie(response: Response, session_id: str) -> None:
    """HttpOnly; Secure; SameSite=Strict; Path=/api; no Domain, no Max-Age (§4.8 Cookie)."""
    response.set_cookie(COOKIE, session_id, path=COOKIE_PATH, secure=True, httponly=True, samesite="strict")


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE, path=COOKIE_PATH, secure=True, httponly=True, samesite="strict")


def guard_error(code: str) -> JSONResponse:
    status, message = ERRORS[code] if code in ERRORS else (500, INTERNAL[1])
    return JSONResponse(error_body(code, message), status_code=status)


class SessionGuard:
    """Refuses a request of NEEDS_ORIGIN whose Origin is not in allowed_origins (403), then a request
    of NEEDS_SESSION without a valid session (401, cookie cleared). A valid session is put in the
    request state (STATE_KEY) and its last_seen_at is updated. A store error answers 503
    sign_in_unavailable on the /session routes and 500 internal_error elsewhere (§4.8)."""

    def __init__(self, app: ASGIApp, accounts: AccountRepository, allowed_origins: tuple[str, ...],
                 idle_minutes: int):
        self.app = app
        self.accounts = accounts
        self.allowed_origins = frozenset(allowed_origins)
        self.idle_minutes = idle_minutes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        route = (scope["method"], scope["path"])
        headers = Headers(scope=scope)
        if route in NEEDS_ORIGIN and headers.get("origin") not in self.allowed_origins:
            await guard_error("origin_not_allowed")(scope, receive, send)
            return
        if route in NEEDS_SESSION:
            session_id = session_id_of(headers)
            session = None
            if session_id is not None:
                now = utc_now()
                try:
                    session = await run_in_threadpool(self.accounts.touch_session, session_hash(session_id),
                                                      iso(now), idle_before(now, self.idle_minutes))
                except STORE_ERRORS as exc:
                    log.error("session check failed, error=%s", type(exc).__name__)
                    code = "sign_in_unavailable" if scope["path"] == SESSION_PATH else INTERNAL[0]
                    await guard_error(code)(scope, receive, send)
                    return
            if session is None:
                response = guard_error("session_expired")
                clear_session_cookie(response)
                await response(scope, receive, send)
                return
            scope.setdefault("state", {})[STATE_KEY] = session
        await self.app(scope, receive, send)
