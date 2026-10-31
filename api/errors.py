"""One error shape for every error (docs/api_design.md §6).

FastAPI's default bodies (validation 422, HTTPException, unknown routes, unhandled 500) are
replaced by {"error": {"code", "message", "request_id"}}. Messages are fixed sentences: no
exception text, stack trace or internal path ever reaches the client, and no error carries a
verdict (NFR-06). request_id is null unless the error says otherwise (§6: only POST /analyses
errors carry the upload's ID); the X-Request-ID header is set on every response by the middleware.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.middleware import HEADER, request_id_of

log = logging.getLogger("api")

# HTTP status -> (§6 code, message) for errors raised by the framework.
HTTP_ERRORS = {
    400: ("invalid_request", "The request is not valid."),
    401: ("session_expired", "No valid session."),
    403: ("access_denied", "This session is not allowed to do this."),
    404: ("not_found", "The requested resource was not found."),
    405: ("invalid_request", "This method is not allowed on this path."),
}
INTERNAL = ("internal_error", "An unexpected error occurred.")


class ApiError(Exception):
    """An error the API raises on purpose, answered with its own status and §6 code."""

    def __init__(self, status: int, code: str, message: str, request_id: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id


def error_body(code: str, message: str, request_id: str | None = None) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(error_body(exc.code, exc.message, exc.request_id), status_code=exc.status)


async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code, message = HTTP_ERRORS.get(exc.status_code, INTERNAL)
    # Keep framework headers such as Allow on 405; never the detail text.
    return JSONResponse(error_body(code, message), status_code=exc.status_code, headers=exc.headers)


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(error_body(*HTTP_ERRORS[400]), status_code=400)


async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    # Runs in Starlette's outermost ServerErrorMiddleware, outside the request-ID middleware,
    # so the header is set here. The traceback goes to the server log only.
    request_id = request_id_of(request)
    log.exception("unhandled error, request_id=%s", request_id)
    headers = {HEADER: request_id} if request_id else None
    return JSONResponse(error_body(*INTERNAL), status_code=500, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled_error)
