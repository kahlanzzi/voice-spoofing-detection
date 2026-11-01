"""Upload intake for POST /api/analyses: docs/api_design.md §4.3 steps 1-5.

Size protection in three layers:
1. ContentLengthGuard rejects a Content-Length above the limit before the body is read.
2. Starlette's RequestBodyLimitMiddleware (same limit) stops a body without Content-Length while
   it streams. It must sit inside the guard: on its own it answers a too-large Content-Length with
   a plain-text 413, which is not the §6 shape.
3. received_upload counts the file bytes while writing the temp file: the exact max_size_bytes cap.
Layers 1 and 2 allow MULTIPART_OVERHEAD_BYTES above max_size_bytes for the multipart framing, so a
file the browser accepts (File.size <= max_size_bytes) is never refused for its envelope.

The original file name is read only for its extension, then dropped: never logged, never stored.
The temp file has a random name and is deleted when the request ends, on success and on every error.
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path, PurePath
from typing import AsyncIterator

import soundfile as sf
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from api.config import ApiConfig, resolve_path
from api.errors import ApiError, error_body
from api.middleware import STATE_KEY

log = logging.getLogger("api")

ANALYSES_PATH = "/api/analyses"
MULTIPART_OVERHEAD_BYTES = 64 * 1024
CHUNK_BYTES = 64 * 1024
DECODE_BLOCK_FRAMES = 65536

# §6 code -> (HTTP status, fixed message).
ERRORS = {
    "invalid_request": (400, "The request must hold one file part named file."),
    "file_too_large": (413, "The file is larger than the size limit."),
    "unsupported_format": (415, "The file format is not accepted."),
    "empty_file": (422, "The recording is empty."),
    "corrupt_audio": (422, "The recording could not be decoded."),
    "recording_too_long": (422, "The recording is longer than the duration limit."),
}


def body_limit(config: ApiConfig) -> int:
    """The whole-body limit for layers 1 and 2."""
    return config.max_size_bytes + MULTIPART_OVERHEAD_BYTES


def upload_error(code: str, request_id: str | None) -> ApiError:
    status, message = ERRORS[code]
    return ApiError(status, code, message, request_id=request_id)


class ContentLengthGuard:
    """Layer 1: a §6 413 for POST /api/analyses when Content-Length is above the limit."""

    def __init__(self, app: ASGIApp, max_body_size: int):
        self.app = app
        self.max_body_size = max_body_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == ANALYSES_PATH:
            value = Headers(scope=scope).get("content-length")
            if value is not None:
                code = None
                try:
                    if int(value) > self.max_body_size:
                        code = "file_too_large"
                except ValueError:
                    code = "invalid_request"
                if code is not None:
                    request_id = scope.get("state", {}).get(STATE_KEY)
                    status, message = ERRORS[code]
                    await JSONResponse(error_body(code, message, request_id), status_code=status)(scope, receive, send)
                    return
        await self.app(scope, receive, send)


class _Undecodable(Exception):
    pass


def decoded_length(path: Path) -> tuple[int, int]:
    """(frames, sample rate) after decoding the whole file with soundfile, in bounded memory."""
    try:
        with sf.SoundFile(path) as f:
            rate = f.samplerate
            frames = sum(len(block) for block in f.blocks(blocksize=DECODE_BLOCK_FRAMES))
    except (sf.SoundFileError, RuntimeError) as exc:
        raise _Undecodable from exc
    if rate <= 0:
        raise _Undecodable
    return frames, rate


@asynccontextmanager
async def received_upload(request: Request, config: ApiConfig, request_id: str | None) -> AsyncIterator[Path]:
    """Run §4.3 steps 1-5 and yield the path of a temp file holding a valid recording.

    Raises ApiError with the §6 code on the first failed check. The temp file is deleted on exit.
    """
    form = None
    temp_path = None
    try:
        try:
            form = await request.form(max_files=1)
        except StarletteHTTPException as exc:  # 413 from layer 2, 400 for malformed multipart
            raise upload_error("file_too_large" if exc.status_code == 413 else "invalid_request", request_id) from None
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise upload_error("invalid_request", request_id)
        extension = PurePath(upload.filename or "").suffix.lower()[1:]
        if extension not in config.formats:
            raise upload_error("unsupported_format", request_id)

        temp_dir = resolve_path(config.temp_dir) if config.temp_dir else None
        fd, name = tempfile.mkstemp(prefix="upload_", suffix="." + extension, dir=temp_dir)
        temp_path = Path(name)
        size = 0
        with os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(CHUNK_BYTES):
                size += len(chunk)
                if size > config.max_size_bytes:
                    raise upload_error("file_too_large", request_id)
                out.write(chunk)
        if size == 0:
            raise upload_error("empty_file", request_id)

        try:
            frames, rate = await run_in_threadpool(decoded_length, temp_path)
        except _Undecodable:
            raise upload_error("corrupt_audio", request_id) from None
        if frames == 0:
            raise upload_error("empty_file", request_id)
        if frames / rate > config.max_duration_s:
            raise upload_error("recording_too_long", request_id)
        yield temp_path
    finally:
        if form is not None:
            await form.close()
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:  # open decision 4: no fallback decided yet; the server log only
                log.error("temp upload not deleted, request_id=%s", request_id)
