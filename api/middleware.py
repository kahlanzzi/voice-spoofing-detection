"""Request IDs (docs/api_design.md §2): req_ + 32 lower-case hex characters from a random UUID.

Made by the server for every request and returned in the X-Request-ID header. An X-Request-ID sent
by the client is ignored: the browser never makes one. Pure ASGI middleware, so streaming and
exception handling are not changed.
"""
from __future__ import annotations

import uuid

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

HEADER = "X-Request-ID"
STATE_KEY = "request_id"


def new_request_id() -> str:
    return "req_" + uuid.uuid4().hex


def request_id_of(request: Request) -> str | None:
    """The ID the middleware gave this request, or None if it never ran."""
    return getattr(request.state, STATE_KEY, None)


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = new_request_id()
        scope.setdefault("state", {})[STATE_KEY] = request_id
        header = (HEADER.lower().encode("latin-1"), request_id.encode("latin-1"))

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), header]
            await send(message)

        await self.app(scope, receive, send_with_id)
