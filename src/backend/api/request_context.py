"""Request correlation without buffering request bodies or streaming responses."""

import logging
from time import perf_counter
from uuid import UUID, uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


REQUEST_ID_HEADER = "X-Request-ID"
logger = logging.getLogger("uvicorn.error")


def normalize_request_id(header_value: str | None) -> str:
    """Accept only bounded UUIDs from clients; generate an ID for other values."""
    if header_value and len(header_value) <= 36:
        try:
            return str(UUID(header_value))
        except ValueError:
            pass
    return str(uuid4())


class RequestContextMiddleware:
    """Attach a request ID to HTTP state, response headers and completion logs.

    State is local to each request. This pure ASGI middleware forwards messages
    directly, so future SSE responses do not need a different correlation path.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = normalize_request_id(Headers(scope=scope).get(REQUEST_ID_HEADER))
        scope.setdefault("state", {})["request_id"] = request_id
        method = scope["method"]
        path = scope["path"]
        started_at = perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            logger.info(
                "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
                request_id,
                method,
                path,
                status_code,
                (perf_counter() - started_at) * 1000,
            )
