"""Give each HTTP request an ID and record its status and elapsed time."""

import logging
from time import perf_counter
from uuid import UUID, uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
logger = logging.getLogger("uvicorn.error")


def normalize_request_id(header_value: str | None) -> str:
    """Return a normalized UUID from the incoming header, or generate a new UUID.

    header_value may be missing or invalid. Accept UUID strings up to 36 characters;
    replace other values rather than raising a validation error. Return a string.
    """
    if header_value and len(header_value) <= 36:
        try:
            return str(UUID(header_value))
        except ValueError:
            pass
    return str(uuid4())


class RequestContextMiddleware:
    """Add X-Request-ID to HTTP request state, response headers and completion logs.

    app is the ASGI application to call next. Forward body chunks as they arrive,
    so uploads and streaming responses are not buffered by this middleware.
    Non-HTTP events pass through unchanged.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Save the ASGI application to call for each request; do not run it yet."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI request and log its duration even if the application fails.

        scope contains request metadata; receive reads incoming events; send writes
        response events. For HTTP, save a request ID in state and wrap send to add its
        header. Return None and let application errors propagate to error middleware.
        """
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
            """Add the request ID to response-start headers and forward every event unchanged
            otherwise.

            message is one ASGI response event. Record the HTTP status when headers start;
            do not read or accumulate body events.
            """
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
