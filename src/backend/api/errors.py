"""Return consistent JSON errors for v1 routes without exposing private exception text."""

from http import HTTPStatus
import logging
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from src.backend.api.request_context import REQUEST_ID_HEADER

logger = logging.getLogger("uvicorn.error")

HTTP_ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "UPLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    500: "INTERNAL_SERVER_ERROR",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT",
}


class ErrorDetail(BaseModel):
    """Describe a public API failure with a code, message and request ID.

    details may contain safe field-level information, never file contents, credentials
    or raw database exceptions. The request ID lets the user match an error to logs.
    """

    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Wrap one ErrorDetail under the error key used by every v1 failure response."""

    error: ErrorDetail


class ApiError(Exception):
    """Tell the v1 error handler which safe HTTP error to return.

    status_code sets the HTTP status; code is a stable error identifier; message is
    text the user may see. details and headers are optional response additions.
    Constructing/raising this exception does not send a response by itself. Never
    put credentials, raw SQL errors or uploaded contents in its public fields.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Store the requested status, public code/message, optional details and headers.

        Default details to an empty dictionary. The registered handler serializes these
        values when the exception reaches the v1 application.
        """
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details if details is not None else {}
        self.headers = headers


def error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build a JSON error response and include its request ID in body and headers.

    request supplies the middleware's request ID, or a new UUID is generated if missing.
    status_code, code and message describe the public failure; details and headers
    are copied into the response. Return JSONResponse without logging body content.
    """
    request_id = getattr(request.state, "request_id", None) or str(uuid4())
    payload = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
            details=details if details is not None else {},
        )
    )
    response_headers = dict(headers or {})
    # ServerErrorMiddleware can send a 500 outside the request-ID send wrapper.
    response_headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=response_headers,
    )


async def handle_api_error(request: Request, exception: ApiError) -> JSONResponse:
    """Convert an ApiError into JSON using its public status, details and headers.

    request supplies the request ID. Return the shared error response; do not expose
    any exception fields beyond the explicitly provided public values.
    """
    return error_response(
        request,
        status_code=exception.status_code,
        code=exception.code,
        message=exception.message,
        details=exception.details,
        headers=exception.headers,
    )


async def handle_http_error(request: Request, exception: HTTPException) -> JSONResponse:
    """Convert a framework HTTPException into the v1 JSON error format.

    Preserve its status and headers. Use string details only for statuses below 500;
    for server errors return the standard HTTP phrase instead of private exception text.
    """
    status_code = exception.status_code
    try:
        message = HTTPStatus(status_code).phrase
    except ValueError:
        message = "HTTP request failed"
    if status_code < 500 and isinstance(exception.detail, str):
        message = exception.detail
    return error_response(
        request,
        status_code=status_code,
        code=HTTP_ERROR_CODES.get(status_code, "HTTP_ERROR"),
        message=message,
        headers=exception.headers,
    )


async def handle_validation_error(
    request: Request, exception: RequestValidationError
) -> JSONResponse:
    # Pydantic's raw errors include input/body and exception objects in ctx.
    # Return only field locations and messages, including for malformed JSON/NaN.
    """Return HTTP 422 with the invalid field locations, codes and validation messages.

    request supplies the request ID and exception supplies validation failures.
    Omit raw input and exception context because they can contain submitted text,
    file data or non-serializable objects. Return the shared JSON error format.
    """
    violations = [
        {"location": list(error["loc"]), "code": error["type"], "message": error["msg"]}
        for error in exception.errors()
    ]
    return error_response(
        request,
        status_code=422,
        code="VALIDATION_ERROR",
        message="Request validation failed",
        details={"violations": violations},
    )


async def handle_unexpected_error(request: Request, exception: Exception) -> JSONResponse:
    """Log an unexpected exception's type and return a generic HTTP 500 response.

    Use the request ID for log matching. Do not log the raw exception text or copy
    it into the response, where it could reveal credentials or user contents.
    """
    logger.error(
        "Unhandled API error request_id=%s exception_type=%s",
        getattr(request.state, "request_id", "unknown"),
        type(exception).__name__,
    )
    return error_response(
        request,
        status_code=500,
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred",
    )


def install_error_handlers(app: FastAPI) -> None:
    """Register public, HTTP, validation and unexpected-error handlers on the given app.

    Call with the v1 sub-application so existing root routes retain their old error
    format. Return None; no requests are handled until the application receives them.
    """
    app.add_exception_handler(ApiError, handle_api_error)
    app.add_exception_handler(HTTPException, handle_http_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
