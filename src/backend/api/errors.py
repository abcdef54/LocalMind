"""Public API v1 errors; legacy FastAPI exception handlers remain unchanged."""

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
    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ApiError(Exception):
    """Represent an intentional client-facing failure with JSON-safe public details.

    Callers must not include credentials, raw database errors or file contents.
    Unexpected exceptions are handled separately with a generic 500 message.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
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
    return error_response(
        request,
        status_code=exception.status_code,
        code=exception.code,
        message=exception.message,
        details=exception.details,
        headers=exception.headers,
    )


async def handle_http_error(request: Request, exception: HTTPException) -> JSONResponse:
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
    """Install only on the v1 sub-application, not the legacy root application."""
    app.add_exception_handler(ApiError, handle_api_error)
    app.add_exception_handler(HTTPException, handle_http_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
