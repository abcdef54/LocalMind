# API v1 foundation (BE-02)

API v1 is mounted as a separate FastAPI application at `/api/v1` so its exception
handlers and OpenAPI contract do not change legacy response bodies.

- Legacy routes and documentation remain at their existing paths, including
  `/chat`, `/conversation`, `/uploads`, `/models`, and `/docs`.
- V1 documentation: `/api/v1/docs`; schema: `/api/v1/openapi.json`.
- No conversation, submission, run or inference endpoints are implemented in v1
  yet. The schema publishes shared `ChatSettings` and `ErrorResponse` components
  even while the v1 paths collection is empty.
- Add completed feature routers in `v1/router.py`. Feature endpoints use the
  existing main application lifespan/database resources; do not add a second
  AI/database startup lifecycle to the mounted application.

The implementation follows FastAPI's [mounted application documentation](https://fastapi.tiangolo.com/advanced/sub-applications/)
and [custom error handlers](https://fastapi.tiangolo.com/tutorial/handling-errors/).

## ChatSettings contract

Use `src.backend.api.v1.schemas.ChatSettings` for v1 input models. The legacy
`database.schemas.ChatRequest` is intentionally unchanged.

| Field | Validation | Default |
|---|---|---|
| `model_name` | Required string, trimmed, 1–255 characters | None |
| `personality` | `general`, `coder`, `researcher`, `genz` | `general` |
| `reasoning_mode` | `low`, `medium`, `high`, `extra` | `low` |
| `temperature` | Finite JSON number in [0, 2]; no strings or booleans | `0.0` |
| `tools_enabled` | JSON boolean, no string/integer coercion | `true` |
| `ranking_group_size` | Strict JSON integer in [2, 7] | `3` |

Unknown fields are rejected. Group size is independent of the mode's candidate
count, including when group size exceeds that count. Registry availability and
whether a model can serve a run require later service checks; schema validation
alone does not load or verify a model.

Group size is a declared v1 contract at this stage. The legacy LangGraph still
uses its original fixed group size; BE-12 implements the configurable algorithm.
This foundation does not advertise working configurable inference or streaming.
The initial schema cap is 7; capability/configuration delivery belongs to BE-16.

## Error contract

Every handled v1 error has this structure:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "request_id": "b416e560-6ef4-4591-ab08-57894641e321",
    "details": {
      "violations": [
        {
          "location": ["body", "ranking_group_size"],
          "code": "greater_than_equal",
          "message": "Input should be greater than or equal to 2"
        }
      ]
    }
  }
}
```

Validation details omit Pydantic's raw input/body and exception context. Invalid
JSON and non-finite numeric values receive the same envelope. Unexpected errors
and HTTP errors with status >= 500 return generic messages, never raw exception
strings. A run that later fails after acceptance is a different lifecycle concern
and must not be represented as an HTTP validation error.

New services can raise `ApiError(status_code=..., code=..., message=..., details=...)`
for an intentional public error. Details must be JSON-safe and contain only
information suitable for the client. Use domain-specific codes such as
`RUN_ACTIVE` or `MODEL_UNAVAILABLE` where applicable.

HTTP errors preserve protocol headers such as `Allow`, `Retry-After` and
`WWW-Authenticate`. Registered HTTP/validation handlers preserve appropriate
status codes; other exceptions become `500 INTERNAL_SERVER_ERROR`.

## Request correlation

The root pure-ASGI middleware handles both v1 and legacy HTTP requests:

- Generates a UUID when no usable `X-Request-ID` is supplied.
- Accepts a client UUID up to 36 characters, normalizes it, and replaces other
  values. This ID is for diagnostics, not authentication or idempotency.
- Stores it in `request.state.request_id` and returns `X-Request-ID` on responses
  passing through the middleware. V1 error handlers also set it explicitly so
  unhandled 500 responses have the same ID in the header and error body.
- Writes request ID, method, path, status and duration through the existing
  `uvicorn.error` logger. It does not read or log request bodies or query strings.
- Logs unexpected v1 exception types with the request ID, without printing raw
  exception strings, SQL parameters or user content.
- Exposes `X-Request-ID` through the root CORS configuration for browser clients.

Legacy HTTP/validation errors retain the old `detail` envelope. The foundation
does not rewrite legacy error handling or claim to sanitize old route debug logs.
The middleware forwards ASGI response messages without buffering the body;
actual SSE/event transport will be implemented in BE-14.

## Tests

```bash
python -m unittest discover -s src/backend/api/tests -v
```

No DB/GPU/model download is needed. Test-only routes exercise the real installed
middleware and handlers; they are never included in the production router. A
production-wiring test calls the existing conversation route with its service
dependency overridden, without entering the application startup lifespan.

The test client uses HTTPX, already available in the current AI dependency stack.
