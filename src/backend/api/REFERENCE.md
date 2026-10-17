# Detailed reference

API v1 is mounted as a separate FastAPI application at `/api/v1` so its exception
handlers and OpenAPI contract do not change legacy response bodies.

- Legacy routes and documentation remain at their existing paths, including
  `/chat`, `/conversation`, `/models`, and `/docs`; the old `/uploads` router was removed.
- V1 documentation: `/api/v1/docs`; schema: `/api/v1/openapi.json`.
- Conversation CRUD, bounded history, multipart submissions and indexing retry
  endpoints are implemented. Submission-to-inference integration remains BE-10.
- Add completed feature routers in `v1/router.py`. Feature endpoints use the
  existing main application lifespan/database resources; do not add a second
  AI/database startup lifecycle to the mounted application.

The implementation follows FastAPI's [mounted application documentation](https://fastapi.tiangolo.com/advanced/sub-applications/)
and [custom error handlers](https://fastapi.tiangolo.com/tutorial/handling-errors/).

## Conversations and history

| Method | Endpoint | Response |
|---|---|---|
| POST | `/api/v1/conversations` | 201; JSON body `{ "title": "Research" }` |
| GET | `/api/v1/conversations?limit=50&before=...` | Conversation page, newest first |
| GET | `/api/v1/conversations/{id}` | Conversation metadata or 404 |
| PATCH | `/api/v1/conversations/{id}` | Updated metadata; JSON body `{ "title": "New title" }` |
| DELETE | `/api/v1/conversations/{id}` | 204; archive metadata, or 409 `RUN_ACTIVE` |
| GET | `/api/v1/conversations/{id}/messages?limit=50&before=...` | Newest history page, messages ordered oldest to newest within the page |

Titles are trimmed, required and limited to 255 characters. Unknown body fields
are rejected. Conversation IDs are UUIDs. Missing/archived conversations return
404 `CONVERSATION_NOT_FOUND`; repeated delete returns 404, not a second success.

Both list APIs return `{ "items": [...], "next_cursor": "...", "has_more": true }`.
Use `next_cursor` as the next request's `before` value; when exhausted it is null
and `has_more` is false. Default limit is 50, allowed range is 1–100. A request
queries at most limit + 1 rows, without OFFSET, a total count or a full-history
fetch. Conversation list ordering is creation time, not most recent activity.

Cursors encode `(created_at, id)` with resource/conversation scope. They remain
usable if the anchor message is deleted or newer messages arrive. Malformed or
wrong-scope cursors return 422 `INVALID_CURSOR`; invalid query limits return the
standard validation error. Cursors are opaque pagination tokens, not access
credentials or signed authorization tokens.

Message output includes model/personality/reasoning metadata, nullable
`submission_id`/`run_id`, and scoped citation metadata. Storage paths and file
bytes are not included. Deleted documents have `deleted: true`; cross-conversation
citation links are excluded. Legacy message timestamps are stored as naive UTC
and serialized with a UTC offset. This task does not change that storage type.

DELETE archives the conversation, its messages and attachment metadata in one
transaction. It does not initialize Chroma or remove bytes inline; the background
document cleanup worker removes blobs/vectors and records completion in PostgreSQL.
The legacy deletion endpoint has been removed. BE-08 retrieval checks both live
conversation and ready undeleted document generations, so archived chats are
excluded from search even before disk/vector cleanup. New run creation must
lock/check the conversation; a foreign key alone does not detect archival.

Pagination does not implement a message cache yet. Its cursor/page-size contract
allows future per-page caching, and LLM context is still built server-side rather
than from the browser's currently loaded pages.

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

Foundation tests need no DB/GPU/model download. Test-only routes exercise the real
installed middleware and handlers; they are never included in the production
router. A production-wiring test calls the existing conversation route with its
service dependency overridden, without entering the application startup lifespan.

Conversation integration tests use the same explicitly disposable PostgreSQL
database as migration tests and drop/recreate its schema before each case:

```bash
export LOCALMIND_MIGRATION_TEST_URL='postgresql+asyncpg://postgres:TEST_PASSWORD@127.0.0.1:TEST_PORT/localmind_migration_tests'
python -m unittest discover -s src/backend/api/tests -v
```

Without that variable the database tests skip. Run API/database suites
sequentially when sharing one test database; do not run them concurrently.

The test client uses HTTPX, already available in the current AI dependency stack.

## Submission uploads

The multipart send and client-key lookup contracts are documented in
[the upload guide](../uploads/README.md). Background indexing/scoped retrieval is
implemented in BE-08; AI integration follows in BE-10. User attachments are
separate from citations in message pages. Attachment snapshots include safe
indexing errors and committed chunk counts. See [indexing](../indexing/README.md)
for `POST /conversations/{id}/submissions/{submission_id}/retry-indexing` (202),
recovery and the removal of the old `/uploads`/legacy deletion paths.


## Documents (BE-09)

All endpoints are scoped to a live conversation. Wrong-chat, missing or deleted
files return 404; repeated DELETE returns 204 while the conversation exists.

| Method | Path under `/api/v1/conversations/{id}/documents` | Result |
|---|---|---|
| GET | empty suffix | Newest-first page, `limit=50` (1–100), optional `before` cursor |
| GET | `/{document_id}` | Filename, byte size, indexing state and safe error |
| GET | `/{document_id}/content` | Preview PDF/text; DOCX downloads |
| GET | `/{document_id}/content?download=true` | Download original bytes |
| DELETE | `/{document_id}` | 204 after deleted status is saved; asynchronous physical cleanup |

Preview/download works before indexing finishes. HTML and source files are served
as text/plain, never executable HTML. Responses use no-store, nosniff and sandbox
headers. Content unavailable on disk returns 404 without exposing the path.
A download already opened can finish after deletion; new requests return 404.
Document lists exclude documents marked as deleted; message history retains deleted citation flags.

Deleting an attachment fails its pending/queued run with DOCUMENT_DELETED. It does
not claim an already-running inference has stopped or remove context it already
received. Future retrieval excludes the file; cancellation is a separate feature.
