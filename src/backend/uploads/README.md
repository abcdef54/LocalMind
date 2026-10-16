# Submission uploads (BE-06 / BE-07)

Files are accepted only when the client sends a submission. Selecting/dropping
files in the future composer must remain browser memory only. This backend does
not implement that UI yet. The legacy `/uploads` endpoint remains for the old UI.

## Reading the implementation

Start with `api/v1/submissions.py:create_submission`: reserve a receiving slot,
check the conversation, create a temporary batch, receive the body, validate the
JSON, then call the acceptance service. Its `finally` always releases storage.

`services/submissions.py:accept_submission` coordinates the database transaction:
validate the file manifest, compute a metadata fingerprint, lock the conversation,
resolve retries, check active runs/filenames, insert records, move files, commit.
The helpers distinguish validation, response construction and DB writes explicitly.

`uploads/multipart.py:SubmissionMultipartReceiver` handles parser callbacks. Its
class docstring explains their order and the meaning of chunk slice offsets.
`uploads/storage.py:SubmissionFileBatch` owns staging/permanent paths and the final
keep/delete decision. Method docstrings describe inputs, results, errors and file
or database effects, including the uncertain-commit case.

## HTTP contract

`POST /api/v1/conversations/{conversation_id}/submissions` returns **202**.
Use `multipart/form-data` with one `payload` JSON string and repeated `files`
parts, in the same order as `files_manifest`. The payload may appear before or
after the files. Text-only sends use the same endpoint without `files` parts.
At least text or one file is required.

```json
{
  "client_submission_id": "2a1b7cb0-2a71-47e7-a559-a4134ef27ff3",
  "content": "Compare these files",
  "settings": {"model_name": "local-model", "reasoning_mode": "high", "ranking_group_size": 3},
  "files_manifest": [
    {"client_file_id": "local-file-1", "name": "report.pdf", "size_bytes": 12345}
  ]
}
```

An optional `Idempotency-Key` header must equal `client_submission_id`. Generate
one UUID per Send and retain it through retries. The response includes
`submission_id`, `client_submission_id`, `conversation_id`, `message_id`, `run_id`,
`status`, and `documents`. Each document has its ID, client file ID, full filename,
MIME, byte size, status and deleted flag. Storage paths are never returned.

`GET /api/v1/conversations/{conversation_id}/submissions/by-client-id/{client_submission_id}`
looks up a send after a lost response; 404 means no committed submission was found
at the time of that lookup. Retrying with the same key is safe even if the first
request is still in flight. Conversation message pages expose user `attachments`
separately from assistant `citations`, including tombstones after deletion.

## Filename duplicates and retries

- Compare the complete filename exactly, including extension and case, within
  one conversation. `report.py`, `report.cpp`, `report.pdf` can coexist.
- Reject duplicate names in a batch or names already attached and not deleted:
  **409 DUPLICATE_FILE_NAME**, with the offending name. No partial acceptance.
- Identical names in other conversations are allowed. Deletion releases a name.
- No content hashes, content comparisons or shared-blob deduplication.
- Same client key + same validated text/settings/manifest returns original IDs
  before checking filename duplicates or active runs. Changed metadata returns
  **409 IDEMPOTENCY_CONFLICT**. A small metadata fingerprint is stored, not a
  file-content hash: changed bytes with unchanged filename/size/key replay the
  original submission. Use a new key for a genuinely new send.

The service locks the conversation during acceptance, including the retry check,
so concurrent sends cannot create duplicate messages. A partial unique index
also guards filenames for new submissions. Existing legacy duplicates are not
rewritten or deleted; the new endpoint checks legacy names too. The legacy
upload endpoint is not governed by the new submission lock/filename policy.

## Limits and receiving

Default limits (override with the corresponding environment variable):

| Variable | Default |
| --- | --- |
| `UPLOAD_ROOT` | `<repo>/.uploads/attachments` |
| `UPLOAD_MAX_FILE_BYTES` | 26,214,400 (25 MiB) |
| `UPLOAD_MAX_BATCH_BYTES` | 104,857,600 (100 MiB) |
| `UPLOAD_MAX_FILES` | 10 |
| `UPLOAD_MAX_PAYLOAD_BYTES` | 65,536 |
| `UPLOAD_MAX_PARALLEL_UPLOADS` | 2 per API process |
| `UPLOAD_IDLE_TIMEOUT_SECONDS` | 30 |
| `UPLOAD_REQUEST_TIMEOUT_SECONDS` | 120 |
| `UPLOAD_ORPHAN_GRACE_SECONDS` | 3,600 |

Read supported extensions in `config.py`; PDF, TXT, Markdown, DOCX and common
source/text extensions are accepted. Only basenames are accepted; directory
separators and control characters are rejected. MIME is derived from extension,
not trusted client headers. Actual parsing/content validation is BE-08.

The route consumes `Request.stream()` directly using python-multipart callbacks;
it never calls `request.form()` or buffers a whole batch. It checks actual file,
batch, payload and header bytes while receiving, before writing excess bytes.
Content-Length can reject requests early but cannot bypass actual byte counting.
File IO runs off the event loop; in-flight writes finish before cancellation
cleanup. A busy receiver returns 429 with Retry-After rather than queues uploads.
Idle and overall receiving timeouts return 408; incomplete multipart bodies,
manifest mismatches and invalid settings return 422. Unsupported extensions
return 415, over-limit bodies return 413. Errors retain the v1 envelope/request ID.

## Storage and acceptance

1. Receive into `staging/<server-submission-uuid>/<server-attachment-uuid>`.
   Original filenames are metadata only, never disk paths.
2. Validate every file and manifest; require a live conversation and no active
   run. All files must pass before creating a successful send.
3. Insert submission, user message, attachment metadata and run in one DB
   transaction. Atomically rename the batch directory to `blobs/<uuid>` before
   committing. Every file is flushed/fsynced before promotion.
4. Return 202 only after commit. A file batch has submission `accepted`, documents
   `pending`, run `pending`. Text-only has submission `ready`, run `queued`.

No file content is parsed, embedded, indexed or sent to an LLM here. BE-08 advances
file/submission readiness; BE-10 registers the AI handler. Until those tasks,
accepted uploads remain pending and queued text runs have no production handler.
Settings availability against the installed-model registry is checked during AI
integration, not by this endpoint yet.

Before commit, any invalid file/manifest/storage operation rolls back metadata
and removes the entire batch. After uncertain commit failure, keep promoted
blobs: deleting immediately could destroy files whose DB commit succeeded.
Recovery checks the DB and removes only old unreferenced batches. Database and
filesystem are not one physical transaction.

Each live batch holds an OS file lock across receiving/promotion/commit/cleanup.
Reconciliation skips locked batches and all committed submissions. It runs on
backend startup; after the grace period it can also be run without loading AI:

```bash
python -m src.backend.uploads.cleanup
```

The Docker backend already mounts `.uploads`, so no new volume/container is
needed. Use a shared local filesystem for API processes; file locking here targets
Linux/WSL/Docker. Physical deletion of committed documents belongs to BE-09.

## Migration and verification

Revision `0004_submission_attachments` extends the existing attachments table.
Legacy records keep paths/citations and become ready (deleted tombstones become
deleted). New attachments must link to a user message in the same submission and
conversation. Status, error shape, nonnegative size, storage/client identities
and live filename uniqueness are protected by constraints. New links are nullable
only for legacy compatibility. Staging itself has no DB record.

Back up DB and `.uploads` together before migration. Downgrading to 0003 drops
new attachment ownership/status metadata; it is not a way to preserve new upload
history and does not remove physical files. The frozen legacy migration stays
unchanged.

Tests use temporary filesystem directories and a disposable PostgreSQL 15 DB,
without models/GPU. Run suites sequentially:

```bash
export LOCALMIND_MIGRATION_TEST_URL='postgresql+asyncpg://postgres:TEST_PASSWORD@127.0.0.1:TEST_PORT/localmind_migration_tests'
python -m unittest discover -s src/backend/database/tests -v
python -m unittest discover -s src/backend/api/tests -v
python -m unittest discover -s src/backend/execution/tests -v
```
