# Detailed reference

Files are accepted only when the client sends a submission. Selecting/dropping
files in the future composer must remain browser memory only. This backend does
not implement that UI yet. BE-08 removed the legacy `/uploads` endpoint; the new UI
must use submissions rather than uploading files on selection.

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
separately from assistant `citations`, including references to documents marked as deleted.

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
rewritten or deleted; the submission endpoint checks legacy names too.
There is no parallel legacy upload endpoint.

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

The submission request does not parse/embed files or call an LLM. The BE-08
background worker advances file/submission readiness and queues the reply only
when every file is ready. BE-10 will register the AI handler; queued runs still
have no production generation handler. See [indexing](../indexing/README.md) for
limits, failure recovery and the retry-indexing endpoint.
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
Linux/WSL/Docker. Committed document deletion is handled by `document_cleanup.py` (BE-09).

## Migration and verification

Revision `0004_submission_attachments` extends the existing attachments table.
Legacy records keep paths/citations and become ready (records already marked as deleted retain that status). New attachments must link to a user message in the same submission and
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


## Deleted document cleanup (BE-09)

The API commits deleted_at/status=deleted before returning. File metadata and
citation links remain. A worker started in app.py polls at one-second intervals,
reading up to 20 deleted-document records due for cleanup per pass. Failed or busy files retry after 30
seconds. PostgreSQL cleanup_attempted_at, cleanup_completed_at and cleanup_error
survive restart; there is no in-memory-only deletion queue or Redis requirement.

Cleanup acquires the same per-document OS lock as indexing. It removes all vector
generations in the scoped collection and matching UUIDs in the old `langchain`
collection, then unlinks the file. Missing bytes are already-cleaned success.
A crash between steps repeats those idempotent operations. Unknown commit results
also retry safely. Only a successful DB commit records completion. Generic error
codes omit paths, content and private exception messages.

New files resolve through storage_key under UPLOAD_ROOT; legacy paths must stay
inside the repository's .uploads directory. Paths escaping through symlinks are
rejected and retained for operator inspection. No unrelated directory is scanned
or deleted. Empty shared batch/lock directories may remain.

Conversation archival queues its documents marked as deleted through the same worker.
Shutdown stops new cleanup and waits for any current native deletion to finish
before releasing its lock; it cannot safely kill a running Chroma thread.
