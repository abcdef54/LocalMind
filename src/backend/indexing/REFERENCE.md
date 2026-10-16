# Detailed reference

Files saved by the submission API are indexed after HTTP acceptance. The backend
starts `IndexingWorker` alongside the AI run executor. No Redis, extra worker
container or new embedding dependency is required. Indexing uses CPU embeddings;
LLM execution remains separate and is connected to submissions in BE-10.

## Reading the code

- `store.py` reserves a pending attachment, renews its deadline and commits
  ready/failed states. It queues the reply only after every attachment is ready.
- `worker.py` runs parsing and embedding outside DB transactions, keeps its lease
  alive and handles shutdown/recovery. A file lock prevents two processes from
  indexing the same document while an old native write is still finishing.
- `parser_process.py` extracts PDF/DOCX/UTF-8 text in a separate process.
  `parsing.py` launches, times out and cancels that process.
- `vectors.py` splits text and upserts attempt-specific chunks in Chroma.
- `retrieval.py` reads ready document generations from PostgreSQL before and
  after vector search. In `resources.py`, `get_vector_embedding_model()` loads the
  shared CPU MiniLM model; `get_document_vector_index()` opens Chroma using it.

Each argument, return state and storage effect is documented in these modules.

## File and send lifecycle

```text
Send accepted:       file pending, submission accepted, run pending
Worker reserves:     file indexing, submission indexing
Every file succeeds: files ready, submission ready, run queued
Any file fails:      file failed, submission failed, run failed
```

Other pending files can finish even after a sibling failed. The run cannot start
with a partially indexed batch. A failure stores a safe error code/message, never
raw document contents, library errors or filesystem paths. Document snapshots in
submission lookup and message history include `error` and `chunk_count`.

A scanned/blank PDF with no extractable text fails `EMPTY_DOCUMENT`; this task does
not add OCR. Parser detection uses the original filename, so UUID-only disk names
work. Allowed text/source extensions share UTF-8 parsing; supported code formats
use language-aware splitting. No file-content hashes or content deduplication
are introduced; duplicate upload policy still compares complete filenames.

## Retry API

```http
POST /api/v1/conversations/{conversation_id}/submissions/{submission_id}/retry-indexing
```

Returns **202** with the same submission/message/run/file IDs. Only failed files
return to pending; already-ready files retain their vector generations. No file
bytes are uploaded again. The reply waits again until all files are ready.

- 404: missing/deleted conversation or send outside that conversation.
- 409: send is not a failed indexing operation, another run is active, sibling
  indexing is still in progress, a file was deleted, or cancellation was requested.

Errors follow the v1 envelope. Failed files are not retried automatically; expired
in-progress indexing is recovered automatically because it can safely rebuild its
vectors. A successfully completed AI run is never restarted by document reindexing.

## Partial writes, crashes and shutdown

Each claim receives a new `index_generation`, owner token and increasing attempt.
Chunk IDs are `<attachment_id>:<generation>:<position>`. Repeated batch upserts in
one attempt replace the same IDs. Chroma metadata includes conversation ID,
attachment ID, generation and original filename; it does not include disk paths.

Vectors become retrievable only when PostgreSQL commits that exact generation as
ready. Partial writes and stale workers cannot make a different attempt visible.
A successful attempt deletes older generations; known failures remove their own
partial generation. If cleanup fails, those old vectors stay invisible and can be
removed by a later successful retry or document deletion.

On an uncertain ready commit, keep the vectors: PostgreSQL may have committed even
if the acknowledgement was lost. Retrying/recovery checks DB ownership rather than
deleting a potentially accepted index. A crashed worker's expired indexing lease
returns to pending and a fresh generation rebuilds it.

Admission limits currently leased jobs across workers with the same DB/config.
Workers must share local upload/lock storage on Linux/WSL/Docker. Per-document OS
locks protect against an expired worker's remaining native IO; local task limits
also keep a worker from starting more jobs while a native batch finishes.

Shutdown stops claiming, allows a short drain, kills/waits for a cancelled parser,
and waits for the current embedding/write batch before releasing its file lock.
Repeated cancellation cannot skip an in-flight thread write. Slow native IO can
remain tracked after the shutdown wait; it cannot commit a stale ready generation.
An external hard process kill releases its OS locks; Linux parent-death signalling
also terminates its parser child.

## Conversation isolation

The retriever requires a backend UUID and never has an unfiltered fallback:

1. Select only ready, undeleted attachment generations from a live conversation.
2. Search Chroma with both `conversation_id` and approved attachment:generation keys.
3. Read DB eligibility again, removing matches deleted/replaced during search.

The LangGraph document tool is created with that server-side conversation bound
in a closure. Model arguments contain only `query` and `k_returns`; an extra
conversation ID is rejected. Standard LangChain tool invocation preserves schema
validation and citation artifacts. Legacy `/chat` now supplies its validated chat
UUID too, so there is no remaining global-search route into document contents.
Tool-free summaries do not need a conversation scope.

Run-specific document-version snapshots, final answer persistence and live events
are still BE-10/BE-14 work. This task supplies the indexing/readiness gate and
conversation-scoped tool, not a complete submission-to-answer implementation.

## Limits

All settings can be overridden using `INDEXING_<FIELD_NAME>` environment variables.

| Field | Default |
| --- | --- |
| `MAX_CONCURRENCY` | 1 leased indexing job |
| `POLL_SECONDS` | 1 |
| `LEASE_SECONDS` | 60 |
| `HEARTBEAT_SECONDS` | 5 |
| `DATABASE_TIMEOUT_SECONDS` | 5 |
| `SHUTDOWN_GRACE_SECONDS` | 10 |
| `PARSE_TIMEOUT_SECONDS` | 30 |
| `PARSER_MEMORY_MB` | 512 MiB of parser address space |
| `MAX_PAGES` | 500 PDF pages |
| `MAX_TEXT_CHARACTERS` | 2,000,000 Unicode characters |
| `MAX_EXPANDED_BYTES` | 26,214,400 uncompressed DOCX bytes |
| `MAX_CHUNKS` | 5,000 per document |
| `CHUNK_SIZE` | 1,000 characters |
| `CHUNK_OVERLAP` | 100 characters |
| `EMBEDDING_BATCH_SIZE` | 32 chunks |

Page/text/chunk limits apply after upload acceptance and appear as indexing
failures. Upload byte/count/time limits remain configured separately in the upload
module. Raising upload size alone does not remove extraction limits.

## Migration and removed code

Back up PostgreSQL and `.uploads` together before upgrading.
`0005_attachment_indexing` adds attempt/lease/generation/chunk fields. Existing
undeleted documents become pending for reindexing into `localmind_documents_v1`.
Affected sends wait again and previously queued replies return to pending.
Historical completed/failed/cancelled AI outcomes, original messages, filenames,
citations and disk paths are preserved. No migration invokes models or Chroma.
The previous unscoped Chroma collection stays on disk but is never searched.

Removed the old `LocalMindVectorDB`, `AttachmentService`, `/uploads` routes and
unused attachment CRUD/DTO helpers. Also removed legacy conversation deletion
which bypassed the new run/indexing guards. Use the existing
`DELETE /api/v1/conversations/{id}` for metadata archival; physical document/vector
removal and new document list/preview/download/delete endpoints are BE-09.

**Frontend impact:** old upload-on-drop, old document drawer actions and legacy
chat-deletion calls no longer work. The frontend overhaul must use submissions
and v1 conversation deletion; document drawer APIs follow in BE-09. The old text
chat remains available with scoped retrieval, but v1 submissions do not yet
produce assistant answers until BE-10 registers the AI handler.

## Verification

Tests use real PostgreSQL 15 and Chroma with deterministic fake embeddings,
including real bounded parser subprocesses. No GPU/model download is required.
The DB name must be exactly `localmind_migration_tests`; cases destroy its schema.
Run DB-sharing suites sequentially.

```bash
export LOCALMIND_MIGRATION_TEST_URL='postgresql+asyncpg://postgres:TEST_PASSWORD@127.0.0.1:TEST_PORT/localmind_migration_tests'
python -m unittest discover -s src/backend/indexing/tests -v
python -m unittest discover -s src/backend/database/tests -v
python -m unittest discover -s src/backend/api/tests -v
python -m unittest discover -s src/backend/execution/tests -v
```
