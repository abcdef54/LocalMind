# Model management API and lifecycle

All routes below are under `/api/v1`. Existing `/models` routes were removed; the legacy frontend model screen requires the new UI/API integration.

| Method | Route | Result |
| --- | --- | --- |
| GET | `/models` | Registry metadata only; no disk scan/config writes |
| GET | `/models/{id}` | Public metadata, without `local_path` |
| GET | `/models/catalog?query=Qwen&limit=20` | Hugging Face GGUF search, limit 1..100 |
| GET | `/models/quants?repo_id=owner/repository` | Single/split quantizations |
| POST | `/model-downloads` | 202 task; no download in the request handler |
| GET | `/model-downloads?limit=50` | Recent persisted tasks, limit 1..100 |
| GET | `/model-downloads/{id}` | Progress, result model ID or safe failure code |
| DELETE | `/models/{id}` | 202 deleting metadata; worker performs durable cleanup |
| GET | `/capabilities` | Upload, reasoning, settings, history/event and download limits |

Create/retry example:

```json
{"repo_id":"Qwen/Qwen2.5-3B-Instruct-GGUF","gguf_filename":"qwen2.5-3b-instruct-q4_k_m.gguf"}
```

The same source identifies one task. An active/completed task is returned unchanged;
failed/interrupted tasks are explicitly requeued, preserving their ID and incrementing
`attempt` only when reserved. Removed/missing installations may be downloaded again.
A repeated POST does not refresh an already-ready model to a newer remote revision.
POST is the retry action; there is no automatic retry loop or model-download SSE.
After refresh, list recent tasks and poll the relevant task ID.

Tasks move through `queued → downloading → registering → completed`, or end in
`failed`/`interrupted`. Failed download staging is retained for explicit retry and is excluded from Git.
Byte counts report fully downloaded model parts; they are
not a real-time network transfer meter. A single-part download may remain at zero
until that part finishes. Repository revision and manifest are pinned before download;
each required part must have its expected size and GGUF header before publication.
This validates completeness, not numerical quality, tool support or GPU compatibility.
No whole-file hash pass is added.

Installations have `configuring`, `ready`, `missing` or `deleting` status. Ready means
files were checked and llama-swap configuration was written, not loaded into GPU RAM
or confirmed healthy by a remote inference call. Missing models remain visible.
Root-level complete legacy GGUFs are imported by the worker, including split parts;
partial downloads live in `.downloads/<task-id>` and never enter the root-level scan.
Completed downloads live in `managed/<installation-id>`; unrelated installations
with colliding inference names are rejected rather than overwritten.

One worker polls inside the backend process; downloads run in owned subprocesses.
A shared Linux/WSL `flock` serializes download/publication/config/deletion operations
across backend processes using the same model volume. The child inherits the lock,
and Linux parent-death signaling kills it when its worker process exits. Graceful
shutdown and download timeout kill/reap the child before recording a terminal task
and releasing storage. Recovery marks abandoned downloads interrupted, or completed
when their recorded installation already became ready. It never requeues them alone.
No AI-generation concurrency slot is occupied by downloading.

Deletion protects the default model and any model referenced by a pending/queued/
running/cancel-requested run or an interrupted run with unconfirmed compute ownership.
Submissions/retries share a PostgreSQL transaction lock with the deletion decision.
Known missing/configuring/deleting models are rejected; names managed by an external
inference server without a local registry row retain existing admission behavior.
The worker removes config first, then managed bytes and the registry row; a cleanup
failure keeps deleting metadata for a later retry. Busy storage returns 409
`MODEL_STORAGE_BUSY` instead of waiting behind a long download.

Configuration preserves the existing file inode because Docker currently bind-mounts
that exact file. Names/YAML values/shell paths are quoted, and identical content is
not rewritten. A partial write after a crash is rebuilt by worker reconciliation.
Model/config changes require every writer to use the same shared volume and lock.
Local/manual GGUF modification during inference remains outside this API's control.

Settings: `MODEL_ROOT`, `MODEL_CONFIG_PATH`, `MODEL_POLL_SECONDS` (1),
`MODEL_DOWNLOAD_TIMEOUT_SECONDS` (14400), `MODEL_CATALOG_TIMEOUT_SECONDS` (30),
`MODEL_MAX_QUEUED_DOWNLOADS` (20). Paths should match Docker's model/config mounts.
Backend startup never automatically downloads the default GGUF; capabilities supplies
its suggested source for an explicit download action. Embedding initialization is
still handled by the existing indexing startup and is outside this model-download task.

Verification: 211 backend tests passed using disposable PostgreSQL, real Chroma and
fake download subprocesses; no live Hugging Face model transfer or GPU inference was run.
