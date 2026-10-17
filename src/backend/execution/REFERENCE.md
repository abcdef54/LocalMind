# Detailed reference

The backend owns one in-process dispatcher using the existing PostgreSQL `runs`
table. No Redis, worker container or migration is required for this step. The
dispatcher starts/stops in the main FastAPI lifespan after schema initialization.

**Production integration:** app.py registers GenerateReply, which invokes the real
LangGraph pipeline for queued, ready submissions. The old direct POST /chat route
has been removed, so HTTP generation no longer bypasses shared admission.

## Claim and admission

- Claim only `queued` runs whose submission is `ready`, conversation is not
  archived and cancellation flag is false. `pending`/indexing work is untouched.
- Order by `(created_at, id)` and use `FOR UPDATE SKIP LOCKED` for the candidate
  row. A short PostgreSQL transaction-level advisory lock serializes the shared
  count-and-claim admission decision between executor instances.
- Default concurrency is **one**. All participants sharing the same LocalMind DB
  must use the same configured limit. This does not control direct llama-swap
  requests or GPU usage by other applications.
- Claim commits `running`, a fresh per-claim owner token, incremented attempt,
  start time and lease expiry before invoking the handler. No DB session or row
  lock is held throughout generation.
- Lease calculations and comparisons use PostgreSQL `clock_timestamp()` to avoid
  worker-clock disagreement. Owner and attempt are checked on heartbeat/result
  writes. Expired leases cannot be renewed.

The locking choices follow the PostgreSQL 15 documentation on
[row/advisory locks](https://www.postgresql.org/docs/15/explicit-locking.html) and
[SKIP LOCKED](https://www.postgresql.org/docs/15/sql-select.html).

## Handler contract

Provide an async handler when constructing `RunExecutor`. It receives a frozen
`RunClaim` containing IDs, a copied settings snapshot and lease identity; it
returns `RunResult(content=...)`. It must yield to the event loop so heartbeat,
shutdown and other HTTP requests can continue.

The executor publishes one assistant message and marks the run `completed` in a
single transaction, only if the lease remains owned and no cancel is pending.
The handler must **not** independently publish messages or other unguarded
durable results. Publication after lease loss returns false instead of creating
a second answer. Completion saves model/personality/reasoning/temperature settings,
validated citations and compact generation metadata in the same transaction.
Summary watermark/token-budget improvements remain BE-11.

An ordinary handler exception records a generic `HANDLER_FAILED`, clears its
lease and does not retry inference automatically. Ordinary return/exception means
the handler has finished external compute and cleaned up its resources. If that
cannot be guaranteed, raise `ComputeStateUnknown` instead: the run becomes
`interrupted` and continues to occupy its admission slot. Logs/error records do
not include raw exception strings, settings or user content.

## Recovery and shutdown

| Situation | Result |
|---|---|
| API refresh/browser disconnect | Does not remove the queued/running record; HTTP run APIs come later |
| Queued job survives backend restart | Remains queued; a registered handler can claim it after restart |
| Running/cancel-requested lease expired or missing | Mark `interrupted`; retain or assign an unresolved owner |
| Stale handler returns after expiry/interruption | Result write is fenced; no assistant message is published |
| Heartbeat loses DB access/ownership | Stop the handler cooperatively; retain the slot and record interruption when DB permits |
| Graceful shutdown | Stop dispatch, allow in-flight work to drain for the grace period |
| Drain timeout | Record `SHUTDOWN_TIMEOUT`, cancel handlers, retain unresolved slots |
| Handler resists cancellation | Shutdown stops waiting after a bounded cancellation wait; track the task and keep its slot |
| Cancel flag/status observed | Interrupt conservatively; this is not the BE-15 confirmed-cancellation API |

Python task cancellation is not proof that a GPU, thread or native call has
stopped. Retaining the slot prevents new admitted work from overlapping compute
whose state is unknown. It also means the queue can deliberately pause after a
crash until an adapter/operator confirms stoppage. No interrupted inference is
silently put back into the queue.

After verifying external compute has stopped, an adapter/operator can call:

```python
released = await store.acknowledge_interruption(
    run_id,
    expected_owner=recorded_lease_owner,
)
```

This clears only the matching interrupted owner. It does not requeue the run,
rewrite its answer or claim to verify GPU state itself. There is no public HTTP
acknowledgement/retry endpoint in BE-05. Inspect held slots with a read-only query:

```sql
SELECT id, status, lease_owner, lease_expires_at, error
FROM runs
WHERE lease_owner IS NOT NULL
ORDER BY created_at, id;
```

The lease protocol fences DB publication, not arbitrary handler side effects;
exactly-once external computation is not promised. Recovery polls the durable DB
state even if a previous Python task no longer exists. `BackgroundTask` records
for indexing/download/delete jobs will be introduced with their own feature
tasks; this implementation intentionally handles only `Run` records.

## Configuration

Set these optional environment variables (Docker `.env` is already loaded):

| Variable | Default |
|---|---|
| `RUN_EXECUTOR_MAX_CONCURRENCY` | `1` |
| `RUN_EXECUTOR_POLL_SECONDS` | `1` |
| `RUN_EXECUTOR_LEASE_SECONDS` | `30` |
| `RUN_EXECUTOR_HEARTBEAT_SECONDS` | `5` |
| `RUN_EXECUTOR_SHUTDOWN_GRACE_SECONDS` | `10` |
| `RUN_EXECUTOR_DATABASE_TIMEOUT_SECONDS` | `5` |

Values must be finite and positive; concurrency must be a positive integer, and
heartbeat interval plus DB timeout must be shorter than the lease. Invalid
configuration fails before expensive startup initialization. Keep the default
single slot on the current GPU until actual model/context benchmarks justify a
change. A process using different limits cannot provide a common global cap.

No live handler mutation or multi-replica deployment is needed now. The initial
deployment remains one API process; extra executor instances are tested to verify
DB ownership/admission, not to recommend increasing GPU concurrency.

## Tests

Without a test database, configuration/lifespan tests run and DB tests skip:

```bash
python -m unittest discover -s src/backend/execution/tests -v
```

For integration tests, use the explicitly disposable database from the migration
test instructions. **Its public schema is dropped/recreated for each case.**

```bash
export LOCALMIND_MIGRATION_TEST_URL='postgresql+asyncpg://postgres:TEST_PASSWORD@127.0.0.1:TEST_PORT/localmind_migration_tests'
python -m unittest discover -s src/backend/execution/tests -v
```

Run suites sequentially when sharing that database. Tests use event-controlled
fake handlers and real PostgreSQL locking/transactions; they never load a model
or use the GPU. Coverage includes contention, shared admission, SKIP LOCKED,
lease fencing, heartbeat, orphan recovery, result publication, handler failures,
shutdown drain/timeout, cancellation-resistant handlers and FastAPI lifecycle.


## GenerateReply input and output (BE-10)

Start in handler.py: `_load_input()` reads the owned running run, validates its
stored ChatSettings, checks the live chat/ready submission and original user message,
and verifies all submitted files remain ready with an index generation. A
short DB read ends before inference starts. The newest 20 prior messages are
selected by (created_at, id), then reversed to chronological order. Current input
is appended once, with uploaded filenames for file-only sends. Future messages,
other chats and deleted messages are excluded. Existing summary text is prepended;
this step does not regenerate summaries or implement a token-budget memory policy.

`__call__()` passes the saved settings and server-bound conversation UUID to
LocalMindAI.chat(). RunResult contains nonblank text, deduplicated citation UUIDs
and at most 8 KiB of JSON metadata. The production graph returns only candidate
count, winner ID, ranking counts/latency, parse-fallback flag and tournament counts;
it does not persist prompts, tool contents or losing answers. Message history
exposes this generation_metadata field; historical messages return null.

Completion locks the conversation before run/document rows, validates lease and
cancellation again, and keeps only same-chat, ready, undeleted citations. A source
deleted during generation is omitted at final persistence. Foreign/missing
citations cannot be linked. The lease is checked again after writes, before the
transaction can commit; a late attempt rolls back its assistant message.

Each candidate request may call tools. Empty responses and tool-loop exhaustion
are failures, not assistant text. Generation waits for all sibling requests before
raising an ordinary candidate failure. Connection failures, timeouts and HTTP 5xx
model failures become COMPUTE_STATE_UNKNOWN: remote stoppage is unconfirmed, so
the slot remains held until an operator verifies compute stopped and acknowledges
interruption. Model requests have a 120-second timeout and SDK retries disabled;
no application retry regenerates a failed/interrupted answer automatically.

No run/SSE/cancel endpoints are added here. Use the existing submission lookup for
saved IDs and paginated message history for completed answers; live run/event APIs
arrive in later tasks. ranking_group_size remains a saved contract until BE-12;
the current tournament algorithm still groups at most three candidates per match.

## Example queued task record

An AI task is a row in `runs`, shown below as abbreviated JSON. The question is
stored in the linked `messages` row; the run is not a serialized Python function.
Timestamps and optional fields are omitted here for readability.

```json
{
  "id": "30000000-0000-4000-8000-000000000001",
  "conversation_id": "10000000-0000-4000-8000-000000000001",
  "submission_id": "20000000-0000-4000-8000-000000000001",
  "message_id": "40000000-0000-4000-8000-000000000001",
  "status": "queued",
  "settings_snapshot": {
    "model_name": "qwen2.5-3b-instruct-q4_k_m",
    "personality": "general",
    "reasoning_mode": "high",
    "temperature": 0.7,
    "tools_enabled": true,
    "ranking_group_size": 3
  },
  "attempt": 0,
  "cancel_requested": false,
  "lease_owner": null,
  "lease_expires_at": null,
  "result_message_id": null,
  "error": null
}
```

```json
{
  "id": "40000000-0000-4000-8000-000000000001",
  "conversation_id": "10000000-0000-4000-8000-000000000001",
  "submission_id": "20000000-0000-4000-8000-000000000001",
  "role": "user",
  "content": "Explain how tournament-based reasoning works."
}
```

`claim_next()` requires a queued, uncancelled, unowned run, a ready submission,
a live conversation and a free shared execution slot. It changes the run to
running, increments attempt, assigns a unique owner and sets lease expiry to
DB time plus 30 seconds by default. GenerateReply then loads the linked input.
File sends start pending; the indexing worker queues them only after all files
are ready. Repeated checks find persisted rows even after a backend restart;
interrupted inference is not automatically restarted.
