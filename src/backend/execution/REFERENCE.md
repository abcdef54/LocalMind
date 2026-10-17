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
BE-11 prepares bounded context and incremental summaries using the same run slot.

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
| Cancel flag/status observed | Keep renewing; cooperative checkpoints stop new calls, and the executor confirms cancellation only after all siblings finish |

Python task cancellation is not proof that a GPU, thread or native call has
stopped. Retaining the slot prevents new admitted work from overlapping compute
whose state is unknown. It also means the queue can deliberately pause after a
crash until an adapter/operator confirms stoppage. No interrupted inference is
silently put back into the queue; HTTP retry requires an explicit request after stoppage is confirmed.

After verifying external compute has stopped, an adapter/operator can call:

```python
released = await store.release_interrupted_run_slot_after_compute_stop_confirmed(
    run_id,
    expected_owner=recorded_lease_owner,
)
```

This clears only the matching interrupted owner. It does not requeue the run,
rewrite its answer or claim to verify GPU state itself. There is no public HTTP compute-stoppage acknowledgement endpoint. BE-15 exposes
retry but rejects interrupted owners until stoppage has been confirmed separately. Inspect held slots with a read-only query:

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
short DB read ends before inference starts. ConversationMemory then updates older
history summaries using saved timestamp/UUID coverage and a version check. Raw
history is a chronological suffix after that boundary (at most 20 messages),
shrunk further to fit the configured budget; current input is appended once.
Legacy summaries without a coverage boundary are rebuilt, not blindly reused.
See [memory flow and configuration](../memory/README.md).

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

BE-14/15 expose run snapshots, result lookup, event replay/SSE and cancel/retry.
Submission lookup still resolves saved IDs after a lost acceptance response. BE-12 passes saved ranking_group_size into the graph;
the default is three, with strict validation from two through seven.

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

`reserve_next_queued_run_for_worker()` requires a queued, uncancelled, unowned run, a ready submission,
a live conversation and a free shared execution slot. It changes the run to
running, increments attempt, assigns a unique owner and sets lease expiry to
DB time plus 30 seconds by default. GenerateReply then loads the linked input.
File sends start pending; the indexing worker queues them only after all files
are ready. Repeated checks find persisted rows even after a backend restart;
interrupted inference is not automatically restarted.


## Live run events, SSE and lifecycle (BE-13 / BE-14 / BE-15)

The worker writes progress directly to PostgreSQL through RunProgressRecorder. SSE readers
poll that durable log separately; there is no dependency on a browser connection,
no unbounded per-viewer queue and no GPU wait for animation. Progress writes may
briefly serialize on the run row. Failed persistence fails the pipeline rather than
claiming a complete research trace with missing decisions.

### Client workflow

1. Submit and receive the run_id in HTTP 202, or recover it through submission lookup.
2. GET `/api/v1/runs/{run_id}`. Store the returned canonical status, visualization
   and last_event_sequence S.
3. GET `/api/v1/runs/{run_id}/events?after=S`. Apply only sequences greater than the
   last applied sequence. Each SSE ID is `<run UUID>:<sequence>`.
4. EventSource reconnects with Last-Event-ID, which overrides the older `after`
   query parameter left in the original URL. IDs from another run return 422.
5. On `run.completed`, GET the result_url (`/api/v1/runs/{run_id}/result`) to read
   the exact saved assistant answer without scanning history pages. Close the
   EventSource on terminal state; intentional server EOF is not a new inference.
6. A 410 EVENT_HISTORY_EXPIRED before streaming, or stream.reset during streaming,
   means refetch the snapshot and reconnect after its sequence. A future cursor
   returns 409. Do not silently skip missing events or rerun inference for animation.

All run endpoints accept optional `conversation_id` to assert the expected chat;
a mismatch, missing run or archived conversation returns 404. Runs are globally
identified by UUID; this local single-user app does not introduce authentication.
Only public status/settings/IDs are returned, never lease owner tokens or disk paths.

### Ordered event envelope

```json
{
  "id": "30000000-0000-4000-8000-000000000001:12",
  "schema_version": 1,
  "sequence": 12,
  "run_id": "30000000-0000-4000-8000-000000000001",
  "conversation_id": "10000000-0000-4000-8000-000000000001",
  "attempt": 1,
  "type": "group.completed",
  "timestamp": "2026-10-17T12:00:00Z",
  "payload": {
    "round_id": "round-1",
    "round_index": 1,
    "group_id": "round-1-group-1",
    "candidate_ids": ["candidate-4", "candidate-1", "candidate-5"],
    "winner_id": "candidate-1",
    "eliminated_ids": ["candidate-4", "candidate-5"],
    "selection_method": "llm_ranker",
    "parse_failed": false,
    "duration_ms": 350.0
  }
}
```

Lifecycle events: run.pending, run.queued, run.started, run.completed, run.failed,
run.cancel_requested, run.cancelled, run.interrupted, run.compute_stopped and
run.retried. Indexing emits documents.indexing/ready/failed for pending runs.
Generation emits generation.started, candidate.completed and generation.completed.
Selection emits tournament.started, round.started/completed, group.started/completed,
candidate.bye, ranking.fallback and tournament.completed.

Candidate UUID/display index are allocated before generation. Each completion is
emitted immediately after its own model/tool loop finishes; gather still returns
pool entries in original input order. No prompt, sampling call, bracket shuffle
or ranker policy is changed for animation. Group/round IDs are scoped to run and
attempt. Byes make no ranking request. A fallback is explicitly labeled
fallback_first_candidate rather than presented as a confirmed ranker win. Low mode
has no match/round events; its selection completion says ranking_skipped=true.
Durations measure wall-clock orchestration time, not isolated GPU compute.

### Snapshot, retention and resource bounds

Run.event_sequence, Run.visualization and each RunEvent are updated in one
transaction under the run row lock. Concurrent completions cannot duplicate
sequences or overwrite counters. The projection contains phase, stable candidates,
completed count, seeding, rounds/groups, survivors and selected winner. Terminal
failure/cancellation/interruption clears the winner and stops candidate states.
Run status, result ID and visualization are read from one canonical run row.
Current document statuses are additional live metadata, not a frozen file snapshot.

Keep the newest 1000 events per run; sequence never resets. Projection persists
beyond event retention. Retry resets only the current visualization, retaining
old attempt events until retention removes them. Existing runs at migration 0009
start with sequence zero and no invented historical events. Their saved outcomes
remain readable. Event payloads are at most 16 KiB and exclude full candidate
answers, raw tool content, private exception strings and model thoughts.

JSON replay `/event-log?after=0&limit=100` reads at most 100 ascending events.
SSE reads at most 100 at a time, polls every 0.5 seconds when idle, sends a comment
heartbeat every 15 seconds, disables caching/proxy buffering and holds no DB
transaction while waiting or yielding. Slow viewers affect only their own response.
At most 32 SSE connections per API process are accepted; excess gets 429. These
initial bounds are server constants; multi-host pub/sub and configurable limits
can be added in the operations task. DB failures close/reset the stream for resync.

### Cancel, retry and uncertain compute

POST `/api/v1/runs/{run_id}/cancel` returns 202 and the committed snapshot.
Queued/pending unclaimed inference becomes cancelled immediately. For running
inference, record cancel_requested and keep renewing the lease. Check cancellation
before/after each model request and between tool calls; generation waits for all
sibling requests before raising cooperative cancellation. Only then does the
executor commit cancelled and release its slot. A normal handler result that
loses a race with cancel is discarded, never saved as an answer.

This adapter does not have a verified llama-swap GPU-abort protocol. Cancelling
an HTTP request is not proof of remote stoppage. A request still in flight may
wait until its 120-second timeout; transport uncertainty becomes interrupted,
retaining its slot. Shutdown/lease loss remain conservative. Operators can use
RunStore.release_interrupted_run_slot_after_compute_stop_confirmed only AFTER independently verifying compute
stopped; the public API cannot assert this on behalf of the caller.

POST `/api/v1/runs/{run_id}/retry` with `{"expected_attempt": 1}` reuses the saved
input, settings, run/message/file IDs. Permit failed/cancelled/interrupted runs
only with no reserved owner/result and no other active run in the chat. A newer
user question also rejects retry (RUN_INPUT_SUPERSEDED), avoiding reuse of a newer
conversation summary for an older question; submit a new message instead. Stale,
active, completed or compute-unconfirmed requests return 409; booleans and extra
body fields return 422. The next claim increases attempt; sequence continues.
Ready files retain vectors; unready files return pending. Deleted files or active
indexing reject retry. Repeated retry of an already queued/running run cannot
start a duplicate task. No automatic model retry is introduced.
