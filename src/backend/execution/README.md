# Durable AI generation (BE-05 / BE-10)

This app uses a database-backed job queue. Unlike a synchronous chat endpoint,
where the Send request calls inference and waits for the answer, this app's Send
request saves the message and an AI run in PostgreSQL, then returns HTTP 202.
The HTTP endpoint does not call the AI generation handler.

A background worker repeatedly polls the database for eligible runs, reserves one,
and calls the AI handler separately from the original HTTP request. It runs as an
async task inside the backend Python process, not a separate OS process. When no
work or execution slot is available, it waits before checking again. Once the API
has accepted a send, closing the browser does not cancel that saved work.

```text
Synchronous request flow:
Browser → HTTP endpoint → AI inference → answer in the same HTTP response

This application's flow:
Browser → HTTP endpoint → save message/run in PostgreSQL → HTTP 202
                                  ↑
Background worker polls PostgreSQL → reserve run → AI inference → save answer
Browser → message API → read the saved answer (live SSE delivery comes later)
```

When the user submits a new question without files, the API saves the question in
`messages` and creates a new run record in the `runs` table with this format.
This example is shown as JSON; creation/update timestamps are omitted.

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
  "started_at": null,
  "finished_at": null,
  "result_message_id": null,
  "error": null
}
```

The API returns HTTP 202 after saving the send. A background worker inside the
backend Python process checks PostgreSQL about once per second by default. It is
an async task running alongside HTTP handling, not a separate OS process.

The worker looks for eligible runs where:

```text
run.status = queued
submission.status = ready
conversation is not deleted
cancellation is not requested
run has no reservation owner
a concurrency slot is available
```

After reserving one, it changes these fields:

```json
{
  "status": "running",
  "attempt": 1,
  "lease_owner": "<worker-id>:<unique-reservation-token>",
  "lease_expires_at": "<database time + 30 seconds>",
  "started_at": "<database time>"
}
```

It then passes the run's IDs and reservation details to `GenerateReply`, which
loads the linked message and saved settings. Calling `self.handler(claim)` in
`worker.py` invokes `GenerateReply.__call__()` because the registered handler is a
`GenerateReply` instance. The handler calls `LocalMindAI.chat()` to run LangGraph;
actual LLM inference runs in the separate llama-swap server. A heartbeat renews
the reservation while generation continues.

For a send containing files, the run initially has status `pending`. It becomes
`queued` only after all its uploaded files are ready.

When generation finishes, the handler returns the answer and citation IDs to the
executor. `RunStore.complete()` rechecks the reservation and saves the assistant
message, citations, settings, generation metadata and completed run in one
transaction. The frontend retrieves the saved answer through the message API;
it does not arrive through the already-finished Send request. Live SSE delivery
is planned for a later task.

```text
User presses Send
        ↓
POST /api/v1/conversations/{id}/submissions → create_submission()
Save user message + submission + run in PostgreSQL → return HTTP 202
        ↓
Text-only → run queued; files → run pending until all indexing succeeds

app.py startup → executor.start() → schedule RunExecutor._dispatch()
        ↓
RunStore.claim_next() checks PostgreSQL for eligible queued work
No task / no free slot? → wait about 1 second → check again
        ↓ task found
Reserve slot: run running + owner token + lease expiry + attempt
        ↓
create_task(_execute(claim)) → await self.handler(claim)
self.handler is GenerateReply; calling it invokes GenerateReply.__call__()
        ↓
_load_input(): saved settings + summary + latest 20 prior messages + input once
        ↓
LocalMindAI.chat() → LangGraph → invoke_model() → model.ainvoke()
HTTP request to llama-swap → candidates + optional RAG → tournament winner
        ↓
Return RunResult to executor; RunStore.complete() rechecks lease/cancel/citations
        ↓
Commit assistant + citations + settings + metadata + completed run together
        ↓
Frontend GET /api/v1/conversations/{id}/messages → saved answer
Live SSE delivery is planned; the worker does not push into the old Send response
```

[Configuration and failure/recovery details](REFERENCE.md).
