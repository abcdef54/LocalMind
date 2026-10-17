# Memory configuration and examples

`config.py` defines budgets, `budget.py` estimates request size, and `service.py`
reads history, invokes summary inference and saves coverage. All inference stays
inside the active GenerateReply run, so it shares executor admission and heartbeat.
No separate queue, Redis service or background summary process is introduced.

## Example: no repeated or skipped messages

Suppose messages 1–40 exist and message 41 is the current question. With the default
20-message raw limit and enough budget, summarize messages 1–20, retain 21–40 raw,
and append 41. The saved boundary is message 20's `(created_at, id)`. When the next
question arrives, read only messages after that boundary for further summarization.
Never include messages 1–20 raw alongside their summary.

Ordering compares timestamp and UUID together, including equal timestamps. The
boundary need not refer to a row that still exists; it remains a logical ordering
position. Current/future messages, other conversations and deleted rows are excluded
from summary input. Original history is never truncated or rewritten by memory.

## What is stored

The conversation retains `summary` plus:

- `summary_through_created_at` and `summary_through_message_id`: last whole message
  covered by the summary, saved atomically with its text.
- `summary_version`: increases after each successful batch; a version mismatch
  rejects a stale result instead of overwriting a newer summary.
- `message_count`: updated when accepted user messages and final answers are inserted.
- `unsummarized_message_count`: incremented with message writes and recomputed after
  each summary checkpoint. It is not used to skip rows or choose source ordering.

Migration 0008 preserves legacy summary text but assigns no invented coverage.
The service ignores that text until rebuilding a trusted summary from saved rows.
The migration also rebuilds old counters from live message records.

## Budgets

| Environment variable | Default |
|---|---|
| `MEMORY_CONTEXT_TOKENS` | `DEFAULT_CONTEXT_LENGTH_K * 1024` (normally 32768) |
| `MEMORY_REPLY_TOKENS` | 2048 |
| `MEMORY_TOOL_TOKENS` | 4096 |
| `MEMORY_SUMMARY_TOKENS` | 2048 |
| `MEMORY_RECENT_MESSAGES` | 20 |
| `MEMORY_BATCH_MESSAGES` | 20 |

Set context capacity to match the actual model server; this task does not discover
per-model capabilities. Values must be positive integers and reserves must leave
space for input. Startup validates them before model initialization.

These are conservative admission estimates, not measured model tokens: UTF-8 byte
length plus serialized message framing is used without tokenizer downloads. This
can retain less text than the model's true tokenizer would allow, especially with
Unicode. It avoids relying on the old character-divided-by-four debug estimate.
The reply reserve also sets the generation client max_tokens; explicit per-call
output limits (such as summary calls) are reserved again during request checks.
The personality system instruction, summary, current input and raw history all
contribute to input size. Tool definitions and results are rechecked before every
model invocation; an oversized expanded prompt fails before sending another request.
No automatic truncation silently discards a user question or retrieved source.

Summary inference uses one tool-free model call per bounded batch, capped at 512
output tokens (or the smaller configured summary budget). Its output must be
nonblank and fit the conservative summary estimate, otherwise coverage is unchanged.
A single old message too large for a batch is summarized in Unicode-safe pieces;
only after all pieces succeed is its whole-message boundary committed. A retry may
recompute pieces of an uncommitted message, but never reprocess a committed batch.
Very long imported histories can require several sequential summary calls before
answer generation; each database read remains bounded by batch size.

## Failure and concurrency

Do not hold a database transaction during summary inference. Saving checks the live
conversation, expected summary version and current unexpired, uncancelled run owner.
Lost leases, cancellation or concurrent summary changes reject the late write.
Ordinary summary failure fails the run without an assistant answer; already-saved
batches remain valid. Transport uncertainty retains the execution slot, just like
answer inference. Summary updates can remain saved even if later answer generation
fails; they summarize prior messages independently of that answer.
