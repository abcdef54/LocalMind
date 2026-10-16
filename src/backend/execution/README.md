# Durable run executor (BE-05)

This is the durable AI job dispatcher. The execution path below requires a registered handler; production currently has none.

```text
Startup → worker.py → RunExecutor.start()
        ↓
Recover expired execution records
        ↓
Handler registered?
  No → leave queued runs waiting (current production)
  Yes
        ↓
store.py → claim a queued run with a ready submission
Check shared concurrency; set running, owner and lease
        ↓
worker.py invokes handler; heartbeat renews reservation
        ↓
Handler returns RunResult
        ↓
store.py checks ownership, expiry and cancellation
        ↓
Save assistant message + completed status in one transaction

Unknown compute state / lost lease → interrupted; retain slot
Release slot only after compute is confirmed stopped
```

[Configuration, contracts and troubleshooting](REFERENCE.md).
