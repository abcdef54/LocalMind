# Submission uploads (BE-06 / BE-07)

This module receives file bytes, stores them safely and cleans up deleted documents. It does not parse documents or create embeddings.

```text
Browser selects/drops files (future UI: browser memory only)
        ↓ user presses Send
api/v1/submissions.py → create_submission()
Reserve receiving capacity and create SubmissionFileBatch
        ↓
multipart.py → SubmissionMultipartReceiver
Stream payload + files; enforce byte/count/time limits
        ↓
storage.py stages files under server-generated paths
        ↓
services/submissions.py → accept_submission()
Validate manifest, retries and duplicate filenames; insert DB records
        ↓
Move staged files into permanent blob storage before DB commit
        ↓
Commit succeeds → keep files; return HTTP 202
Commit fails definitely → remove this batch's files
Commit outcome uncertain → keep files for reconciliation
        ↓
Indexing worker processes accepted files separately

Document DELETE → save deleted status in DB → hidden from retrieval
        ↓
document_cleanup.py checks PostgreSQL for documents marked as deleted
        ↓
Acquire indexing lock → remove vectors + bytes → mark cleanup complete
Busy/error → retry later; citation metadata stays
```

[Configuration, contracts and troubleshooting](REFERENCE.md).
