# API v1 (BE-02 / BE-03)

The mounted v1 application validates requests and delegates work. Read a route and then its service rather than reading every schema first.

```text
HTTP /api/v1 request
        ↓
application.py: request ID + standard exception handling
        ↓
v1/router.py selects the feature route
        ↓
v1/schemas.py validates input; dependencies provide DB session
        ↓
Conversation route → services/conversation_history.py
Submission route  → uploads receiver → services/submissions.py
Document route  → services/documents.py → metadata / content / mark as deleted
Run route       → services/runs.py + execution/replay.py → state / events / control
Indexing retry    → indexing/store.py → reset failed files
        ↓
Service commits changes / reads a bounded page
        ↓
Response schema → JSON response
Validation/domain failure → standard error envelope
```

[Configuration, contracts and troubleshooting](REFERENCE.md).
