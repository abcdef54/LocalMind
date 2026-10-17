# API v1 routes

This directory maps HTTP endpoints to backend operations; it does not run document parsing or inference inside submission acceptance.

```text
router.py
   ├─ conversations.py → CRUD + bounded message pages
   ├─ submissions.py → receive multipart → accept send → HTTP 202
   ├─ documents.py → list / metadata / content / delete
   ├─ runs.py → snapshot / result / replay / SSE / cancel / retry
   └─ indexing.py → retry failed indexing → HTTP 202
        ↓
services / indexing store perform DB operations
        ↓
schemas.py builds the public response
```
