# Application services

Services coordinate business rules across records and storage. New submission/history services and remaining legacy conversation/model services coexist.

```text
New Send → submissions.py → accept_submission()
        ↓
Validate manifest + metadata retry key
        ↓
Lock conversation → resolve retry → check active run / filenames
        ↓
Create user message + submission + run + attachments
        ↓
Move uploaded files → commit → return saved IDs

History request → conversation_history.py
        ↓
Validate live conversation → query cursor-bounded rows
        ↓
Build messages with attachments and scoped citations

Run request → runs.py → snapshot / saved result / cooperative cancel / guarded retry
Document request → documents.py → scoped list / content / mark as deleted
Accepted send → execution/handler.py → AI generation outside HTTP
Legacy /conversation → conversation_services.py → conversation reads/metadata
Legacy /models → model_services.py → model files + registry/config
```
