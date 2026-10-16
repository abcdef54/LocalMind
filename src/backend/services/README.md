# Application services

Services coordinate business rules across records and storage. New submission/history services and legacy chat/model services currently coexist.

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

Legacy /chat → conversation_services.py → DB context/messages/summary
Legacy /models → model_services.py → model files + registry/config
```
