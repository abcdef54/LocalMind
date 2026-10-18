# Remaining legacy routes

Conversation reads/metadata remain during the frontend overhaul. Model APIs use v1.

```text
/conversation → conversations.py → ConversationService → database CRUD
/api/v1/models + /model-downloads → model_management/ → registry / worker / llama-swap

Old POST /chat → removed; cannot bypass executor admission
        ↓
Use POST /api/v1/conversations/{id}/submissions instead
        ↓
HTTP 202 → indexing if needed → queued run → background generation
        ↓
Read saved answer through the v1 paginated message API
```
