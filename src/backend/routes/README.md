# Remaining legacy routes

Conversation reads/metadata and model management remain during the frontend overhaul.

```text
/conversation → conversations.py → ConversationService → database CRUD
/models       → models.py → ModelService → model files / registry / llama-swap

Old POST /chat → removed; cannot bypass executor admission
        ↓
Use POST /api/v1/conversations/{id}/submissions instead
        ↓
HTTP 202 → indexing if needed → queued run → background generation
        ↓
Read saved answer through the v1 paginated message API
```
