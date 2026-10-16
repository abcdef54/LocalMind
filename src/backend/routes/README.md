# Legacy routes

These routes support the existing client during the overhaul. New conversation/submission contracts live under `api/v1`.

```text
Legacy POST /chat → chat.py
        ↓
dependencies.py supplies ConversationService + LocalMindAI
        ↓
Load summary/recent messages → save user message
        ↓
LocalMindAI → LangGraph → answer + citation IDs
        ↓
Resolve citations → save assistant → possibly update summary
        ↓
Return HTTP response

/conversation → conversations.py → ConversationService
/models       → models.py → ModelService
```
