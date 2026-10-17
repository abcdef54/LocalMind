# Backend reading map

Start with the flow, then open the README for the module you need.

```text
app.py startup → migrations → orphan cleanup → model synchronization
        ↓
Initialize embedding model + Chroma + AI graph
        ↓
Start indexing, document cleanup and run executor workers
        ↓
POST /api/v1/conversations/{id}/submissions → accept send → HTTP 202
        ↓
Files pending → indexing → all ready (text-only sends skip indexing)
        ↓
Run queued → execution/handler.py → memory/ → summary + budgeted context + settings
        ↓
LangGraph → N candidates → optional RAG → tournament with saved group size K
        ↓
Guarded transaction → save assistant + citations + metadata → run completed
        ↓
Frontend → run snapshot → SSE after sequence S → GET result_url
Message history remains paginated independently

Shutdown → stop workers; retain slots if remote compute is uncertain
```
