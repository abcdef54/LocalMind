# Backend reading map

Start here for the whole backend, then open the README for the module you want to understand. The two request paths below currently coexist.

```text
app.py → lifespan startup
        ↓
Database migrations → orphan upload cleanup → model synchronization
        ↓
Initialize CPU embedding model + Chroma + AI graph
        ↓
Start indexing worker + durable run executor
        ↓
Serve HTTP requests
   ├─ /api/v1 → submission acceptance → pending files
   │                                ↓
   │                         indexing → ready → run queued
   │                                ↓
   │                         AI handler pending BE-10
   └─ legacy /chat → context → LangGraph → persist answer
        ↓
Shutdown → stop workers and handle outstanding work
```
