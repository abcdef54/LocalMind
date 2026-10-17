# AI modules

The AI package builds replies; document acceptance/indexing is managed by separate backend modules.

```text
execution/handler.py loads saved context and settings
        ↓
langgraph/graph_builder.py → LocalMindAI.chat()
Add personality instruction + bind conversation UUID
        ↓
LangGraph: setup → generate candidates → tournament_select
        ↓                       ↕ optional tool calls
        │                langchain/tools.py
        │                Document retrieval / web tools
        ↓
Winning answer + citation IDs returned to caller
        ↓
RunStore persists assistant, citations and metadata atomically

llms/ manages GGUF files and llama-swap configuration
system_prompt/ loads personality instructions
```
