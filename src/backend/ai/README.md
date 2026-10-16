# AI modules

The AI package builds replies; document acceptance/indexing is managed by separate backend modules.

```text
Legacy /chat prepares conversation context
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
Caller persists assistant message

llms/ manages GGUF files and llama-swap configuration
system_prompt/ loads personality instructions
```
