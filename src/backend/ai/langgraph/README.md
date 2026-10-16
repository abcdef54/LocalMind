# Candidate generation and ranking

This graph generates candidate replies and selects a winner. Candidate count and ranking group size are different settings.

```text
graph_builder.py → LocalMindAI.chat()
Add system prompt and execution configuration
        ↓
START → setup → generate
        ↓
graph_configs.py determines candidate count from reasoning mode
        ↓
Generate candidates (optional tool-call loops per candidate)
        ↓
tournament_select
   ├─ one candidate → use it directly
   └─ multiple → tournament_rank()
                 Shuffle bracket → group candidates → rank each group
                 Single-candidate groups advance without a ranking call
                 Repeat with winners until one remains
        ↓
END → final_answer + citations returned to caller
```

Group size is currently fixed at 3; configurable groups and visualization events are planned.
