# Personality instructions

This module converts a selected personality into the system instruction used by the graph.

```text
LocalMindAI.chat(personality=...)
        ↓
instruction.py → get_localmind_system_instruction_with_personality()
        ↓
Read system_instructions.json → select personality text
        ↓
Replace {current_date_str} with the current date
        ↓
Return instruction → graph_builder.py merges/prepends system message
```
