# Candidate generation and ranking (BE-12)

Reasoning mode sets candidate count N; saved ranking_group_size sets maximum match
size K (strict integer 2–7, default 3). They are independent settings.

```text
GenerateReply → LocalMindAI.chat(ranking_group_size=K)
        ↓
setup → reasoning mode selects N: low=1, medium=3, high=5, extra=7
        ↓
generate → N independent candidates, optional tool calls, stable candidate IDs
        ↓
tournament_select → tournament_rank(group_size=K)
        ↓
Shuffle bracket once; split survivors into full groups of K
Trailing candidates advance as singleton byes (no model call)
If all survivors fit in one match, rank them together
        ↓
Rank each non-singleton group → advance winners → repeat until one remains
        ↓
Answer + citations + metadata (configured K and largest actual match)

N=5, K=2 → 4 ranking calls
N=5, K=3 → 2 ranking calls (unchanged default)
N=5, K=5 or 7 → 1 ranking call
```

Group size does not regenerate candidates, change prompts or add visualization calls.
BE-13 emits real completion/match/bye/fallback events through an optional progress recorder.
BE-14 persists/streams them; frontend animation remains part of the UI overhaul.
