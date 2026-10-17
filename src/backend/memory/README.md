# Conversation memory (BE-11)

Memory is prepared inside the existing AI run, using its reserved execution slot.
It does not depend on which history page the browser has loaded.

```text
GenerateReply loads current input + saved settings
        ↓
ConversationMemory.summarize_old_history_and_build_messages_for_inference() reads the saved summary boundary
        ↓
Keep a recent raw suffix that fits the input budget (at most 20 messages)
        ↓
Older messages after the summary boundary?
   Yes → read a bounded batch → tool-free summary inference
        ↓
   Check run lease + summary version
        ↓
   Commit summary + last covered (created_at, message_id) + remaining count
   Repeat until the older gap is covered
        ↓
Summary + raw messages AFTER that boundary + current question exactly once
        ↓
LocalMindAI.chat() → candidates → tournament → saved answer
```

A failed batch never advances its boundary. Oversized old messages are split;
their boundary advances only after every piece succeeds. Oversized current input
is rejected, never silently truncated. Legacy summaries without a known boundary
are rebuilt from stored messages. Full settings and examples: [REFERENCE.md](REFERENCE.md).
