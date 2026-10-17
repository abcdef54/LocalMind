# Existing frontend

This is the current vanilla JavaScript client. React/Vite/Tailwind is the planned replacement, not the current implementation.

```text
index.html loads app.css and app.js
        ↓
app.js → boot() → render() → initializeApp()
        ↓
User selects conversation/settings or sends a message
        ↓
sendMessage() → legacy POST /chat (removed in BE-10)
        ↓
Needs replacement: v1 submission → HTTP 202 → read saved message pages

Attachment controls still call legacy /uploads endpoints
        ↓
Removed in BE-08: use v1 submissions and document APIs instead
```
