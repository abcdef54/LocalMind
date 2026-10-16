# Existing frontend

This is the current vanilla JavaScript client. React/Vite/Tailwind is the planned replacement, not the current implementation.

```text
index.html loads app.css and app.js
        ↓
app.js → boot() → render() → initializeApp()
        ↓
User selects conversation/settings or sends a message
        ↓
sendMessage() → API wrapper → legacy POST /chat
        ↓
Backend generates/persists reply → client updates messages

Attachment controls still call legacy /uploads endpoints
        ↓
Those endpoints were removed in BE-08: this path needs replacement
```
