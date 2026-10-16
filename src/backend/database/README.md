# Database and persistence

PostgreSQL stores application records and processing state. Uploaded bytes live in filesystem storage; document vectors live in Chroma.

```text
Backend startup → session.py → upgrade_database_schema()
        ↓
migration.py → Alembic → migrations/versions/
Apply pending revisions before accepting requests
        ↓
HTTP dependency / worker opens AsyncSession
        ↓
Services, execution/store.py or indexing/store.py
Read/write models.py records in explicit transactions
        ↓
Caller commits → persistent state
Caller rolls back / closes uncommitted session → discard changes

Conversation → messages + submissions → runs + attachments
Assistant citations reference attachment records
```
