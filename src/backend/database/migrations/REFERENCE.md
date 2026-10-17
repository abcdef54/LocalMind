# Detailed reference

Alembic manages schema changes. Application startup runs `upgrade head` in the
database transaction before initializing models or the AI engine. No AI/model
download is needed to run Alembic commands or migration tests.

Connection handling follows the official [Alembic asyncio cookbook](https://alembic.sqlalchemy.org/en/latest/cookbook.html#using-asyncio-with-alembic),
including a supplied SQLAlchemy connection for application startup.

## Run migrations

From the repository root with dependencies installed and `POSTGRESQL_URL` set in
the environment or root `.env`:

```bash
python -m alembic upgrade head
python -m alembic current
python -m alembic check
```

Docker builds include the configuration and revisions. Once the updated backend
image is built, startup applies migrations automatically. To run only migrations:

```bash
docker compose run --rm --no-deps backend python -m alembic upgrade head
```

The database service must already be running. Run only one migration process at
a time; do not start several backend replicas or a CLI migration concurrently.
Schema changes after this baseline require a database role with DDL privileges.

## Existing databases

Back up PostgreSQL before the first upgrade. Keep the corresponding uploaded
files separately; a database backup does not contain their bytes.

```bash
docker compose exec -T db pg_dump -U postgres -d localmind_db -Fc > localmind-before-migrations.dump
```

Revision `0001_existing_schema` contains a frozen copy of the five existing
tables. It never imports live ORM metadata to decide what to create.

- Empty database: create baseline tables/indexes and record the revision.
- Matching unversioned legacy database: compare tables, columns, types,
  nullability, server defaults, indexes, unique/foreign-key constraints and
  primary keys; record the revision without rewriting application rows.
- Partial or drifted legacy database: fail with schema differences. The enclosing
  PostgreSQL transaction rolls back; missing tables are not silently filled in.
- Versioned database: apply outstanding revisions; repeated upgrades are no-ops
  when already at head. Use `alembic check` to detect ORM/schema drift afterward.

Schema comparison uses Alembic autogenerate plus an explicit primary-key check.
It is not a complete audit of grants, triggers, views or stored procedures.
Use a dedicated LocalMind database with the default `public` schema. Unexpected
tables are rejected instead of being assumed to belong to LocalMind.

If adoption fails, inspect the reported differences and reconcile them explicitly
after a backup. Do not run `alembic stamp head` to hide a mismatch. A plain stamp
records a revision without validating or upgrading the schema.

`downgrade base` **drops all five baseline tables and their data**, including
tables adopted from a legacy database. It is for disposable development/test
databases, not a way to undo baseline adoption while preserving application data.
Restore the pre-migration backup if that is the intended recovery.

## Revisions after the baseline

| Revision | Scope |
|---|---|
| `0002_history_pagination` | Index conversations/messages for cursor queries; require conversation creation timestamps |
| `0003_submissions_runs` | Durable submission/run tables, snapshot/status fields, optional legacy-message links, uniqueness and scoped foreign keys |

Revision 0002 backfills only explicitly NULL conversation creation timestamps
with migration time, since the original creation time is unknown. Existing
non-null timestamps and message storage types are preserved. Message dates remain
naive UTC for compatibility; v1 serializes them as UTC-aware timestamps.

Revision 0003 leaves `submission_id` and `run_id` NULL on preexisting messages.
It does not invent submissions/runs for historic chats or trigger inference.
New submission uniqueness is `(conversation_id, client_submission_id)`. A
fingerprint is a 64-character lowercase SHA-256 hex string; computation and
payload-conflict handling belong to the submission API task, not this schema.

Submissions/runs store JSON object settings snapshots and constrained statuses.
SQL checks ensure JSON object shape, not every ChatSettings field; the service
must validate the complete contract before inserting a snapshot. Runs also have
attempt, lease, cancellation/error and timing fields for the future executor.

One submission has at most one user message/run; one run has at most one assistant
result. Composite foreign keys enforce conversation/submission/result ownership,
including preventing a run from borrowing another run's answer. A partial unique
index permits only one pending/queued/running/cancel_requested run per conversation.
A completed run requires its own result message. Full state transitions, leasing
and retries are implemented in BE-05/BE-10/BE-15, not by the schema alone.

Downgrading to revision 0001 removes new submission/run data and link columns but
preserves legacy tables/rows. It must not be used to retain a new run history.
Downgrading all the way to base still drops the legacy application tables.

Offline SQL is available via `python -m alembic upgrade head --sql`, but that
script is only for a new empty database. Offline mode cannot validate or adopt a
legacy schema.

## Add a subsequent revision

Change ORM models, then generate and review a new revision:

```bash
python -m alembic revision --autogenerate -m "describe the schema change"
python -m alembic upgrade head
python -m alembic check
```

Review generated SQL/data backfills before upgrading. Do not modify the frozen
baseline to add future features. Each subsequent backend task supplies its own
migration and preservation tests.

## Integration tests

Tests require a **disposable** PostgreSQL database named exactly
`localmind_migration_tests`. They drop and recreate its `public` schema before
each test. Never point this variable at a database containing useful data.

```bash
export LOCALMIND_MIGRATION_TEST_URL='postgresql+asyncpg://postgres:TEST_PASSWORD@127.0.0.1:TEST_PORT/localmind_migration_tests'
python -m unittest discover -s src/backend/database/tests -v
```

Without the explicit test URL, integration tests skip. Cases cover empty/legacy
databases, preservation of rows and citation links, repeated startup, schema
drift rejection, downgrade/re-upgrade, async CLI and offline SQL generation.
Submission/run tests also check concurrent writes, idempotency uniqueness,
active-run admission constraints, result ownership and invalid snapshots/states.
Run this suite sequentially with API integration tests when using the same DB.

## Attachment lifecycle (0004)

Revision `0004_submission_attachments` adds nullable submission/message/client/storage
identities and constrained lifecycle state to attachments. Legacy citation links and
paths remain intact; old files default to ready and documents already marked as deleted retain that status.
New file links must match the user message submission/conversation. A partial unique
index protects live filenames for new submission attachments without rewriting old
duplicates. The submission service also checks existing legacy filenames.

Downgrading to 0003 drops these new links and states, so it does not preserve new
upload history. Physical files are not removed by schema downgrades. See the
[upload guide](../../uploads/README.md) for limits and orphan reconciliation.

## Durable indexing (0005)

`0005_attachment_indexing` adds indexing owner/expiry, generation, attempt and chunk
count. Existing undeleted attachments become pending for the new scoped collection.
Affected sends wait for indexing and queued replies return to pending; existing
completed/failed/cancelled run outcomes and original rows/paths remain intact.
Back up DB and `.uploads` together. Startup workers rebuild the index; the migration
itself never loads a model, deletes bytes or invokes Chroma. See the
[indexing guide](../../indexing/README.md) for limits, restart and retry behavior.


## 0006_document_cleanup

Adds cleanup attempt/completion timestamps, a safe error code and a partial index
for pending deleted attachments. Documents already marked as deleted become eligible for cleanup;
metadata/citations remain. Migration performs no filesystem or vector operations.


## 0007_generation_metadata

Adds nullable JSONB generation_metadata to messages and expands model_name from
50 to 255 characters to match ChatSettings. Existing answers remain unchanged.
Downgrade removes metrics and rejects shrinking model names longer than 50 rather
than silently truncating them. No inference, file or vector operations run here.
