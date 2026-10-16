# Database migrations (BE-01)

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
