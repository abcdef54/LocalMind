# Database migrations

Migrations change the schema in revision order and can update existing metadata. They do not run document extraction or embedding.

```text
Startup → session.py → upgrade_database_schema()
        ↓
migration.py passes the existing connection to Alembic
        ↓
env.py selects migration metadata/connection
        ↓
versions/: 0001 → 0002 → 0003 → 0004 → 0005
        ↓
Database schema and revision marker updated
        ↓
Startup continues; background workers handle pending records
```

[Configuration, contracts and troubleshooting](REFERENCE.md).
