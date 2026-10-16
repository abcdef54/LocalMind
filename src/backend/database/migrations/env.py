"""Run Alembic revisions using a startup connection, a CLI connection or SQL output mode."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from src.backend.database.config import get_database_url
from src.backend.database.models import Base

config = context.config
if config.config_file_name and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Write migration SQL without connecting to PostgreSQL.

    Use the configured URL to choose SQL syntax. This output supports a new empty
    database; it cannot inspect or safely adopt an existing unversioned database.
    """
    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_on_connection(connection: Connection) -> None:
    """Apply selected revisions using the supplied synchronous database connection.

    Configure schema comparisons against Base.metadata and let Alembic manage its
    transaction inside the caller's connection. Propagate migration errors; do not
    open or close the supplied connection here.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Open an async PostgreSQL connection for an Alembic CLI command.

    Run the synchronous migration function through that connection, then dispose
    the engine even if migration fails. Errors propagate to make the CLI command fail.
    """
    engine = create_async_engine(get_database_url(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(run_migrations_on_connection)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    supplied_connection = config.attributes.get("connection")
    if supplied_connection is not None:
        run_migrations_on_connection(supplied_connection)
    else:
        asyncio.run(run_async_migrations())
