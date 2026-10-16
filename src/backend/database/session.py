"""Create async database sessions and apply migrations during backend startup."""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.backend.database.config import get_database_url
from src.backend.database.migration import upgrade_database

DB_URL = get_database_url()

engine = create_async_engine(DB_URL, echo=False)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def upgrade_database_schema() -> None:
    """Open a transaction, apply pending revisions and commit if they all succeed.

    Return None. Propagate connection/migration errors and roll back on failure,
    so startup does not accept requests against an incomplete database schema.
    """
    async with engine.begin() as connection:
        await connection.run_sync(upgrade_database)


async def get_async_session() -> AsyncIterator[AsyncSession]:
    """Yield a database session for one request and close it when the request ends.

    The caller controls commits. Closing discards any uncommitted transaction;
    this dependency does not automatically commit successful route operations.
    """
    async with async_session_maker() as session:
        yield session
