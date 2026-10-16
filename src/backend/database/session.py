from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.backend.database.config import get_database_url
from src.backend.database.migration import upgrade_database

DB_URL = get_database_url()

engine = create_async_engine(DB_URL, echo=False)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def upgrade_database_schema() -> None:
    """Run versioned migrations before the application starts accepting requests."""
    async with engine.begin() as connection:
        await connection.run_sync(upgrade_database)


async def get_async_session() -> AsyncIterator[AsyncSession]:
    async with async_session_maker() as session:
        yield session
