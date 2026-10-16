"""Lazy database dependency; importing v1 contracts does not initialize the AI."""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession


async def get_session() -> AsyncIterator[AsyncSession]:
    from src.backend.database.session import async_session_maker

    async with async_session_maker() as session:
        yield session
