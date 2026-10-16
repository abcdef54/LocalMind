"""Provide request-scoped database sessions without loading the AI engine."""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield one async database session and close it after the route finishes.

    Import the session factory only when a request needs it. Services must commit
    their own writes; this dependency does not commit a successful route automatically.
    Closing the session rolls back any transaction left unfinished.
    """
    from src.backend.database.session import async_session_maker

    async with async_session_maker() as session:
        yield session
