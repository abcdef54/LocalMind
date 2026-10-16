"""Delete abandoned upload folders without loading models: python -m src.backend.uploads.cleanup."""

import asyncio

from src.backend.uploads.config import UploadSettings
from src.backend.uploads.storage import remove_orphaned_upload_batches


async def main() -> None:
    """Delete old uncommitted upload folders using the configured database and upload root.

    Read UPLOAD_* and DB configuration from the environment. Keep locked folders and
    folders with saved submissions. Print the number of batch folders removed.
    Do not load AI or delete DB metadata. Always dispose the DB engine; let errors
    propagate so the command exits unsuccessfully when cleanup fails.
    """
    from src.backend.database.session import async_session_maker, engine

    settings = UploadSettings.from_environment()
    try:
        async with async_session_maker() as session:
            removed_batch_count = await remove_orphaned_upload_batches(
                session, settings.root, settings.orphan_grace_seconds
            )
        print(f"Removed {removed_batch_count} orphan upload batches")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
