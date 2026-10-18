from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.api.application import install_api_foundation
from src.backend.database.session import upgrade_database_schema, async_session_maker
from src.backend.execution.config import ExecutorSettings
from src.backend.execution.store import RunStore
from src.backend.execution.handler import GenerateReply
from src.backend.memory.config import MemorySettings
from src.backend.execution.worker import RunExecutor
from src.backend.indexing.config import IndexingSettings
from src.backend.indexing.resources import get_document_vector_index
from src.backend.uploads.document_cleanup import DocumentCleanupWorker
from src.backend.indexing.store import IndexingStore
from src.backend.indexing.worker import IndexingWorker
from src.backend.uploads.config import UploadSettings
from src.backend.uploads.storage import remove_orphaned_upload_batches
from src.backend.routes import conversation_router
from src.backend.routes.dependencies import get_ai
from src.backend.model_management.config import ModelManagementSettings
from src.backend.model_management.store import ModelRegistry
from src.backend.model_management.worker import ModelDownloadWorker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare storage/AI and start the indexing/run workers before serving requests.

    Args:
        app: Root FastAPI application that exposes worker instances on app.state.

    Yields:
        None after migrations, orphan cleanup, background worker
        startup. Indexing prepares documents; the run handler generates queued replies.

    Stop indexing, cleanup, download and run workers when serving ends, including on failure.
    Migration/worker initialization failures propagate. Model downloads never block startup.
    """
    executor_settings = ExecutorSettings.from_environment()
    upload_settings = UploadSettings.from_environment()
    indexing_settings = IndexingSettings.from_environment()
    memory_settings = MemorySettings.from_environment()
    model_settings = ModelManagementSettings.from_environment()
    print()
    print("=" * 60)
    print()
    print("[INFO] Initializing LocalMind backend.")
    print()
    print("=" * 60)
    print()

    print("[INFO] Initializing Database...")
    await upgrade_database_schema()
    async with async_session_maker() as session:
        await remove_orphaned_upload_batches(
            session, upload_settings.root, upload_settings.orphan_grace_seconds
        )
    print("[INFO] Database initialized.")
    print()

    # Eagerly initialize heavy singletons so they are ready before the first request
    print("[INFO] Initializing Vector DB")
    vector_index = get_document_vector_index()
    print("[INFO] Vector DB initialized.")
    print()

    print("[INFO] Initializing AI engine")
    ai = get_ai()
    print("[INFO] AI engine initialized.")
    print()

    executor = RunExecutor(
        RunStore(async_session_maker),
        handler=GenerateReply(async_session_maker, ai, memory_settings),
        settings=executor_settings,
    )
    app.state.run_executor = executor
    indexer = IndexingWorker(
        IndexingStore(async_session_maker),
        vector_index,
        upload_settings.root / "indexing_locks",
        settings=indexing_settings,
    )
    app.state.document_cleanup_worker = DocumentCleanupWorker(
        async_session_maker, vector_index, upload_settings.root
    )
    app.state.indexing_worker = indexer
    model_worker = ModelDownloadWorker(ModelRegistry(async_session_maker, model_settings))
    app.state.model_download_worker = model_worker
    await executor.start()

    print()
    print("=" * 60)
    print("[INFO] LocalMind backend is ready.")
    print("=" * 60)
    print()
    print("[INFO] To stop the backend, press Ctrl+C.")
    print("=" * 60)
    print()

    try:
        await model_worker.start()
        await indexer.start()
        await app.state.document_cleanup_worker.start()
        yield
    finally:
        try:
            await app.state.document_cleanup_worker.stop()
        finally:
            try:
                await indexer.stop()
            finally:
                try:
                    await executor.stop()
                finally:
                    await model_worker.stop()
    print("[INFO] Shutting down LocalMind backend...")
    print()
    print("=" * 60)
    print()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

app.include_router(conversation_router)
install_api_foundation(app)
