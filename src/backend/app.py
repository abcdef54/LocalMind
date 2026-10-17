import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.api.application import install_api_foundation
from src.backend.constants import DEFAULT_MODEL_REPO_ID, DEFAULT_MODEL_GGUF_FILE
from src.backend.database.session import upgrade_database_schema, async_session_maker
from src.backend.execution.config import ExecutorSettings
from src.backend.execution.store import RunStore
from src.backend.execution.handler import GenerateReply
from src.backend.execution.worker import RunExecutor
from src.backend.indexing.config import IndexingSettings
from src.backend.indexing.resources import get_document_vector_index
from src.backend.uploads.document_cleanup import DocumentCleanupWorker
from src.backend.indexing.store import IndexingStore
from src.backend.indexing.worker import IndexingWorker
from src.backend.uploads.config import UploadSettings
from src.backend.uploads.storage import remove_orphaned_upload_batches
from src.backend.routes import conversation_router, models_router
from src.backend.routes.dependencies import get_model_manager, get_ai
from src.backend.services.model_services import ModelService
from src.backend.ai.llms.huggingface import path as cpp_models_dir


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare storage/AI and start the indexing/run workers before serving requests.

    Args:
        app: Root FastAPI application that exposes worker instances on app.state.

    Yields:
        None after migrations, orphan cleanup, model synchronization and worker
        startup. Indexing prepares documents; the run handler generates queued replies.

    Stop indexing, cleanup and run workers when serving ends, including on failure.
    Migration/worker initialization failures propagate; model sync failures are logged.
    """
    executor_settings = ExecutorSettings.from_environment()
    upload_settings = UploadSettings.from_environment()
    indexing_settings = IndexingSettings.from_environment()
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

    # Ensure model directory exists
    print("[INFO] Ensuring model directory exists...")
    os.makedirs(cpp_models_dir, exist_ok=True)
    print("[INFO] Model directory exists.")
    print()

    # Detect first run by scanning for GGUF models
    local_ggufs = [f for f in os.listdir(cpp_models_dir) if f.endswith(".gguf")]

    try:
        async with async_session_maker() as session:
            manager = get_model_manager()
            model_service = ModelService(manager=manager, session=session)

            if not local_ggufs:
                print("[INFO] No GGUF models were found.")
                print()
                print("[INFO] Downloading default model...")
                print()
                print("[INFO] Repository:")
                print(DEFAULT_MODEL_REPO_ID)
                print()
                print("[INFO] Model:")
                print(DEFAULT_MODEL_GGUF_FILE)
                print()
                print("[INFO] This only happens once.")
                print()
                # Block until download completes
                await manager.download_gguf(DEFAULT_MODEL_REPO_ID, DEFAULT_MODEL_GGUF_FILE)
                print("[INFO] Default model downloaded successfully.")

                # Synchronize model in database and regenerate llama-swap configuration
                await model_service.sync_local_models()
                print("[INFO] Model synchronized.")
                print("[INFO] Llama-swap configuration updated.")
                print()
                print("[INFO] Starting LocalMind...")
            else:
                await model_service.sync_local_models()
                print("[INFO] Successfully synced local models.")
                print()
    except Exception as e:
        print(f"[ERROR] Failed to sync local models at startup: {e}")

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
        handler=GenerateReply(async_session_maker, ai),
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
                await executor.stop()
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
app.include_router(models_router)
install_api_foundation(app)
