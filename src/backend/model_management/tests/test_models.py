"""Check model tasks with disposable PostgreSQL, temporary files and fake download subprocesses."""

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import sqlalchemy as sa

from src.backend.api.application import install_api_foundation
from src.backend.api.tests import test_conversations as conversation_tests

from src.backend.api.v1.models import get_model_registry
from src.backend.database.models import (
    InstalledModel,
    ModelDownload,
    Run,
    Conversation,
    Message,
    Submission,
)
from src.backend.model_management.config import ModelManagementSettings
from src.backend.model_management.download_process import download_all_model_parts
from src.backend.model_management.files import (
    resolve_managed_model_path,
    select_all_parts_of_requested_model,
    try_reserving_model_storage,
    validate_complete_model_files,
    write_llama_swap_config_for_installed_models,
)
from src.backend.model_management.schemas import ModelDownloadInput
from src.backend.model_management.store import ModelRegistry
from src.backend.model_management.worker import ModelDownloadWorker

TEST_URL = conversation_tests.TEST_URL


class ModelFilesAndCapabilitiesTests(unittest.TestCase):
    """Validate split manifests, safe configuration and frontend limits without network/model downloads."""

    def test_split_model_selection_requires_all_parts_and_part_one(self):
        """Accept supported split conventions, retaining nested paths and exact quantization names."""
        for parts in [
            ["sub/model-00001-of-00002.gguf", "sub/model-00002-of-00002.gguf"],
            ["model.gguf-00001-of-00002", "model.gguf-00002-of-00002"],
            ["model.gguf.00001-of-00002", "model.gguf.00002-of-00002"],
        ]:
            self.assertEqual(
                select_all_parts_of_requested_model(parts[0], parts + ["other.gguf"]), parts
            )
            with self.assertRaises(ValueError):
                select_all_parts_of_requested_model(parts[1], parts)
            with self.assertRaises(ValueError):
                select_all_parts_of_requested_model(parts[0], parts[:1])

    def test_paths_reject_traversal_and_symlink_escape(self):
        """Reject dangerous source paths before any network call or filesystem mutation."""
        for filename in [
            "/tmp/a.gguf",
            "../a.gguf",
            "x/../a.gguf",
            "x//a.gguf",
            "x\\a.gguf",
            "a.txt",
            "a\n.gguf",
        ]:
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                ModelDownloadInput(repo_id="owner/model", gguf_filename=filename)
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "link").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                resolve_managed_model_path(root, "link/a.gguf")

    def test_config_quotes_names_and_shell_arguments_and_preserves_inode(self):
        """A filename with spaces/quotes remains one shell argument; Docker file mounts keep seeing writes."""
        with tempfile.TemporaryDirectory() as directory:
            settings = ModelManagementSettings(
                root=Path(directory), config_path=Path(directory) / "config.yaml"
            )
            settings.config_path.touch()
            inode = settings.config_path.stat().st_ino
            filename = "weird ' model.gguf"
            write_llama_swap_config_for_installed_models(
                settings, [{"model_name": 'name: with quotes"', "gguf_file": filename}]
            )
            lines = settings.config_path.read_text().splitlines()
            self.assertEqual(json.loads(lines[1].strip()[:-1]), 'name: with quotes"')
            command = shlex.split(json.loads(lines[2].split("cmd: ", 1)[1]))
            self.assertTrue(command[command.index("-m") + 1].endswith("/" + filename))
            self.assertEqual(settings.config_path.stat().st_ino, inode)

    def test_subprocess_download_pins_revision_and_checks_every_part(self):
        """The downloader requests exactly the complete selected model at one immutable revision."""
        with tempfile.TemporaryDirectory() as directory:
            names = ["sub/a-00001-of-00002.gguf", "sub/a-00002-of-00002.gguf"]
            metadata = SimpleNamespace(
                sha="a" * 40, siblings=[SimpleNamespace(rfilename=name, size=8) for name in names]
            )
            calls = []

            def fake_download(**kwargs):
                """Write minimal GGUF fixtures in place of real network downloads."""
                calls.append(kwargs)
                path = Path(kwargs["local_dir"]) / kwargs["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"GGUFdata")
                return str(path)

            with patch("src.backend.model_management.download_process.HfApi") as api, patch(
                "src.backend.model_management.download_process.hf_hub_download",
                side_effect=fake_download,
            ):
                api.return_value.model_info.return_value = metadata
                download_all_model_parts(Path(directory), "owner/model", names[0])
            self.assertEqual([call["filename"] for call in calls], names)
            self.assertTrue(all(call["revision"] == metadata.sha for call in calls))
            progress = json.loads((Path(directory) / "progress.json").read_text())
            self.assertEqual(progress["downloaded_bytes"], 16)
            self.assertEqual(
                validate_complete_model_files(Path(directory) / "files", progress["files"]), 16
            )
            (Path(directory) / "files" / names[-1]).write_bytes(b"GGUF")
            with self.assertRaises(ValueError):
                validate_complete_model_files(Path(directory) / "files", progress["files"])

    def test_capabilities_use_server_configuration_and_do_not_require_models(self):
        """A frontend discovers actual upload limits/defaults without a DB or inference connection."""
        app = FastAPI()
        install_api_foundation(app)
        with patch.dict(
            os.environ,
            {
                "UPLOAD_MAX_FILES": "4",
                "UPLOAD_MAX_FILE_BYTES": "12345",
                "MODEL_MAX_QUEUED_DOWNLOADS": "2",
            },
        ), TestClient(app) as client:
            response = client.get("/api/v1/capabilities")
        self.assertEqual(response.status_code, 200, response.text)
        capabilities = response.json()
        self.assertEqual(capabilities["uploads"]["max_files"], 4)
        self.assertEqual(capabilities["uploads"]["max_file_bytes"], 12345)
        self.assertEqual(
            capabilities["chat"]["ranking_group_size"], {"default": 3, "min": 2, "max": 7}
        )
        self.assertEqual(
            [mode["candidate_count"] for mode in capabilities["chat"]["reasoning_modes"]],
            [1, 3, 5, 7],
        )
        self.assertEqual(capabilities["chat"]["temperature"], {"min": 0, "max": 2})
        self.assertEqual(capabilities["model_downloads"]["max_queued_downloads"], 2)
        self.assertNotIn("root", capabilities["uploads"])


FAKE_DOWNLOAD_SCRIPT = """
import json, pathlib, sys, time
job = pathlib.Path(sys.argv[1])
filename = sys.argv[2]
mode = sys.argv[3]
if mode == "slow":
    time.sleep(60)
if mode == "fail":
    sys.exit(1)
path = job / "files" / filename
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(b"GGUFdata")
progress = {"source_revision": "a" * 40, "files": [{"filename": filename, "size_bytes": 8}], "total_bytes": 8, "downloaded_bytes": 8}
(job / "progress.json").write_text(json.dumps(progress))
"""


@unittest.skipUnless(TEST_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class ModelManagementIntegrationTests(unittest.TestCase):
    """Use real transactions/subprocesses while keeping all model files and downloads disposable."""

    setUpClass = classmethod(conversation_tests.ConversationApiTests.setUpClass.__func__)
    tearDownClass = classmethod(conversation_tests.ConversationApiTests.tearDownClass.__func__)

    def setUp(self):
        """Reset only the guarded test schema and override model dependencies with temporary storage."""
        conversation_tests.ConversationApiTests.setUp(self)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = ModelManagementSettings(
            root=self.root / "models",
            config_path=self.root / "config.yaml",
            poll_seconds=0.03,
            download_timeout_seconds=5,
        )
        self.registry = ModelRegistry(self.sessions, self.settings)
        # The client installed by the shared fixture only overrides DB sessions.
        self.client.app.routes[-1].app.dependency_overrides[
            get_model_registry
        ] = lambda: self.registry
        self.fake_script = self.root / "fake_download.py"
        self.fake_script.write_text(FAKE_DOWNLOAD_SCRIPT)
        self.processes = []
        self.real_subprocess_exec = asyncio.create_subprocess_exec

    def download_request(self, filename="tiny.gguf", repo="owner/model"):
        """Submit a source without awaiting any download; return the HTTP response."""
        return self.client.post(
            "/api/v1/model-downloads", json={"repo_id": repo, "gguf_filename": filename}
        )

    async def wait_for_download_status(self, task_id, status):
        """Poll the persisted task for up to five seconds, failing with its actual state."""
        for _ in range(250):
            task = await self.registry.get_model_download(UUID(str(task_id)))
            if task.status == status:
                return task
            await asyncio.sleep(0.02)
        self.fail(f"Expected {status}; actual task: {task}")

    def fake_subprocess_factory(self, mode="ok"):
        """Build real short-lived child processes that write fixtures instead of downloading models."""

        async def spawn(*args, **kwargs):
            process = await self.real_subprocess_exec(
                sys.executable, str(self.fake_script), args[3], args[5], mode, **kwargs
            )
            self.processes.append(process)
            return process

        return spawn

    def test_download_admission_is_deduplicated_bounded_and_network_free(self):
        """POST returns a queued record immediately; refresh can rediscover it and repeats keep one row."""
        self.registry.settings = replace(self.settings, max_queued_downloads=1)
        with patch(
            "src.backend.ai.llms.huggingface.browse_models",
            side_effect=AssertionError("unexpected network"),
        ):
            first = self.download_request()
            second = self.download_request()
            full = self.download_request("other.gguf")
        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(first.json()["id"], second.json()["id"])
        self.assertEqual(first.json()["status"], "queued")
        self.assertEqual(full.status_code, 429, full.text)
        listing = self.client.get("/api/v1/model-downloads").json()
        self.assertEqual(len(listing), 1)
        self.assertEqual(
            self.client.get("/api/v1/model-downloads/" + first.json()["id"]).status_code, 200
        )
        self.assertEqual(self.client.get(f"/api/v1/model-downloads/{uuid4()}").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/model-downloads?limit=101").status_code, 422)
        self.assertEqual(self.download_request("../bad.gguf").status_code, 422)
        self.assertEqual(self.download_request(repo="https://example.com/model").status_code, 422)

    def test_download_publication_reconnect_and_read_only_model_listing(self):
        """A completed child installs bytes/config once; GET and a repeated POST do not redownload."""
        accepted = self.download_request("nested/tiny.gguf").json()

        async def execute():
            worker = ModelDownloadWorker(self.registry)
            with patch(
                "asyncio.create_subprocess_exec", side_effect=self.fake_subprocess_factory()
            ):
                await worker.start()
                try:
                    return await self.wait_for_download_status(accepted["id"], "completed")
                finally:
                    await worker.stop()

        completed = asyncio.run(execute())
        self.assertEqual(completed.attempt, 1)
        models = self.client.get("/api/v1/models").json()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["status"], "ready")
        self.assertNotIn("local_path", models[0])
        self.assertEqual(models[0]["source_revision"], "a" * 40)
        inode = self.settings.config_path.stat().st_mtime_ns
        with patch.object(
            ModelDownloadWorker,
            "_scan_legacy_model_files",
            side_effect=AssertionError("GET must not scan"),
        ):
            self.assertEqual(self.client.get("/api/v1/models").status_code, 200)
        self.assertEqual(self.settings.config_path.stat().st_mtime_ns, inode)
        self.assertEqual(self.download_request("nested/tiny.gguf").json()["status"], "completed")
        self.assertEqual(len(self.processes), 1)
        self.assertIn("managed/", self.settings.config_path.read_text())

    def test_failure_and_explicit_retry_reuse_task_id(self):
        """A failed child leaves no installed model; repeating POST resets the same task for another attempt."""
        first = self.download_request().json()

        async def execute(mode, expected):
            worker = ModelDownloadWorker(self.registry)
            with patch(
                "asyncio.create_subprocess_exec", side_effect=self.fake_subprocess_factory(mode)
            ):
                await worker.start()
                try:
                    return await self.wait_for_download_status(first["id"], expected)
                finally:
                    await worker.stop()

        failed = asyncio.run(execute("fail", "failed"))
        self.assertEqual(failed.error["code"], "MODEL_DOWNLOAD_FAILED")
        self.assertEqual(self.client.get("/api/v1/models").json(), [])
        retried = self.download_request().json()
        self.assertEqual(retried["id"], first["id"])
        self.assertEqual(retried["status"], "queued")
        self.assertEqual(asyncio.run(execute("ok", "completed")).attempt, 2)

    def test_shutdown_reaps_child_and_preserves_interrupted_task(self):
        """Stopping the worker really terminates its child before releasing the storage lock."""
        accepted = self.download_request().json()

        async def execute():
            worker = ModelDownloadWorker(self.registry)
            with patch(
                "asyncio.create_subprocess_exec", side_effect=self.fake_subprocess_factory("slow")
            ):
                await worker.start()
                try:
                    for _ in range(100):
                        if self.processes:
                            break
                        await asyncio.sleep(0.02)
                    self.assertTrue(self.processes)
                    with try_reserving_model_storage(self.settings) as descriptor:
                        self.assertIsNone(descriptor)
                finally:
                    await worker.stop()
            self.assertIsNotNone(self.processes[0].returncode)
            task = await self.registry.get_model_download(UUID(accepted["id"]))
            self.assertEqual(task.status, "interrupted")
            with try_reserving_model_storage(self.settings) as descriptor:
                self.assertIsNotNone(descriptor)

        asyncio.run(execute())

    def seed_local_model(self, filename, *, default=False, status="ready"):
        """Create one real GGUF fixture and matching registry row; return its UUID."""
        self.settings.root.mkdir(parents=True, exist_ok=True)
        (self.settings.root / filename).write_bytes(b"GGUFdata")
        model_id = uuid4()
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(InstalledModel).values(
                    id=model_id,
                    model_name=Path(filename).stem,
                    display_name=filename,
                    hf_repo="owner/model",
                    gguf_file=filename,
                    local_path=str((self.settings.root / filename).resolve()),
                    quantization="Q4_K_M",
                    size_bytes=8,
                    is_default=default,
                    status=status,
                    files=[{"filename": filename, "size_bytes": 8}],
                )
            )
        return model_id

    def test_deletion_blocks_default_and_active_or_unconfirmed_runs_then_cleans_files(self):
        """Deletion cannot remove an in-use model and persists intent until bytes/config are cleaned."""
        default_id = self.seed_local_model("default.gguf", default=True)
        model_id = self.seed_local_model("removable.gguf")
        self.assertEqual(
            self.client.delete(f"/api/v1/models/{default_id}").json()["error"]["code"],
            "DEFAULT_MODEL_PROTECTED",
        )
        conversation_id, submission_id, message_id, run_id = (uuid4() for _ in range(4))
        with self.engine.begin() as connection:
            connection.execute(sa.insert(Conversation).values(id=conversation_id, title="chat"))
            connection.execute(
                sa.insert(Submission).values(
                    id=submission_id,
                    conversation_id=conversation_id,
                    client_submission_id=uuid4(),
                    fingerprint="a" * 64,
                    settings_snapshot={"model_name": "removable"},
                    status="ready",
                )
            )
            connection.execute(
                sa.insert(Message).values(
                    id=message_id,
                    conversation_id=conversation_id,
                    submission_id=submission_id,
                    content="question",
                    role="user",
                )
            )
            connection.execute(
                sa.insert(Run).values(
                    id=run_id,
                    conversation_id=conversation_id,
                    submission_id=submission_id,
                    message_id=message_id,
                    settings_snapshot={"model_name": "removable"},
                    status="queued",
                )
            )
        self.assertEqual(
            self.client.delete(f"/api/v1/models/{model_id}").json()["error"]["code"], "MODEL_IN_USE"
        )
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Run)
                .where(Run.id == run_id)
                .values(status="interrupted", lease_owner="unknown-compute")
            )
        self.assertEqual(
            self.client.delete(f"/api/v1/models/{model_id}").json()["error"]["code"], "MODEL_IN_USE"
        )
        with self.engine.begin() as connection:
            connection.execute(sa.update(Run).where(Run.id == run_id).values(lease_owner=None))
        response = self.client.delete(f"/api/v1/models/{model_id}")
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["status"], "deleting")
        worker = ModelDownloadWorker(self.registry)
        asyncio.run(worker._reconcile_local_models_and_configuration())
        self.assertFalse((self.settings.root / "removable.gguf").exists())
        self.assertEqual(self.client.get(f"/api/v1/models/{model_id}").status_code, 404)
        self.assertTrue((self.settings.root / "default.gguf").exists())
        self.assertNotIn("removable", self.settings.config_path.read_text())

    def test_recovery_keeps_missing_models_and_imports_only_complete_split_models(self):
        """Startup does not download; complete local files become ready, partial split files stay unregistered."""
        missing_id = self.seed_local_model("missing.gguf", default=True)
        (self.settings.root / "missing.gguf").unlink()
        (self.settings.root / "complete.gguf").write_bytes(b"GGUFdata")
        (self.settings.root / "split-00001-of-00002.gguf").write_bytes(b"GGUFdata")
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(ModelDownload).values(
                    id=uuid4(),
                    repo_id="owner/abandoned",
                    gguf_filename="old.gguf",
                    status="downloading",
                    attempt=1,
                )
            )
        worker = ModelDownloadWorker(self.registry)

        async def recover():
            await worker._reconcile_local_models_and_configuration()
            await worker._recover_abandoned_downloads_and_reserve_next_queued_task()

        asyncio.run(recover())
        models = self.client.get("/api/v1/models").json()
        self.assertEqual({model["model_name"] for model in models}, {"missing", "complete"})
        self.assertEqual(
            next(model for model in models if model["id"] == str(missing_id))["status"], "missing"
        )
        self.assertEqual(
            self.client.get("/api/v1/model-downloads").json()[0]["status"], "interrupted"
        )
        self.assertNotIn("missing", self.settings.config_path.read_text())

    def test_download_timeout_reaps_process_and_keeps_task_retryable(self):
        """A timeout records a failure only after its actual download child has exited."""
        self.registry.settings = replace(self.settings, download_timeout_seconds=0.08)
        accepted = self.download_request().json()

        async def execute():
            worker = ModelDownloadWorker(self.registry)
            with patch(
                "asyncio.create_subprocess_exec", side_effect=self.fake_subprocess_factory("slow")
            ):
                await worker.start()
                try:
                    failed = await self.wait_for_download_status(accepted["id"], "failed")
                    self.assertEqual(failed.error["code"], "DOWNLOAD_TIMEOUT")
                    self.assertIsNotNone(self.processes[0].returncode)
                finally:
                    await worker.stop()

        asyncio.run(execute())
        self.assertEqual(self.download_request().json()["id"], accepted["id"])

    def test_configuration_failure_preserves_installation_and_retry_finishes_without_download(self):
        """A failed config write never reports ready; recovery publishes the existing complete installation."""
        accepted = self.download_request().json()
        real_write = write_llama_swap_config_for_installed_models

        def fail_nonempty_config(settings, models):
            """Simulate a config mount failure after model files and registry metadata exist."""
            if models:
                raise OSError("private mount details")
            return real_write(settings, models)

        async def execute():
            worker = ModelDownloadWorker(self.registry)
            with patch(
                "asyncio.create_subprocess_exec", side_effect=self.fake_subprocess_factory()
            ), patch(
                "src.backend.model_management.worker.write_llama_swap_config_for_installed_models",
                side_effect=fail_nonempty_config,
            ):
                await worker.start()
                try:
                    await self.wait_for_download_status(accepted["id"], "failed")
                finally:
                    await worker.stop()
            models = await self.registry.list_installed_models()
            self.assertEqual(models[0].status, "configuring")
            await worker._reconcile_local_models_and_configuration()

        asyncio.run(execute())
        retried = self.download_request().json()
        self.assertEqual(retried["id"], accepted["id"])
        self.assertEqual(retried["status"], "completed")
        self.assertEqual(len(self.processes), 1)
        self.assertNotIn("private", json.dumps(retried))

    def test_catalog_errors_are_sanitized_and_query_bounds_are_validated(self):
        """External metadata failures return stable errors without leaking source exception text."""
        with patch(
            "src.backend.ai.llms.huggingface.browse_models",
            side_effect=RuntimeError("secret token"),
        ):
            response = self.client.get("/api/v1/models/catalog")
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret", response.text)
        self.assertEqual(self.client.get("/api/v1/models/catalog?limit=101").status_code, 422)
        self.assertEqual(self.client.get("/api/v1/models/quants?repo_id=../bad").status_code, 422)

    def test_name_collision_does_not_replace_an_existing_model(self):
        """Downloading a different repository with an occupied inference name leaves its model intact."""
        existing_id = self.seed_local_model("tiny.gguf", default=True)
        accepted = self.download_request(repo="other/repository").json()

        async def execute():
            worker = ModelDownloadWorker(self.registry)
            with patch(
                "asyncio.create_subprocess_exec", side_effect=self.fake_subprocess_factory()
            ):
                await worker.start()
                try:
                    await self.wait_for_download_status(accepted["id"], "failed")
                finally:
                    await worker.stop()

        asyncio.run(execute())
        models = self.client.get("/api/v1/models").json()
        self.assertEqual([model["id"] for model in models], [str(existing_id)])
        self.assertEqual((self.settings.root / "tiny.gguf").read_bytes(), b"GGUFdata")

    def test_unavailable_local_model_rejects_a_new_send(self):
        """A known missing/deleting model cannot be admitted while deletion or recovery is pending."""
        from src.backend.api.v1.submissions import get_upload_settings
        from src.backend.uploads.config import UploadSettings

        model_id = self.seed_local_model("missing.gguf", status="missing")
        conversation_id = uuid4()
        with self.engine.begin() as connection:
            connection.execute(sa.insert(Conversation).values(id=conversation_id, title="chat"))
        self.client.app.routes[-1].app.dependency_overrides[get_upload_settings] = (
            lambda: UploadSettings(root=self.root / "uploads")
        )
        payload = {
            "client_submission_id": str(uuid4()),
            "content": "question",
            "settings": {"model_name": "missing"},
        }
        response = self.client.post(
            f"/api/v1/conversations/{conversation_id}/submissions",
            files=[("payload", (None, json.dumps(payload)))],
        )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"]["code"], "MODEL_NOT_READY")
        with self.engine.connect() as connection:
            self.assertEqual(connection.scalar(sa.select(sa.func.count()).select_from(Run)), 0)

    def test_shutdown_during_subprocess_creation_still_reaps_the_child(self):
        """Cancellation between OS spawn and returning the process handle cannot leak a child."""
        accepted = self.download_request().json()
        spawn = self.fake_subprocess_factory("slow")

        async def delayed_spawn(*args, **kwargs):
            process = await spawn(*args, **kwargs)
            await asyncio.sleep(0.08)
            return process

        async def execute():
            worker = ModelDownloadWorker(self.registry)
            with patch("asyncio.create_subprocess_exec", side_effect=delayed_spawn):
                await worker.start()
                for _ in range(100):
                    if self.processes:
                        break
                    await asyncio.sleep(0.01)
                await worker.stop()
            self.assertTrue(self.processes)
            self.assertIsNotNone(self.processes[0].returncode)
            self.assertEqual(
                (await self.registry.get_model_download(UUID(accepted["id"]))).status, "interrupted"
            )

        asyncio.run(execute())

    def test_already_exited_download_child_still_publishes_its_progress(self):
        """A child finishing before polling begins must not be mistaken for a failed download."""
        accepted = self.download_request().json()

        async def instant_completed_child(*args, **kwargs):
            directory = Path(args[3])
            filename = args[5]
            target = directory / 'files' / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'GGUFdata')
            (directory / 'progress.json').write_text(json.dumps({'source_revision': 'a' * 40, 'files': [{'filename': filename, 'size_bytes': 8}], 'total_bytes': 8, 'downloaded_bytes': 8}))
            return SimpleNamespace(returncode=0)

        async def execute():
            worker = ModelDownloadWorker(self.registry)
            with patch('asyncio.create_subprocess_exec', side_effect=instant_completed_child):
                await worker.start()
                try:
                    await self.wait_for_download_status(accepted['id'], 'completed')
                finally:
                    await worker.stop()

        asyncio.run(execute())

    def test_shutdown_holds_storage_lock_until_config_thread_finishes(self):
        """Cancelling an async waiter must not release the lock while its native file write is running."""
        import threading
        started = threading.Event()
        release = threading.Event()
        real_write = write_llama_swap_config_for_installed_models

        def blocked_write(settings, models):
            """Pause a native config write while shutdown tries to stop its async owner."""
            started.set()
            release.wait(timeout=2)
            real_write(settings, models)

        async def execute():
            worker = ModelDownloadWorker(self.registry)
            with patch('src.backend.model_management.worker.write_llama_swap_config_for_installed_models', side_effect=blocked_write):
                await worker.start()
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(started.is_set())
                stopping = asyncio.create_task(worker.stop())
                try:
                    await asyncio.sleep(0.03)
                    self.assertFalse(stopping.done())
                    with try_reserving_model_storage(self.settings) as descriptor:
                        self.assertIsNone(descriptor)
                finally:
                    release.set()
                    await stopping
            with try_reserving_model_storage(self.settings) as descriptor:
                self.assertIsNotNone(descriptor)

        asyncio.run(execute())

    def test_ready_model_does_not_consume_queue_capacity(self):
        """An existing installation is returned even while the pending-download queue is full."""
        self.registry.settings = replace(self.settings, max_queued_downloads=1)
        self.seed_local_model('already.gguf', default=True)
        self.assertEqual(self.download_request('pending.gguf').status_code, 202)
        existing = self.download_request('already.gguf')
        self.assertEqual(existing.status_code, 202, existing.text)
        self.assertEqual(existing.json()['status'], 'completed')
        self.assertEqual(existing.json()['attempt'], 0)

    def test_retry_of_registered_files_waits_for_configuration_instead_of_redownloading(self):
        """Retry a config-only failure by finishing publication of the existing files."""
        model_id = self.seed_local_model('registered.gguf', default=True, status='configuring')
        accepted = self.download_request('registered.gguf')
        self.assertEqual(accepted.status_code, 202, accepted.text)
        self.assertEqual(accepted.json()['status'], 'registering')
        worker = ModelDownloadWorker(self.registry)

        async def recover():
            await worker._reconcile_local_models_and_configuration()
            self.assertIsNone(await worker._recover_abandoned_downloads_and_reserve_next_queued_task())

        asyncio.run(recover())
        recovered = self.client.get('/api/v1/model-downloads/' + accepted.json()['id']).json()
        self.assertEqual(recovered['status'], 'completed')
        self.assertEqual(recovered['model_id'], str(model_id))
        self.assertEqual(recovered['downloaded_bytes'], 8)
        self.assertEqual(recovered['attempt'], 0)
