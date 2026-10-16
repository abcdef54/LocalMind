"""Check executor limits and startup/shutdown wiring without loading real models or a database."""

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI

from src.backend.execution.config import ExecutorSettings
from src.backend.execution.store import RunResult


class ExecutorConfigTests(unittest.TestCase):
    """Validate executor configuration and returned answer text using in-memory values."""

    def test_default_limits_and_environment_overrides(self) -> None:
        """Use one worker slot by default and accept a valid concurrency environment override."""
        self.assertEqual(ExecutorSettings().max_concurrency, 1)
        with patch.dict(os.environ, {"RUN_EXECUTOR_MAX_CONCURRENCY": "2"}):
            self.assertEqual(ExecutorSettings.from_environment().max_concurrency, 2)

    def test_invalid_or_unsafe_settings_fail_early(self) -> None:
        """Reject invalid counts, nonfinite times and heartbeat/DB timing that can outlast a
        reservation.
        """
        for overrides in (
            {"max_concurrency": 0},
            {"max_concurrency": True},
            {"poll_seconds": 0},
            {"lease_seconds": float("nan")},
            {"heartbeat_seconds": float("inf")},
            {"shutdown_grace_seconds": -1},
            {"lease_seconds": 10, "heartbeat_seconds": 5, "database_timeout_seconds": 5},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                ExecutorSettings(**overrides)

    def test_empty_result_is_rejected_without_modifying_nonempty_content(self) -> None:
        """Reject missing/blank answers while preserving the whitespace of a nonempty answer."""
        for content in ("", "  ", None):
            with self.subTest(content=content), self.assertRaises(ValueError):
                RunResult(content)
        self.assertEqual(RunResult("  preserved\n").content, "  preserved\n")


class ExecutorLifespanTests(unittest.IsolatedAsyncioTestCase):
    """Replace production dependencies with mocks to check executor startup and final cleanup."""

    async def test_application_lifespan_stops_executor_even_when_its_body_fails(self) -> None:
        """Start the mocked executor and still stop it when the serving context raises an error."""
        with patch.dict(
            os.environ, {"POSTGRESQL_URL": "postgresql+asyncpg://unused:unused@127.0.0.1:1/unused"}
        ):
            from src.backend import app as application

        fake_executor = MagicMock()
        fake_executor.start = AsyncMock()
        fake_executor.stop = AsyncMock()
        fake_session_context = MagicMock()
        fake_session_context.__aenter__ = AsyncMock(return_value=object())
        fake_session_context.__aexit__ = AsyncMock(return_value=False)
        fake_model_service = MagicMock()
        fake_model_service.sync_local_models = AsyncMock()
        test_app = FastAPI()
        with (
            patch.object(application, "upgrade_database_schema", AsyncMock()),
            patch.object(application, "remove_orphaned_upload_batches", AsyncMock()),
            patch.object(application.os, "makedirs"),
            patch.object(application.os, "listdir", return_value=["test.gguf"]),
            patch.object(application, "async_session_maker", return_value=fake_session_context),
            patch.object(application, "ModelService", return_value=fake_model_service),
            patch.object(application, "get_model_manager"),
            patch.object(application, "get_vector_db"),
            patch.object(application, "get_ai"),
            patch.object(application, "RunExecutor", return_value=fake_executor),
            patch.object(
                application.ExecutorSettings, "from_environment", return_value=ExecutorSettings()
            ),
            patch("builtins.print"),
        ):
            with self.assertRaisesRegex(RuntimeError, "lifespan body failure"):
                async with application.lifespan(test_app):
                    self.assertIs(test_app.state.run_executor, fake_executor)
                    fake_executor.start.assert_awaited_once()
                    raise RuntimeError("lifespan body failure")
        fake_executor.stop.assert_awaited_once()
