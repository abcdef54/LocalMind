"""Check chunked multipart receiving, limits and cancellation using fake request events."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import Request

from src.backend.api.errors import ApiError
from src.backend.uploads.config import UploadSettings
from src.backend.uploads.multipart import receive_multipart_submission
from src.backend.uploads.storage import SubmissionFileBatch


def multipart_body(parts, closing=True):
    """Return multipart bytes for (field name, optional filename, content bytes) tuples.

    Use a fixed test boundary. Set closing=False to simulate a truncated body.
    """
    body = bytearray()
    for name, filename, content in parts:
        body.extend(f'--test-boundary\r\nContent-Disposition: form-data; name="{name}"'.encode())
        if filename is not None:
            body.extend(f'; filename="{filename}"'.encode())
        body.extend(b"\r\n\r\n" + content + b"\r\n")
    if closing:
        body.extend(b"--test-boundary--\r\n")
    return bytes(body)


class UploadStreamTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the receiver with tiny chunks and temporary folders, without HTTP servers or a
    DB.
    """

    def setUp(self):
        """Create a temporary batch with small byte limits and register file/folder cleanup."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings = UploadSettings(
            root=Path(self.temporary.name), max_file_bytes=20, max_batch_bytes=30
        )
        self.batch = SubmissionFileBatch(self.settings.root)
        self.batch.create_staging_directory()
        self.addCleanup(self.batch.cleanup_and_release_lock)
        self.received_bytes = 0

    def request(self, body, *, chunk_size=8, headers=None, disconnect=False, delay=0):
        """Return a Request that reads body in configurable chunk_size byte fragments.

        headers can simulate invalid length/type; disconnect sends an early-close event;
        delay pauses each receive call. Track bytes consumed to prove early rejection.
        """
        offset = 0

        async def receive():
            """Return the next fake ASGI body chunk, delayed chunk or disconnect event and update
            byte totals.
            """
            nonlocal offset
            if delay:
                await asyncio.sleep(delay)
            if offset == len(body) and disconnect:
                return {"type": "http.disconnect"}
            chunk = body[offset : offset + chunk_size]
            offset += len(chunk)
            self.received_bytes += len(chunk)
            return {
                "type": "http.request",
                "body": chunk,
                "more_body": offset < len(body) or disconnect,
            }

        request_headers = (
            headers
            if headers is not None
            else [
                (b"content-type", b"multipart/form-data; boundary=test-boundary"),
            ]
        )
        return Request(
            {"type": "http", "method": "POST", "path": "/", "headers": request_headers}, receive
        )

    async def test_large_file_stops_at_limit_without_receiving_the_whole_body(self):
        """Reject after the file limit and prove the rest of a much larger body was never
        consumed.
        """
        body = multipart_body([("payload", None, b"{}"), ("files", "large.txt", b"x" * 10000)])
        with self.assertRaises(ApiError) as caught:
            await receive_multipart_submission(self.request(body), self.batch, self.settings)
        self.assertEqual(caught.exception.status_code, 413)
        self.assertLess(self.received_bytes, 500)
        self.assertLessEqual(
            (self.batch.current_directory / str(self.batch.received_files[0].attachment_id))
            .stat()
            .st_size,
            20,
        )

    async def test_small_chunks_and_payload_after_files_work(self):
        """Receive a file and later JSON correctly even when body chunks are one byte long."""
        body = multipart_body(
            [("files", "file.txt", b"abc"), ("payload", None, b'{"content":"ok"}')]
        )
        payload = await receive_multipart_submission(
            self.request(body, chunk_size=1), self.batch, self.settings
        )
        self.assertEqual(json.loads(payload), {"content": "ok"})
        self.assertEqual(self.batch.received_files[0].size_bytes, 3)

    async def test_truncated_body_and_duplicate_payload_are_rejected(self):
        """Reject a missing final boundary or a second payload field instead of accepting partial
        data.
        """
        for body in (
            multipart_body([("payload", None, b"{}")], closing=False),
            multipart_body([("payload", None, b"{}"), ("payload", None, b"{}")]),
        ):
            with self.assertRaises(ApiError) as caught:
                await receive_multipart_submission(self.request(body), self.batch, self.settings)
            self.assertEqual(caught.exception.status_code, 422)

    async def test_missing_or_false_content_length_cannot_bypass_limits(self):
        """Reject actual oversized file bytes even when the Content-Length header claims only one
        byte.
        """
        body = multipart_body([("payload", None, b"{}"), ("files", "file.txt", b"x" * 50)])
        headers = [
            (b"content-type", b"multipart/form-data; boundary=test-boundary"),
            (b"content-length", b"1"),
        ]
        with self.assertRaises(ApiError) as caught:
            await receive_multipart_submission(
                self.request(body, headers=headers), self.batch, self.settings
            )
        self.assertEqual(caught.exception.status_code, 413)

    async def test_payload_size_and_request_size_are_bounded(self):
        """Limit buffered JSON and reject a declared oversized whole request before reading
        bytes.
        """
        body = multipart_body([("payload", None, b"x" * 1000)])
        settings = replace(self.settings, max_payload_bytes=20)
        with self.assertRaises(ApiError) as caught:
            await receive_multipart_submission(self.request(body), self.batch, settings)
        self.assertEqual(caught.exception.status_code, 413)
        headers = [
            (b"content-type", b"multipart/form-data; boundary=test-boundary"),
            (b"content-length", str(settings.max_request_bytes + 1).encode()),
        ]
        self.received_bytes = 0
        with self.assertRaises(ApiError):
            await receive_multipart_submission(
                self.request(body, headers=headers), self.batch, settings
            )
        self.assertEqual(self.received_bytes, 0)

    async def test_disconnect_and_timeout_close_open_files(self):
        """Return disconnect/timeout errors after partial receiving instead of accepting
        incomplete files.
        """
        body = multipart_body([("files", "file.txt", b"abc")], closing=False)
        with self.assertRaises(ApiError) as caught:
            await receive_multipart_submission(
                self.request(body, disconnect=True), self.batch, self.settings
            )
        self.assertEqual(caught.exception.code, "UPLOAD_DISCONNECTED")
        settings = replace(self.settings, idle_timeout_seconds=0.01)
        with self.assertRaises(ApiError) as caught:
            await receive_multipart_submission(self.request(b"", delay=0.1), self.batch, settings)
        self.assertEqual(caught.exception.code, "UPLOAD_TIMEOUT")

    async def test_upload_setting_validation_and_environment(self):
        """Reject invalid upload limits and read a valid byte-limit environment override."""
        for overrides in (
            {"max_files": 0},
            {"max_file_bytes": True},
            {"idle_timeout_seconds": float("nan")},
        ):
            with self.assertRaises(ValueError):
                replace(self.settings, **overrides)
        with patch.dict("os.environ", {"UPLOAD_MAX_FILE_BYTES": "123"}):
            self.assertEqual(UploadSettings.from_environment().max_file_bytes, 123)

    async def test_cancellation_waits_for_inflight_file_io_before_cleanup(self):
        """Wait for the worker-thread file operation to finish before exposing task cancellation."""
        import threading
        from src.backend.uploads.storage import run_file_operation

        started, released, finished = threading.Event(), threading.Event(), threading.Event()

        def write_file():
            """Signal that fake IO started, wait for release, then signal completion before
            cleanup may proceed.
            """
            started.set()
            released.wait(timeout=2)
            finished.set()

        task = asyncio.create_task(run_file_operation(write_file))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.01)
        self.assertFalse(task.done())
        released.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(finished.is_set())
