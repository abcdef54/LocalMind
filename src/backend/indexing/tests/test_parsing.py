"""Check extraction limits and real parser-process termination without model/DB access."""

import asyncio
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from pypdf import PdfWriter

from src.backend.indexing.config import IndexingSettings
from src.backend.indexing.errors import DocumentIndexingError
from src.backend.indexing.parsing import parse_document
from src.backend.indexing.parser_process import DocumentSection, extract_sections
from src.backend.indexing.vectors import split_extracted_sections_into_document_chunks


class DocumentParsingTests(unittest.IsolatedAsyncioTestCase):
    """Use tiny local inputs to check bounds, filename dispatch and subprocess cleanup."""

    def setUp(self):
        """Create an isolated directory and small content limits for each case."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.file_path = Path(self.temporary.name) / "extensionless-uuid"
        self.settings = IndexingSettings(max_text_characters=100, chunk_size=30, chunk_overlap=5)

    async def test_extensionless_storage_uses_original_filename(self):
        """Parse saved UUID bytes as UTF-8 using the supplied original source filename."""
        self.file_path.write_text('print("hello")', encoding="utf-8")
        sections = await parse_document(self.file_path, "example.py", self.settings)
        self.assertEqual(sections, [DocumentSection('print("hello")')])
        chunks = split_extracted_sections_into_document_chunks(
            sections, "example.py", self.settings
        )
        self.assertEqual(chunks[0].metadata["source"], "example.py")
        self.assertNotIn(self.temporary.name, str(chunks[0].metadata))

    async def test_invalid_empty_and_oversized_inputs_have_safe_codes(self):
        """Reject invalid text, empty content, oversized extraction and unsupported formats."""
        cases = (
            (b"", "empty.txt", "EMPTY_DOCUMENT"),
            (b"x" * 101, "large.txt", "DOCUMENT_TOO_LARGE"),
            (b"\xff\xfe", "binary.txt", "UNREADABLE_DOCUMENT"),
            (b"hello", "file.exe", "UNSUPPORTED_FILE_TYPE"),
            (b"not a pdf", "bad.pdf", "UNREADABLE_DOCUMENT"),
        )
        for content, name, expected_code in cases:
            with self.subTest(name=name):
                self.file_path.write_bytes(content)
                with self.assertRaises(DocumentIndexingError) as caught:
                    await parse_document(self.file_path, name, self.settings)
                self.assertEqual(caught.exception.code, expected_code)
                self.assertNotIn(self.temporary.name, str(caught.exception))

    async def test_pdf_page_and_docx_expansion_limits(self):
        """Reject PDF page count and DOCX expansion before expensive text processing."""
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        writer.write(self.file_path)
        with self.assertRaises(DocumentIndexingError) as caught:
            await parse_document(self.file_path, "report.pdf", replace(self.settings, max_pages=1))
        self.assertEqual(caught.exception.code, "TOO_MANY_PAGES")
        with zipfile.ZipFile(self.file_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", "x" * 10000)
        with self.assertRaises(DocumentIndexingError) as caught:
            await parse_document(
                self.file_path, "report.docx", replace(self.settings, max_expanded_bytes=100)
            )
        self.assertEqual(caught.exception.code, "DOCUMENT_TOO_LARGE")

    async def test_chunk_limit_rejects_excessive_embedding_work(self):
        """Stop splitting when a valid text document would produce too many chunks."""
        with self.assertRaises(DocumentIndexingError) as caught:
            split_extracted_sections_into_document_chunks(
                [DocumentSection("word " * 100)], "text.txt", replace(self.settings, max_chunks=1)
            )
        self.assertEqual(caught.exception.code, "TOO_MANY_CHUNKS")

    async def test_timeout_and_cancellation_kill_parser_process(self):
        """Terminate a real sleeping child on timeout/cancellation and wait for its exit."""
        original_launch = asyncio.create_subprocess_exec
        children = []
        started = asyncio.Event()

        async def launch_sleeping_child(*args, **kwargs):
            """Replace only the launch command with a controlled sleeping subprocess."""
            child = await original_launch(
                sys.executable, "-c", "import time; time.sleep(30)", **kwargs
            )
            children.append(child)
            started.set()
            return child

        with patch(
            "src.backend.indexing.parsing.asyncio.create_subprocess_exec", launch_sleeping_child
        ):
            with self.assertRaises(DocumentIndexingError) as caught:
                await parse_document(
                    self.file_path, "text.txt", replace(self.settings, parse_timeout_seconds=0.05)
                )
            self.assertEqual(caught.exception.code, "DOCUMENT_PARSE_TIMEOUT")
            self.assertIsNotNone(children[0].returncode)
            started.clear()
            task = asyncio.create_task(parse_document(self.file_path, "text.txt", self.settings))
            await started.wait()
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertIsNotNone(children[1].returncode)

    def test_invalid_indexing_settings_fail_before_work(self):
        """Reject unsafe counts, overlaps and reservation timing before a file is parsed."""
        for overrides in (
            {"max_concurrency": 0},
            {"max_pages": True},
            {"parse_timeout_seconds": float("nan")},
            {"chunk_overlap": 1000},
            {"lease_seconds": 10},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(self.settings, **overrides)
        with patch.dict("os.environ", {"INDEXING_MAX_CHUNKS": "123"}):
            self.assertEqual(IndexingSettings.from_environment().max_chunks, 123)
