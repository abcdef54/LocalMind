"""Run bounded document extraction outside the backend process."""

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from src.backend.indexing.config import IndexingSettings
from src.backend.indexing.errors import DocumentIndexingError
from src.backend.indexing.parser_process import DocumentSection


async def parse_document(
    file_path: Path, file_name: str, settings: IndexingSettings
) -> list[DocumentSection]:
    """Extract text in a child process and terminate it on timeout or cancellation.

    Args:
        file_path: Saved attachment path, including extensionless UUID filenames.
        file_name: Original filename that determines which reader to use.
        settings: Extraction limits, subprocess memory limit and timeout seconds.

    Returns:
        Nonempty sections with text and optional PDF page numbers.

    Raises:
        DocumentIndexingError: If the reader fails, exceeds limits or times out.
        asyncio.CancelledError: After killing/waiting for the parser when cancelled.
    """
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "src.backend.indexing.parser_process",
        cwd=Path(__file__).resolve().parents[3],
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    request = json.dumps(
        {
            "file_path": str(file_path),
            "file_name": file_name,
            "settings": asdict(settings),
            "parent_pid": os.getpid(),
        }
    ).encode()
    try:
        try:
            output, _ = await asyncio.wait_for(
                process.communicate(request), settings.parse_timeout_seconds
            )
        except asyncio.TimeoutError:
            raise DocumentIndexingError("DOCUMENT_PARSE_TIMEOUT") from None
        if process.returncode != 0:
            raise DocumentIndexingError("DOCUMENT_PARSE_FAILED")
        response = json.loads(output)
        if "error" in response:
            raise DocumentIndexingError(response["error"])
        return [DocumentSection(**section) for section in response["sections"]]
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
