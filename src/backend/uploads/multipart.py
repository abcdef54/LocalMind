"""Read upload bytes in chunks and reject size limits before writing excess bytes.

The multipart parser calls the receiver for each part's headers and data. Keep
only the small JSON/header fields in memory; write file bytes directly to the
batch's temporary folder. Do not hash, parse documents or insert DB records here.
"""

import asyncio
import mimetypes
import os
from time import monotonic
from typing import BinaryIO
from uuid import uuid4

from fastapi import Request
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header
from starlette.requests import ClientDisconnect

from src.backend.api.errors import ApiError
from src.backend.uploads.config import (
    MAX_MULTIPART_HEADER_BYTES,
    UploadSettings,
    validate_file_name,
)
from src.backend.uploads.storage import ReceivedUploadFile, SubmissionFileBatch, run_file_operation

PARSER_CHUNK_BYTES = 64 * 1024
MAX_HEADERS_PER_PART = 16
MAX_BOUNDARY_BYTES = 200


class SubmissionMultipartReceiver:
    """Write file parts and collect the JSON payload from one multipart request.

    file_batch must have its temporary folder open and locked. settings supplies
    byte/count limits. Append received file metadata in arrival order and count
    actual bytes, rather than trusting declared sizes.

    The parser calls on_part_begin, header callbacks, on_headers_finished,
    on_part_data and on_part_end for each part; on_end marks the final boundary.
    Data callbacks receive chunk bytes plus start/end slice offsets. These callback
    names come from python-multipart. File-writing callbacks run in a worker thread.
    """

    def __init__(self, file_batch: SubmissionFileBatch, settings: UploadSettings) -> None:
        """Prepare empty header/JSON buffers and file counters for file_batch.

        Save settings without creating a file. payload_seen/body_complete start False;
        files open only after their part headers pass validation.
        """
        self.file_batch = file_batch
        self.settings = settings
        self.payload_bytes = bytearray()
        self.payload_seen = False
        self.body_complete = False
        self.total_file_bytes = 0
        self.current_file_handle: BinaryIO | None = None
        self.current_received_file: ReceivedUploadFile | None = None
        self.current_field_name = ""
        self.header_name_bytes = bytearray()
        self.header_value_bytes = bytearray()
        self.part_headers: dict[bytes, bytes] = {}
        self.part_header_bytes = 0

    def on_part_begin(self) -> None:
        """Clear the current part's headers/file metadata before its headers arrive.

        Keep the already received payload, file list and combined byte count.
        """
        self.part_headers = {}
        self.part_header_bytes = 0
        self.header_name_bytes.clear()
        self.header_value_bytes.clear()
        self.current_field_name = ""
        self.current_received_file = None

    def _append_header_fragment(
        self, header_buffer: bytearray, chunk: bytes, start: int, end: int
    ) -> None:
        """Append chunk[start:end] to the supplied header buffer.

        Count all header-name/value bytes in this part. Raise ApiError 413 before
        appending a fragment that exceeds the per-part header byte limit.
        """
        self.part_header_bytes += end - start
        if self.part_header_bytes > MAX_MULTIPART_HEADER_BYTES:
            raise ApiError(413, "UPLOAD_TOO_LARGE", "Multipart headers are too large")
        header_buffer.extend(chunk[start:end])

    def on_header_field(self, chunk: bytes, start: int, end: int) -> None:
        """Append chunk[start:end] to the current header name using the part's byte limit.

        A header name can arrive in several fragments; this callback does not store
        the completed header until on_header_end runs.
        """
        self._append_header_fragment(self.header_name_bytes, chunk, start, end)

    def on_header_value(self, chunk: bytes, start: int, end: int) -> None:
        """Append chunk[start:end] to the current header value using the part's byte limit."""
        self._append_header_fragment(self.header_value_bytes, chunk, start, end)

    def on_header_end(self) -> None:
        """Save the completed header and clear its fragment buffers.

        Compare names in lowercase. Raise ApiError 422 for a repeated header name or
        excessive header count. Return None; no file is opened yet.
        """
        header_name = bytes(self.header_name_bytes).lower()
        if header_name in self.part_headers or len(self.part_headers) >= MAX_HEADERS_PER_PART:
            raise ApiError(422, "INVALID_MULTIPART", "Duplicate or excessive multipart headers")
        self.part_headers[header_name] = bytes(self.header_value_bytes)
        self.header_name_bytes.clear()
        self.header_value_bytes.clear()

    def on_headers_finished(self) -> None:
        """Check which field this part contains and open a temporary file when appropriate.

        Allow one payload field without a filename, or repeated files fields with names.
        Check file count, extension and duplicate full names before opening a server-UUID
        filename. Save the original name only as metadata; infer its display MIME label.

        Return None. Raise ApiError 422 for invalid fields/repeated payload, 413 for too
        many files, 415 for unsupported extensions, or 409 for a duplicate batch filename.
        File creation errors propagate; the caller then rejects and cleans the whole batch.
        """
        disposition, disposition_parameters = parse_options_header(
            self.part_headers.get(b"content-disposition", b"")
        )
        if disposition != b"form-data":
            raise ApiError(422, "INVALID_MULTIPART", "Expected form-data disposition")
        self.current_field_name = disposition_parameters.get(b"name", b"").decode("utf-8")
        if self.current_field_name == "payload" and b"filename" not in disposition_parameters:
            if self.payload_seen:
                raise ApiError(422, "INVALID_MULTIPART", "Only one payload field is allowed")
            self.payload_seen = True
            return
        if self.current_field_name != "files" or b"filename" not in disposition_parameters:
            raise ApiError(422, "INVALID_MULTIPART", "Expected payload or files field")
        if len(self.file_batch.received_files) >= self.settings.max_files:
            raise ApiError(413, "TOO_MANY_FILES", "Too many files in this submission")
        file_name = validate_file_name(disposition_parameters[b"filename"].decode("utf-8"))
        if any(
            received_file.file_name == file_name for received_file in self.file_batch.received_files
        ):
            raise ApiError(
                409,
                "DUPLICATE_FILE_NAME",
                "Duplicate filename in this batch",
                {"file_name": file_name},
            )
        # A client MIME header cannot prove the actual format. Content validation
        # happens during indexing; here we store only an extension-derived label.
        mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        self.current_received_file = ReceivedUploadFile(uuid4(), file_name, mime_type)
        self.file_batch.received_files.append(self.current_received_file)
        temporary_file_path = self.file_batch.current_directory / str(
            self.current_received_file.attachment_id
        )
        self.current_file_handle = temporary_file_path.open("xb")

    def on_part_data(self, chunk: bytes, start: int, end: int) -> None:
        """Store chunk[start:end] in the JSON buffer or the currently open file.

        Count the fragment's actual bytes first. Raise ApiError 413 before buffering or
        writing bytes that cross the JSON, file or combined-file limit. Update file size
        as fragments arrive. Return None; disk write errors propagate to request cleanup.
        """
        fragment_size_bytes = end - start
        if self.current_field_name == "payload":
            if len(self.payload_bytes) + fragment_size_bytes > self.settings.max_payload_bytes:
                raise ApiError(413, "UPLOAD_TOO_LARGE", "Submission payload is too large")
            self.payload_bytes.extend(chunk[start:end])
            return
        # The parser always calls on_headers_finished before delivering file data.
        self.current_received_file.size_bytes += fragment_size_bytes
        self.total_file_bytes += fragment_size_bytes
        if (
            self.current_received_file.size_bytes > self.settings.max_file_bytes
            or self.total_file_bytes > self.settings.max_batch_bytes
        ):
            raise ApiError(413, "UPLOAD_TOO_LARGE", "File or batch exceeds upload limits")
        self.current_file_handle.write(chunk[start:end])

    def on_part_end(self) -> None:
        """Flush, sync to disk and close a completed file; do nothing for a JSON-only part.

        File IO errors propagate, causing rejection of the batch. close_open_file will
        close any remaining handle before the request deletes unaccepted files.
        """
        if self.current_file_handle is not None:
            self.current_file_handle.flush()
            os.fsync(self.current_file_handle.fileno())
            self.current_file_handle.close()
            self.current_file_handle = None

    def on_end(self) -> None:
        """Record the final multipart boundary so an incomplete request cannot be accepted."""
        self.body_complete = True

    def close_open_file(self) -> None:
        """Close any unfinished file after receiving succeeds, fails or is cancelled.

        Do not delete files here; SubmissionFileBatch decides that when the request ends.
        """
        if self.current_file_handle is not None:
            self.current_file_handle.close()
            self.current_file_handle = None


def validate_multipart_headers(request: Request, settings: UploadSettings) -> bytes:
    """Check upload headers and return the declared multipart boundary bytes.

    request supplies Content-Type and optional Content-Length; settings supplies
    the request byte limit. Do not read the body. Raise ApiError 415 for an invalid
    content type/boundary, 422 for invalid length, or 413 for a declared oversized body.
    A small or missing Content-Length does not bypass actual byte counting later.
    """
    media_type, content_type_parameters = parse_options_header(
        request.headers.get("content-type", "")
    )
    boundary = content_type_parameters.get(b"boundary", b"")
    if media_type != b"multipart/form-data" or not boundary or len(boundary) > MAX_BOUNDARY_BYTES:
        raise ApiError(415, "INVALID_MULTIPART", "Expected multipart/form-data with a boundary")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_request_bytes = int(content_length)
        except ValueError:
            raise ApiError(422, "INVALID_CONTENT_LENGTH", "Invalid Content-Length") from None
        if declared_request_bytes < 0:
            raise ApiError(422, "INVALID_CONTENT_LENGTH", "Invalid Content-Length")
        if declared_request_bytes > settings.max_request_bytes:
            raise ApiError(413, "UPLOAD_TOO_LARGE", "Request exceeds upload limits")
    return boundary


async def receive_multipart_submission(
    request: Request, file_batch: SubmissionFileBatch, settings: UploadSettings
) -> bytes:
    """Read a request into temporary files and return the raw JSON payload bytes.

    request must have an unread multipart body. file_batch must already be open and
    locked; append its received_files metadata in upload order. settings limits byte
    counts and receiving time. The caller must validate JSON/manifest and save DB rows.

    Raise ApiError on disconnect (400), timeout (408), duplicate name (409), size/count
    limits (413), file type (415), or invalid/incomplete multipart data (422). Propagate
    disk errors. Always close the current file; leave batch deletion to the caller.
    Do not hash file contents or load the whole batch into memory.
    """
    boundary = validate_multipart_headers(request, settings)
    receiver = SubmissionMultipartReceiver(file_batch, settings)
    callbacks = {
        callback_name: getattr(receiver, callback_name)
        for callback_name in (
            "on_part_begin",
            "on_header_field",
            "on_header_value",
            "on_header_end",
            "on_headers_finished",
            "on_part_data",
            "on_part_end",
            "on_end",
        )
    }
    parser = MultipartParser(boundary, callbacks)
    received_request_bytes = 0
    request_deadline = monotonic() + settings.request_timeout_seconds
    request_chunks = request.stream().__aiter__()
    try:
        while True:
            remaining_seconds = request_deadline - monotonic()
            if remaining_seconds <= 0:
                raise asyncio.TimeoutError
            try:
                request_chunk = await asyncio.wait_for(
                    request_chunks.__anext__(),
                    min(remaining_seconds, settings.idle_timeout_seconds),
                )
            except StopAsyncIteration:
                break
            received_request_bytes += len(request_chunk)
            if received_request_bytes > settings.max_request_bytes:
                raise ApiError(413, "UPLOAD_TOO_LARGE", "Request exceeds upload limits")
            # The HTTP server can deliver a large chunk. Smaller parser writes
            # bound temporary copies and keep blocking file IO off the event loop.
            for chunk_offset in range(0, len(request_chunk), PARSER_CHUNK_BYTES):
                parser_chunk = request_chunk[chunk_offset : chunk_offset + PARSER_CHUNK_BYTES]
                await run_file_operation(parser.write, parser_chunk)
        if not receiver.body_complete or not receiver.payload_seen:
            raise ApiError(422, "INVALID_MULTIPART", "Incomplete multipart body or missing payload")
        return bytes(receiver.payload_bytes)
    except (MultipartParseError, UnicodeDecodeError, ValueError):
        raise ApiError(422, "INVALID_MULTIPART", "Malformed multipart body") from None
    except asyncio.TimeoutError:
        raise ApiError(408, "UPLOAD_TIMEOUT", "Upload timed out") from None
    except ClientDisconnect:
        raise ApiError(400, "UPLOAD_DISCONNECTED", "Upload connection closed") from None
    finally:
        await run_file_operation(receiver.close_open_file)
