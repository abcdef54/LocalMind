"""Expose scoped document metadata, safe content responses and deletion."""

from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Response
from starlette.responses import StreamingResponse

from src.backend.api.errors import ApiError
from src.backend.api.v1.conversations import SessionDependency, PageLimit, BeforeCursor
from src.backend.api.v1.schemas import DocumentPage, DocumentResponse
from src.backend.services import documents
from src.backend.uploads.config import UploadSettings
from src.backend.uploads.storage import run_file_operation

router = APIRouter(prefix="/conversations/{conversation_id}/documents", tags=["documents"])


@router.get("", response_model=DocumentPage)
async def list_documents(
    conversation_id: UUID,
    session: SessionDependency,
    limit: PageLimit = 50,
    before: BeforeCursor = None,
) -> DocumentPage:
    """Return a bounded newest-first page of documents.

    Args:
        conversation_id: Live chat UUID whose undeleted documents are listed.
        session: Request-scoped read session.
        limit: Number of documents per page, from 1 to 100; defaults to 50.
        before: Previous page cursor, or None to start with the newest files.

    Returns:
        Metadata items, has_more and next_cursor; never file paths or bytes.
    """
    return await documents.list_documents(session, conversation_id, limit, before)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    conversation_id: UUID, document_id: UUID, session: SessionDependency
) -> DocumentResponse:
    """Return metadata without reading the file bytes or initializing embeddings.

    Args:
        conversation_id: Live chat UUID that must own the document.
        document_id: Undeleted file UUID whose metadata is requested.
        session: Request-scoped database read session.

    Returns:
        Public filename, byte size, indexing status and safe processing errors.

    Raises:
        ApiError: 404 for a missing, deleted or wrong-conversation document.
    """
    return documents.document_response(
        await documents.get_document(session, conversation_id, document_id)
    )


@router.get("/{document_id}/content")
async def get_document_content(
    conversation_id: UUID,
    document_id: UUID,
    session: SessionDependency,
    download: Annotated[bool, Query()] = False,
) -> StreamingResponse:
    """Stream a stored file with a safe preview type or download disposition.

    Args:
        conversation_id: Live chat UUID that must own the file.
        document_id: Undeleted attachment to read, regardless of indexing status.
        session: Request database session used for scope checks.
        download: True to force attachment download; False previews PDF/UTF-8 text.

    Returns:
        Stream from an already-open file handle, avoiding deletion-before-open races.
        HTML/source files are text/plain; DOCX always downloads. Caching is disabled.

    Raises:
        ApiError: 404 for missing/deleted files or unavailable/unsafe disk paths.
    """
    from urllib.parse import quote

    document = await documents.get_document(session, conversation_id, document_id)
    try:
        path = documents.resolve_document_path(document, UploadSettings.from_environment().root)
        handle = path.open("rb")
    except (OSError, ValueError):
        raise ApiError(
            404, "DOCUMENT_CONTENT_NOT_FOUND", "Document content is unavailable"
        ) from None
    suffix = Path(document.file_name).suffix.lower()
    if suffix == ".pdf":
        media_type = "application/pdf"
    elif suffix == ".docx":
        media_type = "application/octet-stream"
    else:
        media_type = "text/plain; charset=utf-8"
    disposition = "attachment" if download or suffix == ".docx" else "inline"

    async def chunks():
        """Yield bounded reads and close the open handle on completion or disconnect."""
        try:
            while chunk := await run_file_operation(handle.read, 64 * 1024):
                yield chunk
        finally:
            handle.close()

    return StreamingResponse(
        chunks(),
        media_type=media_type,
        headers={
            "Content-Disposition": f"{disposition}; filename*=UTF-8''{quote(document.file_name, safe='')}",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox",
        },
    )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    conversation_id: UUID, document_id: UUID, session: SessionDependency
) -> Response:
    """Hide document_id in conversation_id; commit via session before returning 204.

    Args:
        conversation_id: Live chat UUID that must own the file.
        document_id: Document to hide and queue for physical cleanup.
        session: Request database session used to save the deleted-document state.

    Returns:
        Empty HTTP 204 response after the metadata transaction commits.

    Disk/vector cleanup is asynchronous. Repeating this request is safe and returns
    204 while the conversation exists; wrong-chat and unknown IDs return 404.
    """
    await documents.delete_document(session, conversation_id, document_id)
    return Response(status_code=204)
