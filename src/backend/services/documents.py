"""List, read and hide documents belonging to one live conversation."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.api.errors import ApiError
from src.backend.execution.events import update_state_and_add_run_event
from src.backend.api.v1.pagination import HistoryCursor, decode_cursor, encode_cursor
from src.backend.api.v1.schemas import DocumentPage, DocumentResponse
from src.backend.database.models import Attachment, Run, Submission
from src.backend.services.conversation_history import get_conversation


def document_response(document: Attachment) -> DocumentResponse:
    """Convert a database document into public metadata.

    Args:
        document: Attachment row; filesystem location is deliberately omitted.

    Returns:
        Original filename, byte count, processing state and safe indexing error.
    """
    return DocumentResponse(
        id=document.id,
        client_file_id=document.client_file_id,
        file_name=document.file_name,
        file_type=document.file_type,
        file_size=document.file_size,
        status=document.status,
        deleted=document.deleted_at is not None,
        error=document.error,
        chunk_count=document.chunk_count,
    )


async def get_document(
    session: AsyncSession, conversation_id: UUID, document_id: UUID
) -> Attachment:
    """Read one undeleted file in a live chat, rejecting missing or wrong-chat IDs.

    Args:
        session: Request database session; this function does not commit.
        conversation_id: Chat UUID supplied by the server route.
        document_id: Attachment UUID that must belong to that chat.

    Returns:
        Attachment metadata, including its internal path for backend use only.

    Raises:
        ApiError: 404 if the conversation/document is missing, archived or deleted.
    """
    await get_conversation(session, conversation_id)
    document = await session.scalar(
        select(Attachment).where(
            Attachment.id == document_id,
            Attachment.conversation_id == conversation_id,
            Attachment.deleted_at.is_(None),
        )
    )
    if document is None:
        raise ApiError(404, "DOCUMENT_NOT_FOUND", "Document not found")
    return document


async def list_documents(
    session: AsyncSession, conversation_id: UUID, limit: int, before: str | None
) -> DocumentPage:
    """Read a bounded page of undeleted files, newest first.

    Args:
        session: Request read session.
        conversation_id: Live chat whose files are requested.
        limit: Maximum items to return; route accepts 1 through 100.
        before: Scoped cursor from the previous page, or None for the first page.

    Returns:
        Public metadata plus next cursor; queries at most limit + 1 rows.

    Raises:
        ApiError: 404 for missing chat, 422 for invalid or wrong-chat cursor.
    """
    await get_conversation(session, conversation_id)
    query = select(Attachment).where(
        Attachment.conversation_id == conversation_id, Attachment.deleted_at.is_(None)
    )
    if before is not None:
        cursor = decode_cursor(before, resource="documents", conversation_id=conversation_id)
        timestamp = cursor.created_at.replace(tzinfo=None)
        query = query.where(tuple_(Attachment.created_at, Attachment.id) < (timestamp, cursor.id))
    rows = list(
        (
            await session.scalars(
                query.order_by(Attachment.created_at.desc(), Attachment.id.desc()).limit(limit + 1)
            )
        ).all()
    )
    has_more = len(rows) > limit
    documents = rows[:limit]
    next_cursor = None
    if has_more:
        last = documents[-1]
        next_cursor = encode_cursor(
            HistoryCursor(
                resource="documents",
                conversation_id=conversation_id,
                created_at=last.created_at.replace(tzinfo=timezone.utc),
                id=last.id,
            )
        )
    return DocumentPage(
        items=[document_response(row) for row in documents],
        next_cursor=next_cursor,
        has_more=has_more,
    )


def resolve_document_path(document: Attachment, storage_root: Path) -> Path:
    """Resolve a saved path within managed storage, rejecting symlink escapes.

    Args:
        document: Saved attachment with server-generated storage_key or legacy path.
        storage_root: Configured upload root; legacy paths must be inside UPLOAD_DIR.

    Returns:
        Resolved absolute path, which may no longer exist.

    Raises:
        ValueError: If a key/path escapes the allowed root or targets a directory.
    """
    from src.backend.constants import UPLOAD_DIR

    allowed_root = storage_root if document.storage_key else UPLOAD_DIR
    candidate = (
        allowed_root / document.storage_key if document.storage_key else Path(document.file_path)
    )
    resolved = candidate.resolve()
    if not resolved.is_relative_to(allowed_root.resolve()) or resolved == allowed_root.resolve():
        raise ValueError("Document path outside managed storage")
    if resolved.exists() and not resolved.is_file():
        raise ValueError("Document path is not a regular file")
    return resolved


async def delete_document(session: AsyncSession, conversation_id: UUID, document_id: UUID) -> None:
    """Mark a document as deleted in PostgreSQL so retrieval stops before physical cleanup.

    Args:
        session: Request session with no transaction already open.
        conversation_id: Live chat to lock before changing its document.
        document_id: File UUID in that chat; repeated deletion is a successful no-op.

    Pending/queued sends containing this file become failed. Running inference is
    not reported as stopped; retrieval excludes the file on its next DB check.
    Citations and file metadata remain. Cleanup later removes bytes and vectors.

    Raises:
        ApiError: 404 if chat or matching document does not exist.
    """
    async with session.begin():
        await get_conversation(session, conversation_id, lock=True)
        document = await session.scalar(
            select(Attachment)
            .where(Attachment.id == document_id, Attachment.conversation_id == conversation_id)
            .with_for_update()
        )
        if document is None:
            raise ApiError(404, "DOCUMENT_NOT_FOUND", "Document not found")
        if document.deleted_at is not None:
            return
        document.deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)
        document.status = "deleted"
        document.index_owner = None
        document.index_expires_at = None
        document.extracted_text = None
        if document.submission_id is not None:
            run = await session.scalar(
                select(Run).where(Run.submission_id == document.submission_id).with_for_update()
            )
            if run is not None and run.status in ("pending", "queued"):
                error = {"code": "DOCUMENT_DELETED", "message": "An attached document was deleted."}
                run.status = "failed"
                run.error = error
                run.finished_at = datetime.now(timezone.utc)
                submission = await session.get(Submission, document.submission_id)
                submission.status = "failed"
                submission.error = error
                await update_state_and_add_run_event(session, run, "run.failed", {"error": error})
