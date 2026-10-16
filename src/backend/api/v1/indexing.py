"""Retry document indexing for a failed send without uploading the files again."""

from uuid import UUID

from fastapi import APIRouter

from src.backend.api.v1.conversations import SessionDependency
from src.backend.api.v1.schemas import SubmissionResponse
from src.backend.indexing.store import retry_submission_indexing
from src.backend.services.submissions import build_submission_response

router = APIRouter(prefix="/conversations/{conversation_id}/submissions", tags=["indexing"])


@router.post("/{submission_id}/retry-indexing", response_model=SubmissionResponse, status_code=202)
async def retry_indexing(
    conversation_id: UUID,
    submission_id: UUID,
    session: SessionDependency,
) -> SubmissionResponse:
    """Queue failed document indexing again using the same send/message/file IDs.

    Args:
        conversation_id: Live chat UUID from the URL, required to own the send.
        submission_id: Failed send whose files still exist in permanent storage.
        session: Request-scoped database session used to check and commit the retry.

    Returns:
        HTTP 202 with the reset submission and current document states. Already
        ready files keep their vectors; only failed files return to pending.

    Raises:
        ApiError: 404 for missing resources, 409 for active/cancelled/non-indexing
            failures or documents still indexing/deleted. No file bytes are resent.
    """
    submission = await retry_submission_indexing(session, conversation_id, submission_id)
    return await build_submission_response(session, submission)
