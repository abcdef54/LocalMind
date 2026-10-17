"""Save one press of Send as a submission, user message, AI run and optional files.

Resolve retries by the browser's client_submission_id. Move validated files
before committing DB records. If commit outcome is uncertain, keep those files
until a later database check determines whether their submission was saved.
"""

import hashlib
import json
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.api.errors import ApiError
from src.backend.execution.events import update_state_and_add_run_event
from src.backend.api.v1.schemas import DocumentResponse, SubmissionInput, SubmissionResponse
from src.backend.database.models import (
    ACTIVE_RUN_STATUSES,
    Attachment,
    Conversation,
    Message,
    Run,
    Submission,
)
from src.backend.services.conversation_history import get_conversation
from src.backend.uploads.storage import SubmissionFileBatch, run_file_operation


def validate_submission_files(
    submission_payload: SubmissionInput, file_batch: SubmissionFileBatch
) -> None:
    """Check that the received files match the ordered browser manifest.

    submission_payload contains text/settings and declared file names/sizes/IDs;
    file_batch contains actual names and byte counts. Require text or at least one
    file. Return None after assigning each matched file its client_file_id.
    Raise ApiError 422 for count/order/name/size mismatches or repeated client IDs.
    Do not hash file bytes, move files or write database records.
    """
    if not submission_payload.content.strip() and not file_batch.received_files:
        raise ApiError(422, "EMPTY_SUBMISSION", "Send text or at least one file")
    if len(submission_payload.files_manifest) != len(file_batch.received_files):
        raise ApiError(422, "FILE_MANIFEST_MISMATCH", "File count does not match the manifest")
    seen_client_file_ids: set[str] = set()
    for manifest_entry, received_file in zip(
        submission_payload.files_manifest, file_batch.received_files
    ):
        if (
            manifest_entry.name != received_file.file_name
            or manifest_entry.size_bytes != received_file.size_bytes
        ):
            raise ApiError(
                422,
                "FILE_MANIFEST_MISMATCH",
                "File name, size or order does not match the manifest",
            )
        if manifest_entry.client_file_id in seen_client_file_ids:
            raise ApiError(422, "DUPLICATE_CLIENT_FILE_ID", "Client file IDs must be unique")
        seen_client_file_ids.add(manifest_entry.client_file_id)
        received_file.client_file_id = manifest_entry.client_file_id


def build_submission_fingerprint(submission_payload: SubmissionInput) -> str:
    """Return a stable 64-character hash of the send's small JSON metadata.

    submission_payload supplies text, settings, client key and ordered file manifest.
    Use this to detect changed metadata on retry. File bytes are not read or hashed;
    changed contents with the same name/size/key return the original send on retry.
    No database or filesystem operation is performed.
    """
    canonical_payload_json = json.dumps(
        submission_payload.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical_payload_json.encode("utf-8")).hexdigest()


async def build_submission_response(
    session: AsyncSession, submission: Submission
) -> SubmissionResponse:
    """Read a saved send's run and documents and return their public IDs/states.

    session must see the supplied submission and any newly flushed records. Return
    SubmissionResponse with its user-message/run IDs and documents ordered by ID,
    including deleted markers, without disk paths. Do not commit here. Database
    errors propagate; a missing run raises NoResultFound because acceptance must create one.
    """
    run = (await session.scalars(select(Run).where(Run.submission_id == submission.id))).one()
    documents = (
        await session.scalars(
            select(Attachment)
            .where(Attachment.submission_id == submission.id)
            .order_by(Attachment.id)
        )
    ).all()
    return SubmissionResponse(
        submission_id=submission.id,
        client_submission_id=submission.client_submission_id,
        conversation_id=submission.conversation_id,
        message_id=run.message_id,
        run_id=run.id,
        status=submission.status,
        error=submission.error,
        documents=[
            DocumentResponse(
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
            for document in documents
        ],
    )


async def find_submission_by_client_id(
    session: AsyncSession, conversation_id: UUID, client_submission_id: UUID
) -> SubmissionResponse:
    """Find a saved send when the browser did not receive its original response.

    Use session to search client_submission_id only within conversation_id. Return
    its original server IDs and current states. Raise ApiError 404 if the conversation
    is missing/deleted or no committed send exists under that key at lookup time.
    Do not resend files, create rows or commit changes.
    """
    await get_conversation(session, conversation_id)
    submission = await session.scalar(
        select(Submission).where(
            Submission.conversation_id == conversation_id,
            Submission.client_submission_id == client_submission_id,
        )
    )
    if submission is None:
        raise ApiError(404, "SUBMISSION_NOT_FOUND", "Submission not found")
    return await build_submission_response(session, submission)


async def _check_new_submission_allowed(
    session: AsyncSession, conversation_id: UUID, file_names: list[str]
) -> None:
    """Reject a new send if its chat has an active run or already uses one of its filenames.

    session must hold the conversation row lock. conversation_id selects the chat;
    file_names contains complete original filenames, or is empty for text-only.
    Return None if allowed; raise ApiError 409 RUN_ACTIVE or DUPLICATE_FILE_NAME otherwise.
    Include undeleted legacy documents in name checks. Resolve an existing retry key
    before calling this, since the original send's own run/files must not block its retry.
    """
    active_run_id = await session.scalar(
        select(Run.id)
        .where(Run.conversation_id == conversation_id, Run.status.in_(ACTIVE_RUN_STATUSES))
        .limit(1)
    )
    if active_run_id is not None:
        raise ApiError(409, "RUN_ACTIVE", "This conversation already has an active run")
    if not file_names:
        return
    existing_document = (
        await session.execute(
            select(Attachment.id, Attachment.file_name)
            .where(
                Attachment.conversation_id == conversation_id,
                Attachment.deleted_at.is_(None),
                Attachment.file_name.in_(file_names),
            )
            .limit(1)
        )
    ).first()
    if existing_document is not None:
        raise ApiError(
            409,
            "DUPLICATE_FILE_NAME",
            "This conversation already has a file with this name",
            {"file_name": existing_document.file_name, "document_id": str(existing_document.id)},
        )


async def _create_submission_records(
    session: AsyncSession,
    conversation_id: UUID,
    submission_payload: SubmissionInput,
    file_batch: SubmissionFileBatch,
    submission_fingerprint: str,
) -> Submission:
    """Insert a send, its user message/run and attachment rows without committing.

    Use session's open transaction after conversation/admission checks. Read text and
    settings from submission_payload, IDs/validated file metadata from file_batch,
    and save submission_fingerprint for retries. Return the flushed Submission;
    the related rows are also flushed but not yet committed.

    File sends start accepted with a pending run/documents; text-only sends start
    ready with a queued run. Propagate DB errors so the caller rolls back. Do not
    move files, index documents or call the model here.
    """
    settings_snapshot = submission_payload.settings.model_dump(mode="json")
    has_files = bool(file_batch.received_files)
    submission = Submission(
        id=file_batch.submission_id,
        conversation_id=conversation_id,
        client_submission_id=submission_payload.client_submission_id,
        fingerprint=submission_fingerprint,
        settings_snapshot=settings_snapshot,
        status="accepted" if has_files else "ready",
    )
    session.add(submission)
    await session.flush()
    user_message = Message(
        id=uuid4(),
        conversation_id=conversation_id,
        submission_id=submission.id,
        content=submission_payload.content,
        role="user",
    )
    session.add(user_message)
    await session.flush()
    await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(
            message_count=Conversation.message_count + 1,
            unsummarized_message_count=Conversation.unsummarized_message_count + 1,
        )
    )
    session.add(
        Run(
            id=uuid4(),
            conversation_id=conversation_id,
            submission_id=submission.id,
            message_id=user_message.id,
            settings_snapshot=settings_snapshot,
            status="pending" if has_files else "queued",
        )
    )
    for received_file in file_batch.received_files:
        storage_key = file_batch.get_storage_key(received_file.attachment_id)
        session.add(
            Attachment(
                id=received_file.attachment_id,
                conversation_id=conversation_id,
                submission_id=submission.id,
                message_id=user_message.id,
                client_file_id=received_file.client_file_id,
                file_name=received_file.file_name,
                file_type=received_file.mime_type,
                file_size=received_file.size_bytes,
                storage_key=storage_key,
                file_path=str(file_batch.storage_root / storage_key),
                status="pending",
            )
        )
    await session.flush()
    run = await session.scalar(select(Run).where(Run.submission_id == submission.id))
    await update_state_and_add_run_event(
        session,
        run,
        "run.pending" if has_files else "run.queued",
        {"submission_id": str(submission.id)},
    )
    return submission


async def accept_submission(
    session: AsyncSession,
    conversation_id: UUID,
    submission_payload: SubmissionInput,
    file_batch: SubmissionFileBatch,
) -> SubmissionResponse:
    """Save a complete Send request once, or return the original send on retry.

    session must have no open transaction. conversation_id selects the live chat;
    submission_payload is validated JSON and file_batch holds received temporary files.
    Check the manifest, lock the chat, resolve the client key, check active runs/names,
    insert records, move files to permanent storage and commit. Return the saved IDs
    and current states only after commit; matching retries return the original IDs.

    Raise ApiError for invalid files (422), missing chat (404) or conflicts (409),
    and propagate disk/DB errors. Normally leave failed batches for request cleanup.
    Keep moved files if commit succeeded or its outcome is uncertain. The route must
    always call batch cleanup after this function. No indexing or AI runs here.
    """
    validate_submission_files(submission_payload, file_batch)
    submission_fingerprint = build_submission_fingerprint(submission_payload)
    commit_attempt_started = False
    try:
        async with session.begin():
            # Serialize sends and archive/delete for this conversation. Resolve
            # retries first so their existing run/files do not cause a false conflict.
            await get_conversation(session, conversation_id, lock=True)
            existing_submission = await session.scalar(
                select(Submission).where(
                    Submission.conversation_id == conversation_id,
                    Submission.client_submission_id == submission_payload.client_submission_id,
                )
            )
            if existing_submission is not None:
                if existing_submission.fingerprint != submission_fingerprint:
                    raise ApiError(
                        409,
                        "IDEMPOTENCY_CONFLICT",
                        "This submission key was used with a different payload",
                    )
                return await build_submission_response(session, existing_submission)
            file_names = [received_file.file_name for received_file in file_batch.received_files]
            await _check_new_submission_allowed(session, conversation_id, file_names)
            submission = await _create_submission_records(
                session, conversation_id, submission_payload, file_batch, submission_fingerprint
            )
            if file_batch.received_files:
                await run_file_operation(file_batch.move_to_permanent_storage)
            accepted_submission = await build_submission_response(session, submission)
            # Exiting session.begin() attempts commit. A connection failure here
            # cannot tell us whether PostgreSQL committed; preserve files in that case.
            commit_attempt_started = True
        file_batch.keep_files_after_request = bool(file_batch.received_files)
        return accepted_submission
    except BaseException:
        # Cancellation also enters here, so it cannot bypass the uncertain-commit rule.
        if commit_attempt_started and file_batch.moved_to_permanent_storage:
            file_batch.keep_files_after_request = True
        raise
