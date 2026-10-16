"""Accept one Send request with text and multiple files, and let clients look it up later."""

import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import ValidationError

from src.backend.api.errors import ApiError, ErrorResponse
from src.backend.api.v1.conversations import SessionDependency
from src.backend.api.v1.schemas import SubmissionInput, SubmissionResponse
from src.backend.services import submissions
from src.backend.services.conversation_history import get_conversation
from src.backend.uploads.config import UploadSettings
from src.backend.uploads.multipart import receive_multipart_submission
from src.backend.uploads.storage import SubmissionFileBatch, run_file_operation

router = APIRouter(prefix="/conversations/{conversation_id}/submissions", tags=["submissions"])


def get_upload_settings(request: Request) -> UploadSettings:
    """Return the app's upload limits, reading UPLOAD_* environment overrides once.

    request identifies the v1 app whose state caches the settings. Return validated
    UploadSettings or raise ValueError for invalid configuration. Create no files/DB rows.
    """
    settings = getattr(request.app.state, "upload_settings", None)
    if settings is None:
        settings = UploadSettings.from_environment()
        request.app.state.upload_settings = settings
    return settings


def parse_submission_payload(
    payload_json_bytes: bytes, idempotency_header: str | None
) -> SubmissionInput:
    """Read the multipart JSON field and check its optional Idempotency-Key header.

    payload_json_bytes contains text/settings/file declarations. Return SubmissionInput
    if valid; idempotency_header must match its client_submission_id when supplied.
    Raise ApiError 422 on invalid JSON/fields/settings or a mismatched key. Error details
    show field locations/codes without copying raw submitted contents.
    """
    try:
        submission_payload = SubmissionInput.model_validate_json(payload_json_bytes)
    except ValidationError as error:
        raise ApiError(
            422,
            "VALIDATION_ERROR",
            "Invalid submission payload",
            {
                "violations": [
                    {"location": list(violation["loc"]), "code": violation["type"]}
                    for violation in error.errors()
                ]
            },
        ) from None
    if idempotency_header is not None and idempotency_header != str(
        submission_payload.client_submission_id
    ):
        raise ApiError(
            422, "IDEMPOTENCY_KEY_MISMATCH", "Idempotency-Key must match client_submission_id"
        )
    return submission_payload


UploadSettingsDependency = Annotated[UploadSettings, Depends(get_upload_settings)]


@router.post(
    "",
    response_model=SubmissionResponse,
    status_code=202,
    responses={code: {"model": ErrorResponse} for code in (400, 404, 408, 409, 413, 415, 429)},
    openapi_extra={
        "parameters": [
            {
                "name": "Idempotency-Key",
                "in": "header",
                "required": False,
                "schema": {"type": "string", "format": "uuid"},
                "description": "Must match payload.client_submission_id when supplied",
            }
        ],
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["payload"],
                        "properties": {
                            "payload": {
                                "type": "string",
                                "description": "JSON-encoded SubmissionInput",
                            },
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            },
                        },
                    }
                }
            },
        },
    },
)
async def create_submission(
    conversation_id: UUID,
    request: Request,
    session: SessionDependency,
    settings: UploadSettingsDependency,
) -> SubmissionResponse:
    """Receive files and text for one Send action and return HTTP 202 after saving them.

    conversation_id selects the chat, request supplies multipart bytes, session is the
    request's DB session, and settings supplies receiving/storage limits. Reserve an
    upload slot, check the chat, receive into a temporary batch, validate JSON and call
    the acceptance service. Return saved send/message/run IDs and document states.

    Reject missing chats (404), busy receivers (429) and invalid uploads according to
    the upload error contract. Always release the file lock and delete unaccepted
    files. Keep files saved by the service or awaiting confirmation of a DB commit.
    Do not wait for document indexing or an assistant answer.
    """
    upload_slots = getattr(request.app.state, "upload_slots", None)
    if upload_slots is None:
        upload_slots = asyncio.Semaphore(settings.max_parallel_uploads)
        request.app.state.upload_slots = upload_slots
    if upload_slots.locked():
        raise ApiError(
            429, "UPLOAD_BUSY", "Too many concurrent uploads", headers={"Retry-After": "1"}
        )
    async with upload_slots:
        async with session.begin():
            await get_conversation(session, conversation_id)
        file_batch = SubmissionFileBatch(settings.root)
        try:
            await run_file_operation(file_batch.create_staging_directory)
            payload_json_bytes = await receive_multipart_submission(request, file_batch, settings)
            submission_payload = parse_submission_payload(
                payload_json_bytes, request.headers.get("idempotency-key")
            )
            return await submissions.accept_submission(
                session, conversation_id, submission_payload, file_batch
            )
        finally:
            await run_file_operation(file_batch.cleanup_and_release_lock)


@router.get("/by-client-id/{client_submission_id}", response_model=SubmissionResponse)
async def get_submission_by_client_id(
    conversation_id: UUID,
    client_submission_id: UUID,
    session: SessionDependency,
) -> SubmissionResponse:
    """Return a saved send after the browser loses its original HTTP response.

    Search client_submission_id only in conversation_id using the request session.
    Return the send/message/run IDs and current states, or ApiError 404 if the live
    conversation or committed send is missing. Do not receive files or create a new send.
    """
    return await submissions.find_submission_by_client_id(
        session, conversation_id, client_submission_id
    )
