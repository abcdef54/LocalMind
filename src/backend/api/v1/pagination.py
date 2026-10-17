"""Encode and check the timestamp/ID used to request the next older page."""

import base64
import binascii
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from src.backend.api.errors import ApiError


class HistoryCursor(BaseModel):
    """Store where a conversation or message/document page should continue.

    created_at and id identify the oldest item already returned. resource prevents
    using a conversation-list cursor for messages; conversation_id restricts message
    cursors to one chat. This contains pagination data, not access permission.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    resource: Literal["conversations", "messages", "documents"]
    conversation_id: UUID | None = None
    created_at: datetime
    id: UUID

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, timestamp: datetime) -> datetime:
        """Return the supplied timestamp converted to UTC.

        Raise ValueError if it has no timezone, since a cursor must identify the same
        point in time on every request. Do not change the represented instant.
        """
        if timestamp.tzinfo is None:
            raise ValueError("Cursor timestamp must include a timezone")
        return timestamp.astimezone(timezone.utc)


def encode_cursor(cursor: HistoryCursor) -> str:
    """Return the cursor's JSON as URL-safe base64 text without padding.

    cursor must already be validated. The result is suitable for a before query
    parameter, but it is not encrypted, signed or an authorization token.
    """
    return base64.urlsafe_b64encode(cursor.model_dump_json().encode()).decode().rstrip("=")


def decode_cursor(
    token: str,
    *,
    resource: Literal["conversations", "messages", "documents"],
    conversation_id: UUID | None = None,
) -> HistoryCursor:
    """Read a before token and verify it belongs to the requested list/chat.

    token is base64 JSON; resource must match conversations, messages or documents, and
    conversation_id must match the expected chat (None for the conversation list).
    Return HistoryCursor, or raise ApiError 422 INVALID_CURSOR for malformed or
    mismatched data. The original row may have been deleted; no DB lookup is needed.
    The caller must still check permission to access the requested conversation.
    """
    try:
        if not token or len(token) > 1024:
            raise ValueError("Invalid cursor length")
        encoded = token.encode("ascii")
        decoded = base64.b64decode(
            encoded + b"=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        )
        cursor = HistoryCursor.model_validate_json(decoded)
        if cursor.resource != resource or cursor.conversation_id != conversation_id:
            raise ValueError("Cursor scope mismatch")
        return cursor
    except (ValueError, UnicodeError, binascii.Error, ValidationError) as exception:
        raise ApiError(422, "INVALID_CURSOR", "Invalid pagination cursor") from exception
