"""Opaque keyset cursors, scoped to a resource and conversation."""

import base64
import binascii
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from src.backend.api.errors import ApiError


class HistoryCursor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    resource: Literal["conversations", "messages"]
    conversation_id: UUID | None = None
    created_at: datetime
    id: UUID

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            raise ValueError("Cursor timestamp must include a timezone")
        return timestamp.astimezone(timezone.utc)


def encode_cursor(cursor: HistoryCursor) -> str:
    return base64.urlsafe_b64encode(cursor.model_dump_json().encode()).decode().rstrip("=")


def decode_cursor(
    token: str,
    *,
    resource: Literal["conversations", "messages"],
    conversation_id: UUID | None = None,
) -> HistoryCursor:
    """Validate scope without requiring the anchor row to still exist.

    Cursors are not credentials or signatures. All resource access checks remain
    server-side, even when a client constructs its own valid cursor.
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
