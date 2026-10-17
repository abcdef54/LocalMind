"""Estimate context size without network downloads or model-specific tokenizer files."""

import json
from typing import Any


def estimate_text_tokens(text: str) -> int:
    """Return a conservative token estimate using the text's UTF-8 byte length.

    Args:
        text: Prompt, message or tool schema text, including Unicode content.

    Returns:
        Byte count used as a conservative estimate, not a measured model token count.
        Empty text costs zero; no tokenizer model or network lookup is needed.
    """
    return len(text.encode("utf-8"))


def estimate_messages_tokens(messages: list[Any]) -> int:
    """Estimate message content plus per-message protocol overhead.

    Args:
        messages: Role/content dictionaries or LangChain message objects. Tool-call
            arguments and structured content are included in the estimate.

    Returns:
        Conservative text estimate plus 16 units per message for framing.
    """
    total = 0
    for message in messages:
        if isinstance(message, dict):
            payload = message
        else:
            payload = {"role": message.type, "content": message.content}
            calls = getattr(message, "tool_calls", None)
            if calls:
                payload["tool_calls"] = calls
        total += estimate_text_tokens(json.dumps(payload, ensure_ascii=False, default=str)) + 16
    return total


def split_text_for_summary(text: str, capacity: int) -> list[str]:
    """Split a long historical message without dropping or breaking Unicode characters.

    Args:
        text: Original historical content; it is never truncated in the database.
        capacity: Positive UTF-8 byte budget available for each summary input piece.

    Returns:
        Ordered pieces whose concatenation equals text, each within capacity.

    Raises:
        ValueError: If capacity cannot hold the next character.
    """
    pieces = []
    remaining = text
    while remaining:
        piece = remaining.encode("utf-8")[:capacity].decode("utf-8", errors="ignore")
        if not piece:
            raise ValueError("Summary input budget cannot fit a character")
        pieces.append(piece)
        remaining = remaining[len(piece) :]
    return pieces or [""]
