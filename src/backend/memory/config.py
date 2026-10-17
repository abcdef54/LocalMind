"""Set independent limits for model context, summary and history reads."""

from dataclasses import dataclass, fields
import os

from src.backend.constants import DEFAULT_CONTEXT_LENGTH_K


@dataclass(frozen=True)
class MemorySettings:
    """Limit the context assembled by the backend before model requests.

    Args:
        context_tokens: Configured model context capacity; must match llama-swap.
        reply_tokens: Capacity reserved for the generated reply.
        tool_tokens: Capacity reserved for tool definitions and retrieved context.
        summary_tokens: Maximum conservative token estimate for the saved summary.
        recent_messages: Maximum raw prior messages retained before the current input.
        batch_messages: Maximum older messages fetched for one summary batch.

    Text uses its UTF-8 byte length as a conservative token estimate rather than
    downloading a model tokenizer. These are admission estimates, not usage metrics.
    Actual tool-augmented requests are checked again before invoking the model.
    """

    context_tokens: int = DEFAULT_CONTEXT_LENGTH_K * 1024
    reply_tokens: int = 2048
    tool_tokens: int = 4096
    summary_tokens: int = 2048
    recent_messages: int = 20
    batch_messages: int = 20

    def __post_init__(self) -> None:
        """Reject nonpositive integers and budgets with no room for input/system text."""
        for setting in fields(self):
            value = getattr(self, setting.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Memory {setting.name} must be a positive integer")
        if self.reply_tokens + self.tool_tokens + self.summary_tokens >= self.context_tokens:
            raise ValueError("Memory reserves must leave room for messages and system instructions")

    @classmethod
    def from_environment(cls) -> "MemorySettings":
        """Read optional MEMORY_<FIELD> integer overrides; invalid budgets fail early."""
        defaults = cls()
        return cls(
            **{
                setting.name: int(
                    os.getenv(
                        "MEMORY_" + setting.name.upper(), str(getattr(defaults, setting.name))
                    )
                )
                for setting in fields(cls)
            }
        )
