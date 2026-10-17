"""Check cooperative cancellation at boundaries where no new model call has started."""

from contextvars import ContextVar
from typing import Protocol


class RunCancellationRequested(Exception):
    """Stop further work at a safe boundary; the executor confirms cancellation after all siblings finish."""


class RunControl(Protocol):
    """Check run ownership and cancellation before inference and tool calls."""

    async def check_run_ownership_and_cancellation(self) -> None:
        """Reject cancellation or lost ownership before starting more external work."""
        ...


current_run_control: ContextVar[RunControl | None] = ContextVar("current_run_control", default=None)


async def check_run_control() -> None:
    """Check the current run when present; direct graph benchmarks use no control object."""
    control = current_run_control.get()
    if control is not None:
        await control.check_run_ownership_and_cancellation()
