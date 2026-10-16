"""Read and validate limits for polling, running and stopping AI jobs."""

from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class ExecutorSettings:
    """Set how many AI jobs can run and how long executor operations may wait.

    max_concurrency is the occupied-slot limit shared by workers using the same DB
    and limit. poll_seconds is the queue-check interval. lease_seconds is a run
    reservation's lifetime; heartbeat_seconds is its renewal interval.
    shutdown_grace_seconds allows active handlers to finish before cancellation.
    database_timeout_seconds limits each awaited store call. All times are seconds.
    """

    max_concurrency: int = 1
    poll_seconds: float = 1.0
    lease_seconds: float = 30.0
    heartbeat_seconds: float = 5.0
    shutdown_grace_seconds: float = 10.0
    database_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        """Reject invalid counts/times and reservation settings that cannot renew in time.

        Raise ValueError for a nonpositive/noninteger concurrency, nonfinite or
        nonpositive time, or heartbeat interval plus DB timeout reaching the lease duration.
        """
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise ValueError("Executor max_concurrency must be a positive integer")
        for name in (
            "poll_seconds",
            "lease_seconds",
            "heartbeat_seconds",
            "shutdown_grace_seconds",
            "database_timeout_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Executor {name} must be finite and positive")
        if self.heartbeat_seconds + self.database_timeout_seconds >= self.lease_seconds:
            raise ValueError(
                "Heartbeat interval plus database timeout must be shorter than the lease"
            )

    @classmethod
    def from_environment(cls) -> "ExecutorSettings":
        """Read RUN_EXECUTOR_* environment variables and return validated settings.

        Parse MAX_CONCURRENCY as int and each *_SECONDS value as float. Missing values
        use the defaults shown in this method. Raise ValueError for invalid numbers or
        unsafe timing settings. No worker or database connection is started here.
        """
        prefix = "RUN_EXECUTOR_"
        return cls(
            max_concurrency=int(os.getenv(prefix + "MAX_CONCURRENCY", "1")),
            poll_seconds=float(os.getenv(prefix + "POLL_SECONDS", "1")),
            lease_seconds=float(os.getenv(prefix + "LEASE_SECONDS", "30")),
            heartbeat_seconds=float(os.getenv(prefix + "HEARTBEAT_SECONDS", "5")),
            shutdown_grace_seconds=float(os.getenv(prefix + "SHUTDOWN_GRACE_SECONDS", "10")),
            database_timeout_seconds=float(os.getenv(prefix + "DATABASE_TIMEOUT_SECONDS", "5")),
        )
