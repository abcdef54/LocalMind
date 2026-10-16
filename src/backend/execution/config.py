"""Validated operational limits for the in-process executor."""

from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class ExecutorSettings:
    max_concurrency: int = 1
    poll_seconds: float = 1.0
    lease_seconds: float = 30.0
    heartbeat_seconds: float = 5.0
    shutdown_grace_seconds: float = 10.0
    database_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
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
        prefix = "RUN_EXECUTOR_"
        return cls(
            max_concurrency=int(os.getenv(prefix + "MAX_CONCURRENCY", "1")),
            poll_seconds=float(os.getenv(prefix + "POLL_SECONDS", "1")),
            lease_seconds=float(os.getenv(prefix + "LEASE_SECONDS", "30")),
            heartbeat_seconds=float(os.getenv(prefix + "HEARTBEAT_SECONDS", "5")),
            shutdown_grace_seconds=float(os.getenv(prefix + "SHUTDOWN_GRACE_SECONDS", "10")),
            database_timeout_seconds=float(os.getenv(prefix + "DATABASE_TIMEOUT_SECONDS", "5")),
        )
