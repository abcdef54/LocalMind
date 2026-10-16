"""Limit indexing work independently of upload byte limits and AI inference slots."""

from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class IndexingSettings:
    """Set worker concurrency, parsing limits and embedding batch sizes.

    Args:
        max_concurrency: Maximum indexing jobs admitted by workers sharing the DB.
        poll_seconds: Seconds between queue checks.
        lease_seconds: Seconds until an unrenewed job can be recovered.
        heartbeat_seconds: Seconds between reservation renewals.
        database_timeout_seconds: Maximum seconds to await one database operation.
        shutdown_grace_seconds: Seconds to allow jobs to finish on shutdown.
        parse_timeout_seconds: Maximum seconds allowed for the parser subprocess.
        parser_memory_mb: Address-space limit in MiB for that subprocess on Linux.
        max_pages: Maximum PDF pages accepted for extraction.
        max_text_characters: Maximum extracted characters across the entire document.
        max_expanded_bytes: Maximum uncompressed DOCX bytes before XML parsing.
        max_chunks: Maximum chunks produced from one document.
        chunk_size: Target chunk length in characters, not tokens.
        chunk_overlap: Characters repeated between neighboring chunks.
        embedding_batch_size: Chunks embedded/written together before checking cancellation.

    Byte upload limits alone cannot bound PDF expansion or embedding work. These
    limits reject excessive content as an indexing failure without generating an answer.
    """

    max_concurrency: int = 1
    poll_seconds: float = 1.0
    lease_seconds: float = 60.0
    heartbeat_seconds: float = 5.0
    database_timeout_seconds: float = 5.0
    shutdown_grace_seconds: float = 10.0
    parse_timeout_seconds: float = 30.0
    parser_memory_mb: int = 512
    max_pages: int = 500
    max_text_characters: int = 2_000_000
    max_expanded_bytes: int = 25 * 1024 * 1024
    max_chunks: int = 5000
    chunk_size: int = 1000
    chunk_overlap: int = 100
    embedding_batch_size: int = 32

    def __post_init__(self) -> None:
        """Reject invalid limits and timing combinations before starting a worker.

        Raises:
            ValueError: If a count/time is nonpositive, nonfinite, or overlap is
                outside 0..chunk_size-1, or renewals can outlast the reservation.
        """
        for name, value in vars(self).items():
            if name.endswith("_seconds"):
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
            elif type(value) is not int or value < (0 if name == "chunk_overlap" else 1):
                raise ValueError(f"{name} must be a valid integer limit")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.heartbeat_seconds + self.database_timeout_seconds >= self.lease_seconds:
            raise ValueError("Heartbeat interval plus DB timeout must be shorter than the lease")

    @classmethod
    def from_environment(cls) -> "IndexingSettings":
        """Read INDEXING_<FIELD_NAME> overrides and return validated settings.

        Returns:
            New immutable settings; missing variables retain the defaults above.

        Raises:
            ValueError: If an override cannot be parsed or violates a limit.
        """
        overrides = {}
        for name, default in vars(cls()).items():
            value = os.getenv(f"INDEXING_{name.upper()}")
            if value is not None:
                overrides[name] = type(default)(value)
        return cls(**overrides)
