"""Set allowed file extensions and limits applied while receiving uploaded bytes."""

from dataclasses import dataclass
import math
import os
from pathlib import Path

from src.backend.constants import UPLOAD_DIR

MAX_MULTIPART_HEADER_BYTES = 16 * 1024

SUPPORTED_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".txt",
        ".md",
        ".py",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".json",
        ".csv",
        ".html",
        ".css",
        ".yaml",
        ".yml",
        ".docx",
    }
)


@dataclass(frozen=True)
class UploadSettings:
    """Set where uploads are stored and how much a receiving request may use.

    root contains temporary/permanent batch folders. max_file_bytes limits each file;
    max_batch_bytes limits their combined bytes; max_payload_bytes limits JSON text.
    max_files limits file count and max_parallel_uploads limits concurrent receivers
    per API process. idle_timeout_seconds limits the wait for the next chunk;
    request_timeout_seconds limits receiving overall. orphan_grace_seconds delays
    abandoned-folder deletion. Byte/count limits and times must all be positive.
    """

    root: Path = UPLOAD_DIR / "attachments"
    max_file_bytes: int = 25 * 1024 * 1024
    max_batch_bytes: int = 100 * 1024 * 1024
    max_files: int = 10
    max_payload_bytes: int = 64 * 1024
    max_parallel_uploads: int = 2
    idle_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 120.0
    orphan_grace_seconds: float = 3600.0

    def __post_init__(self) -> None:
        """Raise ValueError for invalid limits before an upload starts.

        Require positive integer byte/count limits, excluding booleans. Require finite,
        positive time values; reject zero, negatives, infinity and NaN.
        """
        for setting_name in (
            "max_file_bytes",
            "max_batch_bytes",
            "max_files",
            "max_payload_bytes",
            "max_parallel_uploads",
        ):
            if type(getattr(self, setting_name)) is not int or getattr(self, setting_name) <= 0:
                raise ValueError(f"{setting_name} must be a positive integer")

        for setting_name in (
            "idle_timeout_seconds",
            "request_timeout_seconds",
            "orphan_grace_seconds",
        ):
            if not math.isfinite(getattr(self, setting_name)) or getattr(self, setting_name) <= 0:
                raise ValueError(f"{setting_name} must be finite and positive")

    @property
    def max_request_bytes(self) -> int:
        """Return the whole request's byte limit, including file bytes, JSON and headers.

        Reserve one header allowance per file plus one for payload. Separate per-file
        and combined-file limits still apply even when the whole request fits.
        """
        part_count = self.max_files + 1
        multipart_overhead_bytes = part_count * MAX_MULTIPART_HEADER_BYTES
        return self.max_batch_bytes + self.max_payload_bytes + multipart_overhead_bytes

    @classmethod
    def from_environment(cls) -> "UploadSettings":
        """Read UPLOAD_* environment overrides and return validated UploadSettings.

        Use defaults for missing variables, Path for UPLOAD_ROOT, integers for byte/count
        limits and floats for seconds. Raise ValueError if parsing or validation fails.
        Do not create directories or connect to the database.
        """
        default_settings = cls()
        environment_overrides = {}
        for setting_name in cls.__dataclass_fields__:
            environment_value = os.getenv(f"UPLOAD_{setting_name.upper()}")
            if environment_value is None:
                continue
            if setting_name == "root":
                parsed_value = Path(environment_value)
            else:
                default_value = getattr(default_settings, setting_name)
                parsed_value = type(default_value)(environment_value)
            environment_overrides[setting_name] = parsed_value
        return cls(**environment_overrides)


def validate_file_name(file_name: str) -> str:
    """Check an original filename and return it unchanged if allowed.

    file_name must be a basename, including its extension. Reject empty/overlong
    names, directory separators and control characters with ApiError 422; reject
    unsupported extensions with 415. Do not compare file contents or check duplicates
    here. Names stay case-sensitive; parsing actual file contents happens during indexing.
    """
    from src.backend.api.errors import ApiError

    has_directory_or_control_characters = any(
        character in "/\\" or ord(character) < 32 or ord(character) == 127
        for character in file_name
    )
    if (
        not file_name
        or len(file_name) > 255
        or file_name in {".", ".."}
        or has_directory_or_control_characters
    ):
        raise ApiError(
            422, "INVALID_FILE_NAME", "Use a filename without a directory or control characters"
        )
    if Path(file_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ApiError(
            415, "UNSUPPORTED_FILE_TYPE", "Unsupported file extension", {"file_name": file_name}
        )
    return file_name
