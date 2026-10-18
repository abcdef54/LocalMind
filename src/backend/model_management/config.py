"""Read filesystem locations and limits for model downloads."""

from dataclasses import dataclass, fields
import math
import os
from pathlib import Path

from src.backend.constants import PROJECT_ROOT, LLAMA_SWAP_CONFIG_PATH


@dataclass(frozen=True)
class ModelManagementSettings:
    """Configure model storage and one-at-a-time background downloads.

    Args:
        root: Shared directory containing existing GGUFs, staging and installations.
        config_path: YAML file watched by llama-swap; may be a Docker file mount.
        poll_seconds: Delay between queue checks and download progress samples.
        download_timeout_seconds: Maximum runtime of one download subprocess.
        catalog_timeout_seconds: Maximum wait for a Hugging Face metadata request.
        max_queued_downloads: Maximum number of pending downloads admitted by the API.
    """

    root: Path = PROJECT_ROOT / "src/backend/ai/llms/cpp_models"
    config_path: Path = LLAMA_SWAP_CONFIG_PATH
    poll_seconds: float = 1.0
    download_timeout_seconds: float = 14400.0
    catalog_timeout_seconds: float = 30.0
    max_queued_downloads: int = 20

    def __post_init__(self) -> None:
        """Reject invalid queue sizes and nonfinite or nonpositive time limits."""
        if type(self.max_queued_downloads) is not int or self.max_queued_downloads < 1:
            raise ValueError("max_queued_downloads must be a positive integer")
        for name in ("poll_seconds", "download_timeout_seconds", "catalog_timeout_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_environment(cls) -> "ModelManagementSettings":
        """Read MODEL_<FIELD> overrides; validate without creating files or downloading."""
        defaults = cls()
        overrides = {}
        for field in fields(cls):
            value = os.getenv(f"MODEL_{field.name.upper()}")
            if value is not None:
                default = getattr(defaults, field.name)
                overrides[field.name] = (
                    Path(value) if isinstance(default, Path) else type(default)(value)
                )
        return cls(**overrides)
