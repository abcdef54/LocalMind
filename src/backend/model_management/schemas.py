"""Validate model sources and expose model/task metadata without local disk paths."""

from datetime import datetime
from pathlib import PurePosixPath
import re
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from huggingface_hub.utils import validate_repo_id

RepoId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
ModelFilename = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
]


def validate_model_repository(repo_id: str) -> str:
    """Return a valid owner/repository identifier; reject URLs and local paths.

    Args:
        repo_id: Hugging Face repository ID, for example Qwen/Qwen2.5-3B-Instruct-GGUF.

    Returns:
        The unchanged identifier after validation.

    Raises:
        ValueError: The ID is malformed or does not contain an owner and repository.
    """
    validate_repo_id(repo_id)
    if repo_id.count("/") != 1:
        raise ValueError("Use an owner/repository identifier")
    return repo_id


def validate_model_filename(filename: str) -> str:
    """Return a safe repository-relative GGUF filename, including supported split names.

    Args:
        filename: Relative filename, optionally within a repository subdirectory.

    Returns:
        The unchanged filename; no path normalization conceals invalid components.

    Raises:
        ValueError: The path is absolute, contains traversal/control characters or is not GGUF.
    """
    if (
        PurePosixPath(filename).is_absolute()
        or "\\" in filename
        or any(part in {"", ".", ".."} for part in filename.split("/"))
        or any(ord(character) < 32 or ord(character) == 127 for character in filename)
        or not re.search(r"\.gguf(?:[-.]\d+-of-\d+)?$", filename, re.I)
    ):
        raise ValueError("Use a relative GGUF filename without traversal or control characters")
    return filename


class ModelDownloadInput(BaseModel):
    """Select one model file or the first part of a split GGUF for a download task."""

    model_config = ConfigDict(extra="forbid")
    repo_id: RepoId
    gguf_filename: ModelFilename
    _repository = field_validator("repo_id")(validate_model_repository)
    _filename = field_validator("gguf_filename")(validate_model_filename)


class InstalledModelResponse(BaseModel):
    """Describe installed files/configuration; ready does not mean loaded into GPU memory."""

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    display_name: str
    model_name: str
    hf_repo: str
    gguf_file: str
    quantization: str
    size_bytes: int
    installed_at: datetime
    is_default: bool
    status: Literal["ready", "missing", "deleting", "configuring"]
    source_revision: str | None


class ModelDownloadResponse(BaseModel):
    """Expose persisted progress; downloaded_bytes counts fully downloaded model parts only."""

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    repo_id: str
    gguf_filename: str
    status: Literal["queued", "downloading", "registering", "completed", "failed", "interrupted"]
    attempt: int
    downloaded_bytes: int
    total_bytes: int | None
    source_revision: str | None
    model_id: UUID | None
    error: dict | None
    created_at: datetime
    updated_at: datetime


class CatalogModelResponse(BaseModel):
    """Expose public search metadata supplied by the existing Hugging Face catalog helper."""

    id: str
    author: str
    name: str
    family: str
    size: str | None = None
    task: str
    downloads: int
    downloads_text: str
    likes: int
    license: str | None = None
    created_at: str | None = None


class ModelQuantResponse(BaseModel):
    """Describe one quantization; split sizes include all parts of the model."""

    filename: str
    quant: str
    size_bytes: int
    size_gb: float
