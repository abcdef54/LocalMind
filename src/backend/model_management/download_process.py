"""Download one pinned GGUF installation in a subprocess owned by the model worker."""

import json
import ctypes
import os
import signal
from pathlib import Path
import sys

from huggingface_hub import HfApi, hf_hub_download

from src.backend.model_management.files import (
    select_all_parts_of_requested_model,
    validate_complete_model_files,
)
from src.backend.model_management.schemas import ModelDownloadInput


def write_download_progress(job_directory: Path, progress: dict) -> None:
    """Atomically publish progress metadata, leaving model bytes in the separate files directory.

    Args:
        job_directory: Worker-created staging directory for this task.
        progress: Pinned revision, file manifest and bytes of fully downloaded parts.
    """
    temporary = job_directory / "progress.tmp"
    temporary.write_text(json.dumps(progress), encoding="utf-8")
    temporary.replace(job_directory / "progress.json")


def download_all_model_parts(job_directory: Path, repo_id: str, gguf_filename: str) -> None:
    """Resolve one repository revision, download all required parts and validate before publication.

    Args:
        job_directory: Isolated writable staging directory; never the live model root.
        repo_id: Validated owner/repository source identifier.
        gguf_filename: Single GGUF or first split part requested by the user.

    Exceptions leave the task incomplete. No installed model or llama-swap config
    is changed here. The parent enforces a timeout and reaps this process on shutdown.
    """
    source = ModelDownloadInput(repo_id=repo_id, gguf_filename=gguf_filename)
    metadata = HfApi().model_info(source.repo_id, files_metadata=True)
    if not metadata.sha:
        raise ValueError("Missing immutable repository revision")
    siblings = {file.rfilename: file for file in metadata.siblings}
    filenames = select_all_parts_of_requested_model(source.gguf_filename, list(siblings))
    files = [{"filename": name, "size_bytes": siblings[name].size} for name in filenames]
    if any(type(file["size_bytes"]) is not int or file["size_bytes"] < 4 for file in files):
        raise ValueError("Missing model part size")
    progress = {
        "source_revision": metadata.sha,
        "files": files,
        "total_bytes": sum(file["size_bytes"] for file in files),
        "downloaded_bytes": 0,
    }
    write_download_progress(job_directory, progress)
    target = job_directory / "files"
    for file in files:
        hf_hub_download(
            repo_id=source.repo_id,
            filename=file["filename"],
            revision=metadata.sha,
            local_dir=target,
        )
        validate_complete_model_files(target, [file])
        progress["downloaded_bytes"] += file["size_bytes"]
        write_download_progress(job_directory, progress)
    validate_complete_model_files(target, files)


if __name__ == "__main__":
    try:
        # Linux/WSL: a worker crash must not leave an orphan network download.
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL) != 0:
            raise OSError("Cannot bind download lifetime to its parent")
        if os.getppid() != int(sys.argv[4]):
            sys.exit(1)
        download_all_model_parts(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
    except Exception:
        # Do not print tokens, signed URLs or source library errors into API-visible logs.
        sys.exit(1)
