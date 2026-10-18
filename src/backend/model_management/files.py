"""Keep downloads isolated from installed files and serialize changes to model storage."""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re
import shlex
from typing import Iterator

from src.backend.constants import DEFAULT_CONTEXT_LENGTH_K, LLAMA_SWAP_PATH_PREFIX
from src.backend.model_management.config import ModelManagementSettings
from src.backend.model_management.schemas import validate_model_filename


@contextmanager
def try_reserving_model_storage(settings: ModelManagementSettings) -> Iterator[int | None]:
    """Try to reserve shared model files/configuration without blocking the event loop.

    Args:
        settings: Shared model root used by every backend worker accessing these files.

    Yields:
        Open lock descriptor, or None if another process is changing model storage.
        A download subprocess inherits the descriptor, so a crashed parent's lock
        remains held until that subprocess exits. Callers must reap children before
        leaving normally; closing a descriptor does not explicitly unlock its child.
    """
    settings.root.mkdir(parents=True, exist_ok=True)
    with (settings.root / ".model-management.lock").open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
            return
        yield lock_file.fileno()


def resolve_managed_model_path(root: Path, relative_name: str) -> Path:
    """Resolve a model path inside root and reject traversal or symlink escapes.

    Args:
        root: Model storage directory shared with llama-swap.
        relative_name: Validated repository-relative or installed GGUF path.

    Returns:
        Absolute path inside root; the file need not exist yet.

    Raises:
        ValueError: The path is malformed or resolves outside model storage.
    """
    validate_model_filename(relative_name)
    resolved = (root / relative_name).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Model path is outside managed storage")
    return resolved


def select_all_parts_of_requested_model(filename: str, available_files: list[str]) -> list[str]:
    """Select exactly the requested GGUF or every numbered part of its split model.

    Args:
        filename: Requested single file or part 1; later parts cannot start an installation.
        available_files: Repository-relative names from one resolved Hugging Face revision.

    Returns:
        Complete ordered list of matching model files, without unrelated quantizations.

    Raises:
        ValueError: The requested file is absent, a split part is missing, or part 1 was not selected.
    """
    validate_model_filename(filename)
    if filename not in available_files:
        raise ValueError("Requested GGUF is absent")
    match = re.fullmatch(r"(.*?)(\d+)-of-(\d+)(\.gguf)?", filename, re.I)
    if not match:
        return [filename]
    prefix, number, total, suffix = match.groups()
    if int(number) != 1 or int(total) < 1 or int(total) > 1000:
        raise ValueError("Choose the first part of a supported split model")
    selected = [
        f'{prefix}{index:0{len(number)}d}-of-{total}{suffix or ""}'
        for index in range(1, int(total) + 1)
    ]
    if not set(selected).issubset(available_files):
        raise ValueError("A split model part is missing")
    for name in selected:
        validate_model_filename(name)
    return selected


def validate_complete_model_files(root: Path, files: list[dict]) -> int:
    """Check that every declared model part exists with its expected size and GGUF header.

    Args:
        root: Staging or installation directory containing the declared relative files.
        files: Nonempty list of filename/size_bytes objects, with unique file names.

    Returns:
        Combined bytes after all checks pass; no whole-model hashing is performed.

    Raises:
        ValueError: A file is missing, duplicated, empty, wrongly sized or not GGUF.
    """
    if not files or len(files) > 1000 or len({file["filename"] for file in files}) != len(files):
        raise ValueError("Invalid model file manifest")
    total = 0
    for file in files:
        path = resolve_managed_model_path(root, file["filename"])
        expected_size = file["size_bytes"]
        if (
            type(expected_size) is not int
            or expected_size < 4
            or not path.is_file()
            or path.stat().st_size != expected_size
        ):
            raise ValueError("Incomplete model file")
        with path.open("rb") as source:
            if source.read(4) != b"GGUF":
                raise ValueError("Invalid GGUF header")
        total += expected_size
    return total


def write_llama_swap_config_for_installed_models(
    settings: ModelManagementSettings, models: list[dict]
) -> None:
    """Write quoted model commands after validating paths, while the caller holds storage's lock.

    Args:
        settings: Destination config file and shared model storage directory.
        models: Model names and GGUF paths relative to model storage; ready/configuring only.

    Docker currently bind-mounts this exact file. Preserve its inode instead of
    replacing it, so llama-swap sees writes through that mount. A crashed partial
    write is rebuilt on the next worker reconciliation. Do not rewrite identical text.
    """
    lines = ["models: {}"] if not models else ["models:"]
    for model in models:
        resolve_managed_model_path(settings.root, model["gguf_file"])
        container_path = LLAMA_SWAP_PATH_PREFIX.rstrip("/") + "/" + model["gguf_file"]
        command = (
            f"llama-server --port ${{PORT}} -m {shlex.quote(container_path)} "
            f"-ngl 99 -c {DEFAULT_CONTEXT_LENGTH_K * 1024} --parallel 1 --flash-attn auto"
        )
        lines.extend(
            [
                f'  {json.dumps(model["model_name"])}:',
                f"    cmd: {json.dumps(command)}",
                "    proxy: http://127.0.0.1:${PORT}",
            ]
        )
    content = "\n".join(lines) + "\n"
    settings.config_path.parent.mkdir(parents=True, exist_ok=True)
    if settings.config_path.exists() and settings.config_path.read_text() == content:
        return
    with settings.config_path.open("w", encoding="utf-8") as destination:
        destination.write(content)
        destination.flush()
