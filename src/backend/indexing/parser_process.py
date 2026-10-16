"""Extract bounded text in a disposable process, without importing models or Chroma."""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import zipfile

from src.backend.indexing.config import IndexingSettings
from src.backend.indexing.errors import DocumentIndexingError
from src.backend.uploads.config import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class DocumentSection:
    """Carry extracted text and its one-based PDF page number, if applicable.

    Args:
        text: Extracted Unicode text, excluding binary source bytes.
        page: PDF page number starting at one, or None for plain text and DOCX.
    """

    text: str
    page: int | None = None


def extract_sections(
    file_path: Path, file_name: str, settings: IndexingSettings
) -> list[DocumentSection]:
    """Read a document using its original extension and enforce extraction limits.

    Args:
        file_path: Actual local path, which need not have a filename extension.
        file_name: Original basename used to select PDF, DOCX or UTF-8 text parsing.
        settings: Limits for pages, total text characters and expanded DOCX bytes.

    Returns:
        Nonempty text sections, preserving PDF page numbers when available.

    Raises:
        DocumentIndexingError: For empty, oversized, unsupported, encrypted or
            unreadable documents. The subprocess caller converts other parse errors.
    """
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentIndexingError("UNSUPPORTED_FILE_TYPE")
    sections = []
    character_count = 0

    def add_section(text: str, page: int | None = None) -> None:
        """Add nonblank text after enforcing the whole-document character limit.

        Args:
            text: Text returned by the selected reader.
            page: One-based PDF page number or None for other file formats.

        Raises:
            DocumentIndexingError: If the combined extracted text exceeds the limit.
        """
        nonlocal character_count
        character_count += len(text)
        if character_count > settings.max_text_characters:
            raise DocumentIndexingError("DOCUMENT_TOO_LARGE")
        if text.strip():
            sections.append(DocumentSection(text, page))

    if extension == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        if reader.is_encrypted:
            raise DocumentIndexingError("ENCRYPTED_DOCUMENT")
        if len(reader.pages) > settings.max_pages:
            raise DocumentIndexingError("TOO_MANY_PAGES")
        for page_number, page in enumerate(reader.pages, start=1):
            add_section(page.extract_text() or "", page_number)
    elif extension == ".docx":
        import docx2txt

        with zipfile.ZipFile(file_path) as archive:
            expanded_bytes = sum(entry.file_size for entry in archive.infolist())
            if expanded_bytes > settings.max_expanded_bytes:
                raise DocumentIndexingError("DOCUMENT_TOO_LARGE")
        add_section(docx2txt.process(str(file_path)))
    else:
        with file_path.open("r", encoding="utf-8-sig") as source:
            add_section(source.read(settings.max_text_characters + 1))
    if not sections:
        raise DocumentIndexingError("EMPTY_DOCUMENT")
    return sections


def main() -> None:
    """Read a parser request from stdin and write bounded JSON text or a safe error code.

    The parent passes file_path, file_name, settings and parent_pid as JSON. Limit
    process address space before importing readers. On Linux, kill this child if
    its backend parent exits, so a crash cannot leave a parser consuming resources.
    No file contents or private exception strings are written to error responses.
    """
    import os
    import signal
    import resource
    import ctypes

    request = json.loads(sys.stdin.read())
    settings = IndexingSettings(**request["settings"])
    memory_bytes = settings.parser_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    if sys.platform == "linux":
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)
        if os.getppid() != request["parent_pid"]:
            raise SystemExit(1)
    try:
        sections = extract_sections(Path(request["file_path"]), request["file_name"], settings)
        response = {"sections": [vars(section) for section in sections]}
    except DocumentIndexingError as error:
        response = {"error": error.code}
    except MemoryError:
        response = {"error": "DOCUMENT_TOO_LARGE"}
    except Exception:
        response = {"error": "UNREADABLE_DOCUMENT"}
    sys.stdout.write(json.dumps(response, ensure_ascii=False))


if __name__ == "__main__":
    main()
