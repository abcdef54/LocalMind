"""Public indexing failures whose codes can be saved without exposing private contents."""


class DocumentIndexingError(Exception):
    """Report a safe reason why a document could not be indexed.

    Args:
        code: Stable public error code, such as DOCUMENT_TOO_LARGE or EMPTY_DOCUMENT.

    Parser/library exception strings must not be copied here; they can contain
    local paths or document text. The worker saves this code with a generic message.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)
