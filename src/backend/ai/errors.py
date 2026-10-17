"""Describe generation failures without pretending a model error is an answer."""


class InferenceStateUnknown(Exception):
    """Report a connection/timeout failure that may leave remote inference running.

    The caller must retain its admission slot until external compute stoppage is
    confirmed. Reconnecting or cancelling a Python task is not that confirmation.
    """
