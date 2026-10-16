"""Read the PostgreSQL URL used by backend startup and Alembic commands."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_database_url() -> str:
    """Return POSTGRESQL_URL from the environment or the repository's .env file.

    Load .env without replacing values already supplied by Docker or the shell.
    Raise RuntimeError if the URL is missing. This reads configuration only; it
    does not connect to PostgreSQL or check the supplied credentials.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("POSTGRESQL_URL")
    if not database_url:
        raise RuntimeError(
            "POSTGRESQL_URL is required. Configure it in the environment or project .env."
        )
    return database_url
