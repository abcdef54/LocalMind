"""Database configuration shared by the application and Alembic CLI."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_database_url() -> str:
    """Read the local configuration without overriding Docker environment variables."""
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("POSTGRESQL_URL")
    if not database_url:
        raise RuntimeError(
            "POSTGRESQL_URL is required. Configure it in the environment or project .env."
        )
    return database_url
