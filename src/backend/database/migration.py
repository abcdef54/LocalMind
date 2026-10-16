"""Apply versioned migrations using an existing application transaction."""

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Connection

from src.backend.database.config import PROJECT_ROOT


def upgrade_database(connection: Connection) -> None:
    """Upgrade to head; propagate errors so startup cannot use an incomplete schema.

    The caller owns the transaction. Legacy databases are adopted only after the
    baseline revision validates their schema; this function never stamps blindly.
    """
    migration_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    migration_config.attributes["connection"] = connection
    migration_config.attributes["configure_logger"] = False
    command.upgrade(migration_config, "head")
