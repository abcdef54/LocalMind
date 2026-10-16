"""Apply pending Alembic database changes using the caller's open connection."""

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Connection

from src.backend.database.config import PROJECT_ROOT


def upgrade_database(connection: Connection) -> None:
    """Apply all pending database revisions using the supplied synchronous connection.

    connection must belong to the caller's transaction. Return None after upgrading
    to the latest revision; the caller commits or rolls back. Existing unversioned
    databases must pass the baseline's schema checks before being adopted.
    Propagate migration errors so backend startup cannot continue with a partial schema.
    """
    migration_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    migration_config.attributes["connection"] = connection
    migration_config.attributes["configure_logger"] = False
    command.upgrade(migration_config, "head")
