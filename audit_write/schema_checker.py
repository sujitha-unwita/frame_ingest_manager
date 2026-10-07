"""Read-only startup checks; schema changes belong to deployment migrations."""

import logging
from typing import Any

from config.audit_models import AuditWrite

LOGGER = logging.getLogger(__name__)


def _quote_identifier(value: str) -> str:
    """Quote a configured SQL identifier; values use query parameters."""
    return '"' + value.replace('"', '""') + '"'


class SchemaNotReadyError(RuntimeError):
    """Required migration history or configured table columns are missing."""


class SchemaChecker:
    """Check recorded migrations and visible columns without changing data.

    This is a readiness check, not a complete schema-diff or permission audit.
    It does not validate column types, constraints, indexes, or write access.
    """

    def __init__(self, settings: AuditWrite) -> None:
        self._settings = settings

    async def check(self, connection: Any) -> None:
        """Use the connection borrowed by the orchestrator for SELECTs only."""
        LOGGER.info("SchemaChecker.check: checking audit schema")
        try:
            settings = self._settings
            migration = settings.migrations
            history = (
                f"{_quote_identifier(migration.history_schema)}."
                f"{_quote_identifier(migration.history_table)}"
            )
            exists = await connection.fetchval("SELECT to_regclass($1)", history)
            if exists is None:
                raise SchemaNotReadyError(
                    "migration history is missing; apply deployment migrations"
                )
            versions = await connection.fetch(
                f"SELECT {_quote_identifier(migration.version_column)} "
                f"AS version FROM {history}"
            )
            applied = {str(row["version"]) for row in versions}
            missing = set(migration.required_versions) - applied
            if missing:
                raise SchemaNotReadyError(
                    f"required migrations are not applied: {sorted(missing)}"
                )
            rows = await connection.fetch(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema=$1", settings.database.schema_name,
            )
            available: dict[str, set[str]] = {}
            for row in rows:
                available.setdefault(row["table_name"], set()).add(
                    row["column_name"]
                )
            for table in settings.tables.values():
                missing_columns = set(table.columns) - available.get(
                    table.table, set()
                )
                if missing_columns:
                    raise SchemaNotReadyError(
                        f"table {table.table!r} is missing or has inaccessible/"
                        f"missing columns: {sorted(missing_columns)}"
                    )
            LOGGER.info("SchemaChecker.check: audit schema is ready")
        except Exception:
            LOGGER.exception("SchemaChecker.check: failed; startup must stop")
            raise
