"""Exercise read-only readiness checks without PostgreSQL dependencies."""

import unittest
from unittest.mock import AsyncMock, Mock

from audit_write.schema_checker import SchemaChecker, SchemaNotReadyError
from config.config_loader import ConfigLoader
from config.audit_models import AuditWrite


class SchemaCheckerTests(unittest.IsolatedAsyncioTestCase):
    """Migration history and configured columns must both be present."""

    def setUp(self) -> None:
        self.settings = AuditWrite.model_validate(
            ConfigLoader().raw_config["audit_write"]
        )
        self.checker = SchemaChecker(self.settings)
        self.versions = [
            {"version": version}
            for version in self.settings.migrations.required_versions
        ]
        self.columns = [
            {"table_name": table.table, "column_name": column}
            for table in self.settings.tables.values()
            for column in table.columns
        ]
        self.connection = Mock(
            fetchval=AsyncMock(return_value="schema_migrations"),
            fetch=AsyncMock(side_effect=[self.versions, self.columns]),
        )

    async def test_ready_schema_uses_selects_only(self) -> None:
        await self.checker.check(self.connection)
        for method in (self.connection.fetch, self.connection.fetchval):
            for call in method.await_args_list:
                self.assertTrue(call.args[0].startswith("SELECT "))

    async def test_missing_history_blocks_startup(self) -> None:
        self.connection.fetchval.return_value = None
        with self.assertLogs("audit_write.schema_checker", level="ERROR"):
            with self.assertRaisesRegex(SchemaNotReadyError, "history is missing"):
                await self.checker.check(self.connection)
        self.connection.fetch.assert_not_awaited()

    async def test_pending_migration_blocks_startup(self) -> None:
        self.connection.fetch.side_effect = [[]]
        with self.assertLogs("audit_write.schema_checker", level="ERROR"):
            with self.assertRaisesRegex(SchemaNotReadyError, "not applied"):
                await self.checker.check(self.connection)

    async def test_missing_column_blocks_startup(self) -> None:
        self.connection.fetch.side_effect = [self.versions, self.columns[1:]]
        with self.assertLogs("audit_write.schema_checker", level="ERROR"):
            with self.assertRaisesRegex(SchemaNotReadyError, "missing columns"):
                await self.checker.check(self.connection)

    async def test_missing_uuid_rename_migration_blocks_startup(self) -> None:
        """An old database must be migrated before the renamed columns are used."""
        self.connection.fetch.side_effect = [[{"version": "20260915000000"}]]
        with self.assertLogs("audit_write.schema_checker", level="ERROR"):
            with self.assertRaisesRegex(SchemaNotReadyError, "20260924000000"):
                await self.checker.check(self.connection)

    async def test_old_uuid_column_names_block_startup(self) -> None:
        """History alone is insufficient if the expected columns are absent."""
        old_columns = [
            {**column, "column_name": "trigger_id"}
            if column["column_name"] == "trigger_primary_id" else column
            for column in self.columns
        ]
        self.connection.fetch.side_effect = [self.versions, old_columns]
        with self.assertLogs("audit_write.schema_checker", level="ERROR"):
            with self.assertRaisesRegex(SchemaNotReadyError, "missing columns"):
                await self.checker.check(self.connection)

    async def test_query_error_is_preserved(self) -> None:
        failure = RuntimeError("connection lost")
        self.connection.fetchval.side_effect = failure
        with self.assertLogs("audit_write.schema_checker", level="ERROR"):
            with self.assertRaises(RuntimeError) as raised:
                await self.checker.check(self.connection)
        self.assertIs(raised.exception, failure)
