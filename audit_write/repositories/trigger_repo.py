"""SQL for the ``trigger`` table."""

from __future__ import annotations

from typing import Any

import asyncpg

from audit_write.repositories.base import BaseRepository, Row


class TriggerRepository(BaseRepository):
    table_key = "trigger"

    async def list_for_roll(
        self,
        roll_id: Any,
        *,
        camera_id: str | None = None,
        limit: int | None = None,
        before: Any = None,
        conn: asyncpg.Connection | None = None,
    ) -> list[Row]:
        filters: dict[str, Any] = {self.col("roll"): roll_id}
        if camera_id is not None:
            filters[self.col("camera")] = camera_id
        return await self.list(filters=filters, limit=limit, before=before, conn=conn)

    async def get_by_sequence(
        self, roll_id: Any, camera_id: str, trigger_sequence: int, *, conn: asyncpg.Connection | None = None
    ) -> Row | None:
        return await self.find_one(
            {
                self.col("roll"): roll_id,
                self.col("camera"): camera_id,
                self.col("sequence"): trigger_sequence,
            },
            conn=conn,
        )

    async def list_for_sequence(
        self, roll_id: Any, trigger_sequence: int, *, conn: asyncpg.Connection | None = None
    ) -> list[Row]:
        """Every camera row for one physical trigger."""
        return await self.list(
            filters={self.col("roll"): roll_id, self.col("sequence"): trigger_sequence},
            limit=self.settings.pagination.max_limit,
            conn=conn,
        )

    async def max_sequence(
        self, roll_id: Any, camera_id: str | None = None, *, conn: asyncpg.Connection | None = None
    ) -> int | None:
        """Highest trigger_sequence recorded for the roll (optionally one camera), or None."""
        filters: dict[str, Any] = {self.col("roll"): roll_id}
        if camera_id is not None:
            filters[self.col("camera")] = camera_id
        return await self.max_value(self.col("sequence"), filters, conn=conn)
