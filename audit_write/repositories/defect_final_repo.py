"""SQL for the ``defect_final`` table: operator-facing defects after deduplication."""

from __future__ import annotations

from typing import Any

import asyncpg

from audit_write.repositories.base import BaseRepository, Row


class DefectFinalRepository(BaseRepository):
    table_key = "defect_final"

    async def list_for_roll(
        self,
        roll_id: Any,
        *,
        limit: int | None = None,
        before: Any = None,
        by_number: bool = True,
        conn: asyncpg.Connection | None = None,
    ) -> list[Row]:
        """Defects of a roll, by defect number ascending (or newest first with ``by_number=False``)."""
        if by_number:
            return await self.list(
                filters={self.col("roll"): roll_id},
                limit=limit,
                before=before,
                order_by=self.col("number"),
                descending=False,
                conn=conn,
            )
        return await self.list(filters={self.col("roll"): roll_id}, limit=limit, before=before, conn=conn)

    async def get_by_number(self, roll_id: Any, defect_no: int, *, conn: asyncpg.Connection | None = None) -> Row | None:
        return await self.find_one({self.col("roll"): roll_id, self.col("number"): defect_no}, conn=conn)

    async def max_number(self, roll_id: Any, *, conn: asyncpg.Connection | None = None) -> int:
        """Highest defect number used on the roll; 0 when it has none."""
        value = await self.max_value(self.col("number"), {self.col("roll"): roll_id}, conn=conn)
        return int(value) if value is not None else 0

    async def count_for_roll(self, roll_id: Any, *, conn: asyncpg.Connection | None = None) -> int:
        return await self.count({self.col("roll"): roll_id}, conn=conn)
