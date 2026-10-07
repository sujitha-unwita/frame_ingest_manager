"""SQL for the ``defect_live`` table: raw detections before deduplication."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import asyncpg

from audit_write.repositories.base import BaseRepository, Row, quote_ident


class DefectLiveRepository(BaseRepository):
    table_key = "defect_live"

    async def lock_final_members(
        self, roll_id: Any, detection_ids: Sequence[Any],
        *, conn: asyncpg.Connection,
    ) -> list[Row]:
        """Read and lock every requested member without pagination truncation."""
        records = await conn.fetch(
            f"SELECT {self.columns_sql} FROM {self.table_sql} "
            f"WHERE {quote_ident(self.col('roll'))} = $1 "
            f"AND {quote_ident(self.pk)} = ANY($2) "
            f"ORDER BY {quote_ident(self.pk)} FOR UPDATE",
            roll_id, list(detection_ids),
        )
        return [dict(record) for record in records]

    async def list_for_roll(
        self, roll_id: Any, *, limit: int | None = None, before: Any = None, conn: asyncpg.Connection | None = None
    ) -> list[Row]:
        return await self.list(filters={self.col("roll"): roll_id}, limit=limit, before=before, conn=conn)

    async def list_for_trigger(self, trigger_primary_id: Any, *, conn: asyncpg.Connection | None = None) -> list[Row]:
        return await self.list(
            filters={self.col("trigger"): trigger_primary_id}, limit=self.settings.pagination.max_limit, conn=conn
        )

    async def list_for_media(self, media_id: Any, *, conn: asyncpg.Connection | None = None) -> list[Row]:
        return await self.list(
            filters={self.col("media"): media_id}, limit=self.settings.pagination.max_limit, conn=conn
        )

    async def list_for_defect(self, defect_id: Any, *, conn: asyncpg.Connection | None = None) -> list[Row]:
        """The detections that were merged into one final defect."""
        return await self.list(
            filters={self.col("defect"): defect_id}, limit=self.settings.pagination.max_limit, conn=conn
        )

    async def list_unmerged(
        self,
        roll_id: Any,
        *,
        limit: int | None = None,
        before: Any = None,
        oldest_first: bool = True,
        conn: asyncpg.Connection | None = None,
    ) -> list[Row]:
        """Detections not yet merged into a final defect — the deduplication queue."""
        return await self.list(
            filters={self.col("roll"): roll_id, self.col("defect"): None},
            limit=limit,
            before=before,
            descending=not oldest_first,
            conn=conn,
        )

    async def merge_into(
        self, detection_ids: Sequence[Any], defect_id: Any | None, *, conn: asyncpg.Connection | None = None
    ) -> list[Row]:
        """Point detections at the final defect they were merged into (``None`` un-merges)."""
        if not detection_ids:
            return []
        return await self.update_where(
            {self.pk: list(detection_ids)}, {self.col("defect"): defect_id}, conn=conn
        )
