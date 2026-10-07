"""SQL for the ``media`` table.

The storage key column holds an S3 object key as plain text.  This module
never talks to S3.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import asyncpg

from audit_write.repositories.base import BaseRepository, Row


class MediaRepository(BaseRepository):
    table_key = "media"

    async def list_for_roll(
        self, roll_id: Any, *, limit: int | None = None, before: Any = None, conn: asyncpg.Connection | None = None
    ) -> list[Row]:
        return await self.list(filters={self.col("roll"): roll_id}, limit=limit, before=before, conn=conn)

    async def list_for_trigger(self, trigger_primary_id: Any, *, conn: asyncpg.Connection | None = None) -> list[Row]:
        return await self.list(
            filters={self.col("trigger"): trigger_primary_id}, limit=self.settings.pagination.max_limit, conn=conn
        )

    async def list_for_defect(self, defect_id: Any, *, conn: asyncpg.Connection | None = None) -> list[Row]:
        """Evidence images attached to a final defect."""
        return await self.list(
            filters={self.col("defect"): defect_id}, limit=self.settings.pagination.max_limit, conn=conn
        )

    async def list_pending_upload(
        self,
        *,
        roll_id: Any = None,
        limit: int | None = None,
        before: Any = None,
        conn: asyncpg.Connection | None = None,
    ) -> list[Row]:
        """Media rows that have no storage key yet."""
        filters: dict[str, Any] = {self.col("storage_key"): None}
        if roll_id is not None:
            filters[self.col("roll")] = roll_id
        return await self.list(filters=filters, limit=limit, before=before, conn=conn)

    async def get_storage_key(self, media_id: Any, *, conn: asyncpg.Connection | None = None) -> str | None:
        """The object key exactly as stored (``None`` if no image or no row)."""
        row = await self.get(media_id, conn=conn)
        return row[self.col("storage_key")] if row else None

    async def set_storage_key(
        self, media_id: Any, key: str, uploaded_at: datetime, *, conn: asyncpg.Connection | None = None
    ) -> Row | None:
        """Record where the image was uploaded."""
        return await self.update(
            media_id, {self.col("storage_key"): key, self.col("uploaded_at"): uploaded_at}, conn=conn
        )

    async def link_to_defect(
        self, media_ids: Sequence[Any], defect_id: Any | None, *, conn: asyncpg.Connection | None = None
    ) -> list[Row]:
        """Attach media rows to a final defect (or detach with ``defect_id=None``)."""
        if not media_ids:
            return []
        return await self.update_where({self.pk: list(media_ids)}, {self.col("defect"): defect_id}, conn=conn)
