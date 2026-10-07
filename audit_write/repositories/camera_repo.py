"""SQL for the ``camera`` table."""

from __future__ import annotations

from typing import Any

import asyncpg

from audit_write.repositories.base import BaseRepository, Row


class CameraRepository(BaseRepository):
    table_key = "camera"

    async def list_active(
        self, *, limit: int | None = None, before: Any = None, conn: asyncpg.Connection | None = None
    ) -> list[Row]:
        return await self.list(
            filters={self.col("active_flag"): True}, limit=limit, before=before, descending=False, conn=conn
        )

    async def set_active(self, camera_id: str, active: bool, *, conn: asyncpg.Connection | None = None) -> Row | None:
        return await self.update(camera_id, {self.col("active_flag"): active}, conn=conn)

    async def register(self, camera_id: str, active: bool = True, *, conn: asyncpg.Connection | None = None) -> Row:
        """Create the camera, or update its active flag if it already exists."""
        row = await self.upsert({self.pk: camera_id, self.col("active_flag"): active}, conn=conn)
        assert row is not None  # DO UPDATE always returns the row
        return row
