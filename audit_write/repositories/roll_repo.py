"""SQL for the ``roll`` table."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import asyncpg

from audit_write.repositories.base import BaseRepository, Row, quote_ident


class RollRepository(BaseRepository):
    table_key = "roll"

    async def lock_for_finalization(
        self, roll_id: Any, *, conn: asyncpg.Connection
    ) -> Row | None:
        """Serialize final-number allocation until the caller's transaction ends."""
        record = await conn.fetchrow(
            f"SELECT {self.columns_sql} FROM {self.table_sql} "
            f"WHERE {quote_ident(self.pk)} = $1 FOR UPDATE", roll_id,
        )
        return dict(record) if record is not None else None

    async def find_by_label(self, label: str, *, conn: asyncpg.Connection | None = None) -> Row | None:
        """The most recent roll with this label."""
        return await self.find_one({self.col("label"): label}, conn=conn)

    async def list_open(
        self, *, limit: int | None = None, before: Any = None, conn: asyncpg.Connection | None = None
    ) -> list[Row]:
        """Rolls that have not ended yet."""
        return await self.list(filters={self.col("ended_at"): None}, limit=limit, before=before, conn=conn)

    async def list_suspect(
        self, *, limit: int | None = None, before: Any = None, conn: asyncpg.Connection | None = None
    ) -> list[Row]:
        return await self.list(filters={self.col("suspect_flag"): True}, limit=limit, before=before, conn=conn)

    async def end_roll(
        self,
        roll_id: Any,
        ended_at: datetime,
        length_mm: float | None = None,
        *,
        conn: asyncpg.Connection | None = None,
    ) -> Row | None:
        changes: dict[str, Any] = {self.col("ended_at"): ended_at}
        if length_mm is not None:
            changes[self.col("length")] = length_mm
        return await self.update(roll_id, changes, conn=conn)

    async def set_suspect(
        self,
        roll_id: Any,
        suspect: bool,
        reason: str | None = None,
        *,
        conn: asyncpg.Connection | None = None,
    ) -> Row | None:
        """Set or clear the suspect flag.  Clearing it also clears the reason."""
        changes = {
            self.col("suspect_flag"): suspect,
            self.col("suspect_reason"): reason if suspect else None,
        }
        return await self.update(roll_id, changes, conn=conn)
