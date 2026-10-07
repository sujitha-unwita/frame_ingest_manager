"""Every SQL statement in the project lives in this package.

Use one repository::

    from audit_write.repositories import RollRepository
    rolls = RollRepository()            # uses the shared Database
    row = await rolls.get(roll_id)

Or all of them at once (handy for an orchestrator)::

    from audit_write.repositories import Repositories
    repos = Repositories.create(db)
    async with db.transaction() as conn:
        defect = await repos.defects_final.insert({...}, conn=conn)
        await repos.defects_live.merge_into(ids, defect["defect_id"], conn=conn)
"""

from __future__ import annotations

from dataclasses import dataclass

from config.audit_models import AuditWrite as Settings

from audit_write.database import Database
from audit_write.repositories.base import BaseRepository, RepositoryError, Row, quote_ident
from audit_write.repositories.camera_repo import CameraRepository
from audit_write.repositories.defect_final_repo import DefectFinalRepository
from audit_write.repositories.defect_live_repo import DefectLiveRepository
from audit_write.repositories.media_repo import MediaRepository
from audit_write.repositories.roll_repo import RollRepository
from audit_write.repositories.trigger_repo import TriggerRepository


@dataclass(frozen=True)
class Repositories:
    rolls: RollRepository
    cameras: CameraRepository
    triggers: TriggerRepository
    media: MediaRepository
    defects_live: DefectLiveRepository
    defects_final: DefectFinalRepository

    @classmethod
    def create(cls, db: Database | None = None, settings: Settings | None = None) -> "Repositories":
        return cls(
            rolls=RollRepository(db, settings),
            cameras=CameraRepository(db, settings),
            triggers=TriggerRepository(db, settings),
            media=MediaRepository(db, settings),
            defects_live=DefectLiveRepository(db, settings),
            defects_final=DefectFinalRepository(db, settings),
        )


__all__ = [
    "BaseRepository",
    "CameraRepository",
    "DefectFinalRepository",
    "DefectLiveRepository",
    "MediaRepository",
    "Repositories",
    "RepositoryError",
    "RollRepository",
    "Row",
    "TriggerRepository",
    "quote_ident",
]
