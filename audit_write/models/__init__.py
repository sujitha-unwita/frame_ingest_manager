"""Pydantic shapes, one module per table.

    from audit_write.models import RollCreate, Roll
"""

from audit_write.models.camera import Camera, CameraCreate, CameraUpdate
from audit_write.models.defect_final import DefectFinal, DefectFinalCreate, DefectFinalUpdate
from audit_write.models.defect_live import DefectLive, DefectLiveCreate, DefectLiveUpdate
from audit_write.models.media import Media, MediaCreate, MediaUpdate
from audit_write.models.roll import Roll, RollCreate, RollUpdate
from audit_write.models.trigger import Trigger, TriggerCreate, TriggerUpdate

# Config table key -> (row, create, update) models.  The tests use it to check
# the models stay in step with audit_write.tables in the main config.yaml.
MODELS_BY_TABLE = {
    "roll": (Roll, RollCreate, RollUpdate),
    "camera": (Camera, CameraCreate, CameraUpdate),
    "trigger": (Trigger, TriggerCreate, TriggerUpdate),
    "media": (Media, MediaCreate, MediaUpdate),
    "defect_live": (DefectLive, DefectLiveCreate, DefectLiveUpdate),
    "defect_final": (DefectFinal, DefectFinalCreate, DefectFinalUpdate),
}

__all__ = [
    "Camera", "CameraCreate", "CameraUpdate",
    "DefectFinal", "DefectFinalCreate", "DefectFinalUpdate",
    "DefectLive", "DefectLiveCreate", "DefectLiveUpdate",
    "Media", "MediaCreate", "MediaUpdate",
    "Roll", "RollCreate", "RollUpdate",
    "Trigger", "TriggerCreate", "TriggerUpdate",
    "MODELS_BY_TABLE",
]
