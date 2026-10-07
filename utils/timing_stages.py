"""Descriptions of instrumentation sites, not adjustable application settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StageDescription:
    """Explain a measurement's ownership, repetition and nested timing scope."""

    label: str
    component: str
    kind: str
    parent: str
    description: str


STAGES = {
    "workers_startup": StageDescription(
        "Start vision and audit workers", "AppOrchestrator.start", "lifecycle", "",
        "Once at startup; includes waiting for both workers to become ready.",
    ),
    "workers_shutdown": StageDescription(
        "Shut down workers", "AppOrchestrator.stop", "lifecycle", "",
        "Once at shutdown. An ok timing row does not prove every job was saved.",
    ),
    "operator_command": StageDescription(
        "Send MCU command and await response", "AppOrchestrator.handle_operator_command",
        "operation", "", "Once per measured operator command; not roll preparation.",
    ),
    "trigger_enqueue": StageDescription(
        "Submit trigger to vision", "AppOrchestrator.on_trigger", "operation", "",
        "Once per submission attempt. An error row is not an accepted trigger.",
    ),
    "vision_queue_wait": StageDescription(
        "Wait before vision begins", "VisionWorker.process", "wait", "",
        "Per trigger; from admission until vision begins processing.",
    ),
    "frame_bundle": StageDescription(
        "Select frame references and create bundle", "FrameIngestManager.create_bundle",
        "operation", "vision_total",
        "Once per trigger for all cameras. Selects paths and assigns metadata; no image bytes are read.",
    ),
    "image_read_decode": StageDescription(
        "Read and decode image for inference", "SahiDetectionEngine._detect_sync",
        "operation", "inference",
        "Once per camera per trigger; cv2.imread reads the file and decodes pixels. Includes OS cache effects; excludes colour conversion and prediction.",
    ),
    "inference": StageDescription(
        "Complete detection for one camera", "SahiDetectionEngine.detect", "total",
        "vision_total",
        "Once per camera per trigger. Includes read/decode, preprocessing, tiling, prediction, SAHI merging and thread scheduling; not GPU-only time.",
    ),
    "coordinate_mapping": StageDescription(
        "Map detections to physical coordinates", "CoordinateMapper.map_detection",
        "operation", "vision_total",
        "One measurement per camera, covering all detections from that frame.",
    ),
    "vision_total": StageDescription(
        "Complete vision work for one trigger", "VisionWorker.process", "total", "",
        "Includes frame bundle, both/all cameras and overhead; excludes earlier queue wait and later audit-queue insertion.",
    ),
    "audit_queue_put": StageDescription(
        "Send vision result to audit queue", "VisionWorker.run", "handoff", "",
        "Once per trigger. Includes waiting for queue space, not audit processing.",
    ),
    "model_startup": StageDescription(
        "Load model and warm up", "SahiDetectionEngine.start", "lifecycle", "",
        "Once at vision startup; not repeated for each frame.",
    ),
    "database_connect": StageDescription(
        "Open database pool", "AuditWorker.run", "lifecycle", "",
        "Once at audit startup.",
    ),
    "schema_check": StageDescription(
        "Check database schema", "SchemaChecker.check", "lifecycle", "",
        "Once at startup; includes acquiring a connection.",
    ),
    "roll_preparation": StageDescription(
        "Create or prepare roll", "AuditWriter.start_roll", "operation", "",
        "Per roll preparation request; includes persistence.",
    ),
    "audit_handoff_wait": StageDescription(
        "Wait between vision and audit", "AuditWorker.process", "wait", "",
        "Per trigger; from vision finishing to audit starting. Includes queue insertion and queue residence.",
    ),
    "frame_read": StageDescription(
        "Read original file bytes for S3", "AuditWorker.read_frame", "operation",
        "audit_total",
        "Per camera with detections; reads encoded bytes again for upload, with file checks and thread scheduling. This is NOT the inference image read.",
    ),
    "frame_audit": StageDescription(
        "Upload and save one camera frame", "AuditWriter.write_frame", "total",
        "audit_total",
        "Per camera with detections. Includes s3_upload and raw_database_transaction; do not add those again.",
    ),
    "s3_upload": StageDescription(
        "Upload whole frame to S3", "AuditWriter._upload_frame", "operation",
        "frame_audit", "Per audited camera frame; includes network and any SDK retries.",
    ),
    "raw_database_transaction": StageDescription(
        "Save trigger, media and raw detections", "AuditWriter.write_frame",
        "operation", "frame_audit",
        "Per audited camera frame; includes database transaction and commit.",
    ),
    "live_event_handoff": StageDescription(
        "Send live defects to control process", "AuditWorker.process", "handoff",
        "audit_total",
        "One timing for all live events of an audited camera frame; not browser delivery.",
    ),
    "deduplication_grading": StageDescription(
        "Merge detections and grade settled defects", "DeduplicationEngine.process_trigger",
        "operation", "audit_total", "Once per audited trigger, across all cameras.",
    ),
    "final_defect_write": StageDescription(
        "Save final defects and member links", "AuditWriter.write_final_defects",
        "operation", "audit_total",
        "Per trigger only when groups settle. Includes all settled groups for that trigger.",
    ),
    "audit_total": StageDescription(
        "Complete audit work for one trigger", "AuditWorker.process", "total", "",
        "All cameras, uploads, raw writes, merging and final writes. Excludes the preceding audit wait.",
    ),
    "trigger_to_audit": StageDescription(
        "Admission to end of audit", "AuditWorker.run", "end-to-end", "",
        "Includes vision, queue waiting and audit. Do not add stage totals to this value.",
    ),
    "panel_queue_wait": StageDescription(
        "Wait before dashboard broadcast", "AppOrchestrator._send_panels", "wait", "",
        "Per dashboard message, including status updates and separate defect events.",
    ),
    "websocket_broadcast": StageDescription(
        "Broadcast dashboard message", "AppOrchestrator._send_panels", "operation", "",
        "Per dashboard message; not confirmation that the browser displayed it.",
    ),
}


def describe_stage(stage: str) -> StageDescription:
    """Keep new or historical stage names visible even without a description."""
    return STAGES.get(
        stage, StageDescription(stage, "Unknown", "unclassified", "",
                                "No description is registered for this stage."),
    )
