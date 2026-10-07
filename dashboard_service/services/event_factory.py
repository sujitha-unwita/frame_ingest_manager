"""LLD SW S3.9 dashboard data contract -- payload builders.

WHERE THIS SITS
===============

    control_service/*   observes something worth telling the operator
         |
         v
    dashboard_service/services/event_factory.py (this file) builds JSON payload
         |
         v
    dashboard_service/services/dashboard_coordinator.py  publish()
         |
         v
    dashboard_service/operator_panel/client.py  format_event()   one line per event

WHY PURE FUNCTIONS
==================

No state, no config, no imports from the rest of the service. The wire format
lives in exactly one place, so when S3.9 changes this is the only file that
changes -- and neither McuLink nor the WebSocket layer can grow its own
slightly-different version of a payload.

Every builder returns a plain dict with an "event" key. The coordinator
serialises it with websocket.send_json(); nothing here does any I/O.

CONTRACT DEVIATIONS ARE FLAGGED, NEVER SILENT
=============================================

Several payloads carry fields S3.9 does not define. Each is documented in its
function's docstring with the reason. A gap in the spec gets closed VISIBLY,
so the next reader can tell a deliberate extension from an accident.

DATA IN   : plain values from control_service/* and DashboardCoordinator
DATA OUT  : dicts, JSON-serialised by the WebSocketHub
CREATED BY: nothing -- module-level functions, called not instantiated
"""

from __future__ import annotations

import time


def defect_event(
    trigger_id: int,
    camera_id: str,
    tile_ids: list[int] | None,
    frame_variant: str | None,
    defect_class: str,
    confidence: float,
    world_x1_mm: float,
    world_y1_mm: float,
    world_x2_mm: float,
    world_y2_mm: float,
    bbox_x1_px: float,
    bbox_y1_px: float,
    bbox_x2_px: float,
    bbox_y2_px: float,
    crop_thumbnail_ref: str,
) -> dict[str, object]:
    """Build a committed live observation, not a numbered final defect.

    trigger_id is the MCU sequence. Pixel bounds are camera-local; millimetre
    bounds come from the coordinate mapper. The legacy crop_thumbnail_ref name
    now holds the whole-frame S3 key, not a crop or a public URL. Unknown tile
    provenance and capture variant are null. Internal detection UUIDs are not
    exposed by this dashboard contract.
    """
    return {
        "event": "defect_event",
        "trigger_id": trigger_id,
        "camera_id": camera_id,
        "tile_ids": sorted(tile_ids) if tile_ids is not None else None,
        "frame_variant": frame_variant,
        "class": defect_class,
        "confidence": confidence,
        "world_x1_mm": world_x1_mm,
        "world_y1_mm": world_y1_mm,
        "world_x2_mm": world_x2_mm,
        "world_y2_mm": world_y2_mm,
        "bbox_x1_px": bbox_x1_px,
        "bbox_y1_px": bbox_y1_px,
        "bbox_x2_px": bbox_x2_px,
        "bbox_y2_px": bbox_y2_px,
        "crop_thumbnail_ref": crop_thumbnail_ref,
    }


def system_status(
    trigger_id: int | None,
    duty_cycle: float | None,
    queue_depth: int,
    capture_paused: bool,
    last_trigger_age_ms: float | None,
    link_state: str,
    last_heartbeat_age_ms: float,
    missed_intervals: int,
    is_armed: bool,
    faults: list[str],
) -> dict[str, object]:
    """Build the health strip -- LLD SW S3.8a "Dashboard heartbeat mirroring".

    Pushed on its OWN 500ms-1s timer, per S3.8a: "not only when a new event
    exists". An opportunistic push tied to defect events goes silent exactly
    when the pipeline stalls -- the moment the operator most needs to see
    something is wrong.

    The payload draws from THREE sources, assembled by dashboard_mirror.py:
        link_state, last_heartbeat_age_ms, missed_intervals  <- heartbeat
        trigger_id, last_trigger_age_ms, capture_paused       <- telemetry/state
        duty_cycle                                           <- GPU monitor
        queue_depth                                          <- processing queue

    DEVIATION FROM S3.9: `missed_intervals` added. S3.8a defines
    LINK_DEGRADED as covering both 1 and 2 missed intervals, which are
    operationally different -- 2 is one beat away from LINK_DOWN. Without
    this field the dashboard cannot tell a hiccup from a link about to drop.

    Args:
        trigger_id: Most recent trigger seen.
        duty_cycle: Last inference's mean device-wide GPU activity, 0..100,
            or None when that inference had no samples. Not a live idle reading.
        queue_depth: Frames waiting for inference.
        capture_paused: True whenever the MCU is not RUNNING.
        last_trigger_age_ms: Time since the last trigger arrived, or None
            if none has ever arrived. None is NOT zero.
        link_state: LINK_UP / LINK_DEGRADED / LINK_DOWN. S3.8a names
            these three and no others.
        last_heartbeat_age_ms: Time since the last heartbeat arrived.
        missed_intervals: Intervals of silence; disambiguates the two
            LINK_DEGRADED bands.
        is_armed: [OURS -- Pooja] True while the MCU reports ARMED, i.e. it
            has self-checked and is waiting for START CAPTURE. Goes False the
            moment it leaves ARMED, so the panel enables the button on true
            and disables it on false without tracking state itself.
        faults: [OURS -- Pooja] Decoded status_flags names, e.g.
            ["encoder_fault"]. EMPTY LIST means no faults, not unknown.
            Carried on every push -- this is the ONLY channel for faults.

    Returns:
        The system_status payload.
    """
    return {
        "event": "system_status",
        "trigger_id": trigger_id,
        "duty_cycle": duty_cycle,
        "queue_depth": queue_depth,
        "capture_paused": capture_paused,
        "last_trigger_age_ms": last_trigger_age_ms,
        "link_state": link_state,
        "last_heartbeat_age_ms": last_heartbeat_age_ms,
        "missed_intervals": missed_intervals,
        "is_armed": is_armed,
        "faults": list(faults),
    }


def capture_paused(paused: bool, source: str) -> dict[str, object]:
    """Build a paused/running transition notice.

    `source` distinguishes an operator pause from a fault-triggered one,
    which S3.9 requires the UI to render differently -- an operator pause is
    normal, a fault pause needs attention.

    NOTE ON TRUTH: emitted when the MCU's OBSERVED state changes, not when a
    command is acknowledged. A command is a request; the heartbeat is the
    truth. See control_service/mcu_state.py for why that matters.

    Args:
        paused: True whenever the MCU is not RUNNING.
        source: One of constants.SOURCE_* -- operator, mcu, fault, snapshot.

    Returns:
        The capture_paused payload.
    """
    return {
        "event": "capture_paused",
        "capture_paused": paused,
        "source": source,
        "timestamp": time.time(),
    }


def command_failed(command: str, cmd_seq: int, reason: str) -> dict[str, object]:
    """Build a notice that the MCU never acknowledged a command.

    Emitted only after every retry is exhausted. The operator pressed a
    button and nothing happened, so they must be told -- a silently dropped
    pause is a safety problem, not a cosmetic one.

    Args:
        command: Command name, e.g. "pause".
        cmd_seq: The sequence that went unacknowledged, or -1 when the
            failure happened before a sequence was allocated.
        reason: Human-readable cause.

    Returns:
        The command_failed payload.
    """
    return {
        "event": "command_failed",
        "command": command,
        "cmd_seq": cmd_seq,
        "reason": reason,
    }


def missed_trigger_alarm(
    expected_trigger_id: int, last_seen_trigger_id: int, gap_count: int
) -> dict[str, object]:
    """Build the alarm for a gap in trigger_id.

    A gap means fabric passed uninspected. LLD SW S3.6 is explicit that this
    must alarm on the panel, "not just to a backend log" -- nobody reads the
    log while the line is running, and the fabric is on the roll by then.

    Args:
        expected_trigger_id: The id that should have arrived next.
        last_seen_trigger_id: The last id actually received.
        gap_count: How many triggers were skipped.

    Returns:
        The missed_trigger_alarm payload.
    """
    return {
        "event": "missed_trigger_alarm",
        "expected_trigger_id": expected_trigger_id,
        "last_seen_trigger_id": last_seen_trigger_id,
        "gap_count": gap_count,
    }


def mcu_reboot(
    previous_uptime_s: int, current_uptime_s: int
) -> dict[str, object]:
    """Build a notice that the MCU restarted unexpectedly.

    HW LLD S3.5.2: "uptime_s reset detects an unexpected MCU reboot even if
    heartbeats never stopped." Without this the crash is invisible -- beats
    resume within 100 ms of the restart, so link_state stays LINK_UP
    throughout and nothing else in the system notices.

    Args:
        previous_uptime_s: Uptime reported on the beat before the reset.
        current_uptime_s: Uptime reported on the first beat after it.

    Returns:
        The mcu_reboot payload.
    """
    return {
        "event": "mcu_reboot",
        "previous_uptime_s": previous_uptime_s,
        "current_uptime_s": current_uptime_s,
        "timestamp": time.time(),
    }


def heartbeat_missed(dropped: int, total_missed: int) -> dict[str, object]:
    """Build a notice that beats were genuinely lost, per the MCU's clock.

    NOT IN S3.9 -- added because it answers a question link_state cannot.

    THE DISTINCTION THIS EXISTS FOR:

        link_state       derived from HOST ARRIVAL time. Polluted by USB
                         latency batching and host scheduling. Answers
                         "how stale is my information?" -- and CANNOT tell
                         a late beat from a lost one.

        heartbeat_missed derived from the MCU's OWN timestamp, written
                         before the frame reaches USB. Immune to both.
                         Answers "was a beat actually lost?" -- definitively,
                         but only once the next beat arrives.

    A LINK_DEGRADED with no heartbeat_missed beside it is a transport
    hiccup -- the beat arrived late, nothing was lost. A heartbeat_missed is
    a real hole in the MCU's output.

    Args:
        dropped: Beats lost immediately before the beat just received.
        total_missed: Running total since the service started.

    Returns:
        The heartbeat_missed payload.
    """
    return {
        "event": "heartbeat_missed",
        "dropped": dropped,
        "total_missed": total_missed,
        "timestamp": time.time(),
    }
