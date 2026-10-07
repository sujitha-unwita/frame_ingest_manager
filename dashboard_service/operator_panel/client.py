"""Panel stand-in -- what the Raspberry Pi touch panel would show.

WHERE THIS SITS
===============

    MCU <--RS-422/USB--> control_service/McuLink
                                  |
                        dashboard_service/routers/websocket.py
                                  |
                          WebSocket (JSON)
                                  |
    dashboard_service/operator_panel/client.py  (this file)  --  mock panel

Started by `python main.py panel`.

THE ONE IDEA WORTH TAKING FROM THIS FILE: SILENCE MUST BE VISIBLE
=================================================================

If the AGX process dies, or the network drops, or the status timer crashes,
this client would otherwise keep displaying the last numbers it received --
forever. The operator would see LINK_UP, a healthy duty cycle and no defects
scrolling past, and would reasonably conclude the fabric is clean.

In fact nothing is being inspected at all.

So there are TWO independent failure detectors, because they catch different
things:

    ServerWatchdog       the socket is OPEN but the AGX has gone quiet.
                         Watches system_status specifically -- the one event
                         with a contract-guaranteed cadence (LLD SW S3.8a
                         "Dashboard heartbeat mirroring") -- so its absence
                         is unambiguous. Also catches the case where defects
                         still flow but the status timer has died.

    ConnectionClosed     the socket itself broke: AGX process died, cable
                         pulled, switch rebooted.

Neither may leave a stale panel looking healthy. This is the same
three-missed-intervals rule the AGX applies to the MCU heartbeat, one layer
up -- and the same reasoning as heartbeat_receiver.py's link bands.

DATA IN   : JSON events from routers/websocket.py over the WebSocket
DATA OUT  : operator_stop_request / operator_resume_request, and printed lines
CREATED BY: main.py panel
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from config.config_loader import AttrDict

try:
    import websockets
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "websockets is required for the dashboard client. Install it with:\n"
        "    python3 -m pip install websockets"
    ) from _exc

#: Friendly name -> the operator_* message shared command model expects.
COMMANDS: dict[str, str] = {
    "pause": "operator_stop_request",
    "resume": "operator_resume_request",
    "start": "operator_start_capture",
}

#: The panel's on-screen transcript, recorded to the log FILE only when one
#: is configured. Never reaches the console -- print() already put it there.
_TRANSCRIPT = logging.getLogger("panel.transcript")


def say(text: str, file: Any = None) -> None:
    """Print one panel line and record it in the log file.

    The panel is a live display: it scrolls away, and a fault you only notice
    afterwards is gone. Every line the operator saw is written to the file so
    the sequence can be read back.

    ALWAYS flushed -- a panel behind a pipe or a redirect must not buffer, or
    the operator sees nothing until the buffer fills. Callers must NOT pass
    flush=; it is not optional here.

    Args:
        text: The line to show. Printed verbatim.
        file: Stream to print to. Defaults to stdout; pass sys.stderr for
            diagnostics that should not mix with the panel display.
    """
    print(text, file=file, flush=True)
    try:
        _TRANSCRIPT.info(text)
    except Exception:  # pragma: no cover - logging must never break the panel
        pass


#: Left margin for lines not tied to a specific trigger.
_GUTTER = " " * 13

# These are panel-side defaults, not AGX config.yaml values. The real
# dashboard owns its own polling/timeout policy; this stand-in uses the values
# documented in DASHBOARD_API.md so it can exercise the same contract.
HTTP_FALLBACK_POLL_INTERVAL_S = 0.5
HTTP_FALLBACK_TIMEOUT_S = 3.0


class HttpFallbackError(OSError):
    """Raised when one HTTP fallback request cannot be completed."""


class PanelEventCursor:
    """Tracks the one event stream/cursor across WebSocket and HTTP modes."""

    def __init__(self) -> None:
        """Start before any retained event has been observed."""
        self.stream_id: str | None = None
        self.last_processed_event_id = 0

    def observe(self, event: Mapping[str, Any]) -> bool:
        """Record an event's delivery metadata and reject replay duplicates.

        Snapshot messages such as system_status intentionally have no event_id
        and are always returned as new display data.
        """
        event_id = event.get("event_id")
        stream_id = event.get("stream_id")
        if event_id is None:
            return True
        if not isinstance(event_id, int) or event_id < 1:
            say(f"{_GUTTER}[invalid event_id {event_id!r}; displaying anyway]")
            return True
        if not isinstance(stream_id, str) or not stream_id:
            say(f"{_GUTTER}[event {event_id} has no stream_id; displaying anyway]")
            return True

        if self.stream_id is not None and stream_id != self.stream_id:
            say(f"{_GUTTER}*** AGX EVENT STREAM RESTARTED; RESYNCING ***")
            self.last_processed_event_id = 0
        self.stream_id = stream_id

        if event_id <= self.last_processed_event_id:
            return False
        self.last_processed_event_id = event_id
        return True


class ServerWatchdog:
    """Flags the panel as stale when the AGX stops sending system_status.

    Attributes:
        timeout_s: Silence tolerated before declaring the display stale.
        check_interval_s: How often the check runs.
        last_status_at: monotonic() of the most recent system_status.
        stale: Whether the display is currently flagged stale.
    """

    def __init__(self, timeout_s: float, check_interval_s: float) -> None:
        """Start the watchdog assuming the link is currently healthy.

        Args:
            timeout_s: Silence tolerated before flagging stale.
            check_interval_s: How often to test that.
        """
        self.timeout_s = timeout_s
        self.check_interval_s = check_interval_s
        # monotonic is fine here: this measures seconds-scale silence, not the
        # sub-interval heartbeat ages that need perf_counter's resolution, and
        # it never crosses into the AGX's clock domain.
        self.last_status_at = time.monotonic()
        self.stale = False
        self.stale_event = asyncio.Event()

    def mark_alive(self) -> None:
        """Reset the timer. Call on every system_status received."""
        self.last_status_at = time.monotonic()
        if self.stale:
            self.stale = False
            say(f"{_GUTTER}*** AGX LINK RECOVERED ***")

    async def run(self) -> None:
        """Watch for silence forever, printing a warning when it starts.

        Raises:
            asyncio.CancelledError: Propagated on shutdown.
        """
        while True:
            try:
                await asyncio.sleep(self.check_interval_s)
                age = time.monotonic() - self.last_status_at
                if age > self.timeout_s and not self.stale:
                    self.stale = True
                    self.stale_event.set()
                    say(
                        f"{_GUTTER}!!! NO STATUS FROM AGX FOR {age:.1f}s -- "
                        f"DISPLAY IS STALE, DO NOT TRUST IT !!!",
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # The watchdog is the last line of defence; it must not be the
                # thing that dies quietly.
                say(
                    f"{_GUTTER}watchdog error: {exc}",
                    file=sys.stderr,
                )


def format_event(event: Mapping[str, Any]) -> str:
    """Render one event as a single readable line.

    Args:
        event: The decoded JSON event.

    Returns:
        A formatted line, falling back to raw JSON for unknown event types so
        a new event added on the AGX is still visible here rather than
        silently swallowed.
    """
    kind = event.get("event")

    try:
        if kind == "defect_event":
            tiles = event["tile_ids"]
            tile_text = ",".join(str(tile) for tile in tiles) if tiles else "unknown"
            return (
                f"[trig {event['trigger_id']:>4}] DEFECT  "
                f"{event['camera_id']} {event['frame_variant']}  "
                f"{event['class']:<12} conf {event['confidence']:.2f}  "
                f"tiles {tile_text:<5} "
                f"x [{event['world_x1_mm']:.1f}, {event['world_x2_mm']:.1f}] mm "
                f"y [{event['world_y1_mm']:.1f}, {event['world_y2_mm']:.1f}] mm"
            )

        if kind == "system_status":
            link = event["link_state"]
            # LINK_DEGRADED covers both 1 and 2 intervals of silence. Show
            # which: 2 is one interval from LINK_DOWN, 1 is a hiccup.
            missed = event.get("missed_intervals", 0)
            if link == "LINK_DEGRADED":
                link = f"{link}({missed})"
            # Both may be None: no trigger has EVER arrived. Rendered as
            # "--" and "never", not as 0 and a process-uptime figure.
            trig = event["trigger_id"]
            trig_text = "  --" if trig is None else f"{trig:>4}"
            age = event["last_trigger_age_ms"]
            age_text = (
                "no trigger yet" if age is None
                else f"last trigger {age:.0f}ms ago"
            )
            # [OURS -- Pooja] ARMED beats PAUSED as a label: both mean "not
            # inspecting", but ARMED means "press START", and PAUSED means
            # "someone stopped a roll in progress". A FAULT list outranks
            # both -- that is the thing an operator must act on.
            faults = event.get("faults") or []
            if faults:
                tail = "  [FAULT: " + ", ".join(faults) + "]"
            elif event.get("is_armed"):
                tail = "  [ARMED -- press s to start]"
            elif event["capture_paused"]:
                tail = "  [PAUSED]"
            else:
                tail = ""

            duty_cycle = event.get("duty_cycle")
            gpu_text = "N/A" if duty_cycle is None else f"{duty_cycle:.0f}%"
            return (
                f"[trig {trig_text}] status  "
                f"GPU avg {gpu_text}  "
                f"queue {event['queue_depth']}  "
                f"{link}  "
                f"hb {event['last_heartbeat_age_ms']:>5.1f}ms  "
                f"missed {missed}  "
                f"{age_text}{tail}"
            )

        if kind == "capture_paused":
            state = "PAUSED" if event["capture_paused"] else "RUNNING"
            return f"{_GUTTER}*** {state} (source: {event['source']}) ***"

        if kind == "command_failed":
            return (
                f"{_GUTTER}!!! COMMAND FAILED: {event['command']} "
                f"(seq {event['cmd_seq']}) -- {event['reason']}"
            )

        if kind == "missed_trigger_alarm":
            return (
                f"{_GUTTER}!!! MISSED TRIGGER: expected "
                f"{event['expected_trigger_id']}, {event['gap_count']} "
                f"trigger(s) of fabric UNINSPECTED !!!"
            )

        if kind == "mcu_reboot":
            return (
                f"{_GUTTER}*** MCU REBOOTED: uptime "
                f"{event['previous_uptime_s']}s -> "
                f"{event['current_uptime_s']}s ***"
            )

        if kind == "heartbeat_missed":
            return (
                f"{_GUTTER}!!! HEARTBEAT LOST: {event['dropped']} beat(s), "
                f"{event['total_missed']} total (MCU clock -- a REAL loss, "
                f"not just a late arrival) !!!"
            )

    except (KeyError, TypeError, ValueError) as exc:
        # A malformed or newly-extended event must still be VISIBLE. Hiding it
        # behind a formatting crash is exactly the stale-display failure this
        # file exists to prevent.
        return (
            f"{_GUTTER}[unformattable {kind} event: {exc}] {json.dumps(event)}"
        )

    return json.dumps(event, separators=(",", ":"))


def _http_base_url(websocket_url: str) -> str:
    """Derive the matching AGX HTTP origin from one WebSocket URL."""
    parts = urlsplit(websocket_url)
    if parts.scheme not in ("ws", "wss") or not parts.netloc:
        raise HttpFallbackError(
            f"cannot derive HTTP fallback URL from WebSocket URL {websocket_url!r}"
        )
    scheme = "https" if parts.scheme == "wss" else "http"
    return urlunsplit((scheme, parts.netloc, "", "", ""))


def _http_json_request(
    method: str,
    url: str,
    body: Mapping[str, object] | None,
    timeout_s: float,
) -> dict[str, Any]:
    """Perform one blocking JSON request for asyncio.to_thread()."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url=url,
        data=data,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HttpFallbackError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (URLError, OSError, TimeoutError) as exc:
        raise HttpFallbackError(f"cannot reach {url}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HttpFallbackError(f"invalid JSON from {url}: {exc}") from exc

    if not isinstance(decoded, dict):
        raise HttpFallbackError(f"expected JSON object from {url}")
    return decoded


class HttpFallbackClient:
    """Panel-side HTTP fallback transport using the documented FastAPI API.

    It shares one PanelEventCursor with normal WebSocket mode.  Thus an event
    seen over WebSocket is not displayed again when HTTP fallback asks for
    events after that cursor.
    """

    def __init__(self, websocket_url: str, cursor: PanelEventCursor) -> None:
        """Create an HTTP fallback client for the AGX serving websocket_url."""
        self.base_url = _http_base_url(websocket_url)
        self.cursor = cursor

    async def request(
        self, method: str, path: str, body: Mapping[str, object] | None = None
    ) -> dict[str, Any]:
        """Run a blocking standard-library HTTP request off the event loop."""
        return await asyncio.to_thread(
            _http_json_request,
            method,
            f"{self.base_url}{path}",
            body,
            HTTP_FALLBACK_TIMEOUT_S,
        )

    async def acknowledge(self) -> None:
        """Confirm every retained event the panel has processed so far."""
        if self.cursor.stream_id is None or self.cursor.last_processed_event_id < 1:
            return
        await self.request(
            "POST",
            "/api/v1/events/ack",
            {
                "stream_id": self.cursor.stream_id,
                "last_processed_event_id": self.cursor.last_processed_event_id,
            },
        )

    async def process_event(
        self, event: Mapping[str, Any], raw_output: bool
    ) -> bool:
        """Display one event once and update the shared event cursor."""
        if not self.cursor.observe(event):
            return False
        say(
            json.dumps(event, separators=(",", ":"))
            if raw_output else format_event(event)
        )
        return True

    async def poll_once(self, raw_output: bool) -> None:
        """Replay events, acknowledge them, then show fresh current status."""
        after = self.cursor.last_processed_event_id
        replay = await self.request("GET", f"/api/v1/events?after={after}")
        stream_id = replay.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id:
            raise HttpFallbackError("event replay response omitted stream_id")

        changed_stream = (
            self.cursor.stream_id is not None
            and stream_id != self.cursor.stream_id
        )
        if changed_stream:
            say(f"{_GUTTER}*** AGX RESTARTED; EVENT HISTORY RESYNC REQUIRED ***")
            self.cursor.stream_id = stream_id
            self.cursor.last_processed_event_id = 0

        raw_events = replay.get("events")
        if not isinstance(raw_events, list):
            raise HttpFallbackError("event replay response has invalid events list")
        for event in raw_events:
            if not isinstance(event, Mapping):
                say(f"{_GUTTER}[invalid replay event: {event!r}]")
                continue
            await self.process_event(event, raw_output)

        if replay.get("resync_required") or changed_stream:
            say(f"{_GUTTER}!!! EVENT HISTORY GAP; CURRENT STATUS RESYNCED !!!")
            latest = replay.get("latest_event_id")
            if not isinstance(latest, int) or latest < 0:
                raise HttpFallbackError("event replay response has invalid latest_event_id")
            self.cursor.last_processed_event_id = latest

        await self.acknowledge()
        status = await self.request("GET", "/api/v1/status")
        await self.process_event(status, raw_output)

    async def send(self, message: str) -> None:
        """WebSocket-like command sender used by existing panel controls."""
        try:
            payload = json.loads(message)
        except json.JSONDecodeError as exc:
            raise HttpFallbackError(f"cannot encode HTTP command: {exc}") from exc
        if not isinstance(payload, dict):
            raise HttpFallbackError("HTTP command must be a JSON object")
        result = await self.request("POST", "/api/v1/commands", payload)
        if not result.get("acknowledged"):
            say(
                f"{_GUTTER}!!! COMMAND UNCONFIRMED: {result.get('command')} -- "
                f"{result.get('reason') or 'no matching ACK'}"
            )


async def send_command(connection: Any, name: str) -> None:
    """Send one operator command by friendly name.

    Args:
        connection: The open WebSocket connection.
        name: A key of COMMANDS, e.g. "pause".
    """
    try:
        await connection.send(json.dumps({"command": COMMANDS[name]}))
        say(f"{_GUTTER}--> sent {name} ({COMMANDS[name]})")
    except (websockets.exceptions.WebSocketException, OSError) as exc:
        say(f"{_GUTTER}!!! could not send {name}: {exc}")


async def interactive_console(connection: Any) -> None:
    """Read keystrokes from stdin so commands can be sent on demand.

    input() is blocking, so it runs in a worker thread via asyncio.to_thread.
    Calling it directly would freeze the receive loop -- the same reasoning
    that puts pyserial's blocking read on a thread in transport.py.

    Args:
        connection: The open WebSocket connection.
    """
    say("commands: s = start capture, p = pause, r = resume, q = quit")
    while True:
        try:
            line = (await asyncio.to_thread(input)).strip().lower()
        except (EOFError, RuntimeError):
            return  # stdin closed -- piped input ran out, or no terminal
        except asyncio.CancelledError:
            raise

        if line in ("s", "start"):
            await send_command(connection, "start")
        elif line in ("p", "pause"):
            await send_command(connection, "pause")
        elif line in ("r", "resume"):
            await send_command(connection, "resume")
        elif line in ("q", "quit"):
            with contextlib.suppress(Exception):
                await connection.close()
            return
        elif line:
            say("commands: s = start capture, p = pause, r = resume, "
                  "q = quit")


async def demo_sequence(connection: Any, pause_at_s: float = 5.0) -> None:
    """Run an unattended demo: pause, then resume.

    Args:
        connection: The open WebSocket connection.
        pause_at_s: Seconds to wait before pausing, and again before resuming.
    """
    await asyncio.sleep(pause_at_s)
    await send_command(connection, "pause")
    await asyncio.sleep(pause_at_s)
    await send_command(connection, "resume")


async def run_once(
    config: AttrDict, options: Mapping[str, Any], cursor: PanelEventCursor
) -> bool:
    """Connect once and stream events until the connection ends.

    Args:
        config: The whole validated configuration.
        options: Runtime options -- url, send, interactive, demo, raw,
            status_timeout.
        cursor: Shared event-delivery cursor, preserved into HTTP fallback.

    Returns:
        True if the caller should try to reconnect, False to stop.
    """
    dashboard = config.dashboard
    url = options.get("url") or dashboard.ws_url

    try:
        connection = await websockets.connect(url)
    except (OSError, websockets.exceptions.WebSocketException) as exc:
        say(
            f"{_GUTTER}!!! CANNOT REACH AGX at {url} "
            f"({type(exc).__name__}: {exc})",
        )
        return True

    say(f"connected to {url}")
    fallback = HttpFallbackClient(url, cursor)
    watchdog = ServerWatchdog(
        timeout_s=(
            options.get("status_timeout") or dashboard.client_status_timeout_s
        ),
        check_interval_s=dashboard.client_status_check_interval_s,
    )

    # Anything that is not the receive loop runs as its own task -- the loop
    # below occupies this coroutine, exactly as the server's endpoint does.
    tasks = [asyncio.create_task(watchdog.run(), name="watchdog")]
    if options.get("interactive"):
        tasks.append(
            asyncio.create_task(interactive_console(connection), name="console")
        )
    if options.get("demo"):
        tasks.append(
            asyncio.create_task(demo_sequence(connection), name="demo")
        )

    raw_output = bool(options.get("raw"))
    try:
        if options.get("send"):
            await send_command(connection, options["send"])

        while True:
            # `recv()` can block forever while the socket remains open. Race
            # it against the watchdog's stale event so an AGX that goes quiet
            # actually enters HTTP fallback rather than merely printing a
            # warning and waiting forever for another WebSocket frame.
            receive_task = asyncio.create_task(
                connection.recv(), name="panel-websocket-receive"
            )
            stale_task = asyncio.create_task(
                watchdog.stale_event.wait(), name="panel-status-stale"
            )
            done, pending = await asyncio.wait(
                (receive_task, stale_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            if stale_task in done:
                # Consume the stale-task result so asyncio cannot report an
                # unobserved exception, then leave this session for fallback.
                stale_task.result()
                say(f"{_GUTTER}switching to HTTP fallback mode")
                return True

            message = receive_task.result()
            try:
                event = json.loads(message)
            except json.JSONDecodeError as exc:
                say(
                    f"{_GUTTER}[unparseable message: {exc}] {message!r}",
                )
                continue

            if event.get("event") == "system_status":
                watchdog.mark_alive()

            if not isinstance(event, Mapping):
                say(f"{_GUTTER}[invalid JSON event: {event!r}]")
                continue
            processed = await fallback.process_event(event, raw_output)
            if processed and event.get("event_id") is not None:
                try:
                    await fallback.acknowledge()
                except HttpFallbackError as exc:
                    # The event remains safely retained on AGX. The next HTTP
                    # fallback poll/acknowledgement will prune it.
                    say(f"{_GUTTER}event acknowledgement deferred: {exc}")
    except (websockets.exceptions.ConnectionClosed, OSError) as exc:
        # The socket itself broke. Distinct from the watchdog's case, where
        # the socket stays open but the AGX goes quiet. Both must be visible;
        # neither may leave a stale panel looking healthy.
        say(
            f"{_GUTTER}!!! CONNECTION TO AGX LOST ({type(exc).__name__}) -- "
            f"DISPLAY IS STALE !!!",
        )
    except asyncio.CancelledError:
        raise
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await connection.close()

    # A one-shot command or a finished demo has nothing left to do.
    one_shot = options.get("send") and not options.get("interactive")
    return not one_shot and not options.get("demo")


async def run_http_fallback(
    config: AttrDict,
    options: Mapping[str, Any],
    cursor: PanelEventCursor,
) -> bool:
    """Poll FastAPI fallback until WebSocket can be used again.

    Returns:
        True when WebSocket should be retried, False for a completed one-shot
        command or demo session.
    """
    url = options.get("url") or config.dashboard.ws_url
    fallback = HttpFallbackClient(url, cursor)
    raw_output = bool(options.get("raw"))
    one_shot = options.get("send") and not options.get("interactive")

    say(f"{_GUTTER}*** HTTP FALLBACK MODE: polling {fallback.base_url} ***")
    tasks: list[asyncio.Task[Any]] = []
    if options.get("interactive"):
        tasks.append(
            asyncio.create_task(interactive_console(fallback), name="console")
        )
    if options.get("demo"):
        tasks.append(
            asyncio.create_task(demo_sequence(fallback), name="demo")
        )

    try:
        if options.get("send"):
            await send_command(fallback, options["send"])
            if one_shot:
                await fallback.poll_once(raw_output)
                return False

        next_probe_at = time.monotonic()
        while True:
            try:
                await fallback.poll_once(raw_output)
            except HttpFallbackError as exc:
                say(f"{_GUTTER}HTTP fallback request failed: {exc}")

            # Polling and WebSocket recovery are independent: HTTP may keep
            # the display current even while the preferred stream is down.
            now = time.monotonic()
            if now >= next_probe_at:
                next_probe_at = now + config.dashboard.client_reconnect_delay_s
                try:
                    probe = await websockets.connect(url)
                except (OSError, websockets.exceptions.WebSocketException):
                    pass
                else:
                    with contextlib.suppress(Exception):
                        await probe.close()
                    say(f"{_GUTTER}*** WEBSOCKET RECOVERED; RETURNING TO LIVE MODE ***")
                    return True
            await asyncio.sleep(HTTP_FALLBACK_POLL_INTERVAL_S)
    except asyncio.CancelledError:
        raise
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def run(config: AttrDict, options: Mapping[str, Any]) -> int:
    """Connect, stream, and reconnect for as long as the panel should run.

    Args:
        config: The whole validated configuration.
        options: Runtime options, as for run_once, plus no_reconnect.

    Returns:
        Process exit code.
    """
    delay = config.dashboard.client_reconnect_delay_s
    cursor = PanelEventCursor()

    while True:
        should_retry = await run_once(config, options, cursor)
        if not should_retry or options.get("no_reconnect"):
            return 0
        if config.dashboard.fallback.available:
            should_retry = await run_http_fallback(config, options, cursor)
            if not should_retry:
                return 0
            # HTTP fallback established that WebSocket is reachable. Re-enter
            # the normal receive loop immediately; no avoidable stale window.
            continue
        say(f"{_GUTTER}reconnecting in {delay:.1f}s...")
        await asyncio.sleep(delay)
