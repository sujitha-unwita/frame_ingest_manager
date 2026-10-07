"""The Transport contract, and the real RS-422 link over USB.

WHERE THIS SITS
===============

    RS-422 (4-wire, full duplex)
            |
    USB serial adapter  (FTDI or CDC-ACM)
            |
    /dev/ttyUSB0                          <-- serial.port in config.yaml
            |
    pyserial Serial object                <-- BLOCKING api
            |
    SerialTransport  (this file)          <-- blocking -> async
            |
    mcu_link.McuLink.run()                <-- the ONE reader
    mcu_command.McuCommandSender          <-- the ONE writer

WHAT A TRANSPORT IS
===================

Just "the thing that gives me bytes and takes my bytes". Four methods:

    async open()                get ready
    async read()  -> bytes      give me whatever arrived
    async write(frame)          put these bytes out
    async close()               release it

Two implementations satisfy the ABC below:

    SerialTransport   this file        the real USB serial port
    MockMcu           simulator/       a simulated MCU, no hardware

McuLink cannot tell them apart. That is the point: parsing, CRC checking,
link health and command retry are the SAME CODE under test against the mock
and in production against the MCU. Only the byte source changes.

WHY THE PORT IS OPENED EXACTLY ONCE, BY EXACTLY ONE PROCESS
===========================================================

A tty is a byte stream with ONE kernel receive buffer. When two processes
read it, the kernel does not duplicate bytes -- each byte is delivered to
exactly one reader. Two readers therefore get disjoint, random halves of the
stream. A 13-byte heartbeat gets sliced 5/8 and NEITHER parser can decode it.

It fails SILENTLY. UnifiedStreamParser does what it is written to do: the ETX
check fails, it byte-slips, discarded_bytes climbs. No exception.

And there is a worse tail. The parser resyncs by scanning for STX (0x02),
which appears constantly inside payloads -- state=2 is PAUSED, command code
0x02 is resume, LEN=2 on every command frame. A fragment can resync onto a
false STX and produce a frame that passes the single-byte CRC by chance one
time in 256. That FABRICATES plausible telemetry, which is strictly worse
than losing it.

`serial.exclusive` in config.yaml turns that silent corruption into a loud
failure at open() -- on POSIX. It is a safety net, not the guarantee. The
guarantee is that McuLink is the only thing that constructs this class.

NOT AN ELECTRICAL PROBLEM: RS-422 is 4-wire, full duplex, point to point,
with separate differential pairs for TX and RX. No bus arbitration, no
driver-enable turnaround, no collision domain -- that is RS-485 half-duplex,
a different thing. The contention is entirely a file-descriptor problem on
the host, which is why it is fixable in software with no hardware change.

WHY A THREAD AND NOT pyserial-asyncio
=====================================

pyserial's read() is BLOCKING. Calling it from a coroutine stalls the whole
event loop -- including the WebSocket broadcast to every connected panel --
for the duration of the read. Three options were considered:

    pyserial-asyncio 0.6          official, but Windows support is a POLLING
                                  fallback rather than native Proactor
                                  integration; the PR to fix that is unmerged.
    pyserial-asyncio-fast 0.16    actively maintained (Home Assistant), but
                                  its PyPI classifiers list POSIX/macOS only.
                                  Development happens on Windows; target is a
                                  Jetson.
    asyncio.to_thread + pyserial  identical on both platforms, no new
                                  dependency.   <-- chosen

The deciding argument is platform parity: one code path that behaves the same
on the Windows dev box and the Jetson, so the real transport can be exercised
locally against a USB-TTL loopback. At 115200 baud with a 50 ms read timeout
that is ~20 thread hops per second -- nowhere near where a dedicated reader
thread would earn its extra complexity.

DATA IN   : config_loader.AttrDict
            command frames from mcu_command.McuCommandSender, via McuLink
DATA OUT  : raw byte chunks, fed to McuLink's UnifiedStreamParser
CREATED BY: control_service.mcu_link.build_transport()
"""

from __future__ import annotations

import serial


import abc
import asyncio
import logging
import os
from pathlib import Path

from config.config_loader import AttrDict

LOGGER = logging.getLogger(__name__)

#: sysfs template for the FTDI latency timer on Linux. Read for the startup
#: warning; never written, because writing needs root.
LATENCY_TIMER_SYSFS = "/sys/bus/usb-serial/devices/{device}/latency_timer"


class TransportError(Exception):
    """Raised when the link cannot be opened or has been lost.

    A distinct type so McuLink can tell a transport failure -- retryable, the
    cable may come back -- from a protocol or programming error, which
    retrying would only repeat.
    """


class Transport(abc.ABC):
    """The four methods McuLink needs from any byte source.

    An abstract base class rather than a typing.Protocol so the contract is
    ENFORCED: a subclass that forgets write() cannot be instantiated at all,
    instead of failing later at the first command. The contract is also
    written down in one place rather than implied by two classes happening to
    match.
    """

    @abc.abstractmethod
    async def open(self) -> None:
        """Make the link usable.

        Raises:
            TransportError: If the link cannot be opened.
        """

    @abc.abstractmethod
    async def read(self) -> bytes:
        """Return the next chunk of bytes.

        Returns:
            Bytes, possibly empty when nothing arrived within the timeout.
            An empty return is NORMAL, not an error -- McuLink simply loops.
            Frames are recovered by the parser across chunk boundaries, so a
            chunk carrying half a frame is fine.

        Raises:
            TransportError: If the link was lost mid-read.
        """

    @abc.abstractmethod
    async def write(self, frame: bytes) -> None:
        """Put one complete frame on the wire.

        Args:
            frame: The complete framed message. Must be written whole --
                partial frames corrupt the MCU's parser.

        Raises:
            TransportError: If the write failed or the link was lost.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Release the link. Must be safe to call more than once."""

    async def reopen(self) -> None:
        """Recover a lost link. Default implementation: close then open.

        Overridden by SerialTransport to add backoff. Present on the base so
        McuLink's reconnect path works against any transport without
        special-casing which one it holds.

        Raises:
            TransportError: If recovery fails.
        """
        await self.close()
        await self.open()

    def snapshot(self) -> dict[str, object]:
        """Return counters, for logging and diagnostics.

        Returns:
            A plain dict describing the link. Not a wire payload.
        """
        return {"transport": type(self).__name__}


class SerialTransport(Transport):
    """Async wrapper around one blocking pyserial port.

    Every blocking call is moved onto a worker thread with asyncio.to_thread,
    so the event loop stays free to serve WebSocket clients while a read is
    parked waiting for bytes.

    Attributes:
        config: Validated serial settings from config.yaml.
        port_name: The device path actually opened.
        reopen_count: How many times the port was reopened after loss.
        bytes_read: Running total of bytes pulled off the wire.
        bytes_written: Running total of bytes put on the wire.
    """

    def __init__(self, config: AttrDict) -> None:
        """Prepare the transport. Does NOT open the port -- call open().

        Opening is deliberately not done in __init__ so construction cannot
        block or raise, and the caller controls when hardware is touched.

        Args:
            config: Validated serial settings from config.yaml.
        """
        self.config = config
        self.port_name = config.port
        self._port: object | None = None
        self._closing = False

        # Serialises writes. Two coroutines writing at once could interleave
        # bytes MID-FRAME and hand the MCU a corrupt command.
        # McuCommandSender already serialises whole commands, so this is
        # defence in depth -- but it is cheap, and a future second writer
        # would otherwise reintroduce the bug silently.
        self._write_lock = asyncio.Lock()

        self.reopen_count = 0
        self.bytes_read = 0
        self.bytes_written = 0

    # ---- lifecycle -------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """Whether the underlying pyserial port is currently open.

        Returns:
            True if a port object exists and reports itself open.
        """
        port = self._port
        if port is None:
            return False
        try:
            return bool(getattr(port, "is_open", False))
        except OSError:
            # The device vanished between the check and the attribute read.
            return False

    async def open(self) -> None:
        """Open the serial port, applying every setting from config.yaml.

        Raises:
            TransportError: If pyserial is not installed, or the port cannot
                be opened -- missing device, permission denied, or already
                held by another process.
        """
        await asyncio.to_thread(self._open_blocking)
        self._log_latency_timer()

    def _open_blocking(self) -> None:
        """Open the port. Runs on a worker thread.

        Raises:
            TransportError: On any failure, with the cause made actionable.
        """

        # config_loader validated these against pyserial's accepted values,
        # so every lookup below is guaranteed to hit.
        bytesize_map = {
            5: serial.FIVEBITS, 6: serial.SIXBITS,
            7: serial.SEVENBITS, 8: serial.EIGHTBITS,
        }
        parity_map = {
            "N": serial.PARITY_NONE, "E": serial.PARITY_EVEN,
            "O": serial.PARITY_ODD, "M": serial.PARITY_MARK,
            "S": serial.PARITY_SPACE,
        }
        stopbits_map = {
            1: serial.STOPBITS_ONE,
            1.5: serial.STOPBITS_ONE_POINT_FIVE,
            2: serial.STOPBITS_TWO,
        }

        kwargs: dict[str, object] = {
            "port": self.port_name,
            "baudrate": self.config.baud,
            "bytesize": bytesize_map[self.config.bytesize],
            "parity": parity_map[self.config.parity],
            "stopbits": stopbits_map[self.config.stopbits],
            "timeout": self.config.read_timeout_s,
            "write_timeout": self.config.write_timeout_s,
        }

        # `exclusive` is POSIX-only -- it applies TIOCEXCL. On Windows the
        # parameter is a no-op, but only because the driver ALREADY enforces
        # exclusivity: a second open there fails with
        # PermissionError(13, 'Access is denied'). Passed only on POSIX so
        # the intent is explicit rather than relying on it being ignored.
        if os.name == "posix" and self.config.exclusive:
            kwargs["exclusive"] = True

        try:
            self._port = serial.Serial(**kwargs)
        except serial.SerialException as exc:
            raise TransportError(
                f"cannot open {self.port_name}: {exc}\n"
                f"  - is the adapter plugged in?  ls -l /dev/serial/by-id/\n"
                f"  - is another process already using it? Only McuLink may "
                f"open this port -- see the README.\n"
                f"  - permission denied? add your user to the 'dialout' group"
            ) from exc
        except (ValueError, OSError) as exc:
            raise TransportError(
                f"cannot open {self.port_name}: {type(exc).__name__}: {exc}"
            ) from exc

        # Discard anything the adapter buffered while we were not listening.
        # Without this the first read returns a stale partial frame, which the
        # parser resyncs past -- harmless, but it shows up as a
        # discarded_bytes count on every startup and masks the real signal.
        try:
            self._port.reset_input_buffer()
            self._port.reset_output_buffer()
        except (OSError, AttributeError) as exc:
            LOGGER.debug("could not flush buffers on open: %s", exc)

        self._closing = False
        LOGGER.info(
            "opened %s at %d baud, %d-%s-%g (exclusive=%s)",
            self.port_name, self.config.baud, self.config.bytesize,
            self.config.parity, self.config.stopbits,
            kwargs.get("exclusive", "n/a on this platform"),
        )

    def _log_latency_timer(self) -> None:
        """Warn if the FTDI latency timer disagrees with config.yaml.

        INFORMATIONAL ONLY -- setting it needs root. The factory default of
        16 ms aliases a perfect 100 ms heartbeat into 96 ms and 112 ms arrival
        buckets, and the 112 ms ones look late even though nothing was lost.
        Reporting it at startup means a run's log always records whether that
        was a factor.

        Never raises: a missing sysfs file just means the device is CDC-ACM
        rather than FTDI, which has no equivalent knob.
        """
        if os.name != "posix":
            LOGGER.debug(
                "latency timer check skipped: not POSIX. On Windows set it in "
                "Device Manager > Ports > Advanced > Latency Timer."
            )
            return

        device = Path(self.port_name).name
        sysfs_path = Path(LATENCY_TIMER_SYSFS.format(device=device))
        try:
            actual = int(sysfs_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            LOGGER.debug(
                "no latency_timer at %s -- device is probably CDC-ACM, which "
                "has no equivalent setting. Arrival jitter there is USB "
                "polling; that is what heartbeat.debounce_samples absorbs.",
                sysfs_path,
            )
            return

        expected = self.config.expected_latency_timer_ms
        if actual == expected:
            LOGGER.info("FTDI latency timer is %d ms, as configured", actual)
            return

        LOGGER.warning(
            "FTDI latency timer is %d ms but config.yaml expects %d ms. A high "
            "value batches short frames and makes a healthy 100 ms heartbeat "
            "read as late. Fix with: echo %d | sudo tee %s",
            actual, expected, expected, sysfs_path,
        )

    async def close(self) -> None:
        """Close the port. Safe to call more than once, and on a failed open.

        Never raises: close is called from shutdown paths and `finally`
        blocks, where an exception would mask whatever caused the shutdown.
        """
        self._closing = True
        port, self._port = self._port, None
        if port is None:
            return
        try:
            await asyncio.to_thread(port.close)
            LOGGER.info("closed %s", self.port_name)
        except (OSError, AttributeError) as exc:
            LOGGER.warning("error closing %s: %s", self.port_name, exc)

    # ---- I/O -------------------------------------------------------------

    async def read(self) -> bytes:
        """Read whatever bytes are available, blocking up to read_timeout_s.

        Returns:
            A chunk of bytes, possibly empty when the timeout expires with
            nothing arriving.

        Raises:
            TransportError: If the port is closed or was lost mid-read. The
                caller decides whether to reconnect.
        """
        if self._port is None:
            raise TransportError(f"{self.port_name} is not open")

        chunk = await asyncio.to_thread(self._read_blocking)
        if chunk:
            self.bytes_read += len(chunk)
        return chunk

    def _read_blocking(self) -> bytes:
        """Perform one blocking read. Runs on a worker thread.

        Reads in_waiting bytes when any are buffered, otherwise asks for one
        byte and lets the port's timeout bound the wait. That drains a full
        buffer in a single call rather than one byte at a time, while still
        parking cheaply when the line is idle.

        Returns:
            The bytes read, possibly empty.

        Raises:
            TransportError: If the device disappeared or the read failed.
        """
        port = self._port
        if port is None:
            raise TransportError(f"{self.port_name} closed during read")

        try:
            pending = port.in_waiting
        except (OSError, AttributeError) as exc:
            raise TransportError(
                f"{self.port_name} lost while checking in_waiting: {exc}"
            ) from exc

        try:
            return port.read(max(1, pending))
        except Exception as exc:
            # pyserial raises SerialException for a vanished device, and the
            # concrete class differs per platform backend. Caught broadly and
            # re-raised as one type, because every case means the same thing
            # to the caller -- the port is gone -- and the caller must not
            # have to import serial to find that out.
            if self._closing:
                return b""  # shutting down; a failing read here is expected
            raise TransportError(
                f"read from {self.port_name} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    async def write(self, frame: bytes) -> None:
        """Write one complete frame, then flush it to the wire.

        Serialised by a lock so two writers can never interleave bytes
        mid-frame and hand the MCU a corrupt command.

        Args:
            frame: The complete framed message.

        Raises:
            TransportError: If the port is closed, the write timed out, or
                the device was lost.
        """
        if self._port is None:
            raise TransportError(f"{self.port_name} is not open")

        async with self._write_lock:
            await asyncio.to_thread(self._write_blocking, frame)
        self.bytes_written += len(frame)

    def _write_blocking(self, frame: bytes) -> None:
        """Write and flush one frame. Runs on a worker thread.

        Args:
            frame: The complete framed message.

        Raises:
            TransportError: On timeout, short write, or device loss.
        """
        port = self._port
        if port is None:
            raise TransportError(f"{self.port_name} closed during write")

        try:
            written = port.write(frame)
            # flush() waits for the OS buffer to drain to the device. Without
            # it, write() returns as soon as the bytes are queued and the ACK
            # timeout would start counting before the command had actually
            # left the host.
            port.flush()
        except Exception as exc:
            raise TransportError(
                f"write to {self.port_name} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if written is not None and written != len(frame):
            # A short write means a partial frame reached the MCU. Its parser
            # discards the fragment, and our retry resends the whole frame
            # with the same sequence -- recoverable, but it must be visible.
            raise TransportError(
                f"short write to {self.port_name}: {written} of "
                f"{len(frame)} bytes"
            )

    # ---- reconnection ----------------------------------------------------

    async def reopen(self) -> None:
        """Close and reopen the port, backing off between attempts.

        Called by McuLink when a read raises TransportError. The adapter may
        have been unplugged, or the Jetson's USB stack may have re-enumerated
        it.

        Raises:
            TransportError: If reconnection is disabled in config.yaml.
            asyncio.CancelledError: On shutdown, propagated.
        """
        if not self.config.reconnect.enabled:
            raise TransportError(
                f"{self.port_name} was lost and serial.reconnect.enabled is "
                f"false"
            )

        await self.close()
        self._closing = False

        delay = self.config.reconnect.initial_delay_s
        attempt = 0
        while True:
            attempt += 1
            try:
                await asyncio.sleep(delay)
                await self.open()
                self.reopen_count += 1
                LOGGER.info(
                    "reopened %s after %d attempt(s) (reopen #%d)",
                    self.port_name, attempt, self.reopen_count,
                )
                return
            except asyncio.CancelledError:
                raise
            except TransportError as exc:
                # Backoff, capped. An adapter that is never coming back must
                # settle into a slow poll rather than pinning a CPU core
                # retrying open() thousands of times a second.
                delay = min(
                    delay * self.config.reconnect.backoff_factor,
                    self.config.reconnect.max_delay_s,
                )
                LOGGER.warning(
                    "reopen attempt %d for %s failed (%s); retrying in %.1fs",
                    attempt, self.port_name, exc, delay,
                )

    def snapshot(self) -> dict[str, object]:
        """Return counters, for logging and diagnostics.

        Returns:
            A plain dict of transport counters. Not a wire payload.
        """
        return {
            "transport": "serial",
            "port": self.port_name,
            "baud": self.config.baud,
            "is_open": self.is_open,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "reopen_count": self.reopen_count,
        }
