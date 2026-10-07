#!/usr/bin/env python3
"""THE entry point for the VisionBar AGX service.

WHERE THIS SITS
===============

    main.py  (this file)          the ONLY thing you run
       |
 +-- serve     -> dashboard_service/app.py + control_service/McuLink
 +-- panel     -> dashboard_service/operator_panel/client.py
       +-- monitor   -> control_service/McuLink, headless, JSON on stdout
       +-- send      -> control_service/McuLink, one command, then exit
       +-- doctor    -> validate config, probe the port, check the latency
                        timer -- touches no hardware unless asked to
       +-- selftest  -> self_test.py

Every subcommand loads config.yaml through the same ConfigLoader, so there
is one place configuration is resolved and one object that carries it.

To see what a config file actually produces without starting anything:

    python config_loader.py

WHAT THIS REPLACES
==================

    heartbeat_receiver.py   \\
    telemetry_receiver.py    >  all three opened /dev/ttyUSB0 independently
    mcu_command_client.py   /

Three processes reading one tty means each byte goes to exactly ONE of them
-- the kernel does not duplicate. A 13-byte heartbeat gets sliced across two
parsers and neither can decode it. Now one process owns the port and the
three jobs are three subcommands, or three tasks inside `serve`.

USAGE
=====

    python main.py serve                  # the full service
    python main.py serve --real           # against real hardware
    python main.py serve --real --debug   # + log every heartbeat/frame
    python main.py panel --interactive    # the operator panel stand-in
    python main.py monitor                # headless, JSON events on stdout
    python main.py send pause             # one command; exit 0=ACK, 2=no ACK
    python main.py doctor                 # check config and port health
    python main.py selftest               # offline tests, no hardware

Global options, valid on every subcommand:

    --config PATH     use a different config.yaml
    --port DEVICE     override serial.port
    --mock / --real   override transport.kind

DATA IN   : argv, config.yaml
DATA OUT  : a running service, or a report, or an exit code
CREATED BY: you, on the command line
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

# working_code must be importable as the package root before anything else is
# imported, so `python /some/other/dir/main.py` works from any cwd.
_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import constants  # noqa: E402
from config.config_loader import (  # noqa: E402
    AttrDict,
    ConfigError,
    ConfigLoader,
    AppConfig,
    describe,
)

LOGGER = logging.getLogger("main")

#: Exit codes, so scripts can branch on the outcome.
EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_BAD_USAGE = 2
EXIT_INTERRUPTED = 130


def configure_logging(config: AppConfig) -> None:
    """Send application logs to stderr and an optional YAML-configured file.

    Called after configuration validation, before importing service modules.
    Relative file paths belong to the selected YAML's directory. A timestamp
    placeholder allows each process run to retain its own log file.
    """
    logging_config = config.logging
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_path = None
    if logging_config.file:
        timestamp = datetime.now().strftime(logging_config.filename_datefmt)
        log_path = Path(logging_config.file.replace("{timestamp}", timestamp))
        if not log_path.is_absolute():
            log_path = config.source_paths[0].parent / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(
            log_path,
            mode=logging_config.file_mode,
            encoding=logging_config.file_encoding,
        ))
    logging.basicConfig(
        level=getattr(logging, logging_config.level),
        format=logging_config.format,
        datefmt=logging_config.datefmt,
        handlers=handlers,
        force=True,
    )
    for name, level in logging_config.quiet_loggers.items():
        logging.getLogger(name).setLevel(getattr(logging, level))
    LOGGER.info("Logging configured; file=%s", log_path or "disabled")


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------

def cmd_serve(config: AttrDict, args: argparse.Namespace) -> int:
    """Run the full service: MCU link plus the dashboard WebSocket server.

    Args:
        config: The whole validated configuration.
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    # Load service dependencies only when this subcommand is selected.
    from app_orchestrator import AppOrchestrator


    orchestrator = AppOrchestrator(config)

    LOGGER.info(
        "listening on http://%s:%d%s",
        config.dashboard.host, config.dashboard.port, config.dashboard.ws_path,
    )
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("Stopped by user", file=sys.stderr)
        return EXIT_INTERRUPTED
    except OSError as exc:
        LOGGER.exception("Service failed with an operating-system error")
        print(
            f"cannot bind {config.dashboard.host}:{config.dashboard.port}: "
            f"{exc}",
            file=sys.stderr,
        )
        return EXIT_RUNTIME_ERROR
    return EXIT_OK


def cmd_panel(config: AttrDict, args: argparse.Namespace) -> int:
    """Run the operator panel stand-in.

    Args:
        config: The whole validated configuration.
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """

    from dashboard_service.operator_panel.client import run as run_client

    options = {
        "url": args.url,
        "send": args.send,
        "interactive": args.interactive,
        "demo": args.demo,
        "raw": args.raw,
        "no_reconnect": args.no_reconnect,
        "status_timeout": args.status_timeout,
    }
    try:
        return asyncio.run(run_client(config, options))
    except KeyboardInterrupt:
        print("Stopped by user", file=sys.stderr)
        return EXIT_INTERRUPTED


async def _run_monitor(config: AttrDict, raw: bool) -> int:
    """Run the MCU link headless, printing every event as JSON.

    The combined replacement for heartbeat_receiver.py and
    telemetry_receiver.py: heartbeats AND telemetry over ONE port owned by
    ONE process.

    Args:
        config: The whole validated configuration.
        raw: Unused placeholder for future formatting options.

    Returns:
        Process exit code.
    """

    from control_service.mcu_link import McuLink, build_transport
    from control_service.transport import TransportError

    async def emit(message: Mapping[str, Any]) -> None:
        """Print one event as a single line of JSON.

        Args:
            message: The event to print.
        """
        print(json.dumps(dict(message), separators=(",", ":")), flush=True)

    link = McuLink(
        config=config, transport=build_transport(config), publish=emit
    )
    try:
        await link.run()
    except TransportError as exc:
        print(f"link error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except asyncio.CancelledError:
        pass
    return EXIT_OK


def cmd_monitor(config: AttrDict, args: argparse.Namespace) -> int:
    """Run the MCU link with no dashboard, streaming JSON events.

    Args:
        config: The whole validated configuration.
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    try:
        return asyncio.run(_run_monitor(config, args.raw))
    except KeyboardInterrupt:
        print("Stopped by user", file=sys.stderr)
        return EXIT_INTERRUPTED


async def _run_send(config: AttrDict, command: str) -> int:
    """Bring the link up, send one command, report, and stop.

    Args:
        config: The whole validated configuration.
        command: The command name to send.

    Returns:
        EXIT_OK if acknowledged, EXIT_BAD_USAGE if not.
    """

    from control_service.mcu_link import McuLink, build_transport

    link = McuLink(config=config, transport=build_transport(config))
    task = asyncio.create_task(link.run(), name="mcu-link")

    try:
        # Let the link open and see a beat or two, so the command is not
        # raced against a port that is still coming up.
        await asyncio.sleep(config.heartbeat.interval_s * 2)
        result = await link.send_command(command)

        print(
            json.dumps(
                {
                    "event": "command_result",
                    "command": result.command,
                    "command_sequence": result.command_sequence,
                    "outcome": "ACK" if result.acknowledged else "NACK",
                    "attempts": result.attempts,
                    "nacked": result.nacked,
                    "reason": result.reason,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        return EXIT_OK if result.acknowledged else EXIT_BAD_USAGE
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


def cmd_send(config: AttrDict, args: argparse.Namespace) -> int:
    """Send one command to the MCU and exit.

    Replaces mcu_command_client.py -- but note the difference that matters:
    the old tool READ the port while waiting for its ACK, swallowing every
    heartbeat that arrived during the 600 ms retry window. This one uses the
    same single-reader architecture as the service.

    Args:
        config: The whole validated configuration.
        args: Parsed CLI arguments.

    Returns:
        EXIT_OK if the MCU acknowledged, EXIT_BAD_USAGE otherwise.
    """
    if args.command_name not in config.commands.codes:
        known = ", ".join(sorted(config.commands.codes))
        print(
            f"unknown command {args.command_name!r}; configured: {known}",
            file=sys.stderr,
        )
        return EXIT_BAD_USAGE

    try:
        return asyncio.run(_run_send(config, args.command_name))
    except KeyboardInterrupt:
        print("Stopped by user", file=sys.stderr)
        return EXIT_INTERRUPTED


def cmd_doctor(config: AttrDict, args: argparse.Namespace) -> int:
    """Check configuration and the environment without running the service.

    Reports, in order:
      1. the resolved config and its key settings
      2. whether required packages are importable
      3. whether the serial port exists and can be opened exclusively
      4. the FTDI latency timer, if there is one

    Args:
        config: The whole validated configuration.
        args: Parsed CLI arguments.

    Returns:
        EXIT_OK if nothing is obviously wrong, EXIT_RUNTIME_ERROR otherwise.
    """
    problems: list[str] = []

    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    for path in config.source_paths:
        print(f"  read: {path}")
    print()
    print(describe(config))

    print()
    print("=" * 70)
    print("DEPENDENCIES")
    print("=" * 70)
    for module, needed_for in (
        ("yaml", "reading the YAML config (required)"),
        ("serial", "the real serial transport"),
        ("fastapi", "the dashboard server"),
        ("uvicorn", "the dashboard server"),
        ("websockets", "the panel client"),
    ):
        try:
            __import__(module)
            print(f"  {module:<12} OK        ({needed_for})")
        except ImportError:
            note = "MISSING"
            print(f"  {module:<12} {note:<9} ({needed_for})")
            if module == "yaml":
                problems.append("PyYAML is required and is not installed")
            elif module == "serial" and not config.transport.is_mock:
                problems.append(
                    "pyserial is required for transport.kind: serial"
                )

    print()
    print("=" * 70)
    print("SERIAL PORT")
    print("=" * 70)
    if config.transport.is_mock:
        print("  transport.kind is 'mock' -- no hardware needed.")
        print("  Re-run with --real to check the port.")
    else:
        problems.extend(_check_serial_port(config))

    print()
    print("=" * 70)
    if problems:
        print(f"{len(problems)} PROBLEM(S) FOUND")
        print("=" * 70)
        for problem in problems:
            print(f"  - {problem}")
        return EXIT_RUNTIME_ERROR

    print("NO PROBLEMS FOUND")
    print("=" * 70)
    return EXIT_OK


def _check_serial_port(config: AttrDict) -> list[str]:
    """Probe the configured serial port and report what is wrong.

    Args:
        config: The whole validated configuration.

    Returns:
        A list of human-readable problems; empty if the port looks healthy.
    """

    problems: list[str] = []
    port_path = Path(config.serial.port)

    print(f"  port      : {config.serial.port}")
    if os.name == "posix":
        if port_path.exists():
            print("  exists    : yes")
        else:
            print("  exists    : NO")
            problems.append(
                f"{config.serial.port} does not exist. Find the right one "
                f"with: ls -l /dev/serial/by-id/"
            )

    try:
        from control_service.transport import SerialTransport, TransportError
    except ImportError as exc:
        return [f"Cannot check serial port: {exc}"]

    transport = SerialTransport(config.serial)
    try:
        asyncio.run(transport.open())
        print("  open      : OK (exclusive)" if config.serial.exclusive
              else "  open      : OK")
        asyncio.run(transport.close())
    except TransportError as exc:
        print("  open      : FAILED")
        problems.append(str(exc).splitlines()[0])
    except Exception as exc:  # pragma: no cover - defensive
        print("  open      : FAILED")
        problems.append(f"{type(exc).__name__}: {exc}")

    return problems


def cmd_selftest(config: AttrDict, args: argparse.Namespace) -> int:
    """Run the offline test suite.

    Args:
        config: The whole validated configuration (unused; self_test loads
            its own so it can exercise the loader).
        args: Parsed CLI arguments.

    Returns:
        The test suite's exit code.
    """

    import self_test

    return self_test.run(verbose=args.verbose, name_filter=args.filter)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def add_global_options(
    parser: argparse.ArgumentParser, *, suppress_defaults: bool
) -> None:
    """Add --config / --port / --mock / --real to a parser.

    Called TWICE for every option: once on the top-level parser and once on
    each subparser. That is what lets both of these work:

        python main.py --real doctor
        python main.py doctor --real

    argparse's subparsers otherwise only accept global options BEFORE the
    subcommand name, which is a genuinely surprising failure -- the error
    message says "unrecognized arguments" and does not hint at the ordering.

    The subparser copies use default=SUPPRESS so that an option NOT given
    after the subcommand leaves the namespace untouched, rather than
    overwriting a value that was given before it with None.

    Args:
        parser: The parser or subparser to extend.
        suppress_defaults: True for subparser copies, False for the top-level
            parser, which owns the real defaults.
    """
    value_default = argparse.SUPPRESS if suppress_defaults else None
    flag_default = argparse.SUPPRESS if suppress_defaults else False

    parser.add_argument(
        "--config", default=value_default, metavar="PATH",
        help="config file (default: config.yaml alongside main.py)",
    )
    parser.add_argument(
        "--port", default=value_default, metavar="DEVICE",
        help="override serial.port",
    )
    parser.add_argument(
        "--log-file", default=value_default, metavar="PATH",
        help="also write every log line, and the panel transcript, to PATH",
    )
    parser.add_argument(
        "--debug", action="store_true", default=flag_default,
        help="log every heartbeat, telemetry frame and ACK (sets "
             "logging.level=DEBUG)",
    )
    transport_group = parser.add_mutually_exclusive_group()
    transport_group.add_argument(
        "--mock", action="store_true", default=flag_default,
        help="force transport.kind = mock (the simulated MCU)",
    )
    transport_group.add_argument(
        "--real", action="store_true", default=flag_default,
        help="force transport.kind = serial (the real RS-422 port)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "VisionBar AGX control service. One process owns the RS-422 "
            "port; heartbeat, telemetry and commands all flow through it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py serve                 run the full service\n"
            "  python main.py serve --real          against real hardware\n"
            "  python main.py serve --real --debug  + log every frame\n"
            "  python main.py panel --interactive --log-file panel.log\n"
            "  python main.py panel --interactive   operator panel\n"
            "  python main.py monitor               headless JSON events\n"
            "  python main.py send pause            one command, then exit\n"
            "  python main.py doctor --real         check config and port\n"
            "  python main.py selftest              offline tests\n"
        ),
    )

    # Global options, applied before config is loaded. Added here AND to
    # every subparser, so they work on either side of the subcommand name.
    add_global_options(parser, suppress_defaults=False)

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    serve = subparsers.add_parser(
        "serve", help="run the MCU link and the dashboard WebSocket server"
    )
    add_global_options(serve, suppress_defaults=True)
    serve.set_defaults(handler=cmd_serve)

    panel = subparsers.add_parser(
        "panel", help="run the operator panel stand-in"
    )
    add_global_options(panel, suppress_defaults=True)
    panel.add_argument(
        "--url", default=None,
        help="WebSocket URL (default: built from the config)",
    )
    panel.add_argument(
        "--send", choices=("start", "pause", "resume"),
        help="send one command on connect (start = START CAPTURE)",
    )
    panel.add_argument(
        "--interactive", action="store_true",
        help="type p / r to pause and resume",
    )
    panel.add_argument(
        "--demo", action="store_true",
        help="auto pause at 5s, resume at 10s",
    )
    panel.add_argument(
        "--raw", action="store_true", help="print unformatted JSON"
    )
    panel.add_argument(
        "--no-reconnect", action="store_true",
        help="exit when the connection drops instead of retrying",
    )
    panel.add_argument(
        "--status-timeout", type=float, default=None, metavar="SECONDS",
        help="seconds without system_status before flagging the panel stale",
    )
    panel.set_defaults(handler=cmd_panel)

    monitor = subparsers.add_parser(
        "monitor",
        help="run the MCU link headless, printing every event as JSON",
    )
    add_global_options(monitor, suppress_defaults=True)
    monitor.add_argument(
        "--raw", action="store_true", help="reserved for future formatting"
    )
    monitor.set_defaults(handler=cmd_monitor)

    send = subparsers.add_parser(
        "send", help="send one command to the MCU and exit"
    )
    add_global_options(send, suppress_defaults=True)
    send.add_argument(
        "command_name", metavar="COMMAND",
        help="pause, resume or stop (whatever commands.codes defines)",
    )
    send.set_defaults(handler=cmd_send)

    doctor = subparsers.add_parser(
        "doctor", help="check configuration, dependencies and the port"
    )
    add_global_options(doctor, suppress_defaults=True)
    doctor.set_defaults(handler=cmd_doctor)

    selftest = subparsers.add_parser(
        "selftest", help="run the offline test suite (no hardware)"
    )
    add_global_options(selftest, suppress_defaults=True)
    selftest.add_argument(
        "--verbose", action="store_true", help="show log output"
    )
    selftest.add_argument(
        "--filter", default=None, metavar="TEXT",
        help="only run tests whose name contains this",
    )
    selftest.set_defaults(handler=cmd_selftest)

    return parser


def build_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Turn --port / --mock / --real into a config-shaped override dict.

    Returned as DATA and passed to ConfigLoader, not written to os.environ.
    An earlier version set environment variables that config_loader read back
    later -- two modules coupled through a global, and a value that appeared
    in the loaded config with no way to tell it had not come from the YAML.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A nested dict in the same shape as config.yaml. Empty if no flag was
        given. Merged over the file and validated exactly like a YAML value.
    """
    overrides: dict[str, Any] = {}
    if args.port:
        overrides.setdefault("serial", {})["port"] = args.port
    if args.mock:
        overrides["transport"] = {"kind": constants.TRANSPORT_MOCK.value}
    if args.real:
        overrides["transport"] = {"kind": constants.TRANSPORT_SERIAL.value}
    if getattr(args, "debug", False):
        overrides.setdefault("logging", {})["level"] = "DEBUG"
    if getattr(args, "log_file", None):
        overrides.setdefault("logging", {})["file"] = args.log_file
    return overrides


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, load configuration, and dispatch to a subcommand.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = build_overrides(args)

    # Make loader diagnostics visible before YAML logging settings are known.
    # configure_logging() replaces this bootstrap setup after validation.
    logging.basicConfig(level=logging.INFO)

    try:
        config = ConfigLoader(args.config, overrides).get_all_configs()
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_BAD_USAGE

    try:
        configure_logging(config)
    except (OSError, ValueError, LookupError):
        LOGGER.exception("Cannot configure logging; service will not start")
        return EXIT_RUNTIME_ERROR

    LOGGER.info("Configuration loaded and validated from %s", config.source_paths)

    # The banner goes to stderr so `monitor` and `send` can be piped into jq
    # without it contaminating the JSON on stdout.
    if args.subcommand not in ("doctor", "selftest"):
        print(describe(config), file=sys.stderr)

    try:
        return int(args.handler(config, args))
    except KeyboardInterrupt:
        print("Stopped by user", file=sys.stderr)
        return EXIT_INTERRUPTED

    except Exception:
        LOGGER.exception("Command %s failed", args.subcommand)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
