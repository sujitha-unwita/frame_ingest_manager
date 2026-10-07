"""Lightweight spawn target: import resource-owning modules only in children."""

import asyncio
import importlib
import logging
import os
from datetime import datetime
from pathlib import Path


def configure_worker_logging(config, role: str) -> None:
    """Give each process its own log file; never share a FileHandler."""
    settings = config.logging
    handlers = [logging.StreamHandler()]
    if settings.file:
        timestamp = datetime.now().strftime(settings.filename_datefmt)
        path = Path(settings.file.replace("{timestamp}", timestamp))
        if not path.is_absolute():
            path = Path(config.log_root) / path
        path = path.with_name(f"{path.stem}-{role}-{os.getpid()}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(
            path, mode=settings.file_mode, encoding=settings.file_encoding,
        ))
    logging.basicConfig(
        level=settings.level.value, format=settings.format, datefmt=settings.datefmt,
        handlers=handlers, force=True,
    )
    for name, level in settings.quiet_loggers.items():
        logging.getLogger(name).setLevel(level.value)


def run_worker(role, config, channels) -> None:
    """Load the selected worker after spawn, without initializing parent CUDA."""
    try:
        configure_worker_logging(config, role)
        module = importlib.import_module(f"runtime.{role}_worker")
        asyncio.run(module.run(config, channels))
    except BaseException:
        # A shared flag still reports failure if the event queue is full.
        channels.failed.set()
        logging.getLogger(__name__).exception("%s worker failed", role)
        raise
