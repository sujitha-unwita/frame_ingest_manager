"""Verify application-wide file logging without database or GPU dependencies."""

import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main as entrypoint
from config.config_loader import load_config


class FileLoggingTests(unittest.TestCase):
    """Keep handlers and output files isolated from real application settings."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        model = self.folder / "model.pt"
        model.touch()
        self.config = load_config(overrides={
            "inference": {"model_path": str(model)},
            "audit_write": {"enabled": False},
            "logging": {
                "file": "logs/test-{timestamp}.log", "quiet_loggers": {},
            },
        })
        self.config.set_source_paths((self.folder / "config.yaml",))
        self.root = logging.getLogger()
        previous_handlers = list(self.root.handlers)
        previous_level = self.root.level
        self.root.handlers = []

        def restore_logging():
            for handler in self.root.handlers:
                handler.close()
            self.root.handlers = previous_handlers
            self.root.setLevel(previous_level)

        self.addCleanup(restore_logging)

    def test_console_and_file_receive_module_logs_and_tracebacks(self):
        console = io.StringIO()
        with patch("sys.stderr", console):
            entrypoint.configure_logging(self.config)
            logger = logging.getLogger("test.application.inference")
            logger.info("model loading")
            try:
                raise RuntimeError("CUDA unavailable test")
            except RuntimeError:
                logger.exception("model startup failed")
        files = list((self.folder / "logs").glob("*.log"))
        self.assertEqual(len(files), 1)
        contents = files[0].read_text(encoding="utf-8")
        for text in ("model loading", "Traceback", "CUDA unavailable test"):
            self.assertIn(text, contents)
            self.assertIn(text, console.getvalue())
        self.assertIn("test.application.inference.test_console_and_file", contents)

    def test_each_start_preserves_previous_file_and_has_only_two_handlers(self):
        entrypoint.configure_logging(self.config)
        logging.getLogger("test.application").info("first run")
        first = next((self.folder / "logs").glob("*.log"))
        # Make the second startup time deterministic on coarse Windows clocks.
        with patch.object(entrypoint, "datetime") as clock:
            clock.now.return_value.strftime.return_value = "second-start"
            entrypoint.configure_logging(self.config)
        logging.getLogger("test.application").info("second run")
        self.assertEqual(len(list((self.folder / "logs").glob("*.log"))), 2)
        self.assertEqual(len(self.root.handlers), 2)
        self.assertIn("first run", first.read_text(encoding="utf-8"))
        self.assertNotIn("second run", first.read_text(encoding="utf-8"))

    def test_null_file_is_console_only(self):
        self.config.logging.file = None
        entrypoint.configure_logging(self.config)
        self.assertEqual(len(self.root.handlers), 1)
        self.assertNotIsInstance(self.root.handlers[0], logging.FileHandler)
        self.assertFalse((self.folder / "logs").exists())

    def test_exclusive_mode_refuses_to_overwrite_existing_log(self):
        self.config.logging.file = str(self.folder / "existing.log")
        entrypoint.configure_logging(self.config)
        with self.assertRaises(FileExistsError):
            entrypoint.configure_logging(self.config)

    def test_main_logs_command_failure_and_returns_error(self):
        with patch.object(entrypoint.ConfigLoader, "get_all_configs", return_value=self.config), \
                patch.object(entrypoint, "cmd_serve", side_effect=RuntimeError("startup failed")):
            result = entrypoint.main(["serve"])
        self.assertEqual(result, entrypoint.EXIT_RUNTIME_ERROR)
        file = next((self.folder / "logs").glob("*.log"))
        self.assertIn("RuntimeError: startup failed", file.read_text(encoding="utf-8"))

    def test_cli_file_override_is_preserved(self):
        args = entrypoint.build_parser().parse_args([
            "serve", "--log-file", "logs/custom-{timestamp}.log",
        ])
        self.assertEqual(
            entrypoint.build_overrides(args)["logging"]["file"],
            "logs/custom-{timestamp}.log",
        )
