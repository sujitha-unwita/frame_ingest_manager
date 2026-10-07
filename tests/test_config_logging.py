"""Validate mandatory model files and configuration-loader diagnostics."""

import tempfile
import logging
import unittest
from pathlib import Path

from config.config_loader import ConfigError, ConfigLoader, load_config


class ConfigurationTests(unittest.TestCase):
    """Configuration validation requires a file, without loading model weights."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Path(self.directory.name) / "test-model.pt"
        self.model.touch()

    def test_valid_model_logs_load_and_validation(self) -> None:
        with self.assertLogs("config.config_loader", level="INFO") as logs:
            config = load_config(overrides={
                "audit_write": {"enabled": False},
                "inference": {
                    "model_path": str(self.model),
                    "layout": {"mm_per_pixel_x": 0.5},
                },
            })
        self.assertEqual(config.inference.model_path, self.model)
        self.assertNotIn("enabled", type(config.inference).model_fields)
        self.assertTrue(any("validated successfully" in line for line in logs.output))

    def test_yaml_log_format_contains_timestamp_module_and_method(self) -> None:
        """The shared formatter identifies the original emitting call site."""
        settings = ConfigLoader().raw_config["logging"]
        formatter = logging.Formatter(settings["format"], settings["datefmt"])
        record = logging.LogRecord(
            name="audit_write.services.audit_writer", level=logging.INFO,
            pathname="audit_writer.py", lineno=209,
            msg="committed=%s", args=(2,), exc_info=None,
            func="write_final_defects",
        )
        formatted = formatter.format(record)
        self.assertIn(
            "audit_write.services.audit_writer.write_final_defects:209", formatted
        )
        self.assertIn("committed=2", formatted)
        self.assertIn("INFO", formatted)
        self.assertRegex(
            formatted,
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \|",
        )

    def test_missing_null_empty_nonexistent_and_directory_are_rejected(self) -> None:
        for value in (None, "", " ", str(self.model.parent / "missing.pt"),
                      str(self.model.parent)):
            with self.subTest(value=value), self.assertLogs(
                "config.config_loader", level="ERROR"
            ), self.assertRaisesRegex(ConfigError, "inference.model_path"):
                load_config(overrides={"inference": {"model_path": value}})
        loader = ConfigLoader()
        del loader.raw_config["inference"]["model_path"]
        with self.assertLogs("config.config_loader", level="ERROR"), \
                self.assertRaisesRegex(ConfigError, "inference.model_path"):
            loader.get_all_configs()

    def test_old_switch_is_rejected(self) -> None:
        with self.assertLogs("config.config_loader", level="ERROR"), \
                self.assertRaisesRegex(ConfigError, "inference.enabled"):
            load_config(overrides={"inference": {
                "model_path": str(self.model), "enabled": False,
            }})

    def test_validation_logs_do_not_expose_values(self) -> None:
        secret = "secret-value-must-not-be-logged"
        with self.assertLogs("config.config_loader", level="ERROR") as logs:
            with self.assertRaises(ConfigError) as error:
                load_config(overrides={"inference": {"model_path": secret}})
        self.assertNotIn(secret, str(error.exception))
        self.assertNotIn(secret, " ".join(logs.output))

    def test_missing_yaml_is_logged(self) -> None:
        with self.assertLogs("config.config_loader", level="ERROR"), \
                self.assertRaises(ConfigError):
            load_config(self.model.parent / "missing.yaml")

    def test_horizontal_calibration_must_be_positive_and_finite(self) -> None:
        for value in (None, 0, -0.1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertLogs(
                "config.config_loader", level="ERROR"
            ), self.assertRaisesRegex(ConfigError, "inference.layout.mm_per_pixel_x"):
                load_config(overrides={"inference": {
                    "model_path": str(self.model),
                    "layout": {"mm_per_pixel_x": value},
                }})
