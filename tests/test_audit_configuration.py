"""Test consolidated audit configuration without database or storage clients."""

import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config.config_loader import (
    ConfigError, ConfigLoader, get_audit_settings, load_config,
)
from audit_write.models import DefectLiveCreate
from audit_write.models.media import MediaCreate
from audit_write.models.trigger import TriggerCreate


class AuditConfigurationTests(unittest.TestCase):
    """One loader supplies connection settings and configurable row rules."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.model = self.folder / "model.pt"
        self.model.touch()
        self.env = self.folder / ".env"
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def load(self, **audit_overrides):
        """Use a valid model path without loading inference dependencies."""
        database = audit_overrides.pop("database", {})
        database["environment_file"] = str(self.env)
        return load_config(overrides={
            "inference": {
                "model_path": str(self.model),
            },
            "audit_write": {"enabled": False, **audit_overrides, "database": database},
        })

    def test_disabled_audit_needs_no_password(self) -> None:
        config = self.load(enabled=False)
        self.assertIs(get_audit_settings(), config.audit_write)
        self.assertEqual(config.audit_write.table("roll").primary_key, "roll_id")
        self.assertNotIn("testing", config.audit_write.model_dump())
        self.assertNotIn("smoke_test", config.audit_write.model_dump())

    def test_spawn_preserves_private_password_without_yaml_serialization(self):
        config = self.load(enabled=False)
        config.audit_write.database.set_password("test-only-password")
        restored = pickle.loads(pickle.dumps(config.audit_write.database))
        self.assertEqual(restored.require_password(), "test-only-password")
        self.assertNotIn("test-only-password", str(restored.model_dump()))

    def test_test_only_sections_are_rejected_by_runtime(self) -> None:
        """Prevent test settings from silently returning to runtime config."""
        for section in ("testing", "smoke_test"):
            with self.subTest(section=section), self.assertLogs(
                "config.config_loader", level="ERROR"
            ), self.assertRaises(ConfigError):
                self.load(**{section: {}})

    def test_application_does_not_read_test_yaml(self) -> None:
        """Only the main YAML is read during application configuration."""
        with patch.object(
            ConfigLoader, "load_yaml", wraps=ConfigLoader.load_yaml
        ) as read:
            config = self.load(enabled=False)
        read.assert_called_once_with(ConfigLoader.DEFAULT_CONFIG_PATH)
        self.assertGreater(config.audit_write.pagination.default_limit, 0)

    def test_separate_test_yaml_preserves_tool_settings(self) -> None:
        """Check relocated settings without connecting to any database."""
        path = Path(__file__).parent / "test.yaml"
        settings = ConfigLoader.load_yaml(path)
        self.assertEqual(set(settings), {"testing", "smoke_test"})
        testing = settings["testing"]
        self.assertNotIn("database_url", testing)
        self.assertEqual(testing["database_url_env"], "TEXTILE_TEST_DATABASE_URL")
        self.assertTrue((path.parent / testing["migrations_dir"]).is_dir())
        self.assertEqual(settings["smoke_test"]["roll_label_prefix"], "SMOKE-ROLL-")

    def test_enabled_audit_requires_password(self) -> None:
        with self.assertRaisesRegex(ConfigError, "password is missing"):
            self.load(enabled=True)

    def test_trigger_uuid_is_distinct_from_numeric_sequence(self) -> None:
        """Models and repository mappings share the renamed database identity."""
        config = self.load(enabled=False)
        roll_id = "00000000-0000-0000-0000-000000000001"
        trigger_primary_id = "00000000-0000-0000-0000-000000000002"
        trigger = TriggerCreate(
            trigger_primary_id=trigger_primary_id, roll_id=roll_id,
            camera_id="camera_3", trigger_sequence=42, encoder_count=100,
        )
        media = MediaCreate(
            trigger_primary_id=trigger_primary_id, roll_id=roll_id,
            camera_id="camera_3", frame_variant="A",
        )
        detection = DefectLiveCreate(
            trigger_primary_id=trigger_primary_id, roll_id=roll_id,
            camera_id="camera_3", defect_class="hole", confidence=0.9,
        )
        self.assertEqual(trigger.trigger_sequence, 42)
        for row in (trigger, media, detection):
            self.assertEqual(str(row.trigger_primary_id), trigger_primary_id)
            self.assertNotIn("trigger_id", row.to_row())
        for name in ("trigger", "media", "defect_live"):
            table = config.audit_write.table(name)
            self.assertIn("trigger_primary_id", table.columns)
            self.assertNotIn("trigger_id", table.columns)
        self.assertEqual(
            config.audit_write.table("trigger").primary_key, "trigger_primary_id"
        )
        for name in ("media", "defect_live"):
            self.assertEqual(
                config.audit_write.table(name).role("trigger"), "trigger_primary_id"
            )

    def test_env_file_password_is_private(self) -> None:
        self.env.write_text("TEXTILE_AUDIT_DB_PASSWORD=file-secret\n")
        config = self.load(enabled=True)
        self.assertEqual(
            config.audit_write.database.require_password(), "file-secret"
        )
        self.assertNotIn("file-secret", repr(config))
        self.assertNotIn("file-secret", config.model_dump_json())
        self.assertNotIn("TEXTILE_AUDIT_DB_PASSWORD", os.environ)

    def test_environment_password_takes_precedence(self) -> None:
        self.env.write_text("TEXTILE_AUDIT_DB_PASSWORD=file-secret\n")
        os.environ["TEXTILE_AUDIT_DB_PASSWORD"] = "environment-secret"
        self.assertEqual(
            self.load(enabled=True).audit_write.database.require_password(),
            "environment-secret",
        )

    def test_url_and_inline_password_are_rejected(self) -> None:
        for name in ("url", "password"):
            with self.subTest(field=name), self.assertLogs(
                "config.config_loader", level="ERROR"
            ), self.assertRaises(ConfigError):
                self.load(database={name: "must-not-be-in-yaml"})

    def test_pool_bounds_are_validated(self) -> None:
        with self.assertLogs("config.config_loader", level="ERROR"), \
                self.assertRaises(ConfigError):
            self.load(database={"pool": {"min_size": 3, "max_size": 2}})

    def test_row_validation_uses_centrally_loaded_rules(self) -> None:
        self.load(validation={"camera_id_max_length": 3})
        with self.assertRaisesRegex(ValueError, "at most 3"):
            DefectLiveCreate(
                roll_id="00000000-0000-0000-0000-000000000001",
                trigger_primary_id="00000000-0000-0000-0000-000000000002",
                camera_id="camera_3", defect_class="hole", confidence=0.9,
            )
        # No additional YAML read is needed to retrieve those same rules.
        with patch("config.config_loader.ConfigLoader.load_yaml") as read:
            self.assertEqual(get_audit_settings().validation.camera_id_max_length, 3)
            read.assert_not_called()
