"""Load YAML, merge CLI overrides, and return the typed ``AppConfig``."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from pydantic import ValidationError

from config.models import AppConfig
from config.audit_models import AuditWrite

LOGGER = logging.getLogger(__name__)

_LOADED_AUDIT_SETTINGS: AuditWrite | None = None

def initialize_audit_settings(audit_write_config: AuditWrite) -> None:
    """Register already-validated audit settings in the current process."""
    global _LOADED_AUDIT_SETTINGS
    _LOADED_AUDIT_SETTINGS = audit_write_config

def get_audit_settings() -> AuditWrite:
    """Return centrally validated settings; never perform an implicit file read."""
    if _LOADED_AUDIT_SETTINGS is None:
        raise ConfigError("load application configuration before using audit models")
    return _LOADED_AUDIT_SETTINGS


class AttrDict(dict):
    """Mutable mapping used only by isolated legacy-style test fixtures."""

    def __getattr__(self, name: str) -> Any:
        """Expose fixture keys through dot access."""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @classmethod
    def wrap(cls, value: Any) -> Any:
        """Recursively wrap mappings used by test fixtures."""
        if isinstance(value, dict):
            return cls({key: cls.wrap(item) for key, item in value.items()})
        if isinstance(value, list):
            return [cls.wrap(item) for item in value]
        return value


class ConfigError(Exception):
    """Raised when configuration cannot be read or validated."""


class ConfigLoader:
    """Read one YAML document and validate it through ``AppConfig``."""

    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

    def __init__(self, config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> None:
        """Read YAML once and merge any command-line overrides."""
        self.config_path = Path(config_path or self.DEFAULT_CONFIG_PATH)
        LOGGER.info("Loading configuration from %s", self.config_path)
        self.raw_config = self.load_yaml(self.config_path)
        if overrides:
            LOGGER.info("Applying command-line configuration overrides")
            self.raw_config = _deep_merge(self.raw_config, overrides)

    @staticmethod
    def load_yaml(path: str | Path) -> dict[str, Any]:
        """Parse one YAML mapping without applying business validation."""
        try:
            with open(path, encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
        except FileNotFoundError as exc:
            LOGGER.error("Configuration file not found: %s", path)
            raise ConfigError(f"config file not found: {path}") from exc
        except (OSError, UnicodeDecodeError) as exc:
            LOGGER.error("Cannot read configuration file: %s", path)
            raise ConfigError(f"cannot read {path}: {exc}") from exc
        except yaml.YAMLError as exc:
            # Parser errors can include source lines containing secrets.
            LOGGER.error("Invalid YAML in configuration file: %s", path)
            raise ConfigError(f"{path} is not valid YAML") from exc
        if not isinstance(raw, dict):
            LOGGER.error("Configuration must contain a YAML mapping: %s", path)
            raise ConfigError(f"{path} must contain a YAML mapping")
        LOGGER.info("YAML configuration read successfully")
        return raw

    def get_all_configs(self) -> AppConfig:
        """Return a complete validated application configuration."""
        global _LOADED_AUDIT_SETTINGS
        LOGGER.info("Validating application configuration")
        try:
            config = AppConfig.model_validate(self.raw_config)
        except ValidationError as exc:
            # Report field locations without echoing user-supplied values.
            fields = [
                ".".join(str(part) for part in error["loc"])
                for error in exc.errors(include_input=False, include_context=False)
            ]
            message = "configuration validation failed for: " + ", ".join(fields)
            LOGGER.error("%s", message)
            raise ConfigError(message) from exc
        if config.frame_ingest.mvs is not None:
            sdk = config.frame_ingest.mvs.sdk
            if not sdk.mvimport_dir.is_absolute():
                sdk.mvimport_dir = self.config_path.resolve().parent / sdk.mvimport_dir
        self._resolve_audit_settings(config)
        config.set_source_paths((self.config_path.resolve(),))
        _LOADED_AUDIT_SETTINGS = config.audit_write
        LOGGER.info("Application configuration validated successfully")
        return config


    def _resolve_audit_settings(self, config: AppConfig) -> None:
        """Resolve paths and password once, without modifying os.environ.

        Environment wins over the local .env file. Relative paths are anchored
        to the selected main YAML, not the process working directory.
        """
        settings = config.audit_write
        database = settings.database
        base = self.config_path.resolve().parent
        env_path = Path(database.environment_file)
        if not env_path.is_absolute():
            env_path = base / env_path
        password = os.environ.get(database.password_env)
        if password is None and env_path.is_file():
            try:
                password = dotenv_values(env_path, interpolate=False).get(
                    database.password_env
                )
            except (OSError, UnicodeError):
                raise ConfigError("cannot read audit environment file") from None
        if settings.enabled and not password:
            raise ConfigError("audit enabled but configured password is missing")
        database.set_password(password)


def _deep_merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """Merge nested overrides without discarding unspecified sibling keys."""
    result = dict(base)
    for key, value in top.items():
        result[key] = _deep_merge(result[key], value) if isinstance(result.get(key), dict) and isinstance(value, dict) else value
    return result


def load_config(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load, merge, and Pydantic-validate application configuration."""
    return ConfigLoader(config_path, overrides).get_all_configs()


def describe(config: AppConfig) -> str:
    """Build the concise startup configuration banner."""
    serial, heartbeat, dashboard = config.serial, config.heartbeat, config.dashboard
    bands = ", ".join(f"<{bound * heartbeat.interval_s * 1000:.0f}ms={name}" for bound, name in heartbeat.bands)
    transport = "MockMcu (simulated, no hardware)" if config.transport.is_mock else f"SerialTransport {serial.port} @ {serial.baud} baud"
    return f"config      : {', '.join(path.name for path in config.source_paths)}\ntransport   : {transport}\nheartbeat   : {heartbeat.interval_s * 1000:.0f}ms; bands {bands}\ndashboard   : ws://{dashboard.host}:{dashboard.port}{dashboard.ws_path}, status every {dashboard.status_push_interval_s * 1000:.0f}ms"


__all__ = ["AppConfig", "AttrDict", "ConfigError", "ConfigLoader", "describe", "load_config"]
