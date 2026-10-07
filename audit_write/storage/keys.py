"""How S3 object keys are built.

The formats live in ``config.yaml`` under ``audit_write.storage.keys``. This is
the only place keys are assembled, so two formats never go live by accident.

    build_key("media", captured_at=dt, roll_label="R-142", trigger_sequence=412,
              camera_id="CAM-01", frame_variant="raw")
    # -> "crops/2026/09/10/R-142/seq-000412-CAM-01-raw.jpg"
"""

from __future__ import annotations

import re
from typing import Any

from config.audit_models import KeyFormat, StorageSettings
from config.config_loader import get_audit_settings as get_settings

# Characters allowed inside one placeholder value; anything else becomes "_".
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class StorageKeyError(ValueError):
    """A key could not be built from the given values."""


def key_format(kind: str, settings: StorageSettings | None = None) -> KeyFormat:
    s = settings or get_settings().storage
    try:
        return s.keys[kind]
    except KeyError:
        raise StorageKeyError(f"no key format '{kind}' in config; known: {sorted(s.keys)}") from None


def safe_segment(value: Any) -> Any:
    """Make a string safe to use inside a key segment (no slashes, spaces, ...)."""
    if isinstance(value, str):
        cleaned = _UNSAFE.sub("_", value).strip(".")
        if not cleaned:
            raise StorageKeyError(f"value {value!r} is empty after removing unsafe characters")
        return cleaned
    return value


def build_key(kind: str, *, settings: StorageSettings | None = None, **values: Any) -> str:
    """Fill the ``kind`` template.  ``prefix`` and ``ext`` default to the configured ones."""
    fmt = key_format(kind, settings)
    fields = {"prefix": fmt.prefix, "ext": fmt.ext}
    fields.update({k: safe_segment(v) for k, v in values.items()})
    try:
        key = fmt.template.format(**fields)
    except KeyError as exc:
        raise StorageKeyError(f"key format '{kind}' needs a value for {exc.args[0]!r}") from None
    except (ValueError, TypeError) as exc:
        raise StorageKeyError(f"key format '{kind}': {exc}") from None

    if key.startswith("/") or "//" in key or ".." in key.split("/"):
        raise StorageKeyError(f"built an invalid key: {key!r}")
    return key


def content_type_for(kind: str, settings: StorageSettings | None = None) -> str:
    s = settings or get_settings().storage
    return key_format(kind, s).content_type or s.default_content_type
