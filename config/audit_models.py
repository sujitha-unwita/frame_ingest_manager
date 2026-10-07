"""Audit configuration models; no file reads or environment lookups."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel, ConfigDict, Field, PrivateAttr, SecretStr, StrictBool,
    field_validator, model_validator,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _blank_to_none(value: Any) -> Any:
    return None if isinstance(value, str) and value.strip() == "" else value


class PoolSettings(_Frozen):
    min_size: int = Field(ge=0)
    max_size: int = Field(ge=1)
    command_timeout_s: float = Field(gt=0)
    max_inactive_connection_lifetime_s: float = Field(ge=0)
    acquire_timeout_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _min_not_above_max(self) -> "PoolSettings":
        if self.min_size > self.max_size:
            raise ValueError("pool.min_size cannot be greater than pool.max_size")
        return self


class DatabaseSettings(_Frozen):
    """Deployment fields from YAML; password is supplied only by the loader."""

    driver: Literal["postgresql"]
    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65535)
    database_name: str = Field(min_length=1)
    username: str = Field(min_length=1)
    password_env: str = Field(min_length=1)
    environment_file: str = Field(min_length=1)
    ssl_mode: Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
    connect_timeout_s: float = Field(gt=0)
    schema_name: str = Field(min_length=1)
    application_name: str
    statement_timeout_ms: int = Field(ge=0)
    statement_cache_size: int = Field(ge=0)
    decode_json: bool
    pool: PoolSettings
    _password: SecretStr | None = PrivateAttr(default=None)

    def set_password(self, password: str | None) -> None:
        """Keep the externally resolved credential out of serialized settings."""
        self._password = SecretStr(password) if password else None

    def require_password(self) -> str:
        """Expose the credential only to the database connection boundary."""
        if self._password is None:
            raise ValueError("audit database password has not been configured")
        return self._password.get_secret_value()


class PaginationSettings(_Frozen):
    default_limit: int = Field(ge=1)
    max_limit: int = Field(ge=1)

    @model_validator(mode="after")
    def _default_not_above_max(self) -> "PaginationSettings":
        if self.default_limit > self.max_limit:
            raise ValueError("pagination.default_limit cannot exceed max_limit")
        return self


class TableConfig(_Frozen):
    table: str
    primary_key: str
    order_by: str
    columns: tuple[str, ...]
    read_only: tuple[str, ...] = ()
    roles: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _names_are_known_columns(self) -> "TableConfig":
        known = set(self.columns)
        if len(known) != len(self.columns):
            raise ValueError(f"table '{self.table}': duplicate column in columns")
        referenced = {
            "primary_key": [self.primary_key],
            "order_by": [self.order_by],
            "read_only": list(self.read_only),
            "roles": list(self.roles.values()),
        }
        for field, names in referenced.items():
            unknown = [n for n in names if n not in known]
            if unknown:
                raise ValueError(f"table '{self.table}': {field} refers to unknown columns {unknown}")
        return self

    @property
    def writable_columns(self) -> tuple[str, ...]:
        return tuple(c for c in self.columns if c not in self.read_only)

    def role(self, name: str) -> str:
        """The column playing ``name`` (see ``roles`` in the YAML)."""
        try:
            return self.roles[name]
        except KeyError:
            raise ValueError(f"table '{self.table}' has no role '{name}' in config") from None


class ValidationSettings(_Frozen):
    confidence_min: float
    confidence_max: float
    frame_variants: tuple[str, ...] = ()
    defect_classes: tuple[str, ...] = ()
    camera_id_max_length: int = Field(ge=1)
    label_max_length: int = Field(ge=1)


class S3ClientSettings(_Frozen):
    max_pool_connections: int = Field(ge=1)
    connect_timeout_s: float = Field(gt=0)
    read_timeout_s: float = Field(gt=0)
    retry_max_attempts: int = Field(ge=0)
    retry_mode: str


class KeyFormat(_Frozen):
    template: str
    prefix: str
    ext: str
    content_type: str | None = None


class StorageSettings(_Frozen):
    bucket: str | None = None
    region: str | None = None
    endpoint_url: str | None = None
    client: S3ClientSettings
    stream_chunk_size: int = Field(ge=1)
    default_content_type: str
    keys: dict[str, KeyFormat]

    blanks = field_validator("bucket", "region", "endpoint_url", mode="before")(_blank_to_none)


class MigrationSettings(_Frozen):
    """Migration history that must exist before audit processing starts."""

    history_schema: str = Field(min_length=1)
    history_table: str = Field(min_length=1)
    version_column: str = Field(min_length=1)
    required_versions: tuple[str, ...] = Field(min_length=1)


class AuditWrite(_Frozen):
    """All audit settings, loaded only by the main configuration loader."""

    enabled: StrictBool
    roll_label_prefix: str = Field(min_length=1)
    frame_key_kind: str = Field(min_length=1)
    model_version: str | None
    database: DatabaseSettings
    migrations: MigrationSettings
    pagination: PaginationSettings
    tables: dict[str, TableConfig]
    validation: ValidationSettings
    storage: StorageSettings

    @model_validator(mode="after")
    def validate_frame_key(self) -> "AuditWrite":
        """Require the selected frame-key format to exist in storage settings."""
        if self.frame_key_kind not in self.storage.keys:
            raise ValueError("frame_key_kind must name a storage.keys entry")
        return self

    def table(self, name: str) -> TableConfig:
        try:
            return self.tables[name]
        except KeyError:
            raise ValueError(f"no table '{name}' in config; known: {sorted(self.tables)}") from None
