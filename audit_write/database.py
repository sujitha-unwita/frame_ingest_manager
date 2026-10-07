"""The connection — the only module that opens a connection to Postgres.

It holds the asyncpg pool.  It never runs application SQL; that lives in
``src/repositories/``.

FastAPI::

    from audit_write.database import get_database

    @asynccontextmanager
    async def lifespan(app):
        db = get_database()
        await db.connect()
        yield
        await db.close()

Scripts, workers or an ML orchestrator::

    async with Database() as db:
        async with db.transaction() as conn:
            ...
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import asyncpg

from config.audit_models import DatabaseSettings
from config.config_loader import get_audit_settings as get_settings


class DatabaseNotConnectedError(RuntimeError):
    """A connection was requested before ``connect()`` or after ``close()``."""


class Database:
    """Owns one asyncpg pool.  Create it once per process, after any fork."""

    def __init__(self, database_config: DatabaseSettings | None = None) -> None:
        self._database_config = database_config or get_settings().database
        self._pool: asyncpg.Pool | None = None

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> "Database":
        """Create the pool.  Calling it again while connected does nothing."""
        if self._pool is not None:
            return self
        database_config = self._database_config
        self._pool = await asyncpg.create_pool(
            host=database_config.host,
            port=database_config.port,
            database=database_config.database_name,
            user=database_config.username,
            password=database_config.require_password(),
            ssl=database_config.ssl_mode,
            timeout=database_config.connect_timeout_s,
            min_size=database_config.pool.min_size,
            max_size=database_config.pool.max_size,
            command_timeout=database_config.pool.command_timeout_s,
            max_inactive_connection_lifetime=database_config.pool.max_inactive_connection_lifetime_s,
            statement_cache_size=database_config.statement_cache_size,
            server_settings={
                "application_name": database_config.application_name,
                "statement_timeout": str(database_config.statement_timeout_ms),
                "search_path": database_config.schema_name,
            },
            init=self._init_connection,
        )
        return self

    async def close(self) -> None:
        if self._pool is not None:
            pool, self._pool = self._pool, None
            await pool.close()

    async def __aenter__(self) -> "Database":
        return await self.connect()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        if self._database_config.decode_json:
            for type_name in ("json", "jsonb"):
                await conn.set_type_codec(
                    type_name, encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

    # -- access --------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._pool is not None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise DatabaseNotConnectedError("database is not connected; call `await db.connect()` first")
        return self._pool

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Borrow a connection from the pool; it is returned on exit."""
        async with self.pool.acquire(timeout=self._database_config.pool.acquire_timeout_s) as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Borrow a connection inside a transaction: commit on success, roll back on error."""
        async with self.connection() as conn:
            async with conn.transaction():
                yield conn

    async def ping(self) -> bool:
        """True if the database answers.  For health checks."""
        async with self.connection() as conn:
            return await conn.fetchval("SELECT 1") == 1


# ---------------------------------------------------------------------------
# Process-wide default instance
# ---------------------------------------------------------------------------

_default: Database | None = None


def get_database() -> Database:
    """The shared :class:`Database` for this process (created lazily, not connected)."""
    global _default
    if _default is None:
        _default = Database()
    return _default


def set_database(db: Database | None) -> None:
    """Replace the shared instance — for tests or custom wiring."""
    global _default
    _default = db


async def get_connection() -> AsyncIterator[asyncpg.Connection]:
    """FastAPI dependency: ``conn = Depends(get_connection)``."""
    async with get_database().connection() as conn:
        yield conn
