"""What every repository shares.

A repository is bound to an ``audit_write.tables`` entry in ``config.yaml``. Table
and column names come from that entry, and every name is checked against its
``columns`` list and quoted before it reaches SQL, so caller-supplied keys can
never inject SQL.

Every method takes plain arguments and returns plain ``dict`` rows (or
``None`` / ``bool`` / ``int``).  Every method also accepts ``conn=``: pass a
connection from ``db.transaction()`` to run several calls atomically::

    async with db.transaction() as conn:
        defect = await defects_final.insert({...}, conn=conn)
        await defects_live.merge_into(ids, defect["defect_id"], conn=conn)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, ClassVar

import asyncpg

from config.audit_models import AuditWrite as Settings, TableConfig
from config.config_loader import get_audit_settings as get_settings
from audit_write.database import Database, get_database

Row = dict[str, Any]
Filters = Mapping[str, Any]

# Postgres accepts at most this many bind parameters in one statement.
_MAX_BIND_PARAMS = 32767


class RepositoryError(ValueError):
    """A repository was called with arguments it cannot turn into a valid query."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def quote_ident(name: str) -> str:
    """Quote a Postgres identifier."""
    return '"' + name.replace('"', '""') + '"'


def record_to_dict(record: asyncpg.Record | None) -> Row | None:
    return dict(record) if record is not None else None


def records_to_dicts(records: Iterable[asyncpg.Record]) -> list[Row]:
    return [dict(r) for r in records]


def as_mapping(data: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Accept a dict, or any Pydantic model (only the fields that were set)."""
    if isinstance(data, Mapping):
        return dict(data)
    dump = getattr(data, "model_dump", None)
    if callable(dump):
        return dump(exclude_unset=True)
    raise RepositoryError(f"expected a mapping or a Pydantic model, got {type(data).__name__}")


class _Params:
    """Collects bind parameters and hands out ``$n`` placeholders."""

    def __init__(self) -> None:
        self.values: list[Any] = []

    def add(self, value: Any) -> str:
        self.values.append(value)
        return f"${len(self.values)}"


# ---------------------------------------------------------------------------
# Base repository
# ---------------------------------------------------------------------------


class BaseRepository:
    """Generic CRUD and pagination for the table named by ``table_key``."""

    #: Key under audit_write.tables in the main YAML; set by each subclass.
    table_key: ClassVar[str]

    def __init__(self, db: Database | None = None, settings: Settings | None = None) -> None:
        self._db = db
        self.settings = settings or get_settings()
        self.config: TableConfig = self.settings.table(self.table_key)
        self._schema = self.settings.database.schema_name

    # -- names ---------------------------------------------------------------

    @property
    def db(self) -> Database:
        return self._db or get_database()

    @property
    def pk(self) -> str:
        return self.config.primary_key

    def col(self, role: str) -> str:
        """Column name for a role defined under ``roles:`` in the YAML."""
        return self.config.role(role)

    @property
    def table_sql(self) -> str:
        return f"{quote_ident(self._schema)}.{quote_ident(self.config.table)}"

    @property
    def columns_sql(self) -> str:
        return ", ".join(quote_ident(c) for c in self.config.columns)

    def _require_columns(self, names: Iterable[str], allowed: Iterable[str], purpose: str) -> None:
        allowed_set = set(allowed)
        unknown = [n for n in names if n not in allowed_set]
        if unknown:
            raise RepositoryError(
                f"{self.config.table}: cannot {purpose} column(s) {unknown}; allowed: {sorted(allowed_set)}"
            )

    def clamp_limit(self, limit: int | None) -> int:
        page = self.settings.pagination
        if limit is None:
            return page.default_limit
        if limit < 1:
            raise RepositoryError("limit must be at least 1")
        return min(limit, page.max_limit)

    # -- connections ---------------------------------------------------------

    @asynccontextmanager
    async def _connection(
        self, conn: asyncpg.Connection | None
    ) -> AsyncGenerator[asyncpg.Connection, None]:
        if conn is not None:
            yield conn
        else:
            async with self.db.connection() as acquired:
                yield acquired

    # -- SQL building --------------------------------------------------------

    def _where(self, filters: Filters | None, params: _Params) -> list[str]:
        """Equality filters.  ``None`` → IS NULL, list/tuple/set → = ANY(...)."""
        if not filters:
            return []
        self._require_columns(filters.keys(), self.config.columns, "filter on")
        clauses = []
        for column, value in filters.items():
            ident = quote_ident(column)
            if value is None:
                clauses.append(f"{ident} IS NULL")
            elif isinstance(value, (list, tuple, set, frozenset)):
                clauses.append(f"{ident} = ANY({params.add(list(value))})")
            else:
                clauses.append(f"{ident} = {params.add(value)}")
        return clauses

    @staticmethod
    def _where_sql(clauses: Sequence[str]) -> str:
        return f" WHERE {' AND '.join(clauses)}" if clauses else ""

    # -- reads ---------------------------------------------------------------

    async def get(self, pk_value: Any, *, conn: asyncpg.Connection | None = None) -> Row | None:
        sql = f"SELECT {self.columns_sql} FROM {self.table_sql} WHERE {quote_ident(self.pk)} = $1"
        async with self._connection(conn) as c:
            return record_to_dict(await c.fetchrow(sql, pk_value))

    async def get_many(self, pk_values: Sequence[Any], *, conn: asyncpg.Connection | None = None) -> list[Row]:
        if not pk_values:
            return []
        sql = f"SELECT {self.columns_sql} FROM {self.table_sql} WHERE {quote_ident(self.pk)} = ANY($1)"
        async with self._connection(conn) as c:
            return records_to_dicts(await c.fetch(sql, list(pk_values)))

    async def exists(self, pk_value: Any, *, conn: asyncpg.Connection | None = None) -> bool:
        sql = f"SELECT EXISTS (SELECT 1 FROM {self.table_sql} WHERE {quote_ident(self.pk)} = $1)"
        async with self._connection(conn) as c:
            return bool(await c.fetchval(sql, pk_value))

    async def find_one(self, filters: Filters, *, conn: asyncpg.Connection | None = None) -> Row | None:
        """The first row matching ``filters`` (in list order), or ``None``."""
        rows = await self.list(filters=filters, limit=1, conn=conn)
        return rows[0] if rows else None

    async def list(
        self,
        *,
        filters: Filters | None = None,
        limit: int | None = None,
        before: Any = None,
        order_by: str | None = None,
        descending: bool = True,
        conn: asyncpg.Connection | None = None,
    ) -> list[Row]:
        """A page of rows.

        ``before`` is the primary key of the last row of the previous page;
        pass it to get the next page.  Sorted by ``order_by`` (default: the
        table's configured ``order_by``) then the primary key, so pages never
        overlap or skip.  The sort column should be NOT NULL.
        """
        order_col = order_by or self.config.order_by
        self._require_columns([order_col], self.config.columns, "order by")

        params = _Params()
        clauses = self._where(filters, params)
        direction = "DESC" if descending else "ASC"
        order_ident, pk_ident = quote_ident(order_col), quote_ident(self.pk)

        if before is not None:
            cmp = "<" if descending else ">"
            clauses.append(
                f"({order_ident}, {pk_ident}) {cmp} "
                f"(SELECT {order_ident}, {pk_ident} FROM {self.table_sql} WHERE {pk_ident} = {params.add(before)})"
            )

        sql = (
            f"SELECT {self.columns_sql} FROM {self.table_sql}{self._where_sql(clauses)}"
            f" ORDER BY {order_ident} {direction}, {pk_ident} {direction}"
            f" LIMIT {params.add(self.clamp_limit(limit))}"
        )
        async with self._connection(conn) as c:
            return records_to_dicts(await c.fetch(sql, *params.values))

    async def count(self, filters: Filters | None = None, *, conn: asyncpg.Connection | None = None) -> int:
        params = _Params()
        sql = f"SELECT count(*) FROM {self.table_sql}{self._where_sql(self._where(filters, params))}"
        async with self._connection(conn) as c:
            return int(await c.fetchval(sql, *params.values))

    async def max_value(
        self, column: str, filters: Filters | None = None, *, conn: asyncpg.Connection | None = None
    ) -> Any:
        """``max(column)`` over the matching rows, or ``None`` if there are none."""
        self._require_columns([column], self.config.columns, "aggregate")
        params = _Params()
        sql = (
            f"SELECT max({quote_ident(column)}) FROM {self.table_sql}"
            f"{self._where_sql(self._where(filters, params))}"
        )
        async with self._connection(conn) as c:
            return await c.fetchval(sql, *params.values)

    # -- writes --------------------------------------------------------------

    async def insert(self, data: Mapping[str, Any] | Any, *, conn: asyncpg.Connection | None = None) -> Row:
        """Insert one row; columns left out take their database default."""
        row = as_mapping(data)
        self._require_columns(row.keys(), self.config.columns, "insert")
        if row:
            params = _Params()
            placeholders = ", ".join(params.add(v) for v in row.values())
            names = ", ".join(quote_ident(k) for k in row)
            sql = f"INSERT INTO {self.table_sql} ({names}) VALUES ({placeholders}) RETURNING {self.columns_sql}"
            args = params.values
        else:
            sql = f"INSERT INTO {self.table_sql} DEFAULT VALUES RETURNING {self.columns_sql}"
            args = []
        async with self._connection(conn) as c:
            return dict(await c.fetchrow(sql, *args))

    async def insert_many(
        self, rows: Sequence[Mapping[str, Any] | Any], *, conn: asyncpg.Connection | None = None
    ) -> list[Row]:
        """Insert many rows atomically and return the inserted rows.

        Every row must set the same columns.  For very large batches where the
        inserted rows are not needed back, :meth:`copy_many` is faster.
        """
        mapped = [as_mapping(r) for r in rows]
        if not mapped:
            return []
        names = list(mapped[0].keys())
        if not names:
            raise RepositoryError(f"{self.config.table}: insert_many needs at least one column per row")
        self._require_columns(names, self.config.columns, "insert")
        if any(set(r.keys()) != set(names) for r in mapped[1:]):
            raise RepositoryError(f"{self.config.table}: every row passed to insert_many must set the same columns")

        per_statement = max(1, _MAX_BIND_PARAMS // len(names))
        names_sql = ", ".join(quote_ident(n) for n in names)
        inserted: list[Row] = []

        async with self._connection(conn) as c:
            async with c.transaction():
                for start in range(0, len(mapped), per_statement):
                    chunk = mapped[start : start + per_statement]
                    params = _Params()
                    values_sql = ", ".join(
                        "(" + ", ".join(params.add(r[n]) for n in names) + ")" for r in chunk
                    )
                    sql = (
                        f"INSERT INTO {self.table_sql} ({names_sql}) VALUES {values_sql}"
                        f" RETURNING {self.columns_sql}"
                    )
                    inserted.extend(records_to_dicts(await c.fetch(sql, *params.values)))
        return inserted

    async def copy_many(
        self, rows: Sequence[Mapping[str, Any] | Any], *, conn: asyncpg.Connection | None = None
    ) -> int:
        """Bulk-load rows with COPY (fastest path).  Returns the number of rows written.

        Rows must set the same columns; no rows are returned, so supply primary
        keys yourself if you need them afterwards.
        """
        mapped = [as_mapping(r) for r in rows]
        if not mapped:
            return 0
        names = list(mapped[0].keys())
        self._require_columns(names, self.config.columns, "insert")
        if any(set(r.keys()) != set(names) for r in mapped[1:]):
            raise RepositoryError(f"{self.config.table}: every row passed to copy_many must set the same columns")
        records = [tuple(r[n] for n in names) for r in mapped]
        async with self._connection(conn) as c:
            await c.copy_records_to_table(
                self.config.table, records=records, columns=names, schema_name=self._schema
            )
        return len(records)

    def _set_clause(self, changes: Mapping[str, Any], params: _Params) -> str:
        allowed = [c for c in self.config.writable_columns if c != self.pk]
        self._require_columns(changes.keys(), allowed, "update")
        return ", ".join(f"{quote_ident(k)} = {params.add(v)}" for k, v in changes.items())

    async def update(
        self, pk_value: Any, changes: Mapping[str, Any] | Any, *, conn: asyncpg.Connection | None = None
    ) -> Row | None:
        """Update one row by primary key.  Returns the new row, or ``None`` if it does not exist."""
        changes = as_mapping(changes)
        if not changes:
            return await self.get(pk_value, conn=conn)
        params = _Params()
        set_sql = self._set_clause(changes, params)
        sql = (
            f"UPDATE {self.table_sql} SET {set_sql} WHERE {quote_ident(self.pk)} = {params.add(pk_value)}"
            f" RETURNING {self.columns_sql}"
        )
        async with self._connection(conn) as c:
            return record_to_dict(await c.fetchrow(sql, *params.values))

    async def update_where(
        self, filters: Filters, changes: Mapping[str, Any] | Any, *, conn: asyncpg.Connection | None = None
    ) -> list[Row]:
        """Update every row matching ``filters`` (which must not be empty)."""
        changes = as_mapping(changes)
        if not filters:
            raise RepositoryError(f"{self.config.table}: update_where refuses to update every row; pass filters")
        if not changes:
            return []
        params = _Params()
        set_sql = self._set_clause(changes, params)
        where_sql = self._where_sql(self._where(filters, params))
        sql = f"UPDATE {self.table_sql} SET {set_sql}{where_sql} RETURNING {self.columns_sql}"
        async with self._connection(conn) as c:
            return records_to_dicts(await c.fetch(sql, *params.values))

    async def upsert(
        self,
        data: Mapping[str, Any] | Any,
        *,
        conflict_columns: Sequence[str] | None = None,
        update_columns: Sequence[str] | None = None,
        conn: asyncpg.Connection | None = None,
    ) -> Row | None:
        """Insert, or on conflict update ``update_columns``.

        ``conflict_columns`` defaults to the primary key and must match a unique
        constraint.  ``update_columns`` defaults to every supplied writable
        column except the conflict columns; if that leaves nothing, the
        conflict is ignored and ``None`` is returned when the row already existed.
        """
        row = as_mapping(data)
        if not row:
            raise RepositoryError(f"{self.config.table}: upsert needs at least one column")
        self._require_columns(row.keys(), self.config.columns, "insert")
        conflict = list(conflict_columns or [self.pk])
        self._require_columns(conflict, self.config.columns, "use as conflict target")
        if update_columns is None:
            update_columns = [k for k in row if k not in conflict and k in self.config.writable_columns]
        self._require_columns(update_columns, row.keys(), "update on conflict (must be supplied)")
        self._require_columns(update_columns, self.config.writable_columns, "update")

        params = _Params()
        names = ", ".join(quote_ident(k) for k in row)
        placeholders = ", ".join(params.add(v) for v in row.values())
        conflict_sql = ", ".join(quote_ident(c) for c in conflict)
        if update_columns:
            action = "DO UPDATE SET " + ", ".join(
                f"{quote_ident(c)} = EXCLUDED.{quote_ident(c)}" for c in update_columns
            )
        else:
            action = "DO NOTHING"
        sql = (
            f"INSERT INTO {self.table_sql} ({names}) VALUES ({placeholders})"
            f" ON CONFLICT ({conflict_sql}) {action} RETURNING {self.columns_sql}"
        )
        async with self._connection(conn) as c:
            return record_to_dict(await c.fetchrow(sql, *params.values))

    async def delete(self, pk_value: Any, *, conn: asyncpg.Connection | None = None) -> bool:
        """Delete one row.  True if a row was deleted."""
        sql = f"DELETE FROM {self.table_sql} WHERE {quote_ident(self.pk)} = $1"
        async with self._connection(conn) as c:
            status = await c.execute(sql, pk_value)
        return status.endswith(" 1")
