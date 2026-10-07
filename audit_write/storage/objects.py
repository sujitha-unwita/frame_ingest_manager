"""Upload and fetch objects.

boto3 is synchronous, so every call runs in a worker thread and never blocks
the event loop.  Deciding who may see an object is the route's job, not this
module's.

    store = ObjectStore(create_s3_client())
    await store.upload(key, data, content_type="image/jpeg")
    return StreamingResponse(store.get(key), media_type="image/jpeg")
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from botocore.exceptions import ClientError

from config.audit_models import StorageSettings
from config.config_loader import get_audit_settings as get_settings
from audit_write.storage.client import StorageNotConfiguredError

_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}


class ObjectNotFoundError(LookupError):
    """No object exists at the requested key."""


class ObjectStore:
    """One bucket, one shared client."""

    def __init__(self, client: Any, storage_config: StorageSettings | None = None, *, bucket: str | None = None) -> None:
        self._client = client
        self._storage_config = storage_config or get_settings().storage
        self._bucket = bucket or self._storage_config.bucket
        if not self._bucket:
            raise StorageNotConfiguredError("audit_write.storage.bucket is not set in config.yaml")

    @property
    def bucket(self) -> str:
        return self._bucket

    async def close(self) -> None:
        """Release the S3 client's HTTP connections at application shutdown."""
        await asyncio.to_thread(self._client.close)

    async def upload(self, key: str, data: bytes, *, content_type: str | None = None) -> str:
        """Store ``data`` at ``key`` and return the key."""
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type or self._storage_config.default_content_type,
        )
        return key

    async def get(self, key: str, *, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        """Stream the object in chunks without loading it all into memory."""
        size = chunk_size or self._storage_config.stream_chunk_size
        body = await self._open(key)
        try:
            while True:
                chunk = await asyncio.to_thread(body.read, size)
                if not chunk:
                    break
                yield chunk
        finally:
            await asyncio.to_thread(body.close)

    async def get_bytes(self, key: str) -> bytes:
        """The whole object in memory — only for small objects."""
        body = await self._open(key)
        try:
            return await asyncio.to_thread(body.read)
        finally:
            await asyncio.to_thread(body.close)

    async def exists(self, key: str) -> bool:
        try:
            await asyncio.to_thread(self._client.head_object, Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if _is_not_found(exc):
                return False
            raise
        return True

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._client.delete_object, Bucket=self._bucket, Key=key)

    async def _open(self, key: str) -> Any:
        try:
            response = await asyncio.to_thread(self._client.get_object, Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if _is_not_found(exc):
                raise ObjectNotFoundError(key) from exc
            raise
        return response["Body"]


def _is_not_found(exc: ClientError) -> bool:
    return str(exc.response.get("Error", {}).get("Code")) in _NOT_FOUND_CODES
