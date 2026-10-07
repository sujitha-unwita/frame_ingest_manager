"""The boto3 S3 client and its configuration.

Build it once at startup (after any fork, e.g. in the FastAPI lifespan) and
reuse it: the client keeps an HTTPS connection pool and caches credentials.
Credentials come from the standard AWS chain (IAM role in production).
"""

from __future__ import annotations

from typing import Any
from dotenv import dotenv_values
from pathlib import Path
import boto3
from botocore.config import Config

from config.audit_models import StorageSettings
from config.config_loader import get_audit_settings as get_settings


class StorageNotConfiguredError(RuntimeError):
    """S3 settings required for this operation are missing."""


def create_s3_client(settings: StorageSettings | None = None) -> Any:
    """Return a configured ``boto3`` S3 client."""
    environment_file = Path(__file__).resolve().parents[2] / ".env"
    env_values = dotenv_values(dotenv_path=environment_file)  # load .env for local development, ignored in production
    access_key = env_values.get("AWS_ACCESS_KEY_ID")
    secret_key = env_values.get("AWS_SECRET_ACCESS_KEY")
    s = settings or get_settings().storage
    if not s.region:
        raise StorageNotConfiguredError("audit_write.storage.region is not set in config.yaml")
    config = Config(
        region_name=s.region,
        max_pool_connections=s.client.max_pool_connections,
        connect_timeout=s.client.connect_timeout_s,
        read_timeout=s.client.read_timeout_s,
        retries={"max_attempts": s.client.retry_max_attempts, "mode": s.client.retry_mode},
    )
    return boto3.client("s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key, endpoint_url=s.endpoint_url, config=config)
