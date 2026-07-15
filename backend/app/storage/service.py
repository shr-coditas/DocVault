from collections.abc import AsyncIterator
from typing import Any

import aioboto3
from botocore.exceptions import ClientError

from app.core.config import Settings, get_settings

CHUNK_SIZE = 1024 * 1024  # 1 MiB


class StorageService:
    """Thin async wrapper around S3-compatible object storage (MinIO in dev).

    The rest of the app never talks to boto directly, so swapping MinIO for
    real S3 is a settings change, not a code change.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._session = aioboto3.Session()

    def _client(self) -> Any:
        return self._session.client(
            "s3",
            endpoint_url=self.settings.s3_endpoint_url,
            aws_access_key_id=self.settings.s3_access_key,
            aws_secret_access_key=self.settings.s3_secret_key,
            region_name=self.settings.s3_region,
        )

    async def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        async with self._client() as s3:
            await s3.put_object(
                Bucket=self.settings.s3_bucket, Key=key, Body=data, ContentType=content_type
            )

    async def stream(self, key: str) -> AsyncIterator[bytes]:
        """Yield the object in chunks — callers never hold the whole file in memory."""
        async with self._client() as s3:
            obj = await s3.get_object(Bucket=self.settings.s3_bucket, Key=key)
            body = obj["Body"]
            try:
                while chunk := await body.read(CHUNK_SIZE):
                    yield chunk
            finally:
                body.close()

    async def delete(self, key: str) -> None:
        async with self._client() as s3:
            await s3.delete_object(Bucket=self.settings.s3_bucket, Key=key)

    async def ensure_bucket(self) -> None:
        """Create the bucket if missing (dev/test convenience; prod uses IaC)."""
        async with self._client() as s3:
            try:
                await s3.head_bucket(Bucket=self.settings.s3_bucket)
            except ClientError:
                await s3.create_bucket(Bucket=self.settings.s3_bucket)


def document_key(workspace_id: str, document_id: str, version: int, filename: str) -> str:
    return f"ws_{workspace_id}/doc_{document_id}/v{version}_{filename}"
