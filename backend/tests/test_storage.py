from collections.abc import Iterator

import pytest
from botocore.exceptions import ClientError
from testcontainers.minio import MinioContainer

from app.core.config import Settings
from app.storage.service import StorageService, document_key


@pytest.fixture(scope="session")
def minio_settings() -> Iterator[Settings]:
    with MinioContainer() as minio:
        cfg = minio.get_config()
        yield Settings(
            s3_endpoint_url=f"http://{cfg['endpoint']}",
            s3_access_key=cfg["access_key"],
            s3_secret_key=cfg["secret_key"],
            s3_bucket="test-bucket",
        )


@pytest.mark.integration
async def test_storage_roundtrip(minio_settings: Settings) -> None:
    storage = StorageService(minio_settings)
    await storage.ensure_bucket()
    await storage.ensure_bucket()  # idempotent

    key = document_key("ws1", "doc1", 1, "hello.txt")
    await storage.put_bytes(key, b"hello docvault", "text/plain")

    chunks = [chunk async for chunk in storage.stream(key)]
    assert b"".join(chunks) == b"hello docvault"

    await storage.delete(key)
    with pytest.raises(ClientError):
        _ = [chunk async for chunk in storage.stream(key)]
