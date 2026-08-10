import pytest
from botocore.exceptions import ClientError

from app.config import Settings
from app.services.storage_service import StorageService, document_key


@pytest.mark.integration
async def test_storage_roundtrip(storage_settings: Settings) -> None:
    storage = StorageService(storage_settings)
    await storage.ensure_bucket()
    await storage.ensure_bucket()  # idempotent

    key = document_key("ws1", "doc1", 1, "hello.txt")
    await storage.put_bytes(key, b"hello docvault", "text/plain")

    chunks = [chunk async for chunk in storage.stream(key)]
    assert b"".join(chunks) == b"hello docvault"
    assert await storage.read_all(key) == b"hello docvault"

    await storage.delete(key)
    with pytest.raises(ClientError):
        _ = [chunk async for chunk in storage.stream(key)]
