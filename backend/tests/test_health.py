import pytest
from httpx import AsyncClient


async def test_health_liveness(client: AsyncClient) -> None:
    resp = await client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert resp.headers.get("x-request-id")


async def test_request_id_is_propagated(client: AsyncClient) -> None:
    resp = await client.get("/health", headers={"X-Request-ID": "test-id-123"})

    assert resp.headers.get("x-request-id") == "test-id-123"


@pytest.mark.integration
async def test_health_readiness_with_database(db_client: AsyncClient) -> None:
    resp = await db_client.get("/health/ready")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}
