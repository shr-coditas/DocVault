import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from starlette.testclient import TestClient

from app.db.base import Base
from app.db.session import get_db
from app.main import create_app
from app.scripts.seed_rbac import sync_rbac_catalog
from app.services.activity_broadcaster import ActivityBroadcaster


class TestBroadcasterUnit:
    def test_subscribe_publish_roundtrip(self) -> None:
        broadcaster = ActivityBroadcaster()
        ws_id = uuid.uuid4()

        async def scenario() -> dict:
            queue = broadcaster.subscribe(ws_id)
            broadcaster.publish(ws_id, {"action": "x"})
            return await queue.get()

        assert asyncio.run(scenario()) == {"action": "x"}

    def test_publish_to_other_workspace_not_delivered(self) -> None:
        broadcaster = ActivityBroadcaster()
        mine, other = uuid.uuid4(), uuid.uuid4()

        async def scenario() -> bool:
            queue = broadcaster.subscribe(mine)
            broadcaster.publish(other, {"action": "x"})
            return queue.empty()

        assert asyncio.run(scenario())

    def test_unsubscribe_stops_delivery(self) -> None:
        broadcaster = ActivityBroadcaster()
        ws_id = uuid.uuid4()

        async def scenario() -> bool:
            queue = broadcaster.subscribe(ws_id)
            broadcaster.unsubscribe(ws_id, queue)
            broadcaster.publish(ws_id, {"action": "x"})
            return queue.empty()

        assert asyncio.run(scenario())

    def test_full_queue_drops_instead_of_blocking(self) -> None:
        broadcaster = ActivityBroadcaster()
        ws_id = uuid.uuid4()

        async def scenario() -> int:
            queue = broadcaster.subscribe(ws_id)
            for i in range(150):  # maxsize is 100 — the rest must be dropped silently
                broadcaster.publish(ws_id, {"n": i})
            return queue.qsize()

        assert asyncio.run(scenario()) == 100


@pytest.mark.integration
class TestActivityFeed:
    """End-to-end over a real WebSocket, TestClient + throwaway Postgres.

    TestClient runs the app on its own event loop, so the DB engine is created
    lazily per-loop inside the override (an engine can't hop loops).
    """

    def _make_app_client(self, postgres_url: str) -> TestClient:
        state: dict = {}

        async def override_get_db() -> AsyncIterator[AsyncSession]:
            loop = asyncio.get_running_loop()
            if loop not in state:
                engine = create_async_engine(postgres_url)
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                factory = async_sessionmaker(engine, expire_on_commit=False)
                async with factory() as session:
                    await sync_rbac_catalog(session)
                state[loop] = (engine, factory)
            _, factory = state[loop]
            async with factory() as session:
                yield session

        app = create_app()
        app.dependency_overrides[get_db] = override_get_db
        return TestClient(app)

    @staticmethod
    def _drop_all(postgres_url: str) -> None:
        async def drop() -> None:
            engine: AsyncEngine = create_async_engine(postgres_url)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            await engine.dispose()

        asyncio.run(drop())

    def test_member_receives_committed_events(self, postgres_url: str) -> None:
        client = self._make_app_client(postgres_url)
        try:
            with client:
                client.post(
                    "/api/v1/auth/register",
                    json={
                        "email": "live@example.com",
                        "password": "supersecret123",
                        "full_name": "Live",
                    },
                )
                token = client.post(
                    "/api/v1/auth/login",
                    json={"email": "live@example.com", "password": "supersecret123"},
                ).json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
                ws_id = client.post(
                    "/api/v1/workspaces", json={"name": "Live"}, headers=headers
                ).json()["id"]

                with client.websocket_connect(
                    f"/api/v1/workspaces/{ws_id}/activity?token={token}"
                ) as socket:
                    client.post(
                        f"/api/v1/workspaces/{ws_id}/folders",
                        json={"name": "Watched"},
                        headers=headers,
                    )
                    event = socket.receive_json()
                    assert event["action"] == "folder.created"
                    assert event["workspace_id"] == ws_id
                    assert event["extra"]["name"] == "Watched"
        finally:
            self._drop_all(postgres_url)

    def test_bad_token_closed_with_1008(self, postgres_url: str) -> None:
        client = self._make_app_client(postgres_url)
        try:
            with client:
                ws_id = uuid.uuid4()
                with client.websocket_connect(
                    f"/api/v1/workspaces/{ws_id}/activity?token=not-a-jwt"
                ) as socket:
                    message = socket.receive()
                    assert message["type"] == "websocket.close"
                    assert message["code"] == 1008
        finally:
            self._drop_all(postgres_url)

    def test_non_member_closed_with_1008(self, postgres_url: str) -> None:
        client = self._make_app_client(postgres_url)
        try:
            with client:
                for email in ("owner2@example.com", "outsider@example.com"):
                    client.post(
                        "/api/v1/auth/register",
                        json={"email": email, "password": "supersecret123", "full_name": "U"},
                    )

                def login(email: str) -> str:
                    resp = client.post(
                        "/api/v1/auth/login",
                        json={"email": email, "password": "supersecret123"},
                    )
                    token: str = resp.json()["access_token"]
                    return token

                owner_token = login("owner2@example.com")
                outsider_token = login("outsider@example.com")
                ws_id = client.post(
                    "/api/v1/workspaces",
                    json={"name": "Private"},
                    headers={"Authorization": f"Bearer {owner_token}"},
                ).json()["id"]

                with client.websocket_connect(
                    f"/api/v1/workspaces/{ws_id}/activity?token={outsider_token}"
                ) as socket:
                    message = socket.receive()
                    assert message["type"] == "websocket.close"
                    assert message["code"] == 1008
        finally:
            self._drop_all(postgres_url)
