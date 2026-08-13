import asyncio
import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocket
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
from app.services.activity_broadcaster import ActivityConnectionManager
from tests.helpers import create_schema, sync_rbac_catalog


class TestActivityManagerUnit:
    def test_connect_adds_websocket(self) -> None:
        manager = ActivityConnectionManager()
        workspace_id = uuid.uuid4()
        websocket = AsyncMock(spec=WebSocket)

        manager.connect(workspace_id, websocket)

        assert websocket in manager.active_connections[workspace_id]

    def test_disconnect_removes_websocket(self) -> None:
        manager = ActivityConnectionManager()
        workspace_id = uuid.uuid4()
        websocket = AsyncMock(spec=WebSocket)

        manager.connect(workspace_id, websocket)
        manager.disconnect(workspace_id, websocket)

        assert workspace_id not in manager.active_connections

    def test_broadcast_sends_to_workspace_connections(self) -> None:
        manager = ActivityConnectionManager()
        workspace_id = uuid.uuid4()
        websocket = AsyncMock(spec=WebSocket)

        manager.connect(workspace_id, websocket)

        asyncio.run(
            manager.broadcast(
                workspace_id,
                {"action": "folder.created"},
            )
        )

        websocket.send_json.assert_awaited_once_with({"action": "folder.created"})

    def test_broadcast_does_not_send_to_other_workspace(self) -> None:
        manager = ActivityConnectionManager()
        workspace_id = uuid.uuid4()
        other_workspace_id = uuid.uuid4()
        websocket = AsyncMock(spec=WebSocket)

        manager.connect(workspace_id, websocket)

        asyncio.run(
            manager.broadcast(
                other_workspace_id,
                {"action": "folder.created"},
            )
        )

        websocket.send_json.assert_not_awaited()

    def test_failed_connection_is_removed(self) -> None:
        manager = ActivityConnectionManager()
        workspace_id = uuid.uuid4()
        websocket = AsyncMock(spec=WebSocket)
        websocket.send_json.side_effect = RuntimeError("connection closed")

        manager.connect(workspace_id, websocket)

        asyncio.run(
            manager.broadcast(
                workspace_id,
                {"action": "folder.created"},
            )
        )

        assert workspace_id not in manager.active_connections


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
                await create_schema(engine)
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
