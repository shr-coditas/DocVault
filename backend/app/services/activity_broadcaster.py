import asyncio
import uuid
from functools import lru_cache
from typing import Any

from fastapi import WebSocket
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

_PENDING_KEY = "pending_activity"
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


class ActivityConnectionManager:
    def __init__(self) -> None:
        self.active_connections: dict[
            uuid.UUID,
            list[WebSocket],
        ] = {}

    def connect(
        self,
        workspace_id: uuid.UUID,
        websocket: WebSocket,
    ) -> None:
        connections = self.active_connections.setdefault(
            workspace_id,
            [],
        )
        connections.append(websocket)

    def disconnect(
        self,
        workspace_id: uuid.UUID,
        websocket: WebSocket,
    ) -> None:
        connections = self.active_connections.get(workspace_id, [])

        if websocket in connections:
            connections.remove(websocket)

        if not connections:
            self.active_connections.pop(workspace_id, None)

    async def broadcast(
        self,
        workspace_id: uuid.UUID,
        message: dict[str, Any],
    ) -> None:
        connections = list(self.active_connections.get(workspace_id, []))

        disconnected: list[WebSocket] = []

        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        for websocket in disconnected:
            self.disconnect(workspace_id, websocket)


@lru_cache
def get_activity_manager() -> ActivityConnectionManager:
    return ActivityConnectionManager()


def stage_activity_event(
    session: AsyncSession,
    payload: dict[str, Any],
) -> None:
    session.sync_session.info.setdefault(
        _PENDING_KEY,
        [],
    ).append(payload)


@event.listens_for(Session, "after_commit")
def publish_after_commit(session: Session) -> None:
    manager = get_activity_manager()

    for payload in session.info.pop(_PENDING_KEY, []):
        workspace_id = payload.get("workspace_id")

        if workspace_id is None:
            continue

        task = asyncio.create_task(
            manager.broadcast(
                uuid.UUID(workspace_id),
                payload,
            )
        )
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)


@event.listens_for(Session, "after_rollback")
def discard_after_rollback(session: Session) -> None:
    session.info.pop(_PENDING_KEY, None)
