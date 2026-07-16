"""In-process pub/sub for workspace activity, published strictly post-commit.

AuditService stages events on the session (`stage_activity_event`); the
`after_commit` listener below publishes them, and `after_rollback` discards
them — so the feed can never report a change that didn't actually happen.
Single-process only by design (matches the one-container deployment); a
Redis/broker-backed broadcaster is the drop-in replacement when scaling out.
"""

import asyncio
import contextlib
import uuid
from functools import lru_cache
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

_PENDING_KEY = "pending_activity"
QUEUE_MAXSIZE = 100


class ActivityBroadcaster:
    def __init__(self) -> None:
        self._subscribers: dict[uuid.UUID, set[asyncio.Queue[dict[str, Any]]]] = {}

    def subscribe(self, workspace_id: uuid.UUID) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self._subscribers.setdefault(workspace_id, set()).add(queue)
        return queue

    def unsubscribe(self, workspace_id: uuid.UUID, queue: asyncio.Queue[dict[str, Any]]) -> None:
        listeners = self._subscribers.get(workspace_id)
        if listeners is not None:
            listeners.discard(queue)
            if not listeners:
                del self._subscribers[workspace_id]

    def publish(self, workspace_id: uuid.UUID, payload: dict[str, Any]) -> None:
        for queue in self._subscribers.get(workspace_id, ()):
            # a slow consumer loses events; it never blocks the app
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(payload)


@lru_cache
def get_broadcaster() -> ActivityBroadcaster:
    return ActivityBroadcaster()


def stage_activity_event(session: AsyncSession, payload: dict[str, Any]) -> None:
    """Queue an event on the session; it publishes only if the commit succeeds."""
    session.sync_session.info.setdefault(_PENDING_KEY, []).append(payload)


@event.listens_for(Session, "after_commit")
def _publish_after_commit(session: Session) -> None:
    for payload in session.info.pop(_PENDING_KEY, []):
        workspace_id = payload.get("workspace_id")
        if workspace_id is not None:
            get_broadcaster().publish(uuid.UUID(workspace_id), payload)


@event.listens_for(Session, "after_rollback")
def _discard_after_rollback(session: Session) -> None:
    session.info.pop(_PENDING_KEY, None)
