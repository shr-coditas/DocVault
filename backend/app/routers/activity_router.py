"""Live workspace activity feed.

WebSocket handling stays in the router (like health): a socket loop has no
request/response shape for a controller to translate.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from jwt import PyJWTError

from app.config import get_settings
from app.dependencies import DbSession
from app.repository.user_repository import UserRepository
from app.services.activity_broadcaster import get_broadcaster
from app.services.permission_service import PermissionService
from app.utils.security import decode_access_token

router = APIRouter(tags=["activity"])

POLICY_VIOLATION = 1008  # WebSocket close code: auth/permission failure


@router.websocket("/workspaces/{workspace_id}/activity")
async def workspace_activity(
    websocket: WebSocket,
    workspace_id: uuid.UUID,
    db: DbSession,
    token: Annotated[str | None, Query()] = None,
) -> None:
    """Push every committed workspace event to connected members.

    Browsers cannot set headers on WebSockets, so the access token arrives as
    a query parameter (?token=...). Non-members are closed with 1008 — same
    information-hiding intent as the REST 404 rule.
    """
    await websocket.accept()

    user = None
    if token is not None:
        try:
            payload = decode_access_token(token, get_settings().jwt_secret)
            user = await UserRepository(db).get(uuid.UUID(payload["sub"]))
        except (PyJWTError, ValueError):
            user = None

    if user is None or not user.is_active:
        await websocket.close(code=POLICY_VIOLATION)
        return
    role = await PermissionService(db).workspace_role_name(user.id, workspace_id)
    if role is None:
        await websocket.close(code=POLICY_VIOLATION)
        return
    await db.rollback()  # auth done — don't pin a DB connection for the socket's lifetime

    broadcaster = get_broadcaster()
    queue = broadcaster.subscribe(workspace_id)
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except (WebSocketDisconnect, RuntimeError):
        pass  # client went away (RuntimeError = send on a closed socket)
    finally:
        broadcaster.unsubscribe(workspace_id, queue)
