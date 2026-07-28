"""🔔 activity popover: live workspace events over the backend WebSocket.

Streamlit reruns the whole script on every interaction, so a persistent
WebSocket doesn't fit that model directly: a daemon thread owns the
connection and hands events to the script through a thread-safe queue,
and an auto-rerunning fragment drains + renders it every few seconds.
"""

import asyncio
import json
import queue as queue_mod
import threading
from typing import Any

import streamlit as st
import websockets
import websockets.exceptions

import api_client as api

_STATE_KEYS = (
    "activity_stop_event",
    "activity_status_box",
    "activity_queue",
    "activity_events",
    "activity_ws_id",
    "activity_thread",
    "activity_history",
)


def _ws_url(workspace_id: str, token: str) -> str:
    base = api.base_url().replace("http://", "ws://").replace("https://", "wss://")
    return f"{base}/workspaces/{workspace_id}/activity?token={token}"


def _run_feed(
    url: str,
    event_queue: "queue_mod.Queue",
    status_box: dict,
    stop_event: threading.Event,
) -> None:
    async def consume() -> None:
        status_box["state"] = "connecting"
        try:
            async with websockets.connect(url) as ws:
                status_box["state"] = "connected"
                while not stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except TimeoutError:
                        continue
                    event_queue.put(json.loads(message))
        except websockets.exceptions.ConnectionClosed as exc:
            reason = exc.reason or (
                "token expired or not a member" if exc.code == 1008 else ""
            )
            status_box["state"] = f"closed (code={exc.code}) {reason}".strip()
            return
        except (
            Exception
        ) as exc:  # demo tool: any failure just surfaces as a status string
            status_box["state"] = f"error: {exc}"
            return
        status_box["state"] = "stopped"

    asyncio.run(consume())


def _start_feed(workspace_id: str) -> None:
    old_stop = st.session_state.get("activity_stop_event")
    if old_stop is not None:
        old_stop.set()
    stop_event = threading.Event()
    status_box = {"state": "connecting"}
    st.session_state["activity_stop_event"] = stop_event
    st.session_state["activity_status_box"] = status_box
    st.session_state["activity_queue"] = queue_mod.Queue()
    st.session_state["activity_events"] = []
    st.session_state["activity_ws_id"] = workspace_id
    st.session_state.pop("activity_history", None)
    thread = threading.Thread(
        target=_run_feed,
        args=(
            _ws_url(workspace_id, st.session_state["access_token"]),
            st.session_state["activity_queue"],
            status_box,
            stop_event,
        ),
        daemon=True,
    )
    st.session_state["activity_thread"] = thread
    thread.start()


def stop_feed() -> None:
    """Tear the socket + state down (workspace exit / logout)."""
    stop_event = st.session_state.get("activity_stop_event")
    if stop_event is not None:
        stop_event.set()
    for key in _STATE_KEYS:
        st.session_state.pop(key, None)


def _render_event(
    when: str, action: str, resource_type: str, extra: dict[str, Any]
) -> None:
    name = extra.get("title") or extra.get("name") or extra.get("file_name") or ""
    line = f"`{(when or '')[11:19]}` **{action}** · {resource_type}"
    if name:
        line += f" — {name}"
    st.markdown(line)


@st.fragment(run_every=3)
def _feed_body(workspace_id: str) -> None:
    q: queue_mod.Queue | None = st.session_state.get("activity_queue")
    events: list[dict[str, Any]] = st.session_state.get("activity_events", [])
    while q is not None:
        try:
            events.insert(0, q.get_nowait())
        except queue_mod.Empty:
            break
    st.session_state["activity_events"] = events[:50]

    state = st.session_state.get("activity_status_box", {}).get("state", "idle")
    icon = {"connected": "🟢", "connecting": "🟡"}.get(state, "🔴")
    status_col, action_col = st.columns([3, 1])
    status_col.caption(f"{icon} live feed: `{state}`")
    if state not in ("connected", "connecting"):
        if action_col.button("Reconnect", key="activity-reconnect"):
            _start_feed(workspace_id)
            st.rerun(scope="fragment")

    if not st.session_state["activity_events"]:
        st.caption("No live events yet — actions by any member appear here instantly.")
    for ev in st.session_state["activity_events"]:
        _render_event(
            ev.get("occurred_at", ""),
            ev.get("action", "?"),
            ev.get("resource_type", ""),
            ev.get("extra") or {},
        )


def bell(workspace_id: str, my_role: str) -> None:
    """The 🔔 popover rendered by the shell on every page."""
    if st.session_state.get("activity_ws_id") != workspace_id:
        _start_feed(workspace_id)
    with st.popover("🔔", help="Workspace activity"):
        st.markdown("**Activity**")
        _feed_body(workspace_id)
        if my_role == "owner":
            st.divider()
            if st.button("Load recent history", key="activity-history-btn"):
                try:
                    st.session_state["activity_history"] = api.get_audit_log(
                        workspace_id, limit=20
                    )
                except api.ApiError as e:
                    st.error(f"Could not load history: {e.detail}")
            history = st.session_state.get("activity_history")
            if history is not None:
                st.caption("Recent history (owners only until per-role feeds land):")
                for row in history:
                    _render_event(
                        row.get("created_at", ""),
                        row.get("action", "?"),
                        row.get("resource_type", ""),
                        row.get("extra") or {},
                    )
