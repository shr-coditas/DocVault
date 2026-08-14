"""Ephemeral UI state for server-persisted DocVault conversations."""

import time

import streamlit as st

SCOPE_WORKSPACE = "workspace"
SCOPE_SELECTED = "selected"

_STATE_KEYS = (
    "chat_workspace_id",
    "chat_scope_mode",
    "chat_document_ids",
    "chat_conversation_id",
    "chat_pending_conversation_id",
    "chat_pending_client_message_id",
    "chat_pending_content",
    "chat_pending_turn_id",
    "chat_pending_started_at",
    "chat_pending_error",
    "chat_notice",
    "_chat_scope_choice",
    "_chat_document_choice",
)


def clear() -> None:
    """Remove all chat data, including widget state."""
    for key in _STATE_KEYS:
        st.session_state.pop(key, None)


def ensure_workspace(workspace_id: str) -> None:
    """Never carry conversation identifiers into another workspace."""
    if st.session_state.get("chat_workspace_id") == workspace_id:
        return
    clear()
    st.session_state["chat_workspace_id"] = workspace_id


def open_document(workspace_id: str, document_id: str) -> None:
    """Prepare a new, lazily-created conversation scoped to one document."""
    clear()
    st.session_state["chat_workspace_id"] = workspace_id
    st.session_state["chat_scope_mode"] = SCOPE_SELECTED
    st.session_state["chat_document_ids"] = [document_id]


def active_conversation_id() -> str | None:
    value = st.session_state.get("chat_conversation_id")
    return str(value) if value else None


def activate_conversation(conversation_id: str) -> None:
    clear_pending()
    st.session_state["chat_conversation_id"] = conversation_id


def new_conversation() -> None:
    """Leave durable history on the server and prepare a fresh conversation."""
    st.session_state.pop("chat_conversation_id", None)
    clear_pending()


def update_scope(mode: str, document_ids: list[str]) -> None:
    """Update the scope used when the next conversation is created."""
    normalized = sorted(set(document_ids)) if mode == SCOPE_SELECTED else []
    st.session_state["chat_scope_mode"] = mode
    st.session_state["chat_document_ids"] = normalized


def start_pending(conversation_id: str, client_message_id: str, content: str) -> None:
    st.session_state["chat_pending_conversation_id"] = conversation_id
    st.session_state["chat_pending_client_message_id"] = client_message_id
    st.session_state["chat_pending_content"] = content
    st.session_state["chat_pending_started_at"] = time.monotonic()
    st.session_state.pop("chat_pending_turn_id", None)
    st.session_state.pop("chat_pending_error", None)


def set_pending_turn(turn_id: str | None) -> None:
    if turn_id:
        st.session_state["chat_pending_turn_id"] = turn_id


def set_pending_error(detail: str) -> None:
    st.session_state["chat_pending_error"] = detail


def pending() -> dict[str, str] | None:
    conversation_id = st.session_state.get("chat_pending_conversation_id")
    client_message_id = st.session_state.get("chat_pending_client_message_id")
    content = st.session_state.get("chat_pending_content")
    if not conversation_id or not client_message_id or not content:
        return None
    result = {
        "conversation_id": str(conversation_id),
        "client_message_id": str(client_message_id),
        "content": str(content),
    }
    turn_id = st.session_state.get("chat_pending_turn_id")
    error = st.session_state.get("chat_pending_error")
    started_at = st.session_state.get("chat_pending_started_at")
    if turn_id:
        result["turn_id"] = str(turn_id)
    if error:
        result["error"] = str(error)
    if isinstance(started_at, (int, float)):
        result["started_at"] = str(started_at)
    return result


def clear_pending() -> None:
    for key in (
        "chat_pending_conversation_id",
        "chat_pending_client_message_id",
        "chat_pending_content",
        "chat_pending_turn_id",
        "chat_pending_started_at",
        "chat_pending_error",
    ):
        st.session_state.pop(key, None)


def notify(message: str) -> None:
    st.session_state["chat_notice"] = message


def take_notice() -> str | None:
    value = st.session_state.pop("chat_notice", None)
    return str(value) if value else None
