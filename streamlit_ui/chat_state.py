"""Ephemeral UI state for server-persisted DocVault conversations."""

import streamlit as st

SCOPE_WORKSPACE = "workspace"
SCOPE_SELECTED = "selected"

_STATE_KEYS = (
    "chat_workspace_id",
    "chat_scope_mode",
    "chat_document_ids",
    "chat_conversation_id",
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
    st.session_state["chat_conversation_id"] = conversation_id


def new_conversation() -> None:
    """Leave durable history on the server and prepare a fresh conversation."""
    st.session_state.pop("chat_conversation_id", None)


def update_scope(mode: str, document_ids: list[str]) -> None:
    """Update the scope used when the next conversation is created."""
    normalized = sorted(set(document_ids)) if mode == SCOPE_SELECTED else []
    st.session_state["chat_scope_mode"] = mode
    st.session_state["chat_document_ids"] = normalized


def notify(message: str) -> None:
    st.session_state["chat_notice"] = message


def take_notice() -> str | None:
    value = st.session_state.pop("chat_notice", None)
    return str(value) if value else None
