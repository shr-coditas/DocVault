"""Session-state helpers for the first-release Ask DocVault experience."""

from typing import Any

import streamlit as st

SCOPE_ALL = "all"
SCOPE_SELECTED = "selected"
MAX_MESSAGES = 40

_STATE_KEYS = (
    "chat_workspace_id",
    "chat_scope_mode",
    "chat_document_ids",
    "chat_messages",
    "_chat_scope_choice",
    "_chat_document_choice",
)


def clear() -> None:
    """Remove all chat data, including widget state."""
    for key in _STATE_KEYS:
        st.session_state.pop(key, None)


def ensure_workspace(workspace_id: str) -> None:
    """Never carry a document scope or messages into another workspace."""
    if st.session_state.get("chat_workspace_id") == workspace_id:
        return
    clear()
    st.session_state["chat_workspace_id"] = workspace_id


def open_document(workspace_id: str, document_id: str) -> None:
    """Start a fresh Ask DocVault thread scoped to one document."""
    clear()
    st.session_state["chat_workspace_id"] = workspace_id
    st.session_state["chat_scope_mode"] = SCOPE_SELECTED
    st.session_state["chat_document_ids"] = [document_id]


def messages() -> list[dict[str, Any]]:
    return st.session_state.setdefault("chat_messages", [])


def append_message(message: dict[str, Any]) -> None:
    history = messages()
    history.append(message)
    if len(history) > MAX_MESSAGES:
        del history[: len(history) - MAX_MESSAGES]


def clear_messages() -> None:
    st.session_state.pop("chat_messages", None)


def update_scope(mode: str, document_ids: list[str]) -> bool:
    """Set the canonical scope and clear messages if an existing scope changed.

    Returns True when a non-empty conversation was cleared.
    """
    normalized = sorted(set(document_ids)) if mode == SCOPE_SELECTED else []
    previous = (
        st.session_state.get("chat_scope_mode"),
        sorted(st.session_state.get("chat_document_ids", [])),
    )
    current = (mode, normalized)
    if previous == current:
        return False
    had_messages = bool(st.session_state.get("chat_messages"))
    st.session_state["chat_scope_mode"] = mode
    st.session_state["chat_document_ids"] = normalized
    clear_messages()
    return had_messages
