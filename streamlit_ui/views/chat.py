"""Ask DocVault with durable conversations and contextual follow-up turns."""

from collections import Counter
from typing import Any

import api_client as api
import chat_state
import streamlit as st

workspace_id = st.session_state["current_workspace_id"]
chat_state.ensure_workspace(workspace_id)


def _open_document(document_id: str) -> None:
    st.session_state["selected_document_id"] = document_id
    st.switch_page("views/documents.py")


def _render_scope_warning(message: dict[str, Any]) -> None:
    unavailable = message.get("unavailable_documents") or []
    if not unavailable:
        return
    names = ", ".join(
        str(
            document.get("title") or document.get("file_name") or "Unavailable document"
        )
        for document in unavailable
    )
    detail = f" Unavailable: {names}." if names else ""
    st.warning(
        "This conversation's selected scope has changed because you no longer "
        f"have access to one or more pinned documents.{detail}"
    )


def _render_sources(message: dict[str, Any]) -> None:
    sources = message.get("sources") or []
    if not sources:
        return
    ordered = sorted(
        sources,
        key=lambda source: (
            source.get("citation_marker") is None,
            source.get("citation_marker") or source.get("retrieval_rank") or 0,
        ),
    )
    st.markdown("**Sources**")
    for source in ordered:
        marker = source.get("citation_marker")
        prefix = f"[{marker}]" if marker is not None else "Context"
        section_path = source.get("section_path")
        chunk_type = str(source.get("chunk_type") or "").replace("_", " ").title()
        label = " · ".join(
            str(part)
            for part in (prefix, source.get("document_title"), section_path, chunk_type)
            if part
        )
        with st.expander(label):
            supplied = (
                "Supplied to the answer model"
                if source.get("supplied_to_model")
                else "Retrieved only"
            )
            st.caption(supplied)
            if st.button(
                "Open document",
                key=f"open-source-{source['id']}",
                width="stretch",
            ):
                _open_document(str(source["document_id"]))


def _render_message(message: dict[str, Any]) -> None:
    role = str(message["role"])
    with st.chat_message(role):
        content = message.get("content")
        status = message.get("status")
        kind = message.get("kind")
        if status == "failed":
            st.error(content or "The answer could not be completed. Please try again.")
        elif kind in {"redacted", "scope_unavailable"}:
            st.warning(content or "This answer is no longer available.")
        else:
            st.markdown(content or "")
        if role == "assistant":
            _render_scope_warning(message)
            _render_sources(message)


def _scope_description(conversation: dict[str, Any]) -> str:
    if conversation["scope_mode"] == chat_state.SCOPE_WORKSPACE:
        return "All accessible documents"
    documents = conversation.get("documents") or []
    if len(documents) == 1:
        return str(documents[0]["title"])
    return f"{len(documents)} selected documents"


def _recent_label(conversation: dict[str, Any]) -> str:
    updated = str(conversation.get("updated_at", "")).replace("T", " ")[:16]
    return f"{_scope_description(conversation)} · {updated}"


@st.dialog("Delete this conversation?")
def _confirm_delete(conversation_id: str) -> None:
    st.write("Its messages and saved source ledger will be permanently deleted.")
    cancel_col, delete_col = st.columns(2)
    if cancel_col.button("Cancel", width="stretch"):
        st.rerun()
    if delete_col.button("Delete", type="primary", width="stretch"):
        try:
            api.delete_conversation(workspace_id, conversation_id)
        except api.ApiError as exc:
            st.error(exc.detail)
            return
        chat_state.new_conversation()
        chat_state.notify("Conversation deleted.")
        st.rerun()


def _load_recent_conversations() -> list[dict[str, Any]]:
    try:
        return api.list_conversations(workspace_id, limit=20).get("items") or []
    except api.ApiError as exc:
        st.sidebar.error(f"Could not load recent conversations: {exc.detail}")
        return []


active_id = chat_state.active_conversation_id()
recent_conversations = _load_recent_conversations()

with st.sidebar:
    st.subheader("Recent conversations")
    if not recent_conversations:
        st.caption("No saved conversations yet.")
    for recent in recent_conversations:
        recent_id = str(recent["id"])
        if st.button(
            _recent_label(recent),
            key=f"recent-conversation-{recent_id}",
            disabled=recent_id == active_id,
            width="stretch",
            help=_scope_description(recent),
        ):
            chat_state.activate_conversation(recent_id)
            st.rerun()

title_col, new_col, delete_col = st.columns([5, 1, 1], vertical_alignment="center")
title_col.title("Ask DocVault")
if new_col.button("New chat", width="stretch"):
    chat_state.new_conversation()
    st.rerun()
if delete_col.button("Delete", width="stretch", disabled=active_id is None):
    assert active_id is not None
    _confirm_delete(active_id)

notice = chat_state.take_notice()
if notice:
    st.success(notice)

conversation: dict[str, Any] | None = None
messages: list[dict[str, Any]] = []
if active_id is not None:
    try:
        conversation = api.get_conversation(workspace_id, active_id)
        messages = api.list_conversation_messages(workspace_id, active_id)
    except api.ApiError as exc:
        if exc.status_code == 404:
            chat_state.new_conversation()
            chat_state.notify("That conversation is no longer available.")
            st.rerun()
        st.error(f"Could not load this conversation: {exc.detail}")
        st.stop()

    pinned_ids = [
        str(document["document_id"]) for document in conversation["documents"]
    ]
    chat_state.update_scope(str(conversation["scope_mode"]), pinned_ids)
    st.caption(
        f"Scope: **{_scope_description(conversation)}**. The scope is fixed for this "
        "conversation; start a new chat to change it."
    )
else:
    st.caption(
        "Choose a search scope. The conversation will be created when you send "
        "the first message, and follow-ups can refer to earlier grounded answers."
    )


scope_mode = str(st.session_state.get("chat_scope_mode", chat_state.SCOPE_WORKSPACE))
selected_ids = [str(value) for value in st.session_state.get("chat_document_ids", [])]
can_ask = True

if conversation is None:
    try:
        documents = api.list_documents(workspace_id, None, every_folder=True)
        folder_tree = api.folder_tree(workspace_id)
    except api.ApiError as exc:
        st.error(f"Could not load available documents: {exc.detail}")
        st.stop()

    folder_paths = {str(item["id"]): item["path"] for item in folder_tree}
    ready_documents = [doc for doc in documents if doc.get("search_status") == "ready"]
    waiting_documents = [
        doc for doc in documents if doc.get("search_status") != "ready"
    ]
    ready_by_id = {str(doc["id"]): doc for doc in ready_documents}
    title_counts = Counter(str(doc["title"]) for doc in ready_documents)

    def document_label(document_id: str) -> str:
        document = ready_by_id[document_id]
        folder = folder_paths.get(str(document.get("folder_id")), "Workspace root")
        suffix = (
            f" · {document['file_name']}"
            if title_counts[str(document["title"])] > 1
            else ""
        )
        return f"{document['title']} — {folder}{suffix}"

    widget_ids = st.session_state.get("_chat_document_choice")
    if isinstance(widget_ids, list):
        st.session_state["_chat_document_choice"] = [
            str(document_id)
            for document_id in widget_ids
            if str(document_id) in ready_by_id
        ]

    if waiting_documents:
        with st.expander(
            f"{len(waiting_documents)} document(s) are not searchable yet"
        ):
            for document in waiting_documents:
                status = document.get("search_status", "waiting_for_index")
                explanation = (
                    "indexing failed"
                    if status == "indexing_failed"
                    else "waiting for indexing"
                )
                st.caption(f"{document['title']} · {explanation}")

    if not ready_documents:
        st.info(
            "There are no searchable documents available to you in this workspace yet. "
            "Upload documents and complete indexing before starting a conversation."
        )
        can_ask = False
    elif len(ready_documents) == 1:
        only_id = str(ready_documents[0]["id"])
        scope_mode = chat_state.SCOPE_SELECTED
        selected_ids = [only_id]
        st.info(f"Searching in **{document_label(only_id)}**")
    else:
        valid_modes = (chat_state.SCOPE_WORKSPACE, chat_state.SCOPE_SELECTED)
        if scope_mode not in valid_modes:
            scope_mode = chat_state.SCOPE_WORKSPACE
        if st.session_state.get("_chat_scope_choice") not in valid_modes:
            st.session_state["_chat_scope_choice"] = scope_mode
        scope_mode = st.radio(
            "Where should DocVault search?",
            valid_modes,
            format_func=lambda value: (
                "All accessible documents"
                if value == chat_state.SCOPE_WORKSPACE
                else "Choose documents"
            ),
            horizontal=True,
            key="_chat_scope_choice",
            help=(
                "All accessible documents means searchable documents in this workspace "
                "that the signed-in user is allowed to read."
            ),
        )
        if scope_mode == chat_state.SCOPE_SELECTED:
            valid_selected = [
                document_id
                for document_id in selected_ids
                if document_id in ready_by_id
            ]
            if "_chat_document_choice" not in st.session_state:
                st.session_state["_chat_document_choice"] = valid_selected
            selected_ids = st.multiselect(
                "Documents",
                options=list(ready_by_id),
                format_func=document_label,
                max_selections=10,
                key="_chat_document_choice",
                placeholder="Select up to 10 documents",
            )
        else:
            selected_ids = []

    chat_state.update_scope(scope_mode, selected_ids)
    if scope_mode == chat_state.SCOPE_SELECTED and not selected_ids:
        st.info("Select at least one document before asking a question.")
        can_ask = False

for message in messages:
    _render_message(message)

if not messages and can_ask:
    st.caption(
        "Try: “What are the main requirements?” Then follow with “Which of those "
        "is highest risk, and why?”"
    )


question = st.chat_input(
    "Ask a question about the conversation's document scope",
    max_chars=4000,
    disabled=not can_ask,
    submit_mode="disable",
)
if question:
    conversation_id = active_id
    try:
        if conversation_id is None:
            created = api.create_conversation(
                workspace_id,
                scope_mode,
                selected_ids if scope_mode == chat_state.SCOPE_SELECTED else None,
            )
            conversation_id = str(created["id"])
            chat_state.activate_conversation(conversation_id)
        with st.spinner("Searching and preparing the answer…"):
            api.submit_conversation_message(
                workspace_id,
                conversation_id,
                question,
            )
    except api.ApiError as exc:
        st.error(f"I could not submit that message: {exc.detail}")
    else:
        st.rerun()
