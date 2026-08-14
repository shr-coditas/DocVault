"""Ask DocVault with durable conversations and contextual follow-up turns."""

import time
import uuid
from collections import Counter
from typing import Any

import api_client as api
import chat_state
import streamlit as st

workspace_id = st.session_state["current_workspace_id"]
chat_state.ensure_workspace(workspace_id)


def _page_text(pages: list[int]) -> str:
    ordered = sorted(set(pages))
    if not ordered:
        return ""
    if len(ordered) == 1:
        return f"page {ordered[0]}"
    return f"pages {ordered[0]}-{ordered[-1]}"


def _span_text(source_spans: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for span in source_spans:
        location = span.get("location") or {}
        page = location.get("page_number")
        paragraph = location.get("paragraph_index")
        row_start = location.get("row_start")
        row_end = location.get("row_end")
        if page is not None:
            parts.append(f"page {page}")
        if paragraph is not None:
            parts.append(f"paragraph {paragraph}")
        if row_start is not None:
            rows = str(row_start) if row_end in (None, row_start) else f"{row_start}-{row_end}"
            parts.append(f"table row {rows}")
    return " · ".join(dict.fromkeys(parts))


def _open_document(document_id: str) -> None:
    st.session_state["selected_document_id"] = document_id
    st.switch_page("views/documents.py")


def _render_scope_warning(message: dict[str, Any]) -> None:
    if not message.get("scope_degraded"):
        return
    unavailable = message.get("unavailable_documents") or []
    names = ", ".join(
        str(document.get("title") or document.get("file_name") or "Unavailable document")
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
        breadcrumb = source.get("breadcrumb") or source.get("heading")
        location = _page_text(source.get("page_numbers") or [])
        label = " · ".join(
            str(part)
            for part in (prefix, source.get("document_title"), breadcrumb, location)
            if part
        )
        with st.expander(label):
            precise_location = _span_text(source.get("source_spans") or [])
            if precise_location:
                st.caption(precise_location)
            if source.get("relocated"):
                st.info("This source was relocated to the document's current index generation.")
            supplied = (
                "Supplied to the answer model"
                if source.get("supplied_to_model")
                else "Retrieved only"
            )
            st.caption(
                f"{supplied} · generation {source['index_generation']} · "
                f"source `{source['logical_key']}`"
            )
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
        if status == "pending":
            st.markdown(content or "Searching the permitted document scope…")
        elif status == "failed":
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

    pinned_ids = [str(document["document_id"]) for document in conversation["documents"]]
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


def _restore_pending_from_messages() -> None:
    if active_id is None or chat_state.pending() is not None:
        return
    users_by_turn = {
        str(message["turn_id"]): message for message in messages if message["role"] == "user"
    }
    pending_assistants = [
        message
        for message in messages
        if message["role"] == "assistant" and message["status"] == "pending"
    ]
    if not pending_assistants:
        return
    assistant = pending_assistants[-1]
    user = users_by_turn.get(str(assistant["turn_id"]))
    if user is None or not user.get("client_message_id") or not user.get("content"):
        return
    chat_state.start_pending(
        active_id,
        str(user["client_message_id"]),
        str(user["content"]),
    )
    chat_state.set_pending_turn(str(assistant["turn_id"]))


_restore_pending_from_messages()
pending = chat_state.pending()
if pending and pending["conversation_id"] != active_id:
    chat_state.clear_pending()
    pending = None

# A terminal message may have appeared between the last poll and this full reload.
if pending:
    terminal_turn_ids = {
        str(message["turn_id"])
        for message in messages
        if message["role"] == "assistant" and message["status"] != "pending"
    }
    if pending.get("turn_id") in terminal_turn_ids:
        chat_state.clear_pending()
        pending = None

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
    waiting_documents = [doc for doc in documents if doc.get("search_status") != "ready"]
    ready_by_id = {str(doc["id"]): doc for doc in ready_documents}
    title_counts = Counter(str(doc["title"]) for doc in ready_documents)

    def document_label(document_id: str) -> str:
        document = ready_by_id[document_id]
        folder = folder_paths.get(str(document.get("folder_id")), "Workspace root")
        suffix = f" · {document['file_name']}" if title_counts[str(document["title"])] > 1 else ""
        return f"{document['title']} — {folder}{suffix}"

    widget_ids = st.session_state.get("_chat_document_choice")
    if isinstance(widget_ids, list):
        st.session_state["_chat_document_choice"] = [
            str(document_id) for document_id in widget_ids if str(document_id) in ready_by_id
        ]

    if waiting_documents:
        with st.expander(f"{len(waiting_documents)} document(s) are not searchable yet"):
            for document in waiting_documents:
                status = document.get("search_status", "waiting_for_index")
                explanation = (
                    "indexing failed" if status == "indexing_failed" else "waiting for indexing"
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
                document_id for document_id in selected_ids if document_id in ready_by_id
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

if pending and not any(
    str(message.get("client_message_id")) == pending["client_message_id"] for message in messages
):
    _render_message(
        {
            "role": "user",
            "status": "complete",
            "kind": "question",
            "content": pending["content"],
        }
    )
    _render_message(
        {
            "role": "assistant",
            "status": "pending",
            "kind": "answer",
            "content": None,
        }
    )

if not messages and not pending and can_ask:
    st.caption(
        "Try: “What are the main requirements?” Then follow with “Which of those "
        "is highest risk, and why?”"
    )


def _handle_turn_response(response: dict[str, Any]) -> None:
    if response.get("status") == "pending":
        chat_state.set_pending_turn(str(response["turn_id"]))
        chat_state.set_pending_error("The answer is still being prepared.")
    else:
        chat_state.clear_pending()


@st.fragment(run_every="2s")
def _poll_pending_turn() -> None:
    current = chat_state.pending()
    if current is None or current["conversation_id"] != chat_state.active_conversation_id():
        return
    try:
        started_at = float(current.get("started_at", time.monotonic()))
        lease_may_be_stale = time.monotonic() - started_at >= 125
        if current.get("turn_id") and not lease_may_be_stale:
            response = api.get_conversation_turn(
                workspace_id,
                current["conversation_id"],
                current["turn_id"],
            )
        else:
            response = api.submit_conversation_message(
                workspace_id,
                current["conversation_id"],
                current["content"],
                current["client_message_id"],
            )
        _handle_turn_response(response)
        if response.get("status") != "pending":
            st.rerun()
    except api.ApiError as exc:
        if exc.status_code in {409, 503, 504}:
            chat_state.set_pending_error(exc.detail)
        else:
            chat_state.clear_pending()
            chat_state.notify(f"The turn could not be resumed: {exc.detail}")
            st.rerun()

    latest = chat_state.pending()
    if latest:
        st.info(latest.get("error", "Preparing the answer…"))


pending = chat_state.pending()
if pending:
    _poll_pending_turn()

question = st.chat_input(
    "Ask a question about the conversation's document scope",
    max_chars=4000,
    disabled=not can_ask or pending is not None,
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
        client_message_id = str(uuid.uuid4())
        chat_state.start_pending(conversation_id, client_message_id, question)
        response = api.submit_conversation_message(
            workspace_id,
            conversation_id,
            question,
            client_message_id,
        )
        _handle_turn_response(response)
    except api.ApiError as exc:
        if exc.status_code in {409, 503, 504} and conversation_id is not None:
            chat_state.set_pending_error(exc.detail)
        else:
            chat_state.clear_pending()
            chat_state.notify(f"I could not submit that message: {exc.detail}")
    st.rerun()
