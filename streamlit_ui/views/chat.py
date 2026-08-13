"""Ask DocVault - explicit document scope, answers, and traceable citations."""

from collections import Counter
from typing import Any

import streamlit as st

import api_client as api
import chat_state

workspace_id = st.session_state["current_workspace_id"]
chat_state.ensure_workspace(workspace_id)


def _page_text(pages: list[int]) -> str:
    if not pages:
        return ""
    ordered = sorted(set(pages))
    if len(ordered) == 1:
        return f"page {ordered[0]}"
    return f"pages {ordered[0]}-{ordered[-1]}"


def _snippet(text: str, maximum: int = 600) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= maximum else f"{compact[: maximum - 1].rstrip()}…"


def _open_document(document_id: str) -> None:
    st.session_state["selected_document_id"] = document_id
    st.switch_page("views/documents.py")


def _render_sources(message: dict[str, Any], message_index: int) -> None:
    citations = message.get("citations") or []
    hits = message.get("hits") or []
    hits_by_chunk = {str(hit.get("chunk_id")): hit for hit in hits}

    if citations:
        st.markdown("**Sources**")
        for source_index, citation in enumerate(citations):
            hit = hits_by_chunk.get(str(citation.get("chunk_id")), {})
            breadcrumb = hit.get("breadcrumb") or citation.get("heading")
            pages = citation.get("page_numbers") or []
            location = _page_text(pages)
            label_parts = [
                f"[{citation.get('marker', source_index + 1)}]",
                citation["document_title"],
            ]
            if breadcrumb:
                label_parts.append(breadcrumb)
            if location:
                label_parts.append(location)
            with st.expander(" · ".join(label_parts)):
                content = hit.get("content")
                if content:
                    st.text(_snippet(content))
                st.caption(
                    f"Generation {citation['index_generation']} · "
                    f"source `{citation['logical_key']}`"
                )
                if st.button(
                    "Open document",
                    key=f"open-citation-{message_index}-{source_index}",
                    width="stretch",
                ):
                    _open_document(str(citation["document_id"]))
        return

    if not hits:
        return
    st.markdown("**Retrieved sources**")
    for source_index, hit in enumerate(hits[:3]):
        breadcrumb = hit.get("breadcrumb") or hit.get("heading")
        pages = _page_text(hit.get("page_numbers") or [])
        label = " · ".join(
            part for part in (hit.get("document_title"), breadcrumb, pages) if part
        )
        with st.expander(label):
            st.text(_snippet(hit.get("content", "")))
            if st.button(
                "Open document",
                key=f"open-hit-{message_index}-{source_index}",
                width="stretch",
            ):
                _open_document(str(hit["document_id"]))


def _render_message(message: dict[str, Any], message_index: int) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("scope"):
            st.caption(f"Searched: {message['scope']}")
        if message["role"] == "assistant":
            _render_sources(message, message_index)


def _assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    answer = response.get("answer")
    if answer:
        return {
            "role": "assistant",
            "content": answer["text"],
            "citations": answer.get("citations", []),
            "hits": response.get("hits", []),
        }
    return {
        "role": "assistant",
        "content": response.get("message") or "DocVault could not produce an answer.",
        "citations": [],
        "hits": response.get("hits", []),
    }


title_col, action_col = st.columns([5, 1], vertical_alignment="center")
title_col.title("Ask DocVault")
if action_col.button("Clear chat", width="stretch", disabled=not chat_state.messages()):
    chat_state.clear_messages()
    st.rerun()

st.caption(
    "Choose where to search, then ask a question. This first release treats each "
    "question independently and always verifies document access on the server."
)

try:
    documents = api.list_documents(workspace_id, None, every_folder=True)
    folder_tree = api.folder_tree(workspace_id)
except api.ApiError as exc:
    st.error(f"Could not load available documents: {exc.detail}")
    st.stop()

folder_paths = {item["id"]: item["path"] for item in folder_tree}
ready_documents = [doc for doc in documents if doc.get("search_status") == "ready"]
waiting_documents = [doc for doc in documents if doc.get("search_status") != "ready"]
ready_by_id = {str(doc["id"]): doc for doc in ready_documents}

# Widget state survives reruns independently of our canonical scope. Remove a
# document immediately if it was deleted, trashed, or became inaccessible.
widget_ids = st.session_state.get("_chat_document_choice")
if isinstance(widget_ids, list):
    st.session_state["_chat_document_choice"] = [
        str(document_id)
        for document_id in widget_ids
        if str(document_id) in ready_by_id
    ]

if waiting_documents:
    with st.expander(f"{len(waiting_documents)} document(s) are not searchable yet"):
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
        "Upload documents and complete indexing before asking questions."
    )
    st.stop()

title_counts = Counter(str(doc["title"]) for doc in ready_documents)


def _document_label(document_id: str) -> str:
    document = ready_by_id[document_id]
    folder = folder_paths.get(document.get("folder_id"), "Workspace root")
    suffix = (
        f" · {document['file_name']}"
        if title_counts[str(document["title"])] > 1
        else ""
    )
    return f"{document['title']} — {folder}{suffix}"


stored_ids = [
    str(document_id)
    for document_id in st.session_state.get("chat_document_ids", [])
    if str(document_id) in ready_by_id
]

if len(ready_documents) == 1:
    only_id = str(ready_documents[0]["id"])
    scope_mode = chat_state.SCOPE_SELECTED
    selected_ids = [only_id]
    st.info(f"Searching in **{_document_label(only_id)}**")
else:
    option_by_mode = {
        chat_state.SCOPE_ALL: "All accessible documents",
        chat_state.SCOPE_SELECTED: "Choose documents",
    }
    stored_mode = st.session_state.get("chat_scope_mode", chat_state.SCOPE_ALL)
    mode_options = list(option_by_mode)
    scope_mode = st.radio(
        "Where should DocVault search?",
        mode_options,
        index=mode_options.index(stored_mode),
        format_func=option_by_mode.get,
        horizontal=True,
        key="_chat_scope_choice",
        help="All accessible documents means searchable documents in this workspace that you are allowed to read.",
    )
    if scope_mode == chat_state.SCOPE_SELECTED:
        selected_ids = st.multiselect(
            "Documents",
            options=list(ready_by_id),
            default=stored_ids,
            format_func=_document_label,
            max_selections=10,
            key="_chat_document_choice",
            placeholder="Select up to 10 documents",
        )
    else:
        selected_ids = []

if chat_state.update_scope(scope_mode, selected_ids):
    st.toast("The document scope changed, so a new chat was started.", icon="🔎")

if scope_mode == chat_state.SCOPE_ALL:
    scope_label = f"All {len(ready_documents)} accessible document(s)"
    query_document_ids: list[str] | None = None
else:
    query_document_ids = selected_ids
    scope_label = (
        _document_label(selected_ids[0])
        if len(selected_ids) == 1
        else f"{len(selected_ids)} selected documents"
    )

for index, message in enumerate(chat_state.messages()):
    _render_message(message, index)

if scope_mode == chat_state.SCOPE_SELECTED and not selected_ids:
    st.info("Select at least one document before asking a question.")
    st.stop()

if not chat_state.messages():
    st.caption(
        "Try asking: “What are the main requirements?”, “Summarize the key risks”, "
        "or “Which document mentions the approval process?”"
    )

question = st.chat_input("Ask a question about the selected documents")
if question:
    user_message = {"role": "user", "content": question, "scope": scope_label}
    chat_state.append_message(user_message)
    _render_message(user_message, len(chat_state.messages()) - 1)

    try:
        with st.chat_message("assistant"):
            with st.spinner(
                "Searching the selected documents and preparing an answer…"
            ):
                response = api.query_documents(
                    workspace_id,
                    question,
                    document_ids=query_document_ids,
                )
        chat_state.append_message(_assistant_message(response))
    except api.ApiError as exc:
        chat_state.append_message(
            {
                "role": "assistant",
                "content": f"I could not complete that request: {exc.detail}",
                "citations": [],
                "hits": [],
            }
        )
    st.rerun()
