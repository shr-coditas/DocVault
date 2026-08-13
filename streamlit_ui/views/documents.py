"""Documents - folder browsing, upload, list + inline preview, lifecycle actions."""

import io
import json
from typing import Any

import streamlit as st

import api_client as api
import chat_state

workspace_id = st.session_state["current_workspace_id"]

# viewers get document:read and nothing else, so every mutating control here
# would 403 for them - hide it rather than let them click into an error
my_role = next(
    (
        w["my_role"]
        for w in st.session_state.get("workspaces", [])
        if w["id"] == workspace_id
    ),
    "viewer",
)
can_edit = my_role in ("owner", "editor")

try:
    tree = api.folder_tree(workspace_id)
except api.ApiError as e:
    st.error(f"Could not load folder tree: {e.detail}")
    st.stop()

# label -> id; placement targets always offer "(workspace root)" -> None.
# Labelled by full path, not name: two folders may share a name under different
# parents, and a name-keyed dict would silently drop one of them.
folder_options: dict[str, str | None] = {"(workspace root)": None}
for item in tree:
    folder_options[item["path"]] = item["id"]

# Browsing is a different question from placement: "All documents" spans every
# folder, while a folder_id of None means specifically the workspace root.
ALL_DOCUMENTS = "All documents"
browse_options: dict[str, str | None] = {ALL_DOCUMENTS: None, "Workspace root": None}
for label, folder_id in folder_options.items():
    if folder_id is not None:
        browse_options[label] = folder_id


def _fmt_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _visibility_badge(doc: dict[str, Any]) -> str:
    """Trailing caption fragment; workspace-visible is the default, so stays quiet."""
    return (
        "" if doc.get("visibility", "workspace") == "workspace" else " · 🔒 restricted"
    )


def _get_bytes(doc: dict[str, Any]) -> bytes | None:
    cache: dict[str, bytes] = st.session_state.setdefault("doc_bytes_cache", {})
    if doc["id"] not in cache:
        try:
            cache[doc["id"]] = api.download_document(workspace_id, doc["id"])
        except api.ApiError as e:
            st.error(f"Could not fetch the file: {e.detail}")
            return None
        while len(cache) > 5:  # bound memory; uploads are capped at 25 MB each
            cache.pop(next(iter(cache)))
    return cache[doc["id"]]


def _preview(doc: dict[str, Any], data: bytes) -> None:
    mime = (doc.get("mime_type") or "").lower()
    name = doc["file_name"].lower()

    def text() -> str:
        return data[:200_000].decode("utf-8", errors="replace")

    if mime == "application/pdf" or name.endswith(".pdf"):
        st.pdf(data, height=620)
    elif mime.startswith("image/"):
        st.image(data)
    elif mime == "text/markdown" or name.endswith(".md"):
        st.markdown(text())
    elif mime == "text/csv" or name.endswith(".csv"):
        import pandas as pd

        try:
            st.dataframe(pd.read_csv(io.BytesIO(data)), width="stretch")
        except Exception:
            st.code(text(), language=None)
    elif mime == "application/json" or name.endswith(".json"):
        try:
            st.json(json.loads(data))
        except Exception:
            st.code(text(), language=None)
    elif mime.startswith("text/"):
        st.code(text(), language=None)
    else:
        st.info("No inline preview for this file type - use Download instead.")


@st.dialog("Rename or move")
def _rename_dialog(doc: dict[str, Any]) -> None:
    current_label = next(
        (label for label, fid in folder_options.items() if fid == doc["folder_id"]),
        "(workspace root)",
    )
    with st.form("rename-form"):
        new_title = st.text_input("Title", value=doc["title"])
        move_label = st.selectbox(
            "Folder",
            list(folder_options),
            index=list(folder_options).index(current_label),
        )
        if st.form_submit_button("Save"):
            try:
                api.update_document(
                    workspace_id,
                    doc["id"],
                    new_title,
                    folder_options[move_label],
                    moved=(move_label != current_label),
                )
                st.rerun()
            except api.ApiError as e:
                st.error(f"Update failed: {e.detail}")


# Two states, matching the API exactly: `restricted` means "opened by whoever
# you share with, person or team alike", `workspace` means everyone here.
RESTRICTED, EVERYONE = "restricted", "workspace"
AUDIENCES = [RESTRICTED, EVERYONE]
AUDIENCE_LABELS = {
    RESTRICTED: "🔒 Restricted",
    EVERYONE: "🌐 Everyone in the workspace",
}
AUDIENCE_HELP = {
    RESTRICTED: "Only you, workspace owners, and the people and teams you share with below.",
    EVERYONE: "Everyone in this workspace. Nothing to share - they already have it.",
}


def _audience_of(visibility: str) -> str:
    return EVERYONE if visibility == "workspace" else RESTRICTED


def _shared_with_sentence(grants: list[dict[str, Any]]) -> str:
    """Spell out who can actually see this, so the dialog doesn't leave the
    owner guessing what the current setting adds up to."""
    people = sum(1 for g in grants if g["principal_type"] == "user")
    teams = sum(1 for g in grants if g["principal_type"] == "team")
    if not grants:
        return "Only you and workspace owners can see this."
    parts = []
    if teams:
        parts.append(f"{teams} team{'s' if teams > 1 else ''}")
    if people:
        parts.append(f"{people} {'person' if people == 1 else 'people'}")
    return f"Shared with {' and '.join(parts)}, plus you and workspace owners."


@st.dialog("Share")
def _share_dialog(doc: dict[str, Any]) -> None:
    st.caption(f"**{doc['title']}**")

    current = _audience_of(doc.get("visibility", "workspace"))
    chosen = st.selectbox(
        "Who can see this document",
        AUDIENCES,
        index=AUDIENCES.index(current),
        format_func=lambda a: AUDIENCE_LABELS[a],
        help="Your workspace role still controls what you may *do* with it.",
    )
    # One slot for everything below the picker. st.empty() replaces its whole
    # subtree on each render; plain early returns leave the previous render's
    # elements behind, so switching to Everyone would still list who it's
    # shared with.
    body = st.empty()
    with body.container():
        st.caption(AUDIENCE_HELP[chosen])
        if chosen != current:
            if st.button("Apply", width="stretch", type="primary"):
                try:
                    api.set_visibility(workspace_id, doc["id"], chosen)
                    st.rerun()
                except api.ApiError as e:
                    st.error(f"Could not change who can see it: {e.detail}")
        elif current == EVERYONE:
            st.info(
                "Everyone in the workspace can already see this - nothing to share."
            )
        else:
            _sharing_section(doc)


def _sharing_section(doc: dict[str, Any]) -> None:
    """Who has access to a restricted document, and how to add to that."""
    # friendly names for principals we may share with / have already shared with
    try:
        members = api.list_members(workspace_id)
        teams = api.list_teams(workspace_id)
    except api.ApiError as e:
        st.error(f"Could not load people and teams: {e.detail}")
        return
    user_names = {m["user_id"]: m["email"] for m in members}
    # team names are only unique per workspace by constraint, but be defensive:
    # a duplicate key would silently make one team unshareable
    team_names: dict[str, str] = {}
    for team in teams:
        label = team["name"]
        if label in team_names.values():
            label = f"{label} ({team['id'][:8]})"
        team_names[team["id"]] = label

    try:
        grants = api.list_grants(workspace_id, doc["id"])
    except api.ApiError as e:
        st.error(f"Could not load who it's shared with: {e.detail}")
        return

    st.caption(_shared_with_sentence(grants))

    st.divider()
    st.markdown("**Has access**")
    if not grants:
        st.caption("Nobody else yet - add a person or a team below.")
    for grant in grants:
        kind, pid = grant["principal_type"], grant["principal_id"]
        label = (user_names if kind == "user" else team_names).get(pid, pid)
        row, action = st.columns([4, 1], vertical_alignment="center")
        row.write(f"{'👤' if kind == 'user' else '👥'} {label}")
        if action.button("Remove", key=f"rm-{kind}-{pid}", width="stretch"):
            try:
                api.remove_grant(workspace_id, doc["id"], kind, pid)
                st.rerun()
            except api.ApiError as e:
                st.error(f"Could not remove: {e.detail}")

    st.divider()
    st.markdown("**Share with**")
    shared = {(g["principal_type"], g["principal_id"]) for g in grants}
    # both kinds are always on offer, and they stack: adding one person to a
    # document a team already has shouldn't mean taking it away from the team
    picked_kind = st.radio(
        "Share with",
        ["A person", "A team"],
        horizontal=True,
        label_visibility="collapsed",
    )
    kind = "team" if picked_kind == "A team" else "user"
    pool = team_names if kind == "team" else user_names
    choices = {name: pid for pid, name in pool.items() if (kind, pid) not in shared}

    if not choices:
        st.caption(f"No more {'teams' if kind == 'team' else 'people'} left to add.")
        return
    pick = st.selectbox("Add", list(choices))
    if st.button("Share", width="stretch", type="primary"):
        try:
            api.add_grant(workspace_id, doc["id"], kind, choices[pick])
            st.rerun()
        except api.ApiError as e:
            st.error(f"Could not share: {e.detail}")


# --- toolbar -------------------------------------------------------------

# Dialogs, not popovers: a popover collapses on the rerun that follows its own
# submit, so validation errors and failures flash past unread - and the widget
# values inside it don't survive to be read either.


@st.dialog("New folder")
def _new_folder_dialog() -> None:
    with st.form("create_folder_form"):
        name = st.text_input("Folder name")
        parent_label = st.selectbox("Parent", list(folder_options))
        submitted = st.form_submit_button("Create")
    if not submitted:
        return
    if not name.strip():
        st.error("Folder name is required.")
        return
    try:
        api.create_folder(workspace_id, name.strip(), folder_options[parent_label])
    except api.ApiError as e:
        st.error(f"Could not create folder: {e.detail}")
        return
    st.session_state["_flash"] = f"Folder '{name.strip()}' created."
    st.rerun()


@st.dialog("Upload a document")
def _upload_dialog() -> None:
    with st.form("upload_form"):
        uploaded = st.file_uploader("Choose a file")
        folder_label = st.selectbox("Folder", list(folder_options))
        title = st.text_input("Title (optional - defaults to the file name)")
        submitted = st.form_submit_button("Upload")
    if not submitted:
        return
    if uploaded is None:
        st.error("Choose a file first.")
        return
    try:
        api.upload_document(
            workspace_id,
            uploaded.name,
            uploaded.getvalue(),
            uploaded.type,
            folder_options[folder_label],
            title or None,
        )
    except api.ApiError as e:
        st.error(f"Upload failed: {e.detail}")
        return
    st.session_state["_flash"] = f"Uploaded {uploaded.name}."
    st.rerun()


if can_edit:
    browse_col, new_folder_col, upload_col = st.columns(
        [3, 1, 1], vertical_alignment="bottom"
    )
else:
    (browse_col,) = st.columns(1)
browse_label = browse_col.selectbox("Browse", list(browse_options))

if can_edit:
    if new_folder_col.button("📂 New folder", width="stretch"):
        _new_folder_dialog()
    if upload_col.button("⬆️ Upload", width="stretch"):
        _upload_dialog()

# --- list + detail -------------------------------------------------------

try:
    documents = api.list_documents(
        workspace_id,
        browse_options[browse_label],
        every_folder=browse_label == ALL_DOCUMENTS,
    )
except api.ApiError as e:
    st.error(f"Could not load documents: {e.detail}")
    documents = []

list_col, detail_col = st.columns([2, 3], gap="large")

with list_col:
    if not documents:
        st.caption("No documents here yet - upload one with the button above.")
    selected_id = st.session_state.get("selected_document_id")
    for doc in documents:
        is_selected = doc["id"] == selected_id
        if st.button(
            f"📄 {doc['title']}",
            key=f"sel-{doc['id']}",
            width="stretch",
            type="primary" if is_selected else "secondary",
        ):
            st.session_state["selected_document_id"] = doc["id"]
            st.rerun()
        st.caption(
            f"`{doc['file_name']}` · {_fmt_size(doc['size_bytes'])}"
            f"{_visibility_badge(doc)}"
        )

selected = next(
    (d for d in documents if d["id"] == st.session_state.get("selected_document_id")),
    None,
)

with detail_col:
    if selected is None:
        st.info("Select a document to preview it.")
    else:
        st.markdown(f"#### {selected['title']}")
        st.caption(
            f"`{selected['file_name']}` · {selected.get('mime_type') or 'unknown type'}"
            f" · {_fmt_size(selected['size_bytes'])}"
            f" · uploaded {str(selected.get('created_at', ''))[:10]}"
            f"{_visibility_badge(selected)}"
        )
        searchable = selected.get("search_status") == "ready"
        if st.button(
            "💬 Ask this document",
            key=f"ask-{selected['id']}",
            type="primary",
            width="stretch",
            disabled=not searchable,
            help=(
                "Open Ask DocVault with this document selected."
                if searchable
                else "This document must finish indexing before it can answer questions."
            ),
        ):
            chat_state.open_document(workspace_id, str(selected["id"]))
            st.switch_page("views/chat.py")
        if not searchable:
            status = selected.get("search_status", "waiting_for_index")
            detail = (
                "Indexing failed."
                if status == "indexing_failed"
                else "Waiting for indexing."
            )
            st.caption(f"Search unavailable · {detail}")
        data = _get_bytes(selected)
        action_cols = st.columns(4)
        if data is not None:
            action_cols[0].download_button(
                "⬇️ Download",
                data=data,
                file_name=selected["file_name"],
                mime=selected.get("mime_type") or "application/octet-stream",
                width="stretch",
            )
        if can_edit:
            if action_cols[1].button("🔗 Share", width="stretch"):
                _share_dialog(selected)
            if action_cols[2].button("✏️ Rename / move", width="stretch"):
                _rename_dialog(selected)
            if action_cols[3].button("🗑️ Trash", width="stretch"):
                try:
                    api.trash_document(workspace_id, selected["id"])
                    st.session_state.pop("selected_document_id", None)
                    st.rerun()
                except api.ApiError as e:
                    st.error(f"Trash failed: {e.detail}")
        st.divider()
        if data is not None:
            _preview(selected, data)
