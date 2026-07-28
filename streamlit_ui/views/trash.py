"""Trash — restore documents or delete them permanently (with confirmation)."""

from typing import Any

import streamlit as st

import api_client as api

workspace_id = st.session_state["current_workspace_id"]

st.subheader("Trash")

# the whole page needs document:delete, which viewers don't have — say so
# rather than letting every call below fail with a red error
my_role = next(
    (
        w["my_role"]
        for w in st.session_state.get("workspaces", [])
        if w["id"] == workspace_id
    ),
    "viewer",
)
if my_role not in ("owner", "editor"):
    st.caption("Your role can't manage the trash.")
    st.stop()


@st.dialog("Delete permanently")
def _confirm_permanent_delete(doc: dict[str, Any]) -> None:
    st.warning(
        f"**{doc['title']}** (`{doc['file_name']}`) and its stored file "
        "will be removed forever."
    )
    confirmed = st.checkbox("I understand this cannot be undone")
    if st.button("Delete permanently", type="primary", disabled=not confirmed):
        try:
            api.delete_document_permanently(workspace_id, doc["id"])
            st.rerun()
        except api.ApiError as e:
            st.error(f"Permanent delete failed: {e.detail}")


try:
    trashed = api.list_trash(workspace_id)
except api.ApiError as e:
    st.error(f"Could not load trash: {e.detail}")
    trashed = []

if not trashed:
    st.caption("Trash is empty.")

for doc in trashed:
    with st.container(border=True):
        info_col, restore_col, delete_col = st.columns(
            [3, 1, 1], vertical_alignment="center"
        )
        info_col.markdown(f"**{doc['title']}**  \n`{doc['file_name']}`")
        if restore_col.button("Restore", key=f"restore-{doc['id']}", width="stretch"):
            try:
                api.restore_document(workspace_id, doc["id"])
                st.rerun()
            except api.ApiError as e:
                st.error(f"Restore failed: {e.detail}")
        if delete_col.button(
            "Delete…", key=f"perm-{doc['id']}", type="primary", width="stretch"
        ):
            _confirm_permanent_delete(doc)
