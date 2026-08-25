"""Workspace picker - shown right after login (unless auto-entered)."""

from typing import Any

import streamlit as st

import api_client as api

st.title("Choose a workspace")

workspaces = st.session_state.get("workspaces") or []

if not workspaces:
    st.info("No workspaces yet - create your first one below.")


@st.dialog("Delete workspace")
def _confirm_delete_workspace(ws: dict[str, Any]) -> None:
    st.warning(
        f"**{ws['name']}** will be removed forever, along with every document, "
        "folder, team, and stored file inside it."
    )
    st.caption("Type the workspace name to confirm.")
    typed = st.text_input("Workspace name", key=f"confirm-name-{ws['id']}")
    if st.button(
        "Delete workspace", type="primary", disabled=typed.strip() != ws["name"]
    ):
        try:
            api.delete_workspace(ws["id"])
        except api.ApiError as e:
            st.error(f"Delete failed: {e.detail}")
            return
        # we may have been standing in it
        if st.session_state.get("current_workspace_id") == ws["id"]:
            for key in (
                "current_workspace_id",
                "selected_document_id",
                "doc_bytes_cache",
            ):
                st.session_state.pop(key, None)
        st.rerun()


for ws in workspaces:
    with st.container(border=True):
        info_col, open_col, delete_col = st.columns(
            [4, 1, 1], vertical_alignment="center"
        )
        info_col.markdown(f"**📁 {ws['name']}** · role: `{ws['my_role']}`")
        if ws.get("description"):
            info_col.caption(ws["description"])
        if open_col.button("Open", key=f"open-{ws['id']}", width="stretch"):
            st.session_state["current_workspace_id"] = ws["id"]
            st.session_state.pop("force_workspace_picker", None)
            st.rerun()
        # workspace:delete is owner-only
        if ws["my_role"] == "owner" and delete_col.button(
            "Delete…", key=f"del-ws-{ws['id']}", width="stretch"
        ):
            _confirm_delete_workspace(ws)

st.divider()
st.subheader("Create a workspace")
with st.form("create_workspace_form"):
    name = st.text_input("Name")
    description = st.text_area("Description", value="")
    submitted = st.form_submit_button("Create")
if submitted:
    if not name.strip():
        st.error("Name is required.")
    else:
        try:
            ws = api.create_workspace(name, description)
            # flash, not st.success: the rerun below would wipe an inline message
            st.session_state["_flash"] = f"Created workspace '{ws['name']}'."
            st.rerun()
        except api.ApiError as e:
            st.error(f"Could not create workspace: {e.detail}")
