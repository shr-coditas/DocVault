"""Members - list workspace members, add by email."""

import streamlit as st

import api_client as api

workspace_id = st.session_state["current_workspace_id"]

# workspace:manage_members is owner-only
my_role = next(
    (
        w["my_role"]
        for w in st.session_state.get("workspaces", [])
        if w["id"] == workspace_id
    ),
    "viewer",
)
can_manage = my_role == "owner"

st.subheader("Members")

try:
    members = api.list_members(workspace_id)
    st.dataframe(
        [
            {"email": m["email"], "full_name": m["full_name"], "role": m["role"]}
            for m in members
        ],
        hide_index=True,
        width="stretch",
    )
except api.ApiError as e:
    st.error(f"Could not load members: {e.detail}")

if not can_manage:
    st.caption("Only a workspace owner can add or remove members.")
    st.stop()

st.subheader("Add a member")
with st.form("add_member_form", clear_on_submit=True):
    st.caption("They must already have a DocVault account.")
    member_email = st.text_input("Email")
    role = st.selectbox("Role", ["viewer", "editor", "owner"])
    add_submitted = st.form_submit_button("Add member")
if add_submitted:
    try:
        api.add_member(workspace_id, member_email, role)
        # flash, not st.success: the rerun below would wipe an inline message
        st.session_state["_flash"] = f"Added {member_email} as {role}."
        st.rerun()
    except api.ApiError as e:
        st.error(f"Could not add member: {e.detail}")
