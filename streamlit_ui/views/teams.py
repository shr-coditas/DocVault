"""Teams - create teams and manage their membership.

A restricted document can be shared with a whole team as well as with named
people, so this page is the other half of the Share dialog on the Documents page.
"""

import streamlit as st

import api_client as api

workspace_id = st.session_state["current_workspace_id"]

# team:manage is owner-only; team:read is granted to every role
my_role = next(
    (
        w["my_role"]
        for w in st.session_state.get("workspaces", [])
        if w["id"] == workspace_id
    ),
    "viewer",
)
can_manage = my_role == "owner"

st.subheader("Teams")
if not can_manage:
    st.caption("You can see teams here; only a workspace owner can change them.")

try:
    teams = api.list_teams(workspace_id)
except api.ApiError as e:
    st.error(f"Could not load teams: {e.detail}")
    st.stop()

try:
    members = api.list_members(workspace_id)
except api.ApiError as e:
    st.error(f"Could not load workspace members: {e.detail}")
    members = []
member_emails = {m["user_id"]: m["email"] for m in members}

if not teams:
    st.info("No teams yet - create one below to share documents with a group.")

for team in teams:
    with st.expander(f"👥 {team['name']}"):
        try:
            team_members = api.list_team_members(workspace_id, team["id"])
        except api.ApiError as e:
            st.error(f"Could not load members: {e.detail}")
            continue

        if not team_members:
            st.caption("No members yet.")
        for tm in team_members:
            row, action = st.columns([4, 1], vertical_alignment="center")
            row.write(f"{tm['email']} - {tm['full_name']}")
            if can_manage and action.button(
                "Remove", key=f"tm-rm-{team['id']}-{tm['user_id']}", width="stretch"
            ):
                try:
                    api.remove_team_member(workspace_id, team["id"], tm["user_id"])
                    st.rerun()
                except api.ApiError as e:
                    st.error(f"Could not remove: {e.detail}")

        if not can_manage:
            continue

        st.divider()
        in_team = {tm["user_id"] for tm in team_members}
        addable = {
            email: uid for uid, email in member_emails.items() if uid not in in_team
        }
        if addable:
            pick = st.selectbox(
                "Add a workspace member",
                list(addable),
                key=f"tm-add-pick-{team['id']}",
            )
            if st.button("Add to team", key=f"tm-add-{team['id']}", width="stretch"):
                try:
                    api.add_team_member(workspace_id, team["id"], addable[pick])
                    st.rerun()
                except api.ApiError as e:
                    st.error(f"Could not add: {e.detail}")
        else:
            st.caption("Every workspace member is already on this team.")

        if st.button("🗑️ Delete team", key=f"tm-del-{team['id']}", width="stretch"):
            try:
                api.delete_team(workspace_id, team["id"])
                st.rerun()
            except api.ApiError as e:
                st.error(f"Could not delete team: {e.detail}")

if can_manage:
    st.subheader("Create a team")
    with st.form("create_team_form", clear_on_submit=True):
        team_name = st.text_input("Team name")
        if st.form_submit_button("Create team"):
            if not team_name.strip():
                st.error("Team name is required.")
            else:
                try:
                    api.create_team(workspace_id, team_name.strip())
                    st.rerun()
                except api.ApiError as e:
                    st.error(f"Could not create team: {e.detail}")
