"""DocVault demo UI — shell: auth gate → workspace gate → contextual nav.

Navigation is computed from session state with st.navigation, so the
sidebar only ever shows what makes sense *right now*:

- logged out            → the sign-in page, nothing else
- no workspace chosen   → the workspace picker (auto-enters a lone workspace)
- inside a workspace    → Documents · Members · Teams · Trash + workspace
                          switcher, with a 🔔 activity popover in the header
"""

import streamlit as st

import activity
import api_client as api
import session

st.set_page_config(page_title="DocVault", page_icon="\U0001f4c1", layout="wide")

st.session_state.setdefault("api_base_url", api.DEFAULT_BASE_URL)


def _logout() -> None:
    activity.stop_feed()
    refresh_token = st.session_state.get("refresh_token")
    if refresh_token:
        api.logout(refresh_token)  # revoke the family; local keys aren't enough
    session.forget()
    for key in (
        "_sid",
        "access_token",
        "refresh_token",
        "user",
        "current_workspace_id",
        "force_workspace_picker",
        "selected_document_id",
        "doc_bytes_cache",
    ):
        st.session_state.pop(key, None)


def _leave_workspace() -> None:
    activity.stop_feed()
    st.session_state["force_workspace_picker"] = True
    for key in ("current_workspace_id", "selected_document_id", "doc_bytes_cache"):
        st.session_state.pop(key, None)


# with st.sidebar.expander("⚙️ Connection"):
#     st.text_input(
#         "API base URL",
#         key="api_base_url",
#         help="Docker Compose runs the API on :8080; a local uvicorn usually runs on :8000.",
#     )

# --- gate 1: authentication ---------------------------------------------

# A reload wipes st.session_state, so rebuild the session from the persistence
# cookie before deciding anyone is signed out.
session.mount()
session.sync()  # apply cookie changes queued by the previous run
if not st.session_state.get("access_token") and not session.restore():
    # The component answers one run late, so its first "no cookies" means
    # "not yet". Hold the page — mounted, so the browser posts the cookies and
    # reruns us — rather than flashing the login screen at a signed-in user.
    if not st.session_state.get("_cookie_probe"):
        st.session_state["_cookie_probe"] = True
        st.info("Restoring your session…")
        st.stop()

if not st.session_state.get("access_token"):
    login_page = st.Page("views/login.py", title="Sign in", icon="🔑")
    st.navigation([login_page], position="hidden").run()
    st.stop()

# confirmations survive the st.rerun() that follows the action they describe;
# st.success() right before a rerun is wiped before anyone can read it
flash = st.session_state.pop("_flash", None)
if flash:
    st.toast(flash, icon="✅")

try:
    workspaces = api.list_workspaces()
    if "user" not in st.session_state:
        st.session_state["user"] = api.me()
except api.ApiError as e:
    if not st.session_state.get("access_token"):  # session expired for good
        st.rerun()
    st.error(f"Could not reach DocVault: {e.detail}")
    st.stop()
st.session_state["workspaces"] = workspaces

# --- gate 2: workspace selection ----------------------------------------

current_id = st.session_state.get("current_workspace_id")
if current_id and not any(w["id"] == current_id for w in workspaces):
    current_id = None  # deleted, or we were removed
    st.session_state.pop("current_workspace_id", None)

if (
    not current_id
    and len(workspaces) == 1
    and not st.session_state.get("force_workspace_picker")
):
    # lone workspace: skip the picker entirely
    current_id = workspaces[0]["id"]
    st.session_state["current_workspace_id"] = current_id

if not current_id:
    picker = st.Page("views/workspace_select.py", title="Choose workspace", icon="🗂️")
    with st.sidebar:
        st.divider()
        st.caption(f"Signed in as **{st.session_state['user'].get('email', '?')}**")
        st.button("Log out", on_click=_logout, width="stretch")
    st.navigation([picker], position="hidden").run()
    st.stop()

current = next(w for w in workspaces if w["id"] == current_id)

# --- shell chrome: switcher, user, header + 🔔 --------------------------

with st.sidebar:
    labels = {f"{w['name']} ({w['my_role']})": w["id"] for w in workspaces}
    label_list = list(labels)
    current_index = next(
        i for i, lbl in enumerate(label_list) if labels[lbl] == current_id
    )
    chosen = st.selectbox("Workspace", label_list, index=current_index)
    if labels[chosen] != current_id:
        activity.stop_feed()
        st.session_state["current_workspace_id"] = labels[chosen]
        for key in ("selected_document_id", "doc_bytes_cache"):
            st.session_state.pop(key, None)
        st.rerun()
    st.button("🗂️ All workspaces", on_click=_leave_workspace, width="stretch")
    st.divider()
    st.caption(f"Signed in as **{st.session_state['user'].get('email', '?')}**")
    st.button("Log out", on_click=_logout, width="stretch")

head_left, head_right = st.columns([0.93, 0.07])
head_left.markdown(f"### 📁 {current['name']}")
with head_right:
    activity.bell(current_id, current["my_role"])

pages = [
    st.Page("views/documents.py", title="Documents", icon="📄", default=True),
    st.Page("views/members.py", title="Members", icon="👥"),
    st.Page("views/teams.py", title="Teams", icon="🤝"),
    st.Page("views/trash.py", title="Trash", icon="🗑️"),
]
st.navigation(pages).run()
