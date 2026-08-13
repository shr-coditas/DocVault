"""Sign-in / register - the only page visible while logged out."""

import streamlit as st

import api_client as api
import session

st.title("📁 DocVault")
st.caption("Sign in to your document workspace.")


def _start_session(email: str, password: str) -> None:
    """Log in and hand the refresh token to the persistence cookie, so a
    browser reload lands back in the app instead of on this page."""
    tokens = api.login(email, password)
    st.session_state["access_token"] = tokens["access_token"]
    st.session_state["refresh_token"] = tokens["refresh_token"]
    session.remember(tokens["refresh_token"])
    st.session_state["user"] = api.me()


tab_login, tab_register = st.tabs(["Login", "Register"])

with tab_login:
    with st.form("login_form"):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        try:
            _start_session(email, password)
            st.rerun()
        except api.ApiError as e:
            st.error(f"Login failed: {e.detail}")

with tab_register:
    with st.form("register_form"):
        reg_full_name = st.text_input("Full name", key="reg_full_name")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input(
            "Password",
            type="password",
            key="reg_password",
            help="At least 8 characters",
        )
        reg_submitted = st.form_submit_button("Register")
    if reg_submitted:
        try:
            api.register(reg_email, reg_password, reg_full_name)
        except api.ApiError as e:
            st.error(f"Registration failed: {e.detail}")
        else:
            # straight in - making a brand-new user retype their credentials
            # in another tab is a pointless first impression
            try:
                _start_session(reg_email, reg_password)
                st.session_state["_flash"] = "Welcome to DocVault!"
                st.rerun()
            except api.ApiError as e:
                st.warning(
                    f"Account created, but signing you in failed: {e.detail}. "
                    "Use the Login tab."
                )
