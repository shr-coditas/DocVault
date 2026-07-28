"""Sign-in that survives a browser reload.

``st.session_state`` is scoped to the browser<->server websocket, so F5 throws
the tokens away and the app falls back to the login page even though the
refresh token is good for another week. To keep the session we need something
in the browser itself.

The industry answer would be a backend-set ``HttpOnly; Secure; SameSite``
cookie holding the refresh token: unreadable from JavaScript, so XSS cannot
steal it. That is out of reach here — the browser never talks to the DocVault
API, it talks to Streamlit, which calls the API server-side, and Streamlit can
only *read* cookies natively (writing one needs a JS component, which by
definition cannot set HttpOnly).

So we use the next best thing, the classic server-side session: the browser
gets an opaque random id and nothing else, while the refresh token stays in
this process, keyed by that id. A stolen cookie is still a bearer credential,
but it is not the JWT, and ``forget()`` revokes it server-side immediately.

The store is a plain in-process dict, so it is single-replica and dies with the
server. Redis would be the production shape; see docs/decisions.md.
"""

import os
import secrets
import time
from typing import Any

import extra_streamlit_components as stx
import streamlit as st

COOKIE_NAME = "dv_sid"
_TTL_SECONDS = 7 * 24 * 60 * 60  # match the API's refresh-token lifetime
# Secure by default; the compose demo serves plain http and turns it off there.
_SECURE = os.getenv("COOKIE_SECURE", "true").lower() not in ("0", "false", "no")

_MANAGER = "_cookie_manager"
_PENDING_WRITE = "_cookie_write"
_PENDING_CLEAR = "_cookie_clear"


@st.cache_resource
def _store() -> dict[str, dict[str, Any]]:
    """sid -> {"refresh_token", "expires_at"}, shared across browser sessions."""
    return {}


def mount() -> None:
    """Render the cookie component. Call once at the top of every script run.

    CookieManager is a widget, so it can neither be cached across runs (it would
    hand back stale cookies) nor created twice in one run (same key, duplicate
    widget id). One instance per run, parked in session_state for the rest of
    the run to share.
    """
    st.session_state[_MANAGER] = stx.CookieManager(key="docvault-cookies")


def _cookies() -> stx.CookieManager:
    manager = st.session_state.get(_MANAGER)
    if manager is None:  # a view ran without the shell; mount on demand
        mount()
        manager = st.session_state[_MANAGER]
    return manager


def _prune(store: dict[str, dict[str, Any]]) -> None:
    now = time.time()
    for sid in [sid for sid, row in store.items() if row["expires_at"] <= now]:
        del store[sid]


def remember(refresh_token: str) -> None:
    """Bind the refresh token to this browser's session id.

    Called on login and after every rotation, so the stored token is never the
    stale half of a rotated pair. The id itself only changes at login, so the
    browser is only written to then — and the write is queued rather than done
    here, because callers usually st.rerun() straight afterwards and that would
    throw away the component render that carries it.
    """
    store = _store()
    _prune(store)
    sid = st.session_state.get("_sid")
    if not sid:
        sid = secrets.token_urlsafe(32)
        st.session_state["_sid"] = sid
        st.session_state[_PENDING_WRITE] = sid
    store[sid] = {"refresh_token": refresh_token, "expires_at": time.time() + _TTL_SECONDS}


def forget() -> None:
    """Drop the server-side entry and queue the cookie deletion.

    Safe to call when already signed out, and from a widget callback — where
    rendering the component directly would not be allowed.
    """
    sid = st.session_state.pop("_sid", None)
    if sid:
        _store().pop(sid, None)
    st.session_state.pop(_PENDING_WRITE, None)
    st.session_state[_PENDING_CLEAR] = True


def sync() -> None:
    """Flush queued cookie changes. Runs where components may render."""
    sid = st.session_state.pop(_PENDING_WRITE, None)
    if sid:
        _cookies().set(
            COOKIE_NAME,
            sid,
            max_age=_TTL_SECONDS,
            secure=_SECURE,
            same_site="lax",
            key="dv-set-sid",
        )
    if st.session_state.pop(_PENDING_CLEAR, False):
        try:
            _cookies().delete(COOKIE_NAME, key="dv-del-sid")
        except KeyError:  # never set in this browser
            pass


def restore() -> bool:
    """Rebuild ``st.session_state`` from the cookie. True if we are signed in.

    Imported lazily to avoid a circular import: api_client calls ``remember``
    after a rotation, and we call api_client to perform that rotation.
    """
    import api_client as api

    if st.session_state.get("access_token"):
        return True

    sid = _cookies().get(COOKIE_NAME)
    if not sid:
        return False

    store = _store()
    _prune(store)
    row = store.get(sid)
    if row is None:  # server restarted, or the session was revoked elsewhere
        return False

    st.session_state["_sid"] = sid
    st.session_state["refresh_token"] = row["refresh_token"]
    if not api.try_refresh():  # rotates, and re-persists via remember()
        forget()
        st.session_state.pop("refresh_token", None)
        return False
    return True
