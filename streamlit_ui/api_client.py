"""Thin HTTP wrapper around the DocVault API for the demo UI.

Every call attaches the bearer token from st.session_state. On a 401 the
client silently rotates the refresh token once and retries the request;
only if that also fails are the stored tokens cleared, which sends the
app back to the login screen (access tokens last 15 minutes).

Rotations are handed to `session.remember` so the persisted session keeps
pointing at the live token rather than the revoked one.
"""

import os
from contextlib import suppress
from typing import Any

import httpx
import session
import streamlit as st

# In Docker the API is reachable at http://api:8000 (service name on the compose
# network); running the UI locally it's http://localhost:8080. API_URL overrides.
DEFAULT_BASE_URL = os.getenv("API_URL", "http://localhost:8080/api/v1")


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


def base_url() -> str:
    return st.session_state.get("api_base_url", DEFAULT_BASE_URL)


def _headers() -> dict[str, str]:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _detail_of(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        return body.get("detail") or body.get("title") or resp.text
    except Exception:
        return resp.text


def try_refresh() -> bool:
    """Rotate the refresh token and store the new pair. Also used at page load
    to rebuild a session from the persistence cookie (see session.restore)."""
    refresh_token = st.session_state.get("refresh_token")
    if not refresh_token:
        return False
    try:
        resp = httpx.post(
            f"{base_url()}/auth/refresh",
            json={"refresh_token": refresh_token},
            timeout=10.0,
        )
    except httpx.HTTPError:
        return False
    if resp.status_code >= 400:
        return False
    tokens = resp.json()
    if "access_token" not in tokens or "refresh_token" not in tokens:
        return False
    st.session_state["access_token"] = tokens["access_token"]
    st.session_state["refresh_token"] = tokens["refresh_token"]
    # the old token is now revoked; persist the new one or the next reload
    # would present a dead token and trip the API's reuse detection
    session.remember(tokens["refresh_token"])
    return True


def _request(
    method: str, path: str, *, timeout: float = 10.0, **kwargs: Any
) -> httpx.Response:
    resp = httpx.request(
        method, f"{base_url()}{path}", headers=_headers(), timeout=timeout, **kwargs
    )
    if resp.status_code == 401 and try_refresh():
        resp = httpx.request(
            method, f"{base_url()}{path}", headers=_headers(), timeout=timeout, **kwargs
        )
    if resp.status_code == 401:
        # refresh failed too - drop the session so the shell shows login again
        st.session_state.pop("access_token", None)
        st.session_state.pop("refresh_token", None)
        session.forget()
    if resp.status_code >= 400:
        raise ApiError(resp.status_code, _detail_of(resp))
    return resp


# --- auth ---------------------------------------------------------------


def register(email: str, password: str, full_name: str) -> dict[str, Any]:
    resp = httpx.post(
        f"{base_url()}/auth/register",
        json={"email": email, "password": password, "full_name": full_name},
        timeout=10.0,
    )
    if resp.status_code >= 400:
        raise ApiError(resp.status_code, _detail_of(resp))
    return resp.json()


def login(email: str, password: str) -> dict[str, Any]:
    resp = httpx.post(
        f"{base_url()}/auth/login",
        json={"email": email, "password": password},
        timeout=10.0,
    )
    if resp.status_code >= 400:
        raise ApiError(resp.status_code, _detail_of(resp))
    return resp.json()


def logout(refresh_token: str) -> None:
    """Revoke the whole refresh-token family server-side. Best effort: signing
    out locally must succeed even if the API is unreachable."""
    with suppress(httpx.HTTPError):
        httpx.post(
            f"{base_url()}/auth/logout",
            json={"refresh_token": refresh_token},
            timeout=5.0,
        )


def me() -> dict[str, Any]:
    return _request("GET", "/auth/me").json()


# --- workspaces ---------------------------------------------------------


def list_workspaces() -> list[dict[str, Any]]:
    return _request("GET", "/workspaces").json()


def create_workspace(name: str, description: str = "") -> dict[str, Any]:
    return _request(
        "POST", "/workspaces", json={"name": name, "description": description}
    ).json()


def delete_workspace(workspace_id: str) -> None:
    _request("DELETE", f"/workspaces/{workspace_id}")


def list_members(workspace_id: str) -> list[dict[str, Any]]:
    return _request("GET", f"/workspaces/{workspace_id}/members").json()


def add_member(workspace_id: str, email: str, role: str) -> dict[str, Any]:
    return _request(
        "POST",
        f"/workspaces/{workspace_id}/members",
        json={"email": email, "role": role},
    ).json()


# --- teams --------------------------------------------------------------


def list_teams(workspace_id: str) -> list[dict[str, Any]]:
    return _request("GET", f"/workspaces/{workspace_id}/teams").json()


def create_team(workspace_id: str, name: str) -> dict[str, Any]:
    return _request(
        "POST", f"/workspaces/{workspace_id}/teams", json={"name": name}
    ).json()


def delete_team(workspace_id: str, team_id: str) -> None:
    _request("DELETE", f"/workspaces/{workspace_id}/teams/{team_id}")


def list_team_members(workspace_id: str, team_id: str) -> list[dict[str, Any]]:
    return _request("GET", f"/workspaces/{workspace_id}/teams/{team_id}/members").json()


def add_team_member(workspace_id: str, team_id: str, user_id: str) -> None:
    _request(
        "POST",
        f"/workspaces/{workspace_id}/teams/{team_id}/members",
        json={"user_id": user_id},
    )


def remove_team_member(workspace_id: str, team_id: str, user_id: str) -> None:
    _request("DELETE", f"/workspaces/{workspace_id}/teams/{team_id}/members/{user_id}")


# --- folders ------------------------------------------------------------


def folder_tree(workspace_id: str) -> list[dict[str, Any]]:
    return _request("GET", f"/workspaces/{workspace_id}/folders/tree").json()


def create_folder(
    workspace_id: str, name: str, parent_id: str | None
) -> dict[str, Any]:
    return _request(
        "POST",
        f"/workspaces/{workspace_id}/folders",
        json={"name": name, "parent_id": parent_id},
    ).json()


# --- documents ----------------------------------------------------------


def upload_document(
    workspace_id: str,
    file_name: str,
    file_bytes: bytes,
    content_type: str,
    folder_id: str | None,
    title: str | None,
) -> dict[str, Any]:
    data = {}
    if folder_id:
        data["folder_id"] = folder_id
    if title:
        data["title"] = title
    return _request(
        "POST",
        f"/workspaces/{workspace_id}/documents/upload",
        files={
            "file": (file_name, file_bytes, content_type or "application/octet-stream")
        },
        data=data,
        timeout=60,
    ).json()


def list_documents(
    workspace_id: str, folder_id: str | None, *, every_folder: bool = False
) -> list[dict[str, Any]]:
    """``every_folder`` spans the workspace; otherwise this lists exactly one
    folder, and folder_id=None means the workspace root (not "everywhere")."""
    params: dict[str, Any] = {"scope": "all"} if every_folder else {}
    if folder_id and not every_folder:
        params["folder_id"] = folder_id
    return _request(
        "GET", f"/workspaces/{workspace_id}/documents", params=params
    ).json()


def download_document(workspace_id: str, document_id: str) -> bytes:
    return _request(
        "GET",
        f"/workspaces/{workspace_id}/documents/{document_id}/download",
        timeout=60,
    ).content


def update_document(
    workspace_id: str,
    document_id: str,
    title: str | None,
    folder_id: str | None,
    moved: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if title:
        body["title"] = title
    if moved:
        body["folder_id"] = folder_id
    return _request(
        "PATCH", f"/workspaces/{workspace_id}/documents/{document_id}", json=body
    ).json()


def trash_document(workspace_id: str, document_id: str) -> None:
    _request("DELETE", f"/workspaces/{workspace_id}/documents/{document_id}")


def restore_document(workspace_id: str, document_id: str) -> dict[str, Any]:
    return _request(
        "POST", f"/workspaces/{workspace_id}/documents/{document_id}/restore"
    ).json()


def list_trash(workspace_id: str) -> list[dict[str, Any]]:
    return _request("GET", f"/workspaces/{workspace_id}/documents/trash").json()


def delete_document_permanently(workspace_id: str, document_id: str) -> None:
    _request("DELETE", f"/workspaces/{workspace_id}/documents/{document_id}/permanent")


# --- document questions -------------------------------------------------


def query_documents(
    workspace_id: str,
    question: str,
    *,
    document_ids: list[str] | None = None,
    retrieval_mode: str = "hybrid",
    limit: int = 10,
) -> dict[str, Any]:
    """Ask one independent question within an explicit document scope.

    ``None`` means every accessible document in the workspace. A concrete list
    is converted to the API's singular/plural filters so both branches exercise
    the public contract rather than inventing a Streamlit-only shape.
    """
    body: dict[str, Any] = {
        "query": question,
        "retrieval_mode": retrieval_mode,
        "limit": limit,
    }
    if document_ids is not None:
        unique_ids = list(dict.fromkeys(document_ids))
        if not unique_ids:
            raise ValueError("at least one document must be selected")
        if len(unique_ids) > 10:
            raise ValueError("at most 10 documents may be selected")
        if len(unique_ids) == 1:
            body["document_id"] = unique_ids[0]
        else:
            body["document_ids"] = unique_ids
    try:
        return _request(
            "POST",
            f"/workspaces/{workspace_id}/query",
            json=body,
            timeout=90.0,
        ).json()
    except httpx.TimeoutException as exc:
        raise ApiError(504, "The answer took too long. Please try again.") from exc
    except httpx.HTTPError as exc:
        raise ApiError(
            503, "DocVault is temporarily unavailable. Please try again."
        ) from exc


# --- conversations ------------------------------------------------------


def _conversation_request(
    method: str, path: str, *, timeout: float = 15.0, **kwargs: Any
) -> httpx.Response:
    try:
        return _request(method, path, timeout=timeout, **kwargs)
    except httpx.TimeoutException as exc:
        raise ApiError(504, "The conversation request timed out.") from exc
    except httpx.HTTPError as exc:
        raise ApiError(503, "DocVault is temporarily unavailable.") from exc


def create_conversation(
    workspace_id: str,
    scope_mode: str,
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"scope_mode": scope_mode}
    if document_ids:
        body["document_ids"] = list(dict.fromkeys(document_ids))
    return _conversation_request(
        "POST",
        f"/workspaces/{workspace_id}/conversations",
        json=body,
    ).json()


def list_conversations(
    workspace_id: str,
    *,
    limit: int = 20,
    cursor: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    return _conversation_request(
        "GET",
        f"/workspaces/{workspace_id}/conversations",
        params=params,
    ).json()


def get_conversation(workspace_id: str, conversation_id: str) -> dict[str, Any]:
    return _conversation_request(
        "GET",
        f"/workspaces/{workspace_id}/conversations/{conversation_id}",
    ).json()


def delete_conversation(workspace_id: str, conversation_id: str) -> None:
    _conversation_request(
        "DELETE",
        f"/workspaces/{workspace_id}/conversations/{conversation_id}",
    )


def list_conversation_messages(
    workspace_id: str,
    conversation_id: str,
    *,
    page_limit: int = 100,
    max_pages: int = 10,
) -> list[dict[str, Any]]:
    """Load canonical messages in sequence order with a defensive page bound."""
    messages: list[dict[str, Any]] = []
    after_sequence = 0
    for _ in range(max_pages):
        page = _conversation_request(
            "GET",
            f"/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
            params={"after_sequence": after_sequence, "limit": page_limit},
        ).json()
        items = page.get("items") or []
        messages.extend(items)
        next_sequence = page.get("next_after_sequence")
        if next_sequence is None or not items:
            break
        after_sequence = int(next_sequence)
    return messages


def submit_conversation_message(
    workspace_id: str,
    conversation_id: str,
    content: str,
) -> dict[str, Any]:
    """Submit one synchronous conversation exchange."""
    return _conversation_request(
        "POST",
        f"/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
        json={"content": content},
        timeout=90.0,
    ).json()


# --- sharing ------------------------------------------------------------


def set_visibility(
    workspace_id: str, document_id: str, visibility: str
) -> dict[str, Any]:
    return _request(
        "PUT",
        f"/workspaces/{workspace_id}/documents/{document_id}/visibility",
        json={"visibility": visibility},
    ).json()


def list_grants(workspace_id: str, document_id: str) -> list[dict[str, Any]]:
    return _request(
        "GET", f"/workspaces/{workspace_id}/documents/{document_id}/grants"
    ).json()


def add_grant(
    workspace_id: str, document_id: str, principal_type: str, principal_id: str
) -> dict[str, Any]:
    return _request(
        "POST",
        f"/workspaces/{workspace_id}/documents/{document_id}/grants",
        json={"principal_type": principal_type, "principal_id": principal_id},
    ).json()


def remove_grant(
    workspace_id: str, document_id: str, principal_type: str, principal_id: str
) -> None:
    _request(
        "DELETE",
        f"/workspaces/{workspace_id}/documents/{document_id}"
        f"/grants/{principal_type}/{principal_id}",
    )


# --- audit --------------------------------------------------------------


def get_audit_log(workspace_id: str, limit: int = 20) -> list[dict[str, Any]]:
    return _request(
        "GET", f"/workspaces/{workspace_id}/audit", params={"limit": limit}
    ).json()
