from httpx import AsyncClient, Response

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
WORKSPACES = "/api/v1/workspaces"

PASSWORD = "s3cret-password"


async def signup(client: AsyncClient, email: str, full_name: str = "Test User") -> dict[str, str]:
    """Register + login a user; returns Authorization headers."""
    resp = await client.post(
        REGISTER, json={"email": email, "password": PASSWORD, "full_name": full_name}
    )
    assert resp.status_code == 201, resp.text
    resp = await client.post(LOGIN, json={"email": email, "password": PASSWORD})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def create_workspace(client: AsyncClient, headers: dict[str, str], name: str = "Acme") -> str:
    resp = await client.post(WORKSPACES, json={"name": name}, headers=headers)
    assert resp.status_code == 201, resp.text
    workspace_id: str = resp.json()["id"]
    return workspace_id


async def add_member(
    client: AsyncClient, headers: dict[str, str], workspace_id: str, email: str, role: str
) -> Response:
    return await client.post(
        f"{WORKSPACES}/{workspace_id}/members",
        json={"email": email, "role": role},
        headers=headers,
    )
