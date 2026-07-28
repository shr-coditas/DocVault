import uuid
from datetime import UTC, datetime, timedelta

import jwt
from httpx import AsyncClient

from app.config import get_settings
from app.utils.security import create_access_token

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
ME = "/api/v1/auth/me"
REFRESH = "/api/v1/auth/refresh"
LOGOUT = "/api/v1/auth/logout"


def _payload(**overrides: str) -> dict[str, str]:
    payload = {
        "email": "shraddha@example.com",
        "password": "s3cret-password",
        "full_name": "Shraddha",
    }
    payload.update(overrides)
    return payload


async def test_register_creates_user(db_client: AsyncClient) -> None:
    resp = await db_client.post(REGISTER, json=_payload())

    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "shraddha@example.com"
    assert body["full_name"] == "Shraddha"
    assert body["is_active"] is True
    assert "id" in body
    assert "password" not in body
    assert "hashed_password" not in body


async def test_register_normalizes_email_and_rejects_duplicates(db_client: AsyncClient) -> None:
    first = await db_client.post(REGISTER, json=_payload())
    assert first.status_code == 201

    dup = await db_client.post(REGISTER, json=_payload(email="SHRADDHA@example.com"))
    assert dup.status_code == 409


async def test_register_rejects_short_password(db_client: AsyncClient) -> None:
    resp = await db_client.post(REGISTER, json=_payload(password="short"))
    assert resp.status_code == 422


async def _login(client: AsyncClient) -> dict[str, str]:
    await client.post(REGISTER, json=_payload())
    resp = await client.post(
        LOGIN, json={"email": "shraddha@example.com", "password": "s3cret-password"}
    )
    assert resp.status_code == 200
    body: dict[str, str] = resp.json()
    return body


async def test_login_returns_bearer_token_pair(db_client: AsyncClient) -> None:
    body = await _login(db_client)

    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["refresh_token"]


async def test_login_wrong_password_rejected(db_client: AsyncClient) -> None:
    await db_client.post(REGISTER, json=_payload())

    resp = await db_client.post(
        LOGIN, json={"email": "shraddha@example.com", "password": "wrong-password"}
    )
    assert resp.status_code == 401


async def test_login_unknown_email_rejected(db_client: AsyncClient) -> None:
    resp = await db_client.post(
        LOGIN, json={"email": "nobody@example.com", "password": "s3cret-password"}
    )
    assert resp.status_code == 401


async def test_me_returns_current_user(db_client: AsyncClient) -> None:
    body = await _login(db_client)

    resp = await db_client.get(ME, headers={"Authorization": f"Bearer {body['access_token']}"})

    assert resp.status_code == 200
    assert resp.json()["email"] == "shraddha@example.com"


async def test_me_requires_token(db_client: AsyncClient) -> None:
    resp = await db_client.get(ME)
    assert resp.status_code == 401


async def test_me_rejects_garbage_token(db_client: AsyncClient) -> None:
    resp = await db_client.get(ME, headers={"Authorization": "Bearer not-a-jwt"})
    assert resp.status_code == 401


async def test_me_rejects_expired_token(db_client: AsyncClient) -> None:
    expired = create_access_token(uuid.uuid4(), get_settings().jwt_secret, ttl_minutes=-1)

    resp = await db_client.get(ME, headers={"Authorization": f"Bearer {expired}"})
    assert resp.status_code == 401


async def test_me_rejects_wellformed_token_with_a_bad_subject(db_client: AsyncClient) -> None:
    """Correctly signed but carrying junk in `sub`: a bad credential (401),
    not a server fault (500)."""
    settings = get_settings()
    for claims in ({"sub": "not-a-uuid", "typ": "access"}, {"typ": "access"}):
        token = jwt.encode(
            {**claims, "exp": datetime.now(UTC) + timedelta(minutes=5)},
            settings.jwt_secret,
            algorithm="HS256",
        )
        resp = await db_client.get(ME, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401, resp.text


async def test_refresh_rotates_the_token(db_client: AsyncClient) -> None:
    body = await _login(db_client)

    resp = await db_client.post(REFRESH, json={"refresh_token": body["refresh_token"]})

    assert resp.status_code == 200
    rotated = resp.json()
    assert rotated["refresh_token"] != body["refresh_token"]
    assert rotated["access_token"]


async def test_refresh_reuse_revokes_the_whole_family(db_client: AsyncClient) -> None:
    body = await _login(db_client)
    original = body["refresh_token"]

    first = await db_client.post(REFRESH, json={"refresh_token": original})
    assert first.status_code == 200
    rotated = first.json()["refresh_token"]

    # replaying the already-rotated token = reuse attack
    reuse = await db_client.post(REFRESH, json={"refresh_token": original})
    assert reuse.status_code == 401

    # the legitimate descendant must be dead too (family revoked)
    descendant = await db_client.post(REFRESH, json={"refresh_token": rotated})
    assert descendant.status_code == 401


async def test_logout_revokes_the_refresh_token(db_client: AsyncClient) -> None:
    body = await _login(db_client)

    resp = await db_client.post(LOGOUT, json={"refresh_token": body["refresh_token"]})
    assert resp.status_code == 204

    after = await db_client.post(REFRESH, json={"refresh_token": body["refresh_token"]})
    assert after.status_code == 401


async def test_refresh_rejects_garbage_token(db_client: AsyncClient) -> None:
    resp = await db_client.post(REFRESH, json={"refresh_token": "not-a-real-token"})
    assert resp.status_code == 401
