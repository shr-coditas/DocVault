from httpx import AsyncClient

from tests.test_auth import ME, REGISTER, _payload


async def test_conflict_is_rendered_as_problem_json(db_client: AsyncClient) -> None:
    await db_client.post(REGISTER, json=_payload())
    resp = await db_client.post(REGISTER, json=_payload())

    assert resp.status_code == 409
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert body["title"] == "Conflict"
    assert body["status"] == 409
    assert body["detail"] == "email already registered"
    assert body["instance"] == REGISTER
    assert body["request_id"]


async def test_validation_error_is_problem_json_with_errors(db_client: AsyncClient) -> None:
    resp = await db_client.post(REGISTER, json={"email": "not-an-email"})

    assert resp.status_code == 422
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert body["title"] == "Validation Error"
    assert isinstance(body["errors"], list)
    assert body["errors"]


async def test_unauthorized_problem_carries_www_authenticate(db_client: AsyncClient) -> None:
    resp = await db_client.get(ME)

    assert resp.status_code == 401
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.headers.get("www-authenticate") == "Bearer"
    assert resp.json()["title"] == "Unauthorized"
