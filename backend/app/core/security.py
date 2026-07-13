import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from pwdlib import PasswordHash

JWT_ALGORITHM = "HS256"

_password_hash = PasswordHash.recommended()  # argon2id


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return _password_hash.verify(password, hashed)


def create_access_token(user_id: uuid.UUID, secret: str, ttl_minutes: int) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "jti": uuid.uuid4().hex,
        "iat": now,
        "exp": now + timedelta(minutes=ttl_minutes),
        "typ": "access",
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def generate_refresh_secret() -> str:
    return secrets.token_urlsafe(32)


def hash_refresh_secret(secret_value: str) -> str:
    # sha256, not argon2: the secret is already high-entropy random, and a fast
    # deterministic hash lets us look tokens up without a per-row verify loop
    return hashlib.sha256(secret_value.encode()).hexdigest()


def decode_access_token(token: str, secret: str) -> dict[str, Any]:
    """Raises jwt.PyJWTError on invalid/expired tokens."""
    payload: dict[str, Any] = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    if payload.get("typ") != "access":
        raise jwt.InvalidTokenError("not an access token")
    return payload
