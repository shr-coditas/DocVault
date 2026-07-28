"""Domain errors rendered as RFC 9457 problem+json responses.

Raise these from services/controllers; `app.exceptions.handlers` turns them
into responses.
"""

from collections.abc import Mapping
from typing import ClassVar

from fastapi import status


class AppError(Exception):
    """Domain error rendered as an RFC 9457 problem response."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    title: str = "Internal Server Error"
    headers: ClassVar[Mapping[str, str] | None] = None

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail
        super().__init__(detail)


class UnauthorizedError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    title = "Unauthorized"
    headers: ClassVar[Mapping[str, str] | None] = {"WWW-Authenticate": "Bearer"}


class ForbiddenError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    title = "Forbidden"


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    title = "Not Found"


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    title = "Conflict"


class PayloadTooLargeError(AppError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    title = "Content Too Large"


class UnsupportedMediaTypeError(AppError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    title = "Unsupported Media Type"


__all__ = [
    "AppError",
    "ConflictError",
    "ForbiddenError",
    "NotFoundError",
    "PayloadTooLargeError",
    "UnauthorizedError",
    "UnsupportedMediaTypeError",
]
