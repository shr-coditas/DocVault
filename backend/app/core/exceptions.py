from collections.abc import Mapping
from http import HTTPStatus
from typing import Any, ClassVar

import structlog
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


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


def _problem(
    request: Request,
    status_code: int,
    title: str,
    detail: str | None = None,
    extra: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": "about:blank",
        "title": title,
        "status": status_code,
        "instance": request.url.path,
        "request_id": structlog.contextvars.get_contextvars().get("request_id"),
    }
    if detail is not None:
        body["detail"] = detail
    if extra:
        body.update(extra)
    return JSONResponse(
        body, status_code=status_code, media_type="application/problem+json", headers=headers
    )


async def _app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    return _problem(request, exc.status_code, exc.title, exc.detail, headers=exc.headers)


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    detail = exc.detail if isinstance(exc.detail, str) else None
    return _problem(
        request, exc.status_code, HTTPStatus(exc.status_code).phrase, detail, headers=exc.headers
    )


async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return _problem(
        request,
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "Validation Error",
        "request validation failed",
        extra={"errors": jsonable_encoder(exc.errors())},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
