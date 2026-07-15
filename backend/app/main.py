from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db.session import dispose_engine
from app.exceptions.handlers import register_exception_handlers
from app.middleware.context import RequestContextMiddleware
from app.routers.audit_router import router as audit_router
from app.routers.auth_router import router as auth_router
from app.routers.folder_router import router as folders_router
from app.routers.health_router import router as health_router
from app.routers.team_router import router as teams_router
from app.routers.workspace_router import router as workspaces_router
from app.utils.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(title="DocVault API", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(app)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health_router)

    api_v1 = APIRouter(prefix="/api/v1")
    api_v1.include_router(auth_router)
    api_v1.include_router(workspaces_router)
    api_v1.include_router(folders_router)
    api_v1.include_router(teams_router)
    api_v1.include_router(audit_router)
    app.include_router(api_v1)

    return app


app = create_app()
