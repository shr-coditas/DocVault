from collections.abc import AsyncIterator, Iterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

from app.db.base import Base
from app.db.session import get_db
from app.main import create_app

# import models so Base.metadata contains every table
from app.modules.auth import models as _auth_models  # noqa: F401
from app.modules.users import models as _users_models  # noqa: F401


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """API client with no database behind it (no Docker needed)."""
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """Throwaway Postgres for the whole test session (needs Docker)."""
    with PostgresContainer("postgres:17", driver="asyncpg") as pg:
        yield pg.get_connection_url()


@pytest.fixture
async def db_client(postgres_url: str) -> AsyncIterator[AsyncClient]:
    """API client wired to the throwaway Postgres with a fresh schema per test."""
    engine = create_async_engine(postgres_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
