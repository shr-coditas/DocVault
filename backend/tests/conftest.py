import os
from collections.abc import AsyncIterator, Iterator

os.environ.setdefault("CHUNK_MAX_TOKENS", "400")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("EMBEDDING_BATCH_SIZE", "32")
os.environ.setdefault("EMBEDDING_THREADS", "1")
os.environ.setdefault(
    "EMBEDDING_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)
os.environ.setdefault("ANSWER_MAX_SOURCES", "8")

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.minio import MinioContainer
from testcontainers.postgres import PostgresContainer

from app import models  # noqa: F401  # registers every table on Base.metadata
from app.config import Settings
from app.db.base import Base
from app.db.session import get_db
from app.main import create_app
from app.services.storage_service import StorageService
from tests.helpers import POSTGRES_IMAGE, create_schema, sync_rbac_catalog


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """API client with no database behind it (no Docker needed)."""
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """Throwaway Postgres for the whole test session (needs Docker)."""
    with PostgresContainer(POSTGRES_IMAGE, driver="asyncpg") as pg:
        yield pg.get_connection_url()


@pytest.fixture(scope="session")
def storage_settings() -> Iterator[Settings]:
    """Throwaway MinIO for the whole session (needs Docker).

    Lives here rather than in a test module so the suite starts one container,
    not one per module that happens to need object storage.
    """
    with MinioContainer() as minio:
        cfg = minio.get_config()
        yield Settings(
            s3_endpoint_url=f"http://{cfg['endpoint']}",
            s3_access_key=cfg["access_key"],
            s3_secret_key=cfg["secret_key"],
            s3_bucket="test-bucket",
        )


@pytest.fixture
async def test_storage(storage_settings: Settings) -> StorageService:
    """StorageService pointed at the throwaway MinIO (also used for assertions)."""
    storage = StorageService(storage_settings)
    await storage.ensure_bucket()
    return storage


@pytest.fixture
async def db_client(postgres_url: str) -> AsyncIterator[AsyncClient]:
    """API client wired to the throwaway Postgres with a fresh schema per test."""
    engine = create_async_engine(postgres_url)
    await create_schema(engine)

    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        await sync_rbac_catalog(session)  # prod gets this from the migration seed

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
