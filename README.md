# DocVault

![CI](https://github.com/shr-coditas/DocVault/actions/workflows/ci.yml/badge.svg)

Self-hosted document management API: workspaces, teams, role-based access control,
versioned file storage, sharing, and full-text search.

## Status

Early development — working through the roadmap below.

## Roadmap

- [x] Foundation: FastAPI skeleton, async SQLAlchemy + Alembic, structured logging, request-id middleware, CI
- [x] Auth: registration/login, JWT access tokens + refresh token rotation, RFC 9457 errors
- [x] Workspaces, teams, and role-based access control; audit log
- [ ] Folders and document upload/download (S3-compatible storage); trash & restore
- [ ] Document versioning, tags, metadata
- [ ] Sharing: per-document permissions and visibility levels
- [ ] Full-text search, filters, sorting, pagination
- [ ] Hardening: rate limiting, file validation, seed data, API docs polish

## Quickstart

Requires Docker and [uv](https://docs.astral.sh/uv/).

**Everything in containers (one command):**

```bash
docker compose up -d --build
```

API at http://localhost:8080 (docs at `/docs`), MinIO console at http://localhost:9001.
Migrations run automatically on startup.

**Dev loop (DB + MinIO in Docker, API on the host with hot reload):**

```bash
docker compose up -d db minio createbuckets
cd backend
uv sync
uv run uvicorn app.main:app --reload
```

- API: http://localhost:8000 — interactive docs at `/docs`
- Health: `GET /health` (liveness), `GET /health/ready` (checks DB connectivity)

Configuration is environment-driven with local defaults that match
`docker-compose.yml`; see `backend/.env.example` for overrides.

## Tests

```bash
cd backend
uv run pytest                       # full suite (Docker needed: spins a throwaway Postgres)
uv run pytest -m "not integration"  # unit tests only, no Docker
```

## Stack

FastAPI · Pydantic v2 · SQLAlchemy 2 (async) · PostgreSQL 17 · Alembic · structlog ·
uv · Ruff · mypy · pytest

## Project layout

```
backend/
  app/
    core/        # settings, logging setup
    middleware/  # request-id + access logging
    db/          # engine/session, declarative base, alembic
    modules/     # one package per domain (router / schemas / service / models)
  tests/
```
