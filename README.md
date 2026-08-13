# DocVault

![CI](https://github.com/shr-coditas/DocVault/actions/workflows/ci.yml/badge.svg)

Self-hosted document management API: workspaces, teams, role-based access control,
streaming file storage, and per-document sharing.

## Status

Early development. The document-management core is in place; retrieval and
AI features are next. `AGENTS.md` holds the working plan and current state.

## Roadmap

- [x] Foundation: FastAPI skeleton, async SQLAlchemy + Alembic, structured logging, request-id middleware, CI
- [x] Auth: registration/login, JWT access tokens + refresh token rotation, RFC 9457 errors
- [x] Workspaces, teams, and role-based access control; audit log
- [x] Folders and document upload/download (S3-compatible storage, streaming)
- [x] Document rename/move, trash & restore, permanent delete
- [x] Live workspace activity feed (WebSocket)
- [x] Sharing: per-document permissions and visibility levels
- [ ] Document ingestion: text extraction and chunking
- [ ] Search: semantic, full-text, and hybrid retrieval - filtered by document permissions
- [ ] Chat over your documents, with citations
- [ ] Hardening: rate limiting, pagination, seed data, API docs polish

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

- API: http://localhost:8000 - interactive docs at `/docs`
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

Layered architecture - each request flows through one layer at a time:

```
routers/  →  controller/  →  services/  →  repository/  →  db
(HTTP)       (DTO in/out)    (business)    (all queries)
```

```
backend/
  app/
    config.py        # settings
    dependencies.py  # DB session, current user, permission guards
    exceptions/      # AppError types + problem+json handlers
    utils/           # security (JWT/hashing), logging, rbac catalog
    middleware/      # request-id + access logging
    db/              # engine/session, declarative base, alembic
    models/          # SQLAlchemy tables
    routers/         # endpoint definitions only; delegate to controllers
    controller/      # one folder per domain, each with its own DTOs
      <domain>_controller/
        <domain>_controller.py   # thin: extract request, call service, map to DTO
        dto/                     # Pydantic request/response models
    services/        # business logic and workflows; own the transaction
    repository/      # all database access
    scripts/         # seed utilities
  tests/
```
