# Decision log

Short notes on non-obvious choices. Newest at the bottom.

## 2026-07-10 — foundation

- **Modular monolith** (`app/modules/<domain>`): each domain owns its
  router/schemas/service/models. One deployable, clear seams, no premature
  service split.
- **Async SQLAlchemy 2 + asyncpg**: the API is IO-bound (database + file
  storage), so async end-to-end keeps one consistent execution model.
  `expire_on_commit=False` so returned objects stay usable after commit.
- **Naming conventions on metadata**: deterministic constraint names make
  Alembic migrations reviewable and reversible.
- **Pure-ASGI request middleware** instead of `BaseHTTPMiddleware`: does not
  buffer or break streaming responses, which will matter for file downloads.
- **uv + Ruff + mypy**: one fast toolchain; `uv.lock` is the dependency source
  of truth (no requirements.txt). Pre-commit runs ruff from the project venv so
  hook and CI versions cannot drift.
- **testcontainers over a mocked DB**: integration tests hit a real throwaway
  `postgres:17`, so readiness behavior is actually exercised.
- **PostgreSQL 17**: relational fit for documents/permissions, and native
  full-text search covers the search feature later in this phase without extra
  infrastructure.
