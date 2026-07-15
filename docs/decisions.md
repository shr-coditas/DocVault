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

## 2026-07-13 — authentication

- **pwdlib + Argon2id** for passwords: passlib is unmaintained (breaks with
  bcrypt ≥ 4.1); Argon2id is the current OWASP recommendation.
- **JWT access token (15 min, stateless) + opaque rotating refresh token
  (7 days, stored hashed)**: stateless access keeps request auth off the DB;
  the DB-backed refresh token makes logout/revocation real. Rotation with a
  `family_id` means a replayed (stolen) refresh token revokes the whole chain.
- **Refresh secrets hashed with sha256, not Argon2**: they are high-entropy
  random values (no brute-force risk), and a deterministic hash allows direct
  lookup by `jti` without a slow verify loop. Client token format is
  `{jti}.{secret}`.
- **Same 401 for unknown email and wrong password**: no account enumeration.
- **RFC 9457 `application/problem+json`** for every error (domain errors,
  HTTPException, validation): one machine-readable shape, includes
  `request_id` for log correlation.
- **UUIDv7 primary keys**: globally unique like UUIDv4 but time-ordered, so
  b-tree indexes stay compact.

## 2026-07-13 — workspaces, RBAC, audit

- **Roles/permissions live in the database**, seeded by the migration from a
  single code catalog (`rbac/catalog.py`). Permission checks query the DB —
  not JWT claims — so a role change takes effect immediately instead of when
  the token expires.
- **One authorization choke point**: `require_permission("...")` dependency →
  `PermissionService.require()`. Every workspace-scoped endpoint declares its
  required permission; nothing checks roles ad hoc.
- **404 for non-members, 403 for members without permission**: outsiders must
  not learn that a workspace exists.
- **Audit rows have no foreign keys** on purpose: the trail must survive
  deletion of the workspace/user/resource it describes. Rows are staged in the
  same transaction as the mutation (trail can't disagree with the data) and
  carry the `request_id` for log correlation.
- **Last-owner guard**: a workspace can never lose its final owner via demote
  or removal (409).

## 2026-07-14 — storage, docker, folders

- **MinIO behind a `StorageService` interface**: the app speaks plain S3 API;
  moving from dev (MinIO) to prod (real S3) is a settings change. Objects are
  streamed in 1 MiB chunks — no whole-file buffering.
- **Multi-stage Dockerfile with uv**: dependency layer is cached separately
  from code, image runs as a non-root user, and the container applies
  migrations on startup (`alembic upgrade head && uvicorn`). Compose API port
  is 8080 so a local `uvicorn --reload` on 8000 can run side by side.
- **Folders as an adjacency list + recursive CTEs**: one parent pointer per
  row; the whole tree (with depth + path) comes back in a single recursive
  query, and move operations are O(1) row updates. Cycle prevention checks
  descendants before a move.
- **`UNIQUE ... NULLS NOT DISTINCT`** (PG 15+) on (workspace, parent, name):
  plain UNIQUE treats NULLs as distinct, which would have allowed duplicate
  root-folder names.
- **Folders reuse `document:*` permissions** instead of adding a `folder:*`
  set: folders are document structure, and one less catalog migration.

## 2026-07-15 — layered architecture

- **Switched from domain-first modules to layer-first packages**
  (`routers → controller → services → repository`), aligning with the team's
  standard project structure. Supersedes the modular-monolith layout of
  2026-07-10; still one deployable process. Motivation is consistency across
  the org's codebases, not a runtime concern.
- **Strict layer responsibilities**: routers define endpoints and permission
  guards only; controllers extract the request and map results to DTOs;
  services hold business logic and own the transaction (`commit`); repositories
  hold every SQLAlchemy query. Services no longer touch the session for data
  access, which makes them unit-testable with a fake repository.
- **DTOs live under `controller/dto/`** (request + response Pydantic models),
  matching the reference project. Class names are unchanged, so the OpenAPI
  schema and all API behavior stay identical — the 65-test suite passes as-is.
- **Cross-cutting pieces moved to conventional homes**: `config.py` at the app
  root, shared FastAPI dependencies in `dependencies.py`, error types +
  handlers under `exceptions/`, and security/logging/rbac-catalog under
  `utils/`. The RBAC seed moved to `scripts/`.
