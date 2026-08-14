# Decision log

Short notes on non-obvious choices. Newest at the bottom.

## 2026-07-10 - foundation

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

## 2026-07-13 - authentication

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

## 2026-07-13 - workspaces, RBAC, audit

- **Roles/permissions live in the database**, seeded by the migration from a
  single code catalog (`rbac/catalog.py`). Permission checks query the DB -
  not JWT claims - so a role change takes effect immediately instead of when
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

## 2026-07-14 - storage, docker, folders

- **MinIO behind a `StorageService` interface**: the app speaks plain S3 API;
  moving from dev (MinIO) to prod (real S3) is a settings change. Objects are
  streamed in 1 MiB chunks - no whole-file buffering.
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

## 2026-07-15 - layered architecture

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
  schema and all API behavior stay identical - the 65-test suite passes as-is.
- **Cross-cutting pieces moved to conventional homes**: `config.py` at the app
  root, shared FastAPI dependencies in `dependencies.py`, error types +
  handlers under `exceptions/`, and security/logging/rbac-catalog under
  `utils/`. The RBAC seed moved to `scripts/`.

## 2026-07-15 - documents (streaming upload/download)

- **Metadata in Postgres, bytes in object storage**: the `documents` row is the
  source of truth for listing/access; the file lives in MinIO under a stable
  `storage_key` (`ws_{id}/doc_{id}/v1_{name}`). The row is written in the same
  use case as the object; on an upload failure nothing is committed.
- **Download streams, upload is size-capped**: download wraps
  `StorageService.stream()` in a `StreamingResponse` (1 MiB chunks, never
  buffered - the pure-ASGI middleware guarantees this). Upload reads the file in
  chunks, enforcing `max_upload_size_bytes` (25 MB default, env-overridable) and
  computing sha256 as it goes, then stores in one call. True streaming *upload*
  (multipart) is deferred until large files actually need it.
- **MIME via stdlib `mimetypes`** (guess from filename): zero dependencies and
  no libmagic/native binary to install. Content-sniffing (python-magic) can
  harden this later if needed.
- **`documents.folder_id` is `ON DELETE CASCADE`**: deleting a folder deletes
  its documents (the UI will warn first). Known follow-up: cascade removes the
  rows but not the MinIO objects - object cleanup lands with the trash/delete
  feature.
- **Storage injected via a dependency** (`get_storage_service`): tests override
  it to point `StorageService` at a throwaway MinIO (testcontainers), so uploads
  and downloads are exercised against real object storage, not a mock.

## 2026-07-16 - document lifecycle & activity feed

- **Trash-first delete**: `DELETE /documents/{id}` soft-deletes (`deleted_at`),
  hiding the document from list/get/download (404) while keeping row + object
  restorable. `DELETE /{id}/permanent` actually removes both. Trash listing and
  restore sit behind `document:delete` - the trash is for people who can act
  on it.
- **Row before object on permanent delete**: commit the row deletion first,
  then remove the MinIO object. A failed object delete leaves only a harmless
  orphan; the reverse order could leave a live row pointing at a missing file.
- **Folder delete now cleans up objects**: Postgres cascades can't reach into
  MinIO, so `FolderService.delete` collects the subtree's `storage_key`s before
  the cascade and deletes the objects post-commit - closes the orphan gap noted
  on 2026-07-15.
- **Activity feed publishes strictly post-commit**: `AuditService.record()`
  stages a copy of each audit event on the session; SQLAlchemy `after_commit`
  publishes it to an in-process broadcaster, `after_rollback` discards it. The
  feed can never report a change that didn't happen. One audit source feeds
  both history (DB) and the live stream.
- **WebSocket auth via `?token=` query param**: browsers cannot set headers on
  WebSocket connections. Bad token or non-member closes with 1008 (policy
  violation) - same information-hiding intent as the REST 404 rule.
- **In-process broadcaster, bounded queues**: per-workspace `asyncio.Queue`s
  (maxsize 100, drop-on-full) - a slow consumer loses events instead of
  blocking the app. Single-process by design; a Redis-backed broadcaster is the
  drop-in replacement if the API ever scales horizontally.

## 2026-07-23 - per-document visibility & sharing

- **Three visibility levels** (`documents.visibility`, varchar + CHECK, default
  `workspace`): `private` (owner + explicitly granted users), `team` (members of
  a granted team), `workspace` (any member - the pre-sharing default, so every
  existing document is unaffected). Layered *on top of* the workspace-role
  `document:read` gate: the role says you may read documents in general, the
  visibility says whether you may see *this* one.
- **Grants are binary**: a `document_grants` row (`document_id`, `principal_type`
  user|team, `principal_id`) means "this principal can see the document". It
  carries no per-grant role - what you may *do* is still governed by your
  workspace role. Per-grant roles can be added later without a breaking change.
- **Flat, per-document - no folder inheritance**: a document's visibility never
  derives from its folder, so moving a document between folders can never
  silently change who can see it. Consistent with the project's "permissions are
  always an explicit DB check, never inferred" stance.
- **Workspace owners bypass visibility**: an owner can already delete the
  workspace, manage members, and read the audit log, so hiding documents from
  them is theater. They see every document regardless of visibility/grants.
- **Invisible → 404, not 403**: a document you may not see is hidden exactly like
  a workspace you don't belong to (same information-hiding intent).
- **`document:share` finally wired up**: the permission has existed in the RBAC
  catalog since the workspaces/RBAC work (owner + editor); the visibility and
  grant-management endpoints are its first consumers.
- **Workspace delete now cleans up storage too**: `WorkspaceService.delete`
  collects every document's `storage_key` before the cascade and removes the
  objects post-commit - the same pattern `FolderService.delete` already used.
  Closes the last known place where deleting rows orphaned MinIO objects; the
  service now takes a `StorageService` like the folder/document services do.

## 2026-07-27 - grant symmetry, team membership, session persistence

- **A grant is a grant, whoever it names.** The team-grant check was gated behind
  `visibility == 'team'` while the user-grant check was not, so the two kinds of
  grant behaved differently. Removed the gate in both the single-document path
  (`DocumentService._ensure_can_access`) and the list SQL
  (`DocumentRepository._accessible_condition`) - they must stay in lockstep.
  Consequence: `private` and `team` now mean the same thing, *restricted, opened
  by grants*. Both values are kept (existing rows carry them, and no migration is
  worth it) but the UI presents a single "Restricted" state.

  The bug this fixes was invisible from the API alone: because the Share dialog
  only offered *user* grants at `private` and *team* grants at `team`, adding one
  extra person to a team-shared document forced a switch to `private`, which
  silently made every team grant inert while still listing them under "Shared
  with". Both kinds are now offered at either visibility.

- **Creating a team puts you in it.** `TeamService.create` now inserts a
  `TeamMember` for the actor, mirroring `WorkspaceService.create` seeding the
  creator as owner. An empty team reads as broken, and every comparable product
  (Google Groups, Slack, GitHub teams) does the same. The creator can remove
  themselves, for the admin who set the team up for other people. Note their
  document access never came from team membership - it comes from the workspace
  owner override.

- **Grants don't outlive their principal.** `document_grants.principal_id` is
  polymorphic, so it has no foreign key and nothing cascades. Removing a workspace
  member or deleting a team now explicitly deletes their grants in the same
  transaction; otherwise re-adding a removed user silently restored access to
  documents shared with them before.

- **`GET /documents?scope=all`.** An omitted `folder_id` compiles to
  `folder_id IS NULL`, i.e. the workspace *root* - but the UI's "All documents"
  sent exactly that, so anything filed in a folder looked like a failed upload.
  Added an explicit `scope` parameter rather than overloading the absence of
  `folder_id`; the default preserves the old behaviour.

- **Sign-in survives a reload, via a server-side session.** Tokens lived only in
  `st.session_state`, which Streamlit scopes to the browser↔server websocket, so
  F5 always landed on the login page despite a 7-day refresh token.

  The textbook fix - a backend-set `HttpOnly; Secure; SameSite` cookie holding the
  refresh token - is unreachable here: the browser never talks to the API, it
  talks to Streamlit, which calls the API server-side, and Streamlit can only
  *read* cookies natively (writing one needs a JS component, which by definition
  cannot set HttpOnly). So `streamlit_ui/session.py` uses the classic server-side
  session instead: the browser holds an opaque random id, the refresh token stays
  in the Streamlit process keyed by that id, and logout revokes both locally and
  via `POST /auth/logout` (which was never called before).

  Known limits, accepted for a demo UI: the store is an in-process dict, so it
  dies with the server and does not survive multiple replicas - Redis is the
  production shape. The cookie is still a bearer credential, just not the JWT.

- **Cookie writes are queued, not immediate.** `CookieManager` is a widget: it
  cannot be cached across runs (stale cookies), cannot be created twice in one run
  (duplicate widget id), and its render is discarded if the script calls
  `st.rerun()` straight after - which is exactly what login does. So `remember()`
  and `forget()` only queue, and `session.sync()` at the top of the next run
  flushes. Same reason the New folder and Upload popovers became dialogs: a
  popover collapses on its own submit's rerun, so validation errors flashed past
  unread and folder creation failed silently.

- **The Share dialog offers two choices, not three.** `private` and `team` are
  now indistinguishable, so presenting both was offering the same behaviour
  twice. The picker is Restricted / Everyone in the workspace; a document already
  stored as `team` reads as Restricted and keeps that value (so it doesn't look
  like a pending change), and anything newly restricted is written as `private`.
  The API still accepts all three - no migration.

  "Grant" was database vocabulary in a user-facing dialog; it now says Share.

- **A dialog's variable body lives in one `st.empty()`.** Streamlit does not
  clear elements from a dialog's previous render, so an early `return` on a
  changed branch left the old branch's widgets on screen - switching to "Everyone
  in the workspace" still listed who the document was shared with. `st.empty()`
  replaces its whole subtree, so the branches are genuinely exclusive.

## 2026-07-27 (later) - one tracker, collapsed visibility, upload allowlist

- **`AGENTS.md` is the single tracker.** Project state, the plan, and open issues
  had accumulated across five overlapping documents - a kickoff guide, a
  day-by-day plan file, a risk register, a remediation plan, and a learning
  roadmap - with two conflicting day-numbering schemes in which "Day 7" named
  different work. They are merged into `AGENTS.md` at the repo root and the rest
  deleted. `AGENTS.md` rather than `CLAUDE.md` because it is the cross-vendor
  convention, so the file can be handed to any model without translation.

  Planning is now by named slice, not day number. `docs/decisions.md` keeps
  recording decisions - that is history, a different thing from state. The rest
  of `docs/` is local study reference and is gitignored, which means `AGENTS.md`
  has to stand alone.

- **`visibility` collapsed to `restricted | workspace`** (migration
  `c4d8e1f60a25`), superseding the "no migration is worth it" note above. Once
  grants became symmetric, `private` and `team` were two names for one
  behaviour - a trap for the next reader, and one the AI phase would have built
  more code on top of. Existing rows fold into `restricted`; the CHECK constraint
  is dropped before the rewrite and recreated after, so neither the old nor the
  new form is ever violated mid-migration. The downgrade maps back to `private`;
  the original `team` rows are not recoverable, which is acceptable because they
  behaved identically.

- **Uploads are restricted to `.pdf`, `.txt`, `.md`, `.csv`, `.docx`.** Every
  accepted type must be one the ingestion pipeline can extract text from -
  widening the list widens what has to be parsed, chunked, and embedded. The
  check is on the extension and runs before the body is read, so an unsupported
  file is refused without being buffered. It is a scope control, not a security
  boundary: a file can be renamed, so content sniffing and scanning remain on
  the roadmap.

- **Check-constraint names are bare suffixes.** The metadata naming convention is
  `ck_%(table_name)s_%(constraint_name)s`, so a constraint declared as
  `ck_documents_visibility` renders as `ck_documents_ck_documents_visibility`.
  Worse, migration `a1c7f3e29b84` wrapped the grants constraint in `op.f()`,
  which opts out of the convention, while the model did not - so Alembic and
  `Base.metadata.create_all()` produced *different* names for the same
  constraint, meaning the test schema and the production schema disagreed and
  autogenerate would have proposed a spurious drop/recreate.

  Both models and both migrations now pass the bare suffix. Found by actually
  running an upgrade/downgrade round-trip against a disposable database rather
  than trusting that the migration "looked right" - the reason that step is in
  the ritual.

## 2026-07-29 - document intelligence: pgvector, local embeddings, manual indexing

- **pgvector, not Pinecone**, reaffirming the earlier note against an explicit
  proposal to use Pinecone. The deciding argument is specific to this codebase,
  not a general preference: `DocumentRepository._accessible_condition` is already
  a SQLAlchemy `ColumnElement`, so it composes straight into the `WHERE` of the
  vector search and authorization happens *inside* the index scan. Permission
  changes then take effect on the next query with no re-indexing at all.

  Pinecone would need the allowed-document set materialised in Postgres on every
  query and passed as a metadata filter - which has a size ceiling - or ACL
  attributes mirrored into vector metadata, which is a third copy of the
  visibility rule in a system we cannot transactionally update. It also loses
  transactional chunk upserts and `ON DELETE CASCADE` cleanup. Three new failure
  modes to buy nothing this workload needs.

- **fastembed with `BAAI/bge-small-en-v1.5`, 384 dimensions**, local ONNX. No API
  key, so indexing and the whole test suite run offline, and CI needs no secret.
  Chosen over sentence-transformers because it does not pull PyTorch; it still
  brings `onnxruntime`, `numpy` and `tokenizers`, so the image grows - measure
  before deploying. `embedding_dimensions` must match `vector(384)`, and
  `FastEmbedEmbedder` probes the model at construction and refuses to start on a
  mismatch rather than writing vectors the column would reject.

- **Direct parsers over LangChain**: `pypdf`, `python-docx`, and the stdlib for
  `.txt`/`.md`/`.csv`, plus a splitter we own. Five formats, three small typed
  dependencies, no large transitive tree and no mypy overrides for untyped
  community loaders. A test asserts the handler set equals
  `allowed_upload_extensions`, so a widened allowlist cannot silently admit a
  file nothing can read.

- **Chunking is measured in character offsets, not tokens.** Every chunk records
  `char_start`/`char_end` such that `page.text[char_start:char_end] == content`
  exactly, which is what makes a citation traceable back to its source range. A
  token-based target would make that arithmetic lossy, and a splitter that
  reassembles pieces cannot report offsets honestly - so the implementation works
  purely on `(start, end)` pairs and never joins strings. Chunks never span
  pages, so `page_number` is always meaningful.

- **CSV rows are rendered as `header: value` pairs**, not dumped raw. Once a large
  CSV is chunked, a chunk from the middle carries values whose columns are
  hundreds of lines away; pairing each cell with its header keeps every row
  self-describing.

- **`documents.indexed` is the indexing script's only selector**, with
  `indexed_at`, `index_error` and `index_attempts` alongside it. A bare boolean
  cannot distinguish "never attempted" from "failed permanently", so the script
  would retry an unparseable PDF on every run forever; `index_attempts` against
  `max_index_attempts` is the cap, and `--retry-failed` overrides it.

- **`index_generation` is deliberately redundant today.** Retrieval requires
  `document_chunks.index_generation == documents.index_generation`, so chunks from
  a superseded run are invisible rather than merely deleted. The persist phase is
  a single transaction, so partial chunk sets cannot currently exist and the
  column earns nothing right now. It is here so a future step-wise workflow can
  write chunks incrementally across process restarts and flip visibility
  atomically with one `UPDATE`. Recorded because it looks like dead weight and
  should not be removed, and equally should not be assumed load-bearing yet.

- **`IndexingService` takes a session factory, not a session** - a deliberate
  break from the "services take a session" rule. Indexing runs in three
  transactions: claim (`SELECT ... FOR UPDATE SKIP LOCKED`, bump attempts,
  commit), work (download, parse, chunk, embed - **no** transaction held), persist
  (replace chunks, flip the flag, audit, commit). The middle phase is seconds of
  CPU on a large PDF; holding a pooled connection across it would exhaust the
  pool under any concurrency, and a single injected session cannot express that.
  Each phase is a public method, so a future workflow wraps them as activities
  without changing this service.

- **Processing is a manually-run script, and shaped so orchestration is additive.**
  `uv --directory backend run python -m app.scripts.index_documents`. Sequential
  on purpose - concurrency is what the workflow engine will provide, and a
  half-version here would be thrown away. `SKIP LOCKED` costs nothing for a
  single runner and is exactly the primitive a competing-consumer pool needs, so
  the claim semantics will not have to be redesigned. This is also the repo's
  first runnable script; it follows `alembic/env.py` - `asyncio.run` entrypoint,
  `dispose_engine()` in a `finally`.

- **Document categories are deferred.** No `documents.category`. Metadata
  filtering will therefore use columns that already exist - `folder_id`,
  `mime_type`, `created_at` - supplied by the caller rather than inferred from the
  question. An inferred filter can silently hide a document the user asked for,
  and a classifier mistake should never look like missing data.

- **The pgvector image swap requires recreating the `db_data` volume.** The
  extension ships as a shared library, so `postgres:17` cannot load it; compose
  and the test container both move to `pgvector/pgvector:pg17`. `docker compose
  down -v` destroys local data, so it stays a manual documented step and is never
  scripted.

- **Test schema creation is centralised in `tests.helpers.create_schema`.** Three
  fixtures built schema independently and each needed `CREATE EXTENSION` before
  `create_all`, or `create_all` fails on the `vector` column. The helper also
  imports `app.models` and asserts the metadata is non-empty: without that import
  `Base.metadata` is empty, `create_all` is a *successful no-op*, and the failure
  surfaces much later as "relation does not exist". The MinIO fixture moved to
  `conftest.py` at the same time - it was duplicated in two modules, so the suite
  was starting two containers.

## 2026-07-30 - retrieval: permission-filtered vector search

- **The ACL predicate goes *inside* the vector query, not after it.**
  `DocumentChunkRepository.search_semantic` composes
  `DocumentRepository._accessible_condition` into the same `SELECT` that does the
  cosine ranking. Filtering after a top-k would leak nothing, but it starves: a
  narrowly-permissioned user gets three results where a broadly-permissioned one
  gets twenty, and it gets worse as the corpus grows. Co-locating vectors and
  metadata in one table is what makes this possible in one statement, and is the
  whole reason pgvector was chosen over a dedicated vector database.

- **`hnsw.iterative_scan = strict_order` by default**, set with `SET LOCAL` in the
  same transaction as the query. Plain HNSW returns k candidates and *then* lets
  the ACL predicate thin them; iterative scan keeps walking the graph instead.
  `SET` cannot be parameterised, so the value is interpolated - guarded by an
  allowlist (`off`/`strict_order`/`relaxed_order`), which is a membership test
  rather than an escape. `SET LOCAL` is transaction-scoped, so it cannot leak
  into another request sharing the pooled connection. pgvector 0.8.5 in
  `pgvector/pgvector:pg17`; the GUC is settable before the extension library
  loads, so ordering is not a concern.

- **The owner override moved to `services/document_access.py`.** It was private to
  `DocumentService`, and retrieval needs the identical answer: workspace owner →
  no filter, everyone else → (user id, their teams *in this workspace*).
  `DocumentService._access_filter` now delegates. Written twice it would drift,
  and the failure mode of drift is a document surfacing in search that its own
  listing hides.

- **Four hard filters, one soft one.** Tenant (asserted on both the chunk's
  denormalised `workspace_id` and the document's authoritative one),
  `deleted_at IS NULL`, `chunk.index_generation = document.index_generation`, and
  the visibility predicate - all hard. `min_score` is soft and is a floor, not a
  topic pre-filter: every accessible chunk in the workspace is scored, and the
  threshold exists because vector search always returns its k nearest neighbours
  however far away they are. `tests/test_search.py` asserts both halves of that,
  including that dropping the floor brings the off-topic document back.

- **`ORDER BY` the bare distance operator, ascending.** `score DESC` where
  `score = 1 - distance` is arithmetically identical and cannot be served by the
  HNSW index. The similarity conversion therefore happens once, in the
  repository, so no caller has to remember which direction the number runs.

- **Top-K and the score threshold are per-request query parameters** defaulting to
  settings, with a hard `MAX_LIMIT = 50` published as the router's own bound - so
  an over-large request is a 422 rather than a silent clamp - and re-enforced in
  the service for non-HTTP callers. Both are echoed in the response: they are
  clamped server-side, and comparing two tuning runs requires knowing what was
  actually in force.

- **`?mode=semantic` is a one-member `Literal`.** It documents that a mode axis
  exists and makes `?mode=hybrid` a 422 today instead of silently running a
  semantic search. Lexical, hybrid (RRF) and reranked modes join it later without
  changing the URL shape.

- **Search records no audit row.** It is a `GET`; auditing it would put an INSERT
  and a commit in front of every query and flood the activity feed with
  keystrokes. The service logs hit counts and parameters - never the query text
  and never chunk content. If search auditing is wanted it should be its own
  append path, not a side effect of retrieval.

- **A `document_id` naming an inaccessible document returns an empty result, not
  404.** Every filter still applies, so the narrow search cannot become a way to
  confirm a document exists.

- **Retrieval lands before any LLM is wired in**, deliberately: at this stage the
  only consumer of a retrieval bug is the already-authenticated caller, not a
  third-party provider.

## 2026-07-30 (later) - the query pipeline: guardrails, intent, and the retrieval gate

- **Classification runs before retrieval, not after it.** `POST
  /workspaces/{id}/query` runs guardrails → intent → and only then, for
  `document_question`, `SearchService`. Three of the four intents never embed and
  never touch the index. The saving on a single greeting is small - one local ONNX
  inference and one indexed scan - but it scales with abuse (a caller hammering
  injections is refused at regex cost), it is the same branch that will skip the
  *billed* LLM call from step 7, and a blocked query never gets near retrieved
  document text.

- **The gate decides whether to search, never what may be seen.** Retrieval, when
  it happens, goes through `SearchService` unchanged, with the same access
  filter. `test_query.py` asserts a restricted document stays invisible to a
  viewer whose question *did* retrieve - the two controls are independent, and
  the intent layer must never be mistaken for an authorization layer.

- **`retrieval_performed` is stated, not inferred from empty `hits`.** Searching
  and finding nothing is a different event from never searching, and only one of
  them cost anything. Inferring it would also make the gate's own tests pass for a
  pipeline that had quietly stopped retrieving altogether.

- **The default intent is `document_question`, deliberately.** The two errors are
  not symmetric: a false `out_of_scope` refuses a real question and leaves the
  user with no recourse, while a false `document_question` costs one embedding and
  one scan before the score threshold declines. So the classifier only leaves the
  default on an explicit signal. A rule-based classifier fundamentally cannot
  separate "question about my documents" from "general knowledge question" -
  *the same sentence is either one depending on what the corpus holds* - so it
  recognises only what is decidable from the text: greetings, named non-document
  tasks, and instruction-override attempts.

- **Chitchat patterns are anchored; injection patterns are not.** "Hello, what
  does the leave policy say?" must reach retrieval, so only a query that is
  *nothing but* a pleasantry is chitchat. An injection attempt, by contrast, is
  one wherever it appears in the string, and injection is checked first - dressing
  an attack up as a greeting does not change it.

- **The cross-tenant rule requires a leading imperative.** `show/list/give/fetch`
  before the "other users' documents" noun phrase. Without that, "what does the
  all users' data policy say?" - an ordinary HR question - matched and was
  refused. This was caught by a failing test during the slice, and both the
  attack cases and the false-positive cases are now pinned down.

- **Guardrails decide validity; the classifier decides meaning.** Length,
  emptiness and hidden characters are guardrails. What the query is *about* is
  classification. Keeping the line sharp is what stops the two quietly duplicating
  each other's rules - the failure mode being an attack that one blocks, the other
  reclassifies, and neither logs.

- **Zero-width and control characters are rejected**, since no human types them
  and they are a documented way to hide instructions in text that renders
  innocently. Both the check and its tests spell the characters as codepoints,
  never as literals: an invisible character in source cannot be read or reviewed.

- **The chain is fail-fast and hand-rolled.** Stopping at the first failure saves
  nothing measurable today - every check is string work - but the shape is what
  matters, because a Guardrails AI validator may call a model and must not see
  input that a length check already refused. Hand-rolled for the same reason as
  the embedder: the library version should have to beat a working baseline.
  `GuardrailService(checks=[...])` is the seam the adapter plugs into.

- **Refusals are fixed sentences and never echo the query.** An error message that
  repeats user input is a reflection gadget, and a refusal has no reason to quote
  what it refused. The logs likewise carry intent, decision, reason code and
  query length - never the text, which for a blocked query is the most likely to
  be hostile and the most likely to be shipped onward to a log aggregator.

- **`POST`, not `GET`.** The question travels in the body, keeping it out of proxy
  logs and browser history - the open issue `/search` still has. A blocked or
  declined query is 200 with a decision, not a 4xx: it is a normal outcome of the
  pipeline, and step 7 turns each into a spoken answer rather than a status code.

- **`/search` was left alone.** A search box that refuses "hello" would be a
  regression, and it has no LLM to protect. The gate belongs in front of
  generation, which is what `/query` becomes.

## 2026-08-09 - canonical structure and staged hybrid indexes

- **The canonical extraction product is a hierarchical AST.** Parser-specific
  payloads are retained as compressed object-storage artifacts, while stable
  logical paths, normalized text, typed nodes, source spans and confidence are
  the database contract. This keeps citations traceable without coupling
  chunking to Docling, python-docx, or markdown-it object models.

- **Migration 0010 is rewritten instead of repaired forward.** Revisions 0009
  and 0010 have not reached a persistent/shared environment, so 0010 establishes
  index runs, nodes, contextual chunks, generated FTS and the active embedding
  profile directly. Once shared, the migration becomes immutable.

- **Docling lives in a separate worker image.** API requests need query
  embeddings and reranking, not layout parsing. The worker target additionally
  bakes Docling and its layout/table models so runtime indexing remains offline
  and the later Temporal worker can reuse the exact same service phases.

- **Dense embeddings are contextualized but generation content is not.** Each
  chunk stores source `content`, contextual `embedding_text`, and FTS-oriented
  `lexical_text` separately. Document title and section breadcrumb improve
  retrieval, while generated answers and citations remain grounded in normalized
  source content.

- **A generation is staged behind a durable lease and activated atomically.**
  Extraction, nodes, embeddings and chunks can fail without touching the active
  generation. Activation validates counts, hashes, profile and vector dimensions
  before switching the document pointer in one transaction; the immediately
  previous generation remains available for citation continuity.

- **Hybrid retrieval is two authorization-scoped candidate queries.** Dense and
  lexical SQL both compose the same workspace/deletion/generation/profile/ACL
  predicates before ranking. RRF, deduplication, cross-encoder reranking,
  per-document diversity and structural expansion happen only over authorized
  rows, and expansion re-applies the authoritative joins.

## 2026-08-10 - one explicit semantic score floor

- **`semantic_min_score` is the only score-floor contract.** The deprecated
  `min_score` alias is removed from search and query inputs as well as search
  responses. Hybrid retrieval has semantic, lexical, fusion and rerank scores,
  so an unqualified `min_score` is ambiguous; the remaining name states that the
  floor applies only to the dense candidate branch. Obsolete inputs are rejected
  instead of being silently ignored.

## 2026-08-13 - persistent conversations before an agentic query graph

- **Product conversation history lands before LangGraph.** Persistent turns,
  ownership, scope, provenance and contextual follow-ups are useful without an
  agent loop and give a later graph trusted state to run over. LangGraph
  checkpoints will be internal execution state, never the canonical user-visible
  record.
- **Conversations are private to their creator and hard-deleted on request.** A
  workspace role grants the capability to use document chat, not the right to
  inspect another member's questions. Cross-workspace and cross-creator lookups
  return 404. Conversation-owned rows cascade from the conversation; historical
  document and chunk references do not cascade from document tables.
- **Conversation scope is immutable.** Workspace scope means all documents the
  creator can access at each turn. A selected scope preserves its original set of
  one to ten document identities and frozen names. Changing scope starts another
  conversation instead of changing what earlier turns meant.
- **Selected scope degrades explicitly when access changes.** Each turn computes
  the accessible intersection of the original selection. An empty intersection
  refuses without retrieval; a non-empty subset may answer only from that subset
  and must prominently report the missing documents. It never silently rewrites
  the stored selection and never replaces the retrieval ACL.
- **Historical answers are authorized at read time.** Every retrieved source is
  recorded, including the subset supplied to generation. If access to any source
  that could have influenced generated prose is lost, the assistant answer is
  projected as redacted while the user's question remains. Re-granting access
  restores it because redaction is computed rather than destructive. One batched
  repository query composes the existing SQL accessibility predicate for a page;
  there is no per-source Python authorization copy.
- **Source provenance survives reindexing and deletion.** A source records the
  original chunk id plus `document_id`, `index_generation`, and `logical_key`,
  together with frozen display metadata. The original chunk is preferred; a
  current-generation logical-key match is only a relocated navigation target,
  not a claim that the historical answer used the new chunk.
- **Turn durability uses an idempotency key and the existing lease idiom.** A
  user and assistant placeholder share a stable turn id. The pending assistant
  carries a unique lease token and expiry; takeover replaces the token, and
  finalization is fenced on the token so a superseded execution cannot overwrite
  the winner. User input commits before retrieval or model calls, while sources
  and the final assistant state commit atomically afterward.
- **History helps interpretation, never factual grounding.** Guardrails and
  intent classification inspect the current raw message before any history is
  loaded. A separate contextual-query resolver may turn a safe follow-up into a
  standalone query using bounded, access-safe complete turns. Generation still
  receives only fresh permission-filtered sources; prior assistant prose is not
  evidence.
- **The conversation API becomes the canonical chat surface.** The stateless
  `/query` endpoint stays temporarily as a migration bridge while the new API and
  Streamlit client reach parity. It can then be removed unless non-persistent
  questioning becomes an explicit product requirement; the internal
  `QueryService` remains independent of that HTTP decision.
- **Conversation observability excludes content.** Audit events are limited to
  conversation creation and deletion. Structured logs contain identifiers,
  enumerated reason codes, counts, timings, model names and token usage, never
  raw messages, standalone rewrites, titles, retrieved text or prompts.

## 2026-08-14 - structured follow-up resolution and canonical persistent UI

- **Context resolution uses provider-native structured output.** The resolver
  owns a small validated schema and a separate model client; answer generation
  call accounting remains unchanged. Invalid output, provider errors, and the
  bounded timeout all degrade to the raw safe query rather than failing a turn.
- **Streamlit stores identifiers, not canonical chat content.** Conversations
  and messages reload from the API on each full page run. Session state carries
  only the active conversation and an uncertain submission's idempotency key.
  A two-second fragment polls a pending bubble and performs a full rerun only
  when the terminal server state is available.
- **Resolver quality is measured separately from retrieval and generation.** A
  checked-in synthetic corpus covers reference rewriting, topic shifts,
  ambiguity, raw-query security gates, and access-filtered history. Metrics
  report strict labeled exactness only where a query will actually be searched,
  ambiguity recall, unnecessary clarifications, resolver routing, provider
  failures, scope-widening terms, and latency. Provider failures are never hidden
  inside an aggregate quality score.

## 2026-08-14 (later) - one bounded query graph, with product history outside it

- **LangGraph orchestrates one query turn; it is not the authorization or
  conversation system.** Deterministic input gates remain before history and
  retrieval, and every retrieval/rewrite/decomposition branch calls the existing
  permission-filtered `SearchService` with actor, workspace and immutable scope
  supplied through trusted runtime context. The model never chooses those values.
- **Product conversations and graph checkpoints have different identities.** The
  database conversation/message/source ledger remains canonical. A checkpoint
  thread is keyed by durable `turn_id`, not `conversation_id`, so it can resume a
  leased turn without becoming a second multi-turn memory or silently carrying
  scope between questions. Completed, deleted and expired turns clean up their
  checkpoint threads.
- **The first graph is a parity migration, not an agent loop.** Existing linear
  `QueryService` behavior is expressed as nodes and conditional edges and tested
  beside the old path before corrective retrieval, decomposition or generation
  retries are enabled. The service stays the facade for both stateless `/query`
  and canonical persistent conversations.
- **Parallel retrieval requires independent database sessions.** SQLAlchemy's
  request-scoped `AsyncSession` cannot service concurrent branch queries. LCEL
  fan-out is permitted only through a factory that gives each branch its own
  bounded read session; otherwise the bounded subqueries run sequentially.
- **Output is validated before it is delivered.** Security/citation/grounding
  checks may regenerate once, but a rejected draft is neither persisted nor sent
  over SSE. The first event stream therefore carries sanitized progress and the
  validated terminal answer; low-latency token streaming waits for an explicit
  incremental-safety design rather than exposing bytes a later guardrail cannot
  retract.
- **The Postgres saver owns its internal schema lifecycle.** Its pinned package
  migrations run via an explicit autocommit setup/deployment command, never at API
  startup or inside an Alembic transaction. Checkpoint connections are separate
  from application asyncpg sessions, and checkpoint state stores bounded control
  data/source references rather than credentials, prompts or raw document text.
