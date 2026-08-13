# DocVault demo UI

A Streamlit front end for demoing the DocVault API.

## Run

The whole stack, UI included:

```powershell
docker compose up -d
```

The UI is on <http://localhost:8501> and the API on <http://localhost:8080>.

To run just the UI against an API elsewhere, point `API_URL` at it:

```powershell
$env:API_URL = "http://localhost:8000/api/v1"; uv run --directory streamlit_ui streamlit run app.py
```

## Flow

- **Sign in / register** → **workspace picker** (skipped automatically when
  you belong to exactly one workspace) → workspace pages. Registering signs
  you straight in.
- Sidebar is contextual to the current workspace: switcher on top, then
  **Documents · Members · Teams · Trash**. "All workspaces" returns to the
  picker. Controls your role can't use are hidden rather than shown failing.
- **🔔 popover** in the header shows the live WebSocket activity feed on
  every page (owners can also load recent history from the audit log).
- **Documents**: browse all documents or one folder, upload from the toolbar,
  click a document to open its detail panel - inline preview for PDF, images,
  markdown, CSV, JSON, and plain text; download for everything else.
  Rename/move opens a dialog; permanent delete (in Trash) requires an
  explicit confirmation.
- **Share**: `Restricted` documents are opened up by grants naming either a
  person or a team - both, on the same document, in any combination. Teams are
  resolved live, so adding someone to a team grants them everything already
  shared with it. `Everyone in the workspace` needs no grants.

## Sessions

A reload keeps you signed in. The browser holds an opaque id in the `dv_sid`
cookie; the refresh token itself stays in the Streamlit process, keyed by that
id (`session.py`). That store is an in-process dict, so restarting the
container signs everyone out and it will not work across replicas - see
`docs/decisions.md` for why it isn't the HttpOnly cookie you'd expect, and
what production would use instead.

`COOKIE_SECURE` defaults to true; compose sets it false because the demo is
served over plain http. Set it back to true behind TLS.
