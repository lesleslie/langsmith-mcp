# Tool Profile System Rationale

## Context

langsmith-mcp exposes 15 MCP tools across 7 functional groups (thread
history, prompts, traces, datasets, experiments, billing, health).
Pre-refactor, every tool was registered at module-load via
`@mcp.tool()` decorators. This left no way to scope the tool surface
per deployment:

- **Control-plane / health-probe-only deployments** (e.g. an MCP
  proxy in front of langsmith-mcp) wanted zero tool overhead.
- **Resource-constrained environments** (e.g. CI sandboxes running
  the smoke-test path) wanted only the health-check tool.
- **Full deployments** (the default) wanted every tool.

mcp-common 0.18.0 ships a W0 dispatch surface
(`mcp_common.tools.dispatch._apply_tool_profile`) that maps a
`{SERVER}_TOOL_PROFILE` env var to per-profile registration lists. We
adopted it in the W3.2 wave.

## Decision Rule

**langsmith-mcp uses a 2-tier profile mapping** (Tier-B per the W3
adoption plan). Tier-B is smaller scope than Tier-C's 3-tier — a
3-tier split adds no value for a 15-tool surface.

| Profile | Behavior |
|---------|----------|
| `MINIMAL` | No MCP-registered tool groups. Only the `discover_tools` meta-tool from the W0 helper. Useful for control-plane / health-probe deployments. |
| `FULL` (default) | All 15 langsmith tools + `discover_tools`. Matches pre-refactor behavior. |

**`LANGSMITH_TOOL_PROFILE`** env var controls the profile. Unset → FULL
(matches the `mcp_common` from_env default).

**No `STANDARD` tier**: the surface is small enough that
MINIMAL-vs-FULL is sufficient granularity.

## Production Path

The W2b.3 spline lesson is the keystone: the W0 helper has TWO entry
points, and the production path MUST go through the async helper
(`_apply_tool_profile` via `apply_langsmith_tool_profile`), NOT the
sync wrapper (`apply_tool_profile`).

```
                                ┌─────────────────────────┐
   CLI startup (sync)           │                         │
   ────────────────────►        │  apply_tool_profile     │ ── NO, raises in async
   uvicorn factory              │  (sync wrapper)         │
                                │                         │
                                └─────────────────────────┘

                                ┌─────────────────────────┐
   Tests (async)                │                         │
   ────────────────────►        │  apply_langsmith_       │ ── YES, the W0 path
   Async startup hooks          │  tool_profile (async    │     goes here
                                │  wrapper around async   │
                                │  _apply_tool_profile)   │
                                └─────────────────────────┘
```

Production entry point is `langsmith_mcp.tools.profiles.apply_langsmith_tool_profile`
which `await`s `_apply_tool_profile` (the async helper). The sync
wrapper `apply_tool_profile` is used only in
`langsmith_mcp.main._ensure_dispatched_sync` as a sync-friendly shim
for CLI startup, NEVER as the production dispatch path.

## Lazy Dispatch

The W0 dispatch is **lazy** — not at module import. Running
`asyncio.run(apply_langsmith_tool_profile(mcp))` at module import
breaks pytest-asyncio (which provides a running event loop at
collection). Instead:

- **Sync callers** (CLI startup, `get_app`, uvicorn factory) trigger
  the sync wrapper via `_ensure_dispatched_sync()` on first access.
- **Async callers** (tests, async startup hooks) trigger
  `apply_langsmith_tool_profile(mcp)` via `_ensure_dispatched_async()`
  on first access to `create_app()`.

A `_dispatch_done` sentinel ensures the profile applies exactly once
per process. Reloading the module resets the sentinel so a new
`LANGSMITH_TOOL_PROFILE` value can be exercised in tests.

## Architectural Choices

### MANDATORY_GROUPS / MANDATORY_TOOLS = `set()`

No tools are mandatory at any profile level for langsmith-mcp —
every tool group (including `health_tools` and the `health_check_cli`
MCP-registered health tool) is opt-in per profile. The W0 helper's
mandatory-group and essential-tool subset checks are therefore
vacuous; we pass empty sets explicitly to opt out.

The `/healthz` HTTP route registered via
`mcp_common.health.register_http_health_route` is independent of the
W0 profile dispatch — it lives on the FastMCP app directly, not on
the MCP tool registry, and is always available regardless of
`LANGSMITH_TOOL_PROFILE`.

### `register_<group>_tools(server)` Functions

Each group has a dedicated function that re-registers the
module-level tool async functions on the FastMCP server via
`server.tool(name=...)`. This pattern:

- Preserves the existing module-level functions (so `tests/test_main.py`
  can still import them directly and call them as plain async
  functions — no MCP scaffolding needed for unit tests).
- Lets the W0 dispatch surface drive registration by group key, not
  by individual tool name.
- Matches the W3 graphics-mcp `register_<group>_tools(server)`
  convention so the W3 + W3.1 + W3.2 + W3.3 waves produce a uniform
  shape across Bodai ecosystem.

### 2-Tier (No STANDARD)

15 tools, 7 groups. A `STANDARD` tier would either:

- Be a strict subset of FULL (semantically equivalent to MINIMAL +
  some groups), or
- Be a "default-but-not-everything" midpoint (which has no clear
  semantic meaning in a control-plane context).

We chose 2-tier to keep the surface small and the operator's
mental model simple.

## What This Refactor Did

1. Removed all 15 `@mcp.tool()` decorators from `langsmith_mcp/main.py`.
2. Added 7 `register_<group>_tools(server)` functions in main.py.
3. Added `langsmith_mcp/tools/profiles.py` with `PROFILE_REGISTRATIONS`
   (2-tier), `REGISTRATION_MAP`, `register_all_tool_groups`, and
   `apply_langsmith_tool_profile`.
4. Wired `apply_langsmith_tool_profile` via lazy `get_app()` and
   `create_app()` entry points in main.py. The `mcp` symbol is now
   a `__getattr__` lazy attribute that returns the dispatched server.
5. Bumped `mcp-common` pin to `>=0.18.0` in `pyproject.toml`.
6. Added `tests/unit/test_tool_profile.py` with 12 tests:
   - 3 AST guards (no bare `@mcp.tool()`, no sync wrapper in
     production path, production W0 path goes through async helper)
   - 4 profile mapping tests (MINIMAL empty, FULL has all 7 groups,
     2-tier, REGISTRATION_MAP resolves all keys)
   - 1 MANDATORY invariant test (empty sets explicitly)
   - 4 runtime tests (create_app full, minimal profile, full
     profile, behavioral parity)

## Status

**Adopted 2026-08-18 (W3.2 wave)**. Tests pass; quality gate clean.
Behavioural parity verified: the 15 pre-refactor decorator-registered
tools are all present under FULL profile.

## Cross-References

- mcp-common W0 dispatch: `mcp_common.tools.dispatch._apply_tool_profile`
- Bodai W2b.3 spline lesson: production path MUST use the async helper
- Bodai W3.1 graphics-mcp: 2-arg register fns with lambda binding
  (does NOT apply here — all `register_<group>_tools` take a single
  `server` argument matching the W0 helper signature)
- Bodai W3.3 synxis-crs-mcp / unifi-mcp: next waves, will reuse
  the same pattern
