"""Tool profile adoption tests for langsmith-mcp.

Covers W0 dispatch surface (mcp_common 0.18.0+):
- 2-tier mapping (MINIMAL/FULL)
- AST guards: no bare @mcp.tool() decorators, no sync apply_tool_profile
  call from the main entrypoint
- MANDATORY_TOOLS invariant (vacuously satisfied — no MCP-registered
  health tools, only the /healthz HTTP route)
- Real production-path test: await create_app() exercises the async
  helper directly (W2b.3 spline lesson)
- Behavioral parity: W0 path produces the same tool set as the
  decorator-mode path would have
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# =====================
# AST Guards
# =====================


REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "langsmith_mcp" / "main.py"


def _parse_main() -> ast.Module:
    return ast.parse(MAIN_PY.read_text())


def test_main_has_no_bare_mcp_tool_decorators() -> None:
    """All tool registration goes through register_<group>_tools(server)."""
    tree = _parse_main()
    bare = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            # Bare @mcp.tool() (no args) is forbidden.
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and dec.func.attr == "tool"
                and not dec.args
                and not dec.keywords
            ):
                bare.append(node.name)
    assert bare == [], (
        f"Found bare @mcp.tool() decorators on {bare!r}. "
        f"All tools must be wired through register_<group>_tools(server)."
    )


def test_profiles_uses_async_helper_not_sync_wrapper() -> None:
    """The W0 production path in ``profiles.py`` MUST use the async helper
    ``apply_langsmith_tool_profile`` (which awaits
    ``_apply_tool_profile``) — NOT the sync ``apply_tool_profile``
    wrapper (which raises ``RuntimeError`` in async contexts).

    The sync wrapper IS allowed in ``langsmith_mcp.main._ensure_dispatched_sync``
    as a sync-only entry point for CLI startup, but the production W0
    path lives in ``langsmith_mcp/tools/profiles.py`` and goes through
    the async helper. This AST guard checks the production W0 path.
    """
    profiles_py = REPO_ROOT / "langsmith_mcp" / "tools" / "profiles.py"
    tree = ast.parse(profiles_py.read_text())
    has_async_helper = False
    has_sync_wrapper = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "apply_langsmith_tool_profile"
        ):
            has_async_helper = True
        if (
            isinstance(node, ast.Await)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_apply_tool_profile"
        ):
            has_async_helper = True
    assert has_async_helper, (
        "Production W0 path in profiles.py must invoke "
        "apply_langsmith_tool_profile() (which awaits _apply_tool_profile)."
    )
    # The sync wrapper should not be called from profiles.py.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "apply_tool_profile"
        ):
            has_sync_wrapper = True
    assert not has_sync_wrapper, (
        "Production W0 path in profiles.py must NOT call the sync "
        "apply_tool_profile() wrapper — that raises in async contexts."
    )


def test_main_invokes_async_apply_langsmith_tool_profile() -> None:
    """The production path awaits the W0 helper.

    This is the W2b.3 spline keystone: tests that mock the dispatch
    helper can mask a real production bug. We structurally assert that
    the call site is wrapped in ``ast.Await`` (i.e.
    ``await apply_langsmith_tool_profile(server)``), not just a bare
    call. A bare call would silently run the coroutine and discard the
    result, dropping the dispatch.
    """
    tree = _parse_main()
    awaited_calls = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Await)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "apply_langsmith_tool_profile"
        )
    ]
    assert awaited_calls, (
        "main.py must `await apply_langsmith_tool_profile(...)` at least "
        "once (W2b.3 spline keystone — the production W0 path goes through "
        "the async helper, not the sync wrapper)."
    )


# =====================
# Profile mapping
# =====================


def test_minimal_registrations_empty() -> None:
    """MINIMAL profile registers no tool groups (only discover_tools
    meta-tool comes from the W0 helper itself)."""
    from langsmith_mcp.tools.profiles import MINIMAL_REGISTRATIONS

    assert MINIMAL_REGISTRATIONS == []


def test_full_registrations_covers_all_seven_groups() -> None:
    """FULL profile must enumerate all 7 register_<group>_tools() fns."""
    from langsmith_mcp.tools.profiles import FULL_REGISTRATIONS

    assert set(FULL_REGISTRATIONS) == {
        "thread_history_tools",
        "prompt_tools",
        "trace_tools",
        "dataset_tools",
        "experiment_tools",
        "billing_tools",
        "health_tools",
    }


def test_profile_registrations_is_two_tier() -> None:
    """Tier-B is 2-tier (MINIMAL/FULL). STANDARD tier must not be used."""
    from mcp_common.tools import ToolProfile

    from langsmith_mcp.tools.profiles import PROFILE_REGISTRATIONS

    assert set(PROFILE_REGISTRATIONS.keys()) == {
        ToolProfile.MINIMAL,
        ToolProfile.FULL,
    }


def test_registration_map_resolves_all_seven_keys() -> None:
    """Every key in FULL_REGISTRATIONS must resolve to a register fn."""
    from langsmith_mcp.tools.profiles import (
        FULL_REGISTRATIONS,
        _build_registration_map,
    )

    reg_map = _build_registration_map()
    for key in FULL_REGISTRATIONS:
        assert key in reg_map, f"registration_map missing {key!r}"
        assert callable(reg_map[key]), f"registration_map[{key!r}] not callable"


# =====================
# MANDATORY invariant
# =====================


def test_mandatory_and_essential_invariants_held() -> None:
    """No tools are mandatory at any profile level for langsmith-mcp —
    every tool group (including ``health_tools``) is opt-in per
    profile. The MANDATORY_GROUPS / MANDATORY_TOOLS subsets are
    therefore vacuous; profiles pass empty sets explicitly to opt out
    of the subset check (W0 spec)."""
    import inspect

    from langsmith_mcp.tools.profiles import apply_langsmith_tool_profile

    src = inspect.getsource(apply_langsmith_tool_profile)
    assert "mandatory_groups=set()" in src, (
        "apply_langsmith_tool_profile must pass empty mandatory_groups "
        "since no tools are mandatory at any profile level."
    )
    assert "essential_tool_names=set()" in src, (
        "apply_langsmith_tool_profile must pass empty essential_tool_names "
        "since no tools are mandatory at any profile level."
    )


# =====================
# Production-path test (W2b.3 lesson)
# =====================


@pytest.mark.asyncio
async def test_create_app_registers_every_tool() -> None:
    """Real production-path test: await create_app() exercises the async
    helper directly with no mocking of the dispatch surface.

    This is the W2b.3 spline keystone: tests that mock _apply_tool_profile
    can mask a real production bug. We exercise the real helper and
    assert the registered tool set matches the FULL profile.
    """
    import importlib

    from langsmith_mcp import main as main_mod

    importlib.reload(main_mod)
    app = await main_mod.create_app()
    tools = {t.name for t in await app.list_tools()}

    expected_full = {
        "create_dataset",
        "create_examples",
        "fetch_runs",
        "get_billing_usage",
        "get_dataset",
        "get_experiment",
        "get_prompt",
        "get_thread_history",
        "health_check_cli",
        "list_datasets",
        "list_examples",
        "list_experiments",
        "list_projects",
        "list_prompts",
        "push_prompt",
        "discover_tools",
    }
    assert tools == expected_full


@pytest.mark.asyncio
async def test_minimal_profile_only_keeps_discover_tools(monkeypatch) -> None:
    """Under LANGSMITH_TOOL_PROFILE=minimal, only the discover_tools
    meta-tool remains registered.

    Reload the module and explicitly call ``create_app()`` (async path)
    so the dispatch happens via the W2b.3-compliant async helper, not
    the sync wrapper (which raises inside a running event loop).
    """
    monkeypatch.setenv("LANGSMITH_TOOL_PROFILE", "minimal")
    import importlib

    from langsmith_mcp import main as main_mod

    importlib.reload(main_mod)
    # Use the async path; the sync path raises in this test's loop.
    app = await main_mod.create_app()
    tools = {t.name for t in await app.list_tools()}
    assert tools == {"discover_tools"}


@pytest.mark.asyncio
async def test_full_profile_keeps_every_langsmith_tool(monkeypatch) -> None:
    """Under LANGSMITH_TOOL_PROFILE=full (default), all 15 langsmith
    tools + discover_tools meta-tool are registered."""
    monkeypatch.setenv("LANGSMITH_TOOL_PROFILE", "full")
    import importlib

    from langsmith_mcp import main as main_mod

    importlib.reload(main_mod)
    app = await main_mod.create_app()
    tools = {t.name for t in await app.list_tools()}
    # 15 langsmith tools + discover_tools
    assert len(tools) == 16
    assert "discover_tools" in tools
    assert "get_thread_history" in tools
    assert "health_check_cli" in tools


# =====================
# Behavioral parity (W1.3 lesson)
# =====================


@pytest.mark.asyncio
async def test_behavioral_parity_with_pre_refactor_decorator_mode() -> None:
    """W1.3 akosha lesson: behavioral parity, not just name parity.

    The pre-refactor code registered 15 tools via @mcp.tool() decorators.
    The W0 dispatch path must register the same 15 tools (plus the
    discover_tools meta-tool). Compare the actual registered tool set
    against the explicit pre-refactor baseline.
    """
    from langsmith_mcp.main import create_app

    app = await create_app()
    tools = {t.name for t in await app.list_tools()}

    # Pre-refactor baseline: every tool that used to have @mcp.tool()
    # decorator in main.py.
    pre_refactor_tools = {
        "get_thread_history",
        "list_prompts",
        "get_prompt",
        "push_prompt",
        "fetch_runs",
        "list_projects",
        "list_datasets",
        "get_dataset",
        "list_examples",
        "create_dataset",
        "create_examples",
        "list_experiments",
        "get_experiment",
        "get_billing_usage",
        "health_check_cli",
    }
    missing = pre_refactor_tools - tools
    assert not missing, (
        f"Behavioral parity violated. Pre-refactor tools missing from "
        f"W0 dispatch path: {sorted(missing)}"
    )
