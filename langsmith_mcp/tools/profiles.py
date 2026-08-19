"""Tool profile registration groups for langsmith-mcp MCP server.

Maps ``ToolProfile`` levels to specific ``register_<group>_tools()`` call
lists, controlling which tools are exposed at startup based on the
``LANGSMITH_TOOL_PROFILE`` environment variable.

Profile tiers (2-tier, Tier-B — langsmith-mcp is small enough that a
3-tier split adds no value):

    MINIMAL:  No MCP-registered tool groups (only the /healthz HTTP route
              registered via ``mcp_common.health.register_http_health_route``
              and the ``discover_tools`` meta-tool from the W0 helper).
              Useful for control-plane / health-probe-only deployments.
    FULL:     All 15 langsmith tools across 7 groups (thread_history,
              prompts, traces, datasets, experiments, billing, health).
              Default behavior — matches pre-refactor inline registration.

The dispatch surface (``PROFILE_REGISTRATIONS`` + ``REGISTRATION_MAP`` +
``register_all_tool_groups`` + ``apply_langsmith_tool_profile``) is consumed
by ``langsmith_mcp.main.get_app`` which delegates to
``mcp_common.tools.dispatch._apply_tool_profile``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp_common.tools import ToolProfile
from mcp_common.tools.dispatch import ALL_TOOLS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastmcp import FastMCP

# Canonical list of every register_<group>_tools group key + the matching
# attribute name on ``langsmith_mcp.main``. The order matches the
# pre-refactor decorator registration order in ``main.py`` and is
# preserved across all three call sites
# (FULL_REGISTRATIONS, _build_registration_map, register_all_tool_groups)
# so adding a new group requires editing only this constant.
_GROUP_REGISTRY: list[tuple[str, str]] = [
    ("thread_history_tools", "register_thread_history_tools"),
    ("prompt_tools", "register_prompt_tools"),
    ("trace_tools", "register_trace_tools"),
    ("dataset_tools", "register_dataset_tools"),
    ("experiment_tools", "register_experiment_tools"),
    ("billing_tools", "register_billing_tools"),
    ("health_tools", "register_health_tools"),
]

MINIMAL_REGISTRATIONS: list[str | Callable[[FastMCP], Awaitable[None] | None]] = []

FULL_REGISTRATIONS: list[str | Callable[[FastMCP], Awaitable[None] | None]] = [
    key for key, _ in _GROUP_REGISTRY
]

PROFILE_REGISTRATIONS: dict[
    ToolProfile,
    list[str | Callable[[FastMCP], Awaitable[None] | None]] | type[ALL_TOOLS],
] = {
    ToolProfile.MINIMAL: MINIMAL_REGISTRATIONS,
    ToolProfile.FULL: FULL_REGISTRATIONS,
}


def _build_registration_map() -> dict[
    str, Callable[[FastMCP], Awaitable[None] | None]
]:
    """Build the {group_key: register_fn(app)} map.

    Local imports keep ``langsmith_mcp.tools.profiles`` importable without
    forcing every ``register_<group>_tools`` function in ``langsmith_mcp.main``
    to be resolved at module import time. Called by
    ``apply_langsmith_tool_profile`` (not eagerly at import) because
    ``main`` is imported eagerly by ``langsmith_mcp.__init__`` and
    ``tests.test_main``.

    All ``register_<group>_tools`` functions take a single ``(mcp)``
    argument matching the W0 helper's expected signature, so no lambda
    binding is required (the W3.1 graphics-mcp lesson does not apply).
    """
    from langsmith_mcp import main as _main

    return {
        key: getattr(_main, attr_name)
        for key, attr_name in _GROUP_REGISTRY
    }


def register_all_tool_groups(server: FastMCP) -> None:
    """Bulk register every langsmith-mcp tool group (called at FULL profile).

    Used as ``register_all_fn`` for the W0 helper. Imports each
    ``register_<group>_tools`` directly (not via ``REGISTRATION_MAP``
    iteration) so that adding a new group requires editing only the
    ``_GROUP_REGISTRY`` constant — the canonical source of truth
    (matches the W2a Crackerjack pattern).
    """
    from langsmith_mcp import main as _main

    for _key, attr_name in _GROUP_REGISTRY:
        getattr(_main, attr_name)(server)


async def apply_langsmith_tool_profile(server: FastMCP) -> None:
    """Apply the LANGSMITH_TOOL_PROFILE dispatch to ``server`` at startup.

    Async because the W0 helper is async; called from
    ``langsmith_mcp.main.get_app`` via
    ``await apply_langsmith_tool_profile(app)``. The sync
    ``apply_tool_profile`` wrapper raises ``RuntimeError`` in any async
    context, so this async path is the only correct entry point — the
    W2b.3 spline lesson is the keystone of this rule.

    No tools are mandatory at any profile level for langsmith-mcp —
    every tool group (including ``health_tools``) is opt-in per
    profile. The MANDATORY_GROUPS / MANDATORY_TOOLS invariants are
    therefore vacuous; we pass empty sets explicitly to opt out of
    the subset check.
    """
    from mcp_common.tools.dispatch import _apply_tool_profile

    await _apply_tool_profile(
        server,
        profile_env_var="LANGSMITH_TOOL_PROFILE",
        registrations=PROFILE_REGISTRATIONS,
        registration_map=_build_registration_map(),
        register_all_fn=register_all_tool_groups,
        mandatory_groups=set(),
        essential_tool_names=set(),
    )


__all__ = [
    "FULL_REGISTRATIONS",
    "MINIMAL_REGISTRATIONS",
    "PROFILE_REGISTRATIONS",
    "_GROUP_REGISTRY",
    "_build_registration_map",
    "apply_langsmith_tool_profile",
    "register_all_tool_groups",
]
