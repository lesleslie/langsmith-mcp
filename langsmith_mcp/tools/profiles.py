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

MINIMAL_REGISTRATIONS: list[str | Callable[[FastMCP], Awaitable[None] | None]] = []

FULL_REGISTRATIONS: list[str | Callable[[FastMCP], Awaitable[None] | None]] = [
    "thread_history_tools",
    "prompt_tools",
    "trace_tools",
    "dataset_tools",
    "experiment_tools",
    "billing_tools",
    "health_tools",
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
    from langsmith_mcp.main import (
        register_billing_tools,
        register_dataset_tools,
        register_experiment_tools,
        register_health_tools,
        register_prompt_tools,
        register_thread_history_tools,
        register_trace_tools,
    )

    return {
        "thread_history_tools": register_thread_history_tools,
        "prompt_tools": register_prompt_tools,
        "trace_tools": register_trace_tools,
        "dataset_tools": register_dataset_tools,
        "experiment_tools": register_experiment_tools,
        "billing_tools": register_billing_tools,
        "health_tools": register_health_tools,
    }


def register_all_tool_groups(server: FastMCP) -> None:
    """Bulk register every langsmith-mcp tool group (called at FULL profile).

    Used as ``register_all_fn`` for the W0 helper. Imports each
    ``register_<group>_tools`` directly (not via ``REGISTRATION_MAP``
    iteration) so that adding a new group requires editing both this
    function and the ``FULL_REGISTRATIONS`` list — the redundancy is
    intentional: each is the ground-truth for a separate concern
    (matches the W2a Crackerjack pattern).
    """
    from langsmith_mcp.main import (
        register_billing_tools,
        register_dataset_tools,
        register_experiment_tools,
        register_health_tools,
        register_prompt_tools,
        register_thread_history_tools,
        register_trace_tools,
    )

    register_thread_history_tools(server)
    register_prompt_tools(server)
    register_trace_tools(server)
    register_dataset_tools(server)
    register_experiment_tools(server)
    register_billing_tools(server)
    register_health_tools(server)


async def apply_langsmith_tool_profile(server: FastMCP) -> None:
    """Apply the LANGSMITH_TOOL_PROFILE dispatch to ``server`` at startup.

    Async because the W0 helper is async; called from
    ``langsmith_mcp.main.get_app`` via
    ``await apply_langsmith_tool_profile(app)``. The sync
    ``apply_tool_profile`` wrapper raises ``RuntimeError`` in any async
    context, so this async path is the only correct entry point — the
    W2b.3 spline lesson is the keystone of this rule.

    langsmith-mcp exposes no MCP-registered health tools (only the
    /healthz HTTP route via ``mcp_common.health.register_http_health_route``),
    so the MANDATORY_GROUPS / MANDATORY_TOOLS invariants are vacuously
    satisfied. We pass empty sets explicitly to opt out of the subset
    check.
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
    "_build_registration_map",
    "apply_langsmith_tool_profile",
    "register_all_tool_groups",
]