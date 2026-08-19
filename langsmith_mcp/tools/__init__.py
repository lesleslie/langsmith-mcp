"""Tool profile registration groups for langsmith-mcp MCP server.

Maps ``ToolProfile`` levels to specific ``register_<group>_tools()`` call
lists, controlling which tools are exposed at startup based on the
``LANGSMITH_TOOL_PROFILE`` environment variable.
"""

from __future__ import annotations