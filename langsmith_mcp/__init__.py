"""LangSmith MCP Server.

MCP server for LangSmith observability integration providing:
- Conversation history retrieval
- Prompt management (list, get, push)
- Traces & Runs fetching
- Datasets & Examples management
- Experiments & Evaluations
- Usage & Billing information
"""

__author__ = "Les Leslie"

from importlib.metadata import version as _importlib_version

__version__ = _importlib_version("langsmith-mcp")

from langsmith_mcp.config import LangSmithSettings
from langsmith_mcp.main import mcp

__all__ = ["LangSmithSettings", "mcp"]
