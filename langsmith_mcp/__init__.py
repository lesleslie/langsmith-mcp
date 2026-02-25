"""LangSmith MCP Server.

MCP server for LangSmith observability integration providing:
- Conversation history retrieval
- Prompt management (list, get, push)
- Traces & Runs fetching
- Datasets & Examples management
- Experiments & Evaluations
- Usage & Billing information
"""

__version__ = "0.1.0"
__author__ = "Les Leslie"

from langsmith_mcp.config import LangSmithSettings
from langsmith_mcp.main import mcp

__all__ = ["mcp", "LangSmithSettings"]
