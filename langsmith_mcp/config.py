"""LangSmith MCP Server Configuration.

Uses mcp-common patterns with Oneiric layered configuration.
"""

from typing import Annotated

from mcp_common.config import MCPServerSettings
from pydantic import Field


class LangSmithSettings(MCPServerSettings):
    """LangSmith MCP Server Configuration.

    Configuration follows Oneiric layered loading:
    1. Default values (below)
    2. settings/langsmith.yaml (committed)
    3. settings/local.yaml (gitignored, local dev)
    4. Environment variables LANGSMITH_*
    """

    model_config = {
        "env_prefix": "LANGSMITH_",
        "env_file": ".env",
        "extra": "ignore",
    }

    # Required: LangSmith API key
    api_key: Annotated[
        str,
        Field(
            ...,
            description="LangSmith API key for authentication",
            min_length=1,
        ),
    ]

    # Optional: Workspace configuration
    workspace_id: Annotated[
        str | None,
        Field(
            default=None,
            description="LangSmith workspace ID (uses default if not specified)",
        ),
    ]

    # API endpoint configuration
    api_endpoint: Annotated[
        str,
        Field(
            default="https://api.smith.langchain.com",
            description="LangSmith API endpoint",
        ),
    ]

    # Pagination defaults (character-based pagination for large responses)
    max_chars_per_page: Annotated[
        int,
        Field(
            default=25000,
            ge=1000,
            le=100000,
            description="Maximum characters per page for paginated responses",
        ),
    ]

    preview_chars: Annotated[
        int,
        Field(
            default=100,
            ge=50,
            le=500,
            description="Number of characters to show in previews",
        ),
    ]

    # Feature toggles
    features_enabled: Annotated[
        set[str],
        Field(
            default={
                "conversation",
                "prompts",
                "traces",
                "datasets",
                "experiments",
                "billing",
            },
            description="Enabled feature categories",
        ),
    ]

    # HTTP client configuration
    http_timeout: Annotated[
        float,
        Field(
            default=30.0,
            ge=5.0,
            le=120.0,
            description="HTTP request timeout in seconds",
        ),
    ]

    max_retries: Annotated[
        int,
        Field(
            default=3,
            ge=0,
            le=5,
            description="Maximum retry attempts for failed requests",
        ),
    ]

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled."""
        return feature in self.features_enabled

    def get_masked_api_key(self) -> str:
        """Get masked API key for safe logging."""
        if not self.api_key or len(self.api_key) <= 4:
            return "***"
        return f"...{self.api_key[-4:]}"
