"""LangSmith MCP Server Configuration.

from __future__ import annotations
Uses mcp-common patterns with Oneiric layered configuration.
"""

import os
from pathlib import Path
from typing import Annotated, Any, ClassVar

import yaml
from oneiric.core.config import OneiricMCPConfig
from pydantic import Field


class LangSmithSettings(OneiricMCPConfig):
    """LangSmith MCP Server Configuration.

    Configuration follows Oneiric layered loading:
    1. Default values (below)
    2. settings/langsmith.yaml (committed)
    3. settings/local.yaml (gitignored, local dev)
    4. Environment variables LANGSMITH_*
    """

    model_config = {  # type: ignore
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

    LEGACY_ENV_PREFIX: ClassVar[str] = "LANGSMITH"

    @classmethod
    def load(
        cls,
        server_name: str = "langsmith",
        config_path: Path | None = None,
        env_prefix: str | None = None,
    ) -> "LangSmithSettings":
        """Load settings with layered configuration.

        Backward-compatible with the MCPBaseSettings.load() signature.

        Priority (highest to lowest):
        1. Explicit config_path (if provided)
        2. Environment variables ({env_prefix}_{FIELD})
        3. settings/local.yaml (gitignored)
        4. settings/{server_name}.yaml
        5. Defaults defined in the class

        Args:
            server_name: Server identifier (default: 'langsmith')
            config_path: Optional explicit config file path
            env_prefix: Environment variable prefix (default: 'LANGSMITH')
        """
        if env_prefix is None:
            env_prefix = cls.LEGACY_ENV_PREFIX

        data: dict[str, Any] = {"server_name": server_name}

        # Layer 1: settings/{server_name}.yaml
        server_yaml = Path("settings") / f"{server_name}.yaml"
        if server_yaml.exists():
            with server_yaml.open() as f:
                yaml_data = yaml.safe_load(f)
            if isinstance(yaml_data, dict):
                data.update(yaml_data)

        # Layer 2: settings/local.yaml
        local_yaml = Path("settings") / "local.yaml"
        if local_yaml.exists():
            with local_yaml.open() as f:
                local_data = yaml.safe_load(f)
            if isinstance(local_data, dict):
                data.update(local_data)

        # Layer 3: Environment variables
        for field_name in cls.model_fields:
            env_var = f"{env_prefix}_{field_name.upper()}"
            if env_var in os.environ:
                env_value: str | Path | None = os.environ[env_var]
                field_def = cls.model_fields[field_name]
                field_type = field_def.annotation
                field_args: tuple[Any, ...] = ()
                try:
                    from typing import get_args as _get_args

                    field_args = _get_args(field_type)
                except Exception:
                    pass
                if field_type is Path or (field_args and Path in field_args):
                    env_value = Path(env_value) if env_value else None
                data[field_name] = env_value

        # Layer 4: Explicit config path (highest priority)
        if config_path is not None and config_path.exists():
            with config_path.open() as f:
                explicit_data = yaml.safe_load(f)
            if isinstance(explicit_data, dict):
                data.update(explicit_data)

        return cls.model_validate(data)
