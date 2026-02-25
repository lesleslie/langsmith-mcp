#!/usr/bin/env python3
"""LangSmith MCP Server - Oneiric CLI Entry Point."""

from typing import Any, cast

from mcp_common.cli import MCPServerCLIFactory
from mcp_common.server import BaseOneiricServerMixin, create_runtime_components
from oneiric.core.config import OneiricMCPConfig
from oneiric.runtime.mcp_health import HealthStatus

from langsmith_mcp.main import mcp, validate_api_key_at_startup


class LangSmithConfig(OneiricMCPConfig):
    """LangSmith MCP Server Configuration."""

    http_port: int = 3048
    http_host: str = "127.0.0.1"
    enable_http_transport: bool = True

    class Config:
        env_prefix = "LANGSMITH_MCP_"
        env_file = ".env"


class LangSmithMCPServer(BaseOneiricServerMixin):
    """LangSmith MCP Server with Oneiric integration."""

    def __init__(self, config: LangSmithConfig):
        self.config = config  # type: ignore[assignment]
        self.mcp = mcp  # Use the existing FastMCP instance

        # Initialize runtime components using mcp-common helper
        self.runtime = create_runtime_components(
            server_name="langsmith-mcp", cache_dir=".oneiric_cache"
        )

        # Expose runtime components as convenience attributes
        self.snapshot_manager = self.runtime.snapshot_manager
        self.cache_manager = self.runtime.cache_manager
        self.health_monitor = self.runtime.health_monitor

    @property
    def _cfg(self) -> LangSmithConfig:
        """Helper to access config with correct type."""
        return cast(LangSmithConfig, self.config)

    async def startup(self) -> None:
        """Server startup lifecycle hook."""
        # Validate API key at startup
        validate_api_key_at_startup()

        # Initialize runtime components
        await self.runtime.initialize()

        # Create startup snapshot with custom components
        await self._create_startup_snapshot(
            custom_components={
                "api_key": {
                    "status": "validated",
                    "timestamp": self._get_timestamp(),
                },
            }
        )

        print("✅ LangSmith MCP Server started successfully")
        print(f"   Listening on {self._cfg.http_host}:{self._cfg.http_port}")
        print(f"   Cache directory: {self.runtime.cache_dir}")

    async def shutdown(self) -> None:
        """Server shutdown lifecycle hook."""
        # Create shutdown snapshot
        await self._create_shutdown_snapshot()

        # Clean up runtime components
        await self.runtime.cleanup()

        print("👋 LangSmith MCP Server shutdown complete")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        # Build base health components using mixin helper
        base_components = await self._build_health_components()

        # Add LangSmith-specific health checks
        from langsmith_mcp.config import LangSmithSettings

        try:
            settings = LangSmithSettings()
            api_key_available = bool(settings.api_key)
        except Exception:
            api_key_available = False

        base_components.append(
            self.runtime.health_monitor.create_component_health(
                name="api_key",
                status=HealthStatus.HEALTHY
                if api_key_available
                else HealthStatus.UNHEALTHY,
                details={"configured": api_key_available},
            )
        )

        # Create health response
        return self.runtime.health_monitor.create_health_response(base_components)  # type: ignore

    def get_app(self) -> Any:
        """Get the ASGI application."""
        return self.mcp.http_app

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        import time

        return time.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    """Main entry point for LangSmith MCP Server."""

    # Create CLI factory using mcp-common's enhanced factory
    cli_factory = MCPServerCLIFactory.create_server_cli(
        server_class=LangSmithMCPServer,
        config_class=LangSmithConfig,
        name="langsmith-mcp",
    )

    # Create and run CLI
    cli_factory.create_app()()


if __name__ == "__main__":
    main()
