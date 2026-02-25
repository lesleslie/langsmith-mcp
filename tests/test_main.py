"""Tests for LangSmith MCP Server."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langsmith_mcp.config import LangSmithSettings
from langsmith_mcp.main import health_check


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    with patch.dict(
        os.environ,
        {
            "LANGSMITH_API_KEY": "test-api-key-12345678",
        },
    ):
        yield LangSmithSettings()


@pytest.fixture
def mock_client():
    """Create mock LangSmith client."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.list_projects = AsyncMock(return_value={"projects": []})
    return client


class TestConfig:
    """Tests for configuration."""

    def test_settings_defaults(self, mock_settings):
        """Test default settings values."""
        assert mock_settings.api_endpoint == "https://api.smith.langchain.com"
        assert mock_settings.max_chars_per_page == 25000
        assert mock_settings.http_timeout == 30.0
        assert "prompts" in mock_settings.features_enabled

    def test_masked_api_key(self, mock_settings):
        """Test API key masking."""
        masked = mock_settings.get_masked_api_key()
        assert masked == "...5678"
        assert "test-api-key" not in masked

    def test_feature_enabled(self, mock_settings):
        """Test feature toggle check."""
        assert mock_settings.is_feature_enabled("prompts") is True
        assert mock_settings.is_feature_enabled("nonexistent") is False


class TestHealthCheck:
    """Tests for health check tool."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_client):
        """Test successful health check."""
        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            with patch("langsmith_mcp.main.get_settings") as mock_get_settings:
                mock_get_settings.return_value = MagicMock(
                    api_endpoint="https://api.smith.langchain.com",
                    get_masked_api_key=lambda: "...5678",
                    features_enabled={"prompts", "traces"},
                    max_chars_per_page=25000,
                )

                result = await health_check()

                assert result["status"] == "healthy"
                assert result["server"] == "langsmith-mcp"
                assert "config" in result

    @pytest.mark.asyncio
    async def test_health_check_degraded(self):
        """Test degraded health check when API unreachable."""
        with patch(
            "langsmith_mcp.main._get_client",
            side_effect=Exception("Connection failed"),
        ):
            with patch("langsmith_mcp.main.get_settings") as mock_get_settings:
                mock_get_settings.return_value = MagicMock(
                    api_endpoint="https://api.smith.langchain.com",
                    get_masked_api_key=lambda: "...5678",
                    features_enabled={"prompts", "traces"},
                    max_chars_per_page=25000,
                )

                result = await health_check()

                assert result["status"] == "degraded"
                assert "error" in result["connectivity"]
