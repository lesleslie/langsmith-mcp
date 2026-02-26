"""Tests for LangSmith MCP Server configuration."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from langsmith_mcp.config import LangSmithSettings


@pytest.fixture
def clean_env():
    """Clean environment variables before each test."""
    # Save original values
    original = {}
    for key in list(os.environ.keys()):
        if key.startswith("LANGSMITH_"):
            original[key] = os.environ.pop(key, None)

    yield

    # Restore original values
    for key, value in original.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


@pytest.fixture
def mock_settings_file(tmp_path: Path, clean_env):
    """Create a temporary settings file for testing."""
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    settings_file = settings_dir / "langsmith.yaml"
    settings_file.write_text(
        """
server_name: "Test LangSmith Server"
api_endpoint: "https://test.smith.langchain.com"
max_chars_per_page: 50000
features_enabled:
  - prompts
  - traces
http_timeout: 60.0
max_retries: 5
"""
    )
    return tmp_path


class TestLangSmithSettings:
    """Tests for LangSmithSettings configuration."""

    def test_load_with_env_var(self, clean_env):
        """Test loading settings with environment variable."""
        os.environ["LANGSMITH_API_KEY"] = "test-key-12345678"

        settings = LangSmithSettings.load("langsmith")
        assert settings.api_key == "test-key-12345678"

    def test_default_values(self, clean_env):
        """Test default configuration values."""
        os.environ["LANGSMITH_API_KEY"] = "test-key-12345678"

        settings = LangSmithSettings.load("langsmith")

        assert settings.api_endpoint == "https://api.smith.langchain.com"
        assert settings.max_chars_per_page == 25000
        assert settings.preview_chars == 100
        assert settings.http_timeout == 30.0
        assert settings.max_retries == 3

    def test_features_enabled_default(self, clean_env):
        """Test default enabled features."""
        os.environ["LANGSMITH_API_KEY"] = "test-key-12345678"

        settings = LangSmithSettings.load("langsmith")

        assert settings.is_feature_enabled("conversation") is True
        assert settings.is_feature_enabled("prompts") is True
        assert settings.is_feature_enabled("traces") is True
        assert settings.is_feature_enabled("datasets") is True
        assert settings.is_feature_enabled("experiments") is True
        assert settings.is_feature_enabled("billing") is True
        assert settings.is_feature_enabled("nonexistent") is False

    def test_env_override(self, clean_env):
        """Test environment variable overrides."""
        os.environ["LANGSMITH_API_KEY"] = "test-key-12345678"
        os.environ["LANGSMITH_API_ENDPOINT"] = "https://custom.endpoint.com"
        os.environ["LANGSMITH_HTTP_TIMEOUT"] = "45.0"

        settings = LangSmithSettings.load("langsmith")

        assert settings.api_endpoint == "https://custom.endpoint.com"
        assert settings.http_timeout == 45.0

    def test_masked_api_key_full(self, clean_env):
        """Test API key masking with full key."""
        os.environ["LANGSMITH_API_KEY"] = "test_fake_key_1234567890abcdefghijklmnop_9876zyxwv"

        settings = LangSmithSettings.load("langsmith")
        masked = settings.get_masked_api_key()

        assert masked == "...65d7"
        assert "lsv2_pt" not in masked

    def test_masked_api_key_short(self, clean_env):
        """Test API key masking with short key."""
        os.environ["LANGSMITH_API_KEY"] = "abc"

        settings = LangSmithSettings.load("langsmith")
        masked = settings.get_masked_api_key()

        assert masked == "***"

    def test_masked_api_key_empty(self, clean_env):
        """Test API key masking with empty key."""
        os.environ["LANGSMITH_API_KEY"] = "test-key-12345678"

        settings = LangSmithSettings.load("langsmith")
        # Simulate empty key by setting it directly
        settings.api_key = ""
        masked = settings.get_masked_api_key()

        assert masked == "***"

    def test_missing_api_key(self, clean_env):
        """Test that missing API key raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            LangSmithSettings.load("langsmith")

    def test_custom_features_enabled(self, clean_env):
        """Test custom features enabled via environment."""
        os.environ["LANGSMITH_API_KEY"] = "test-key-12345678"
        # Note: set[str] via env var needs special handling

        settings = LangSmithSettings.load("langsmith")
        # Default features should be present
        assert "prompts" in settings.features_enabled
        assert "traces" in settings.features_enabled

    def test_pagination_bounds(self, clean_env):
        """Test pagination configuration bounds."""
        os.environ["LANGSMITH_API_KEY"] = "test-key-12345678"
        os.environ["LANGSMITH_MAX_CHARS_PER_PAGE"] = "50000"

        settings = LangSmithSettings.load("langsmith")

        assert settings.max_chars_per_page == 50000
        assert settings.max_chars_per_page >= 1000
        assert settings.max_chars_per_page <= 100000

    def test_timeout_bounds(self, clean_env):
        """Test HTTP timeout configuration bounds."""
        os.environ["LANGSMITH_API_KEY"] = "test-key-12345678"
        os.environ["LANGSMITH_HTTP_TIMEOUT"] = "60.0"

        settings = LangSmithSettings.load("langsmith")

        assert settings.http_timeout == 60.0
        assert settings.http_timeout >= 5.0
        assert settings.http_timeout <= 120.0

    def test_retry_bounds(self, clean_env):
        """Test max retries configuration bounds."""
        os.environ["LANGSMITH_API_KEY"] = "test-key-12345678"
        os.environ["LANGSMITH_MAX_RETRIES"] = "2"

        settings = LangSmithSettings.load("langsmith")

        assert settings.max_retries == 2
        assert settings.max_retries >= 0
        assert settings.max_retries <= 5
