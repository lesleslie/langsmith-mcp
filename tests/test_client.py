"""Tests for LangSmith API Client."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from langsmith_mcp.client import LangSmithAPIError, LangSmithClient
from langsmith_mcp.config import LangSmithSettings


@pytest.fixture
def settings():
    """Create test settings."""
    with patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key-12345678"}):
        return LangSmithSettings.load("langsmith")


@pytest.fixture
def mock_httpx_client():
    """Create mock httpx client."""
    client = MagicMock(spec=httpx.AsyncClient)
    client.request = AsyncMock()
    client.aclose = AsyncMock()
    return client


class TestLangSmithAPIError:
    """Tests for LangSmithAPIError."""

    def test_error_with_all_fields(self):
        """Test error with all fields."""
        error = LangSmithAPIError(
            message="API error",
            status_code=400,
            details={"error": "Bad request"},
        )

        assert str(error) == "API error"
        assert error.status_code == 400
        assert error.details == {"error": "Bad request"}

    def test_error_minimal(self):
        """Test error with minimal fields."""
        error = LangSmithAPIError(message="Simple error")

        assert str(error) == "Simple error"
        assert error.status_code is None
        assert error.details == {}

    def test_error_inheritance(self):
        """Test that error inherits from MCPServerError."""
        from mcp_common.exceptions import MCPServerError

        error = LangSmithAPIError(message="Test")
        assert isinstance(error, MCPServerError)


class TestLangSmithClient:
    """Tests for LangSmithClient."""

    def test_init(self, settings):
        """Test client initialization."""
        client = LangSmithClient(settings)
        assert client.settings == settings
        assert client._client is None

    @pytest.mark.asyncio
    async def test_initialize(self, settings):
        """Test client initialization creates httpx client."""
        client = LangSmithClient(settings)
        await client.initialize()

        assert client._client is not None
        assert client._client.base_url == settings.api_endpoint
        assert "x-api-key" in client._client.headers
        assert client._client.headers["x-api-key"] == settings.api_key

        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, settings):
        """Test async context manager pattern."""
        async with LangSmithClient(settings) as client:
            assert client._client is not None

        # Client should be closed after context exit
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close(self, settings):
        """Test client close."""
        client = LangSmithClient(settings)
        await client.initialize()
        assert client._client is not None

        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_get_client_not_initialized(self, settings):
        """Test _get_client raises error when not initialized."""
        client = LangSmithClient(settings)

        with pytest.raises(Exception):  # MCPServerError
            client._get_client()


class TestLangSmithClientRequests:
    """Tests for LangSmithClient API requests."""

    @pytest.mark.asyncio
    async def test_list_prompts(self, settings, mock_httpx_client):
        """Test list_prompts request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"prompts": []}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        client = LangSmithClient(settings)
        client._client = mock_httpx_client

        result = await client.list_prompts(limit=50, offset=10)

        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert call_args.kwargs["method"] == "GET"
        assert call_args.kwargs["url"] == "/v1/prompts"
        assert call_args.kwargs["params"]["limit"] == 50
        assert call_args.kwargs["params"]["offset"] == 10
        assert result == {"prompts": []}

    @pytest.mark.asyncio
    async def test_get_prompt(self, settings, mock_httpx_client):
        """Test get_prompt request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"name": "test-prompt", "content": "Hello"}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        client = LangSmithClient(settings)
        client._client = mock_httpx_client

        result = await client.get_prompt("test-prompt", version="v1")

        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert call_args.kwargs["method"] == "GET"
        assert call_args.kwargs["url"] == "/v1/prompts/test-prompt"
        assert call_args.kwargs["params"]["version"] == "v1"

    @pytest.mark.asyncio
    async def test_push_prompt(self, settings, mock_httpx_client):
        """Test push_prompt request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"version": "v2"}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        client = LangSmithClient(settings)
        client._client = mock_httpx_client

        result = await client.push_prompt(
            "test-prompt",
            {"content": "New content", "metadata": {}},
        )

        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert call_args.kwargs["method"] == "POST"
        assert call_args.kwargs["url"] == "/v1/prompts/test-prompt"

    @pytest.mark.asyncio
    async def test_fetch_runs(self, settings, mock_httpx_client):
        """Test fetch_runs request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"runs": []}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        client = LangSmithClient(settings)
        client._client = mock_httpx_client

        result = await client.fetch_runs(project_id="proj-123", limit=50)

        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert call_args.kwargs["params"]["project_id"] == "proj-123"
        assert call_args.kwargs["params"]["limit"] == 50

    @pytest.mark.asyncio
    async def test_list_projects(self, settings, mock_httpx_client):
        """Test list_projects request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"projects": []}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        client = LangSmithClient(settings)
        client._client = mock_httpx_client

        result = await client.list_projects()

        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert call_args.kwargs["url"] == "/v1/projects"

    @pytest.mark.asyncio
    async def test_get_thread_history(self, settings, mock_httpx_client):
        """Test get_thread_history request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"messages": []}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        client = LangSmithClient(settings)
        client._client = mock_httpx_client

        result = await client.get_thread_history(
            thread_id="thread-123",
            project_name="my-project",
            page_number=2,
            max_chars=50000,
        )

        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert "threads/thread-123/history" in call_args.kwargs["url"]
        assert call_args.kwargs["params"]["project_name"] == "my-project"
        assert call_args.kwargs["params"]["page_number"] == 2

    @pytest.mark.asyncio
    async def test_list_datasets(self, settings, mock_httpx_client):
        """Test list_datasets request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"datasets": []}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        client = LangSmithClient(settings)
        client._client = mock_httpx_client

        result = await client.list_datasets()

        mock_httpx_client.request.assert_called_once()
        assert result == {"datasets": []}

    @pytest.mark.asyncio
    async def test_get_dataset(self, settings, mock_httpx_client):
        """Test get_dataset request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "ds-123", "name": "Test Dataset"}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        client = LangSmithClient(settings)
        client._client = mock_httpx_client

        result = await client.get_dataset("ds-123")

        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert call_args.kwargs["url"] == "/v1/datasets/ds-123"

    @pytest.mark.asyncio
    async def test_create_dataset(self, settings, mock_httpx_client):
        """Test create_dataset request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "ds-new", "name": "New Dataset"}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        client = LangSmithClient(settings)
        client._client = mock_httpx_client

        result = await client.create_dataset(
            name="New Dataset",
            description="Test description",
            data_type="kv",
        )

        mock_httpx_client.request.assert_called_once()
        call = mock_httpx_client.request.call_args
        assert call.kwargs["method"] == "POST"
        assert call.kwargs["json"]["name"] == "New Dataset"

    @pytest.mark.asyncio
    async def test_list_experiments(self, settings, mock_httpx_client):
        """Test list_experiments request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"experiments": []}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        client = LangSmithClient(settings)
        client._client = mock_httpx_client

        result = await client.list_experiments(dataset_id="ds-123")

        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert call_args.kwargs["params"]["dataset_id"] == "ds-123"

    @pytest.mark.asyncio
    async def test_get_billing_usage(self, settings, mock_httpx_client):
        """Test get_billing_usage request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"usage": {}, "costs": {}}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        client = LangSmithClient(settings)
        client._client = mock_httpx_client

        result = await client.get_billing_usage(
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert call_args.kwargs["params"]["start_date"] == "2024-01-01"


class TestLangSmithClientErrors:
    """Tests for LangSmithClient error handling."""

    @pytest.mark.asyncio
    async def test_http_status_error(self, settings):
        """Test HTTP status error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_response.json.return_value = {"error": "Not found"}

        mock_httpx = MagicMock()
        mock_httpx.request = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "404", request=MagicMock(), response=mock_response
            )
        )

        client = LangSmithClient(settings)
        client._client = mock_httpx

        with pytest.raises(LangSmithAPIError) as exc_info:
            await client.list_prompts()

        assert exc_info.value.status_code == 404
        assert "Not found" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_request_error(self, settings, mock_httpx_client):
        """Test network request error handling."""
        from mcp_common.exceptions import MCPServerError

        mock_httpx_client.request = AsyncMock(
            side_effect=httpx.RequestError("Connection failed")
        )

        client = LangSmithClient(settings)
        client._client = mock_httpx_client

        with pytest.raises(MCPServerError) as exc_info:
            await client.list_prompts()

        assert "Connection failed" in str(exc_info.value)
