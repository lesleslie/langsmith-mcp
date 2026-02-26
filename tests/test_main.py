"""Tests for LangSmith MCP Server main tools."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langsmith_mcp.client import LangSmithAPIError
from langsmith_mcp.main import (
    _handle_error,
    get_billing_usage,
    get_dataset,
    get_experiment,
    get_prompt,
    get_thread_history,
    health_check,
    list_datasets,
    list_examples,
    list_experiments,
    list_projects,
    list_prompts,
    push_prompt,
    fetch_runs,
    create_dataset,
    create_examples,
)
from langsmith_mcp.config import LangSmithSettings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    with patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key-12345678"}):
        settings = LangSmithSettings.load("langsmith")
        yield settings


@pytest.fixture
def mock_client():
    """Create mock LangSmith client."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_get_settings(mock_settings):
    """Mock get_settings function."""
    with patch("langsmith_mcp.main.get_settings", return_value=mock_settings):
        yield mock_settings


class TestHandleError:
    """Tests for error handling."""

    def test_langsmith_api_error(self):
        """Test handling LangSmithAPIError."""
        error = LangSmithAPIError(
            message="API failed",
            status_code=400,
            details={"reason": "Bad request"},
        )

        result = _handle_error(error, "test_operation")

        assert result["status"] == "error"
        assert result["error"] == "API failed"
        assert result["status_code"] == 400
        assert result["details"]["reason"] == "Bad request"

    def test_mcp_server_error(self):
        """Test handling MCPServerError."""
        from mcp_common.exceptions import MCPServerError

        error = MCPServerError("Server error")

        result = _handle_error(error, "test_operation")

        assert result["status"] == "error"
        assert result["error"] == "Server error"

    def test_generic_error(self):
        """Test handling generic exception."""
        error = ValueError("Unexpected error")

        result = _handle_error(error, "test_operation")

        assert result["status"] == "error"
        assert "Unexpected error" in result["error"]


class TestHealthCheck:
    """Tests for health check tool."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_client, mock_get_settings):
        """Test successful health check."""
        mock_client.list_projects = AsyncMock(return_value={"projects": []})

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            result = await health_check()

        assert result["status"] == "healthy"
        assert result["server"] == "langsmith-mcp"
        assert result["connectivity"] == "ok"
        assert "config" in result

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, mock_get_settings):
        """Test degraded health check when API unreachable."""
        with patch(
            "langsmith_mcp.main._get_client",
            side_effect=Exception("Connection failed"),
        ):
            result = await health_check()

        assert result["status"] == "degraded"
        assert "error" in result["connectivity"]


class TestPromptTools:
    """Tests for prompt management tools."""

    @pytest.mark.asyncio
    async def test_list_prompts(self, mock_client):
        """Test list_prompts tool."""
        mock_client.list_prompts = AsyncMock(
            return_value={"prompts": [{"name": "test-prompt"}]}
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            result = await list_prompts(limit=50, offset=0)

        assert result["status"] == "success"
        assert "prompts" in result["data"]

    @pytest.mark.asyncio
    async def test_get_prompt(self, mock_client):
        """Test get_prompt tool."""
        mock_client.get_prompt = AsyncMock(
            return_value={"name": "test-prompt", "content": "Hello {{name}}"}
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            from langsmith_mcp.main import PromptInput

            result = await get_prompt(
                PromptInput(prompt_identifier="test-prompt", version="v1")
            )

        assert result["status"] == "success"
        mock_client.get_prompt.assert_called_once_with(
            prompt_identifier="test-prompt",
            version="v1",
        )

    @pytest.mark.asyncio
    async def test_push_prompt(self, mock_client):
        """Test push_prompt tool."""
        mock_client.push_prompt = AsyncMock(
            return_value={"version": "v2", "owner": "test-prompt"}
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            from langsmith_mcp.main import PushPromptInput

            result = await push_prompt(
                PushPromptInput(
                    prompt_identifier="test-prompt",
                    content="New content",
                    metadata={"author": "test"},
                )
            )

        assert result["status"] == "success"
        mock_client.push_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_prompts_error(self):
        """Test list_prompts error handling."""
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(
            side_effect=LangSmithAPIError("API error", status_code=500)
        )
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            result = await list_prompts()

        assert result["status"] == "error"
        assert "API error" in result["error"]


class TestTraceTools:
    """Tests for trace/run tools."""

    @pytest.mark.asyncio
    async def test_fetch_runs(self, mock_client):
        """Test fetch_runs tool."""
        mock_client.fetch_runs = AsyncMock(
            return_value={"runs": [{"id": "run-123"}]}
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            from langsmith_mcp.main import RunsInput

            result = await fetch_runs(
                RunsInput(project_id="proj-123", limit=100)
            )

        assert result["status"] == "success"
        mock_client.fetch_runs.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_projects(self, mock_client):
        """Test list_projects tool."""
        mock_client.list_projects = AsyncMock(
            return_value={"projects": [{"id": "proj-123", "name": "Test Project"}]}
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            result = await list_projects(limit=50)

        assert result["status"] == "success"
        assert "projects" in result["data"]


class TestDatasetTools:
    """Tests for dataset tools."""

    @pytest.mark.asyncio
    async def test_list_datasets(self, mock_client):
        """Test list_datasets tool."""
        mock_client.list_datasets = AsyncMock(
            return_value={"datasets": [{"id": "ds-123"}]}
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            result = await list_datasets()

        assert result["status"] == "success"
        assert "datasets" in result["data"]

    @pytest.mark.asyncio
    async def test_get_dataset(self, mock_client):
        """Test get_dataset tool."""
        mock_client.get_dataset = AsyncMock(
            return_value={"id": "ds-123", "name": "Test Dataset"}
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            from langsmith_mcp.main import DatasetInput

            result = await get_dataset(DatasetInput(dataset_id="ds-123"))

        assert result["status"] == "success"
        mock_client.get_dataset.assert_called_once_with(dataset_id="ds-123")

    @pytest.mark.asyncio
    async def test_list_examples(self, mock_client):
        """Test list_examples tool."""
        mock_client.list_examples = AsyncMock(
            return_value={"examples": [{"id": "ex-123"}]}
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            from langsmith_mcp.main import DatasetInput

            result = await list_examples(DatasetInput(dataset_id="ds-123"))

        assert result["status"] == "success"
        mock_client.list_examples.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_dataset(self, mock_client):
        """Test create_dataset tool."""
        mock_client.create_dataset = AsyncMock(
            return_value={"id": "ds-new", "name": "New Dataset"}
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            from langsmith_mcp.main import CreateDatasetInput

            result = await create_dataset(
                CreateDatasetInput(
                    name="New Dataset",
                    description="Test description",
                    data_type="kv",
                )
            )

        assert result["status"] == "success"
        mock_client.create_dataset.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_examples(self, mock_client):
        """Test create_examples tool."""
        mock_client.create_examples = AsyncMock(
            return_value={"created": 2}
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            from langsmith_mcp.main import CreateExamplesInput

            result = await create_examples(
                CreateExamplesInput(
                    dataset_id="ds-123",
                    examples=[{"input": "test", "output": "result"}],
                )
            )

        assert result["status"] == "success"
        mock_client.create_examples.assert_called_once()


class TestExperimentTools:
    """Tests for experiment tools."""

    @pytest.mark.asyncio
    async def test_list_experiments(self, mock_client):
        """Test list_experiments tool."""
        mock_client.list_experiments = AsyncMock(
            return_value={"experiments": [{"id": "exp-123"}]}
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            from langsmith_mcp.main import ExperimentsInput

            result = await list_experiments(ExperimentsInput(dataset_id="ds-123"))

        assert result["status"] == "success"
        mock_client.list_experiments.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_experiment(self, mock_client):
        """Test get_experiment tool."""
        mock_client.get_experiment = AsyncMock(
            return_value={"id": "exp-123", "results": []}
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            from langsmith_mcp.main import DatasetInput

            result = await get_experiment(DatasetInput(dataset_id="exp-123"))

        assert result["status"] == "success"
        mock_client.get_experiment.assert_called_once_with(experiment_id="exp-123")


class TestBillingTools:
    """Tests for billing tools."""

    @pytest.mark.asyncio
    async def test_get_billing_usage(self, mock_client):
        """Test get_billing_usage tool."""
        mock_client.get_billing_usage = AsyncMock(
            return_value={"usage": {"api_calls": 1000}, "costs": {"total": 10.00}}
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            from langsmith_mcp.main import BillingInput

            result = await get_billing_usage(
                BillingInput(start_date="2024-01-01", end_date="2024-01-31")
            )

        assert result["status"] == "success"
        mock_client.get_billing_usage.assert_called_once()


class TestThreadHistory:
    """Tests for conversation thread history."""

    @pytest.mark.asyncio
    async def test_get_thread_history(self, mock_client):
        """Test get_thread_history tool."""
        mock_client.get_thread_history = AsyncMock(
            return_value={
                "thread_id": "thread-123",
                "messages": [{"role": "user", "content": "Hello"}],
                "page_info": {"current": 1, "total": 1},
            }
        )

        with patch("langsmith_mcp.main._get_client", return_value=mock_client):
            from langsmith_mcp.main import ThreadHistoryInput

            result = await get_thread_history(
                ThreadHistoryInput(
                    thread_id="thread-123",
                    project_name="my-project",
                    page_number=1,
                    max_chars_per_page=25000,
                )
            )

        assert result["status"] == "success"
        mock_client.get_thread_history.assert_called_once()
