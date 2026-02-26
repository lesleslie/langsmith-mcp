"""LangSmith MCP Server - FastMCP Tools.

Provides MCP tools for LangSmith observability integration.
"""

import logging
import sys
from typing import Annotated, Any

from fastmcp import FastMCP
from mcp_common.exceptions import MCPServerError
from pydantic import BaseModel, Field

from langsmith_mcp.client import LangSmithAPIError
from langsmith_mcp.config import LangSmithSettings

logger = logging.getLogger(__name__)

# Initialize settings
_settings: LangSmithSettings | None = None


def get_settings() -> LangSmithSettings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = LangSmithSettings.load("langsmith")
    return _settings


def validate_api_key_at_startup() -> None:
    """Validate LangSmith API key at server startup."""
    try:
        settings = get_settings()
        if not settings.api_key or not settings.api_key.strip():
            print("\n❌ LangSmith API Key Validation Failed", file=sys.stderr)
            print("   LANGSMITH_API_KEY environment variable is not set", file=sys.stderr)
            print("   Set it with: export LANGSMITH_API_KEY='your-key-here'", file=sys.stderr)
            sys.exit(1)

        print(f"\n✅ LangSmith MCP Server Ready", file=sys.stderr)
        print(f"   API Key: {settings.get_masked_api_key()}", file=sys.stderr)
        print(f"   Endpoint: {settings.api_endpoint}", file=sys.stderr)
        print(f"   Features: {', '.join(sorted(settings.features_enabled))}\n", file=sys.stderr)

    except Exception as e:
        print(f"\n❌ Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)


# Initialize FastMCP server
mcp = FastMCP(
    name="LangSmith Observability",
    instructions="MCP server for LangSmith observability - traces, prompts, datasets, and experiments",
)


# =====================
# Input Models
# =====================

class ThreadHistoryInput(BaseModel):
    """Input for thread history retrieval."""

    thread_id: Annotated[str, Field(min_length=1, description="Thread ID to retrieve")]
    project_name: Annotated[str, Field(min_length=1, description="Project name containing the thread")]
    page_number: Annotated[int, Field(ge=1, default=1, description="Page number for pagination")]
    max_chars_per_page: Annotated[
        int,
        Field(
            ge=1000,
            le=100000,
            default=25000,
            description="Maximum characters per page",
        ),
    ]


class PromptInput(BaseModel):
    """Input for prompt operations."""

    prompt_identifier: Annotated[str, Field(min_length=1, description="Prompt name or ID")]
    version: Annotated[str | None, Field(default=None, description="Specific version (latest if None)")]


class PushPromptInput(BaseModel):
    """Input for pushing a prompt."""

    prompt_identifier: Annotated[str, Field(min_length=1, description="Prompt name or ID")]
    content: Annotated[str, Field(min_length=1, description="Prompt content")]
    metadata: Annotated[dict[str, Any] | None, Field(default=None, description="Optional metadata")]


class RunsInput(BaseModel):
    """Input for fetching runs."""

    project_id: Annotated[str | None, Field(default=None, description="Filter by project ID")]
    trace_id: Annotated[str | None, Field(default=None, description="Filter by trace ID")]
    run_id: Annotated[str | None, Field(default=None, description="Get specific run by ID")]
    limit: Annotated[int, Field(ge=1, le=1000, default=100, description="Maximum results")]
    offset: Annotated[int, Field(ge=0, default=0, description="Pagination offset")]


class DatasetInput(BaseModel):
    """Input for dataset operations."""

    dataset_id: Annotated[str, Field(min_length=1, description="Dataset ID")]


class CreateDatasetInput(BaseModel):
    """Input for creating a dataset."""

    name: Annotated[str, Field(min_length=1, max_length=255, description="Dataset name")]
    description: Annotated[str | None, Field(default=None, description="Optional description")]
    data_type: Annotated[str, Field(default="kv", description="Data type (kv, chat, etc.)")]


class CreateExamplesInput(BaseModel):
    """Input for creating examples."""

    dataset_id: Annotated[str, Field(min_length=1, description="Dataset ID")]
    examples: Annotated[
        list[dict[str, Any]],
        Field(min_length=1, description="List of example data"),
    ]


class ExperimentsInput(BaseModel):
    """Input for listing experiments."""

    dataset_id: Annotated[str | None, Field(default=None, description="Filter by dataset ID")]
    limit: Annotated[int, Field(ge=1, le=1000, default=100, description="Maximum results")]
    offset: Annotated[int, Field(ge=0, default=0, description="Pagination offset")]


class BillingInput(BaseModel):
    """Input for billing queries."""

    start_date: Annotated[str | None, Field(default=None, description="Start date (ISO format)")]
    end_date: Annotated[str | None, Field(default=None, description="End date (ISO format)")]


# =====================
# Helper Functions
# =====================

async def _get_client():
    """Get or create LangSmith client."""
    from langsmith_mcp.client import LangSmithClient

    settings = get_settings()
    client = LangSmithClient(settings)
    await client.initialize()
    return client


def _handle_error(e: Exception, operation: str) -> dict[str, Any]:
    """Handle errors and return structured error response."""
    if isinstance(e, LangSmithAPIError):
        logger.error(f"LangSmith API error in {operation}: {e.message}", extra={"details": e.details})
        return {
            "status": "error",
            "error": e.message,
            "status_code": e.status_code,
            "details": e.details,
        }
    elif isinstance(e, MCPServerError):
        logger.error(f"MCP error in {operation}: {e}")
        return {"status": "error", "error": str(e)}
    else:
        logger.exception(f"Unexpected error in {operation}")
        return {"status": "error", "error": f"Unexpected error: {e}"}


# =====================
# Conversation History Tools
# =====================

@mcp.tool()
async def get_thread_history(input_data: ThreadHistoryInput) -> dict[str, Any]:
    """Retrieve message history for a conversation thread.

    Uses character-based pagination to handle large conversation histories.
    Each page contains up to max_chars_per_page characters of message content.

    Args:
        input_data: Thread history parameters with pagination

    Returns:
        Dictionary with thread history, messages, and pagination info
    """
    try:
        client = await _get_client()
        async with client:
            result = await client.get_thread_history(
                thread_id=input_data.thread_id,
                project_name=input_data.project_name,
                page_number=input_data.page_number,
                max_chars=input_data.max_chars_per_page,
            )
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "get_thread_history")


# =====================
# Prompt Management Tools
# =====================

@mcp.tool()
async def list_prompts(
    limit: Annotated[int, Field(ge=1, le=1000, default=100)] = 100,
    offset: Annotated[int, Field(ge=0, default=0)] = 0,
) -> dict[str, Any]:
    """List all prompts in the LangSmith workspace.

    Returns prompt metadata including names, versions, and creation dates.
    Use get_prompt to retrieve the full prompt content.

    Args:
        limit: Maximum number of prompts to return
        offset: Offset for pagination

    Returns:
        Dictionary with list of prompts and pagination info
    """
    try:
        client = await _get_client()
        async with client:
            result = await client.list_prompts(limit=limit, offset=offset)
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "list_prompts")


@mcp.tool()
async def get_prompt(input_data: PromptInput) -> dict[str, Any]:
    """Get a specific prompt by identifier.

    Retrieves the prompt content and metadata. If version is not specified,
    returns the latest version.

    Args:
        input_data: Prompt identifier and optional version

    Returns:
        Dictionary with prompt content, metadata, and version info
    """
    try:
        client = await _get_client()
        async with client:
            result = await client.get_prompt(
                prompt_identifier=input_data.prompt_identifier,
                version=input_data.version,
            )
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "get_prompt")


@mcp.tool()
async def push_prompt(input_data: PushPromptInput) -> dict[str, Any]:
    """Push a new prompt version to LangSmith.

    Creates a new version of an existing prompt or creates a new prompt
    if it doesn't exist. Useful for version control of prompts.

    Args:
        input_data: Prompt identifier, content, and optional metadata

    Returns:
        Dictionary with created prompt version details
    """
    try:
        client = await _get_client()
        async with client:
            prompt_data = {
                "content": input_data.content,
                "metadata": input_data.metadata or {},
            }
            result = await client.push_prompt(
                prompt_identifier=input_data.prompt_identifier,
                prompt_data=prompt_data,
            )
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "push_prompt")


# =====================
# Traces & Runs Tools
# =====================

@mcp.tool()
async def fetch_runs(input_data: RunsInput) -> dict[str, Any]:
    """Fetch runs/traces from LangSmith for debugging and analysis.

    Retrieves execution traces including inputs, outputs, and metadata.
    Filter by project, trace, or specific run ID for targeted analysis.

    Args:
        input_data: Filter parameters and pagination options

    Returns:
        Dictionary with runs data including inputs, outputs, and metadata
    """
    try:
        client = await _get_client()
        async with client:
            result = await client.fetch_runs(
                project_id=input_data.project_id,
                trace_id=input_data.trace_id,
                run_id=input_data.run_id,
                limit=input_data.limit,
                offset=input_data.offset,
            )
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "fetch_runs")


@mcp.tool()
async def list_projects(
    limit: Annotated[int, Field(ge=1, le=1000, default=100)] = 100,
    offset: Annotated[int, Field(ge=0, default=0)] = 0,
) -> dict[str, Any]:
    """List all projects in the LangSmith workspace.

    Projects are containers for traces and runs. Use project IDs
    to filter traces when using fetch_runs.

    Args:
        limit: Maximum number of projects to return
        offset: Offset for pagination

    Returns:
        Dictionary with list of projects
    """
    try:
        client = await _get_client()
        async with client:
            result = await client.list_projects(limit=limit, offset=offset)
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "list_projects")


# =====================
# Dataset Tools
# =====================

@mcp.tool()
async def list_datasets(
    limit: Annotated[int, Field(ge=1, le=1000, default=100)] = 100,
    offset: Annotated[int, Field(ge=0, default=0)] = 0,
) -> dict[str, Any]:
    """List all datasets in the LangSmith workspace.

    Datasets contain examples used for evaluation and testing.
    Use get_dataset to retrieve specific dataset details.

    Args:
        limit: Maximum number of datasets to return
        offset: Offset for pagination

    Returns:
        Dictionary with list of datasets
    """
    try:
        client = await _get_client()
        async with client:
            result = await client.list_datasets(limit=limit, offset=offset)
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "list_datasets")


@mcp.tool()
async def get_dataset(input_data: DatasetInput) -> dict[str, Any]:
    """Get a specific dataset by ID.

    Returns dataset metadata, schema, and statistics.

    Args:
        input_data: Dataset ID

    Returns:
        Dictionary with dataset details
    """
    try:
        client = await _get_client()
        async with client:
            result = await client.get_dataset(dataset_id=input_data.dataset_id)
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "get_dataset")


@mcp.tool()
async def list_examples(
    input_data: DatasetInput,
    limit: Annotated[int, Field(ge=1, le=1000, default=100)] = 100,
    offset: Annotated[int, Field(ge=0, default=0)] = 0,
) -> dict[str, Any]:
    """List examples in a dataset.

    Examples are individual test cases containing inputs and expected outputs.

    Args:
        input_data: Dataset ID
        limit: Maximum number of examples to return
        offset: Offset for pagination

    Returns:
        Dictionary with list of examples
    """
    try:
        client = await _get_client()
        async with client:
            result = await client.list_examples(
                dataset_id=input_data.dataset_id,
                limit=limit,
                offset=offset,
            )
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "list_examples")


@mcp.tool()
async def create_dataset(input_data: CreateDatasetInput) -> dict[str, Any]:
    """Create a new dataset in LangSmith.

    Datasets are used for evaluation and testing of LLM applications.

    Args:
        input_data: Dataset name, description, and data type

    Returns:
        Dictionary with created dataset details
    """
    try:
        client = await _get_client()
        async with client:
            result = await client.create_dataset(
                name=input_data.name,
                description=input_data.description,
                data_type=input_data.data_type,
            )
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "create_dataset")


@mcp.tool()
async def create_examples(input_data: CreateExamplesInput) -> dict[str, Any]:
    """Add examples to a dataset.

    Examples contain inputs and expected outputs for evaluation.

    Args:
        input_data: Dataset ID and list of examples

    Returns:
        Dictionary with created examples
    """
    try:
        client = await _get_client()
        async with client:
            result = await client.create_examples(
                dataset_id=input_data.dataset_id,
                examples=input_data.examples,
            )
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "create_examples")


# =====================
# Experiment Tools
# =====================

@mcp.tool()
async def list_experiments(input_data: ExperimentsInput) -> dict[str, Any]:
    """List experiments in the workspace.

    Experiments are evaluations run against datasets to test
    LLM application performance.

    Args:
        input_data: Filter parameters and pagination options

    Returns:
        Dictionary with list of experiments
    """
    try:
        client = await _get_client()
        async with client:
            result = await client.list_experiments(
                dataset_id=input_data.dataset_id,
                limit=input_data.limit,
                offset=input_data.offset,
            )
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "list_experiments")


@mcp.tool()
async def get_experiment(input_data: DatasetInput) -> dict[str, Any]:
    """Get a specific experiment by ID.

    Returns experiment configuration, results, and metrics.

    Args:
        input_data: Experiment ID (uses dataset_id field)

    Returns:
        Dictionary with experiment details and results
    """
    try:
        client = await _get_client()
        async with client:
            result = await client.get_experiment(experiment_id=input_data.dataset_id)
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "get_experiment")


# =====================
# Billing Tools
# =====================

@mcp.tool()
async def get_billing_usage(input_data: BillingInput) -> dict[str, Any]:
    """Get billing and usage information from LangSmith.

    Returns usage metrics including API calls, tokens, and costs.
    Useful for cost tracking and budget management.

    Args:
        input_data: Optional date range for usage query

    Returns:
        Dictionary with billing and usage data
    """
    try:
        client = await _get_client()
        async with client:
            result = await client.get_billing_usage(
                start_date=input_data.start_date,
                end_date=input_data.end_date,
            )
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "get_billing_usage")


# =====================
# Health Check
# =====================

@mcp.tool()
async def health_check() -> dict[str, Any]:
    """Check LangSmith MCP server health and configuration.

    Returns server status, configuration summary, and connectivity check.

    Returns:
        Dictionary with health status
    """
    try:
        settings = get_settings()

        # Test connectivity by listing projects (lightweight operation)
        client = await _get_client()
        async with client:
            await client.list_projects(limit=1)
            connectivity = "ok"
    except Exception as e:
        connectivity = f"error: {e}"
        settings = get_settings()

    return {
        "status": "healthy" if connectivity == "ok" else "degraded",
        "server": "langsmith-mcp",
        "version": "0.1.0",
        "config": {
            "endpoint": settings.api_endpoint,
            "api_key_masked": settings.get_masked_api_key(),
            "features_enabled": list(settings.features_enabled),
            "max_chars_per_page": settings.max_chars_per_page,
        },
        "connectivity": connectivity,
    }


# Run validation when module is executed directly
if __name__ == "__main__":
    validate_api_key_at_startup()
