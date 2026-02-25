# CLAUDE.md

This file provides guidance for Claude Code when working with code in this repository.

## Project Overview

LangSmith MCP Server is a Model Context Protocol server for LangSmith observability integration. It provides tools for retrieving conversation history, managing prompts, analyzing traces, and tracking usage.

## Ecosystem Context

This server is part of the **Bodai Ecosystem**:

| Component | Role | Port | Integration |
|-----------|------|------|-------------|
| **LangSmith MCP** | Observability | 3048 | This server |
| **Mahavishnu** | Orchestrator | 8680 | Cost aggregation |
| **Akosha** | Seer | 8682 | Trace pattern detection |
| **Session-Buddy** | Builder | 8678 | Thread correlation |

## Architecture

### Technology Stack

- **FastMCP**: MCP server implementation with `@mcp.tool()` decorators
- **mcp-common**: Configuration base classes (`MCPServerSettings`), CLI patterns
- **Oneiric**: Layered configuration loading, runtime management
- **httpx**: Async HTTP client for LangSmith API

### Key Patterns

**Configuration** (`config.py`):
- Extends `MCPServerSettings` from mcp-common
- Uses Oneiric layered loading: defaults → YAML → env vars
- Pydantic validation for all settings

**API Client** (`client.py`):
- Async context manager pattern (`async with client:`)
- Retry logic with tenacity
- Structured error handling with `LangSmithAPIError`

**MCP Tools** (`main.py`):
- Pydantic input models for validation
- Consistent error handling via `_handle_error()`
- Lazy client initialization

## Development Commands

### Environment Setup

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=langsmith_mcp --cov-report=html

# Run specific test
pytest tests/test_client.py -v
```

### Code Quality

```bash
# Format code
ruff format langsmith_mcp/

# Lint
ruff check langsmith_mcp/
ruff check --fix langsmith_mcp/

# Type check
pyright langsmith_mcp/

# All checks via Crackerjack
crackerjack run
```

### MCP Server

```bash
# Start MCP server
python -m langsmith_mcp

# Or using Oneiric CLI
langsmith-mcp start

# Check health
langsmith-mcp health
```

## Configuration Files

### settings/langsmith.yaml

Main configuration file with Oneiric patterns:

```yaml
server_name: "LangSmith MCP Server"
api_endpoint: "https://api.smith.langchain.com"

pagination:
  max_chars_per_page: 25000

features_enabled:
  - conversation
  - prompts
  - traces
  - datasets
  - experiments
  - billing
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LANGSMITH_API_KEY` | LangSmith API key | Yes |
| `LANGSMITH_WORKSPACE_ID` | Workspace ID | No |
| `LANGSMITH_API_ENDPOINT` | API endpoint | No |
| `LANGSMITH_MCP_HTTP_PORT` | Server port | No (default: 3048) |

## Key Implementation Notes

### Character-Based Pagination

LangSmith uses character-based pagination for large responses:

```python
@mcp.tool()
async def get_thread_history(input_data: ThreadHistoryInput) -> dict:
    # Uses max_chars_per_page for pagination
    result = await client.get_thread_history(
        thread_id=input_data.thread_id,
        project_name=input_data.project_name,
        page_number=input_data.page_number,
        max_chars=input_data.max_chars_per_page,
    )
```

### Error Handling Pattern

All tools follow this pattern:

```python
@mcp.tool()
async def tool_name(input_data: InputModel) -> dict[str, Any]:
    try:
        client = await _get_client()
        async with client:
            result = await client.operation(...)
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "tool_name")
```

### Feature Toggles

Check feature availability before using:

```python
settings = get_settings()
if settings.is_feature_enabled("billing"):
    result = await client.get_billing_usage()
```

## MCP Tool Categories

### Conversation History (1 tool)
- `get_thread_history` - Retrieve threaded messages with pagination

### Prompts (3 tools)
- `list_prompts` - List all prompts
- `get_prompt` - Get specific prompt with optional version
- `push_prompt` - Push new prompt version

### Traces (2 tools)
- `fetch_runs` - Fetch runs/traces with filters
- `list_projects` - List all projects

### Datasets (5 tools)
- `list_datasets` - List all datasets
- `get_dataset` - Get dataset details
- `list_examples` - List examples in dataset
- `create_dataset` - Create new dataset
- `create_examples` - Add examples to dataset

### Experiments (2 tools)
- `list_experiments` - List experiments
- `get_experiment` - Get experiment details

### Billing (1 tool)
- `get_billing_usage` - Get usage and costs

### Health (1 tool)
- `health_check` - Server health status

## Common Tasks

### Adding a New Tool

1. Create input model in `main.py`:
```python
class NewToolInput(BaseModel):
    param: Annotated[str, Field(min_length=1, description="Parameter")]
```

2. Add client method in `client.py`:
```python
async def new_operation(self, param: str) -> dict[str, Any]:
    return await self._request("GET", f"/v1/endpoint/{param}")
```

3. Create MCP tool in `main.py`:
```python
@mcp.tool()
async def new_tool(input_data: NewToolInput) -> dict[str, Any]:
    try:
        client = await _get_client()
        async with client:
            result = await client.new_operation(input_data.param)
            return {"status": "success", "data": result}
    except Exception as e:
        return _handle_error(e, "new_tool")
```

### Updating Configuration

1. Add field to `LangSmithSettings` in `config.py`
2. Update `settings/langsmith.yaml` with default
3. Document in README.md

## Security

- API key loaded from environment variable only
- Never log full API key (use `get_masked_api_key()`)
- All inputs validated with Pydantic
- HTTP requests use timeout and retry limits

## Related Documentation

- [LangSmith API Docs](https://docs.smith.langchain.com/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [mcp-common](https://github.com/lesleslie/mcp-common)
- [Oneiric](https://github.com/lesleslie/oneiric)
