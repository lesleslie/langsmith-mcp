# LangSmith MCP Server

[![Code style: crackerjack](https://img.shields.io/badge/code%20style-crackerjack-000042)](https://github.com/lesleslie/crackerjack)
[![Runtime: oneiric](https://img.shields.io/badge/runtime-oneiric-6e5494)](https://github.com/lesleslie/oneiric)
[![Framework: FastMCP](https://img.shields.io/badge/framework-FastMCP-0ea5e9)](https://github.com/jlowin/fastmcp)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python: 3.13+](https://img.shields.io/badge/python-3.13%2B-green)](https://www.python.org/downloads/)

MCP server for [LangSmith](https://www.langchain.com/langsmith) observability integration. Provides tools for retrieving conversation history, managing prompts, analyzing traces, and tracking usage.

## Features

| Category | Tools | Purpose |
|----------|-------|---------|
| **Conversation History** | `get_thread_history` | Retrieve threaded message history with character-based pagination |
| **Prompt Management** | `list_prompts`, `get_prompt`, `push_prompt` | Manage LangSmith prompts with versioning |
| **Traces & Runs** | `fetch_runs`, `list_projects` | Debug LLM calls, analyze execution traces |
| **Datasets** | `list_datasets`, `get_dataset`, `list_examples`, `create_dataset`, `create_examples` | Evaluation datasets management |
| **Experiments** | `list_experiments`, `get_experiment` | A/B testing and evaluation results |
| **Billing** | `get_billing_usage` | Cost tracking and usage metrics |
| **Health** | `health_check_cli` | Server health and connectivity status |

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Configuration

### Environment Variables

The server reads two distinct env-var families:

- **`LANGSMITH_*`** — application config (consumed by `LangSmithSettings` in `langsmith_mcp/config.py`)
- **`LANGSMITH_MCP_*`** — HTTP server lifecycle (consumed by `LangSmithConfig` in `langsmith_mcp/__main__.py`)

#### Application config (`LANGSMITH_*`)

```bash
# Required
export LANGSMITH_API_KEY="your-api-key-here"

# Optional — application config
export LANGSMITH_WORKSPACE_ID="your-workspace-id"
export LANGSMITH_API_ENDPOINT="https://api.smith.langchain.com"

# Pagination defaults
export LANGSMITH_MAX_CHARS_PER_PAGE=25000     # 1000..100000
export LANGSMITH_PREVIEW_CHARS=100            # 50..500

# Feature toggles (set / unset to enable / disable)
export LANGSMITH_FEATURES_ENABLED="conversation,prompts,traces,datasets,experiments,billing"

# HTTP client behavior
export LANGSMITH_HTTP_TIMEOUT=30.0            # 5.0..120.0 seconds
export LANGSMITH_MAX_RETRIES=3                # 0..5
```

#### HTTP server lifecycle (`LANGSMITH_MCP_*`)

```bash
# Optional — HTTP transport (mcp-common MCPServerCLIFactory)
export LANGSMITH_MCP_HTTP_PORT=3048
export LANGSMITH_MCP_HTTP_HOST="127.0.0.1"
export LANGSMITH_MCP_ENABLE_HTTP_TRANSPORT=true
```

### Configuration File

Edit `settings/langsmith.yaml` for persistent configuration:

```yaml
server_name: "LangSmith MCP Server"
api_endpoint: "https://api.smith.langchain.com"

# Pagination (flat top-level keys — there is no `pagination:` namespace)
max_chars_per_page: 25000
preview_chars: 100

features_enabled:
  - conversation
  - prompts
  - traces
  - datasets
  - experiments
  - billing

# HTTP client configuration
http_timeout: 30.0
max_retries: 3

# HTTP server (read by `LANGSMITH_MCP_*` env vars / pyproject [tool.langsmith-mcp])
http_port: 3048
http_host: "127.0.0.1"
enable_http_transport: true
```

## Usage

### Start MCP Server

```bash
# Using the CLI
langsmith-mcp start

# Or directly
python -m langsmith_mcp
```

### MCP Tools

#### Get Thread History

```python
# Retrieve conversation history with pagination
result = await get_thread_history({
    "thread_id": "thread_abc123",
    "project_name": "my-project",
    "page_number": 1,
    "max_chars_per_page": 25000
})
```

#### Manage Prompts

```python
# List all prompts
prompts = await list_prompts(limit=100)

# Get specific prompt
prompt = await get_prompt({
    "prompt_identifier": "my-prompt",
    "version": "v1.0.0"  # Optional
})

# Push new prompt version
result = await push_prompt({
    "prompt_identifier": "my-prompt",
    "content": "You are a helpful assistant...",
    "metadata": {"category": "system"}
})
```

#### Analyze Traces

```python
# List projects
projects = await list_projects()

# Fetch runs/traces
runs = await fetch_runs({
    "project_id": "proj_abc123",
    "limit": 100
})
```

#### Manage Datasets

```python
# List datasets
datasets = await list_datasets()

# Create dataset
dataset = await create_dataset({
    "name": "Test Dataset",
    "description": "Evaluation dataset",
    "data_type": "kv"
})

# Add examples
examples = await create_examples({
    "dataset_id": "ds_abc123",
    "examples": [
        {"input": "Hello", "output": "Hi there!"},
        {"input": "Goodbye", "output": "See you later!"}
    ]
})
```

#### Track Usage

```python
# Get billing usage
usage = await get_billing_usage({
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
})
```

## Integration with Mahavishnu Ecosystem

LangSmith MCP integrates with the Bodai ecosystem:

| Component | Integration |
|-----------|-------------|
| **Mahavishnu** | Cost tracking → Routing metrics budget alerts |
| **Akosha** | Trace analysis → Pattern detection across LLM calls |
| **Session-Buddy** | Thread history → Session correlation |

### Example: Cost Integration with Mahavishnu

```python
# In Mahavishnu's CostOptimizer
async def aggregate_costs(self) -> dict:
    """Combine routing costs + LangSmith billing."""
    routing_costs = await self.get_routing_costs()

    # Call LangSmith MCP for billing data
    langsmith_result = await langsmith_mcp.get_billing_usage({})
    langsmith_costs = langsmith_result.get("data", {})

    return self._merge_cost_reports(routing_costs, langsmith_costs)
```

## Development

### Run Tests

```bash
pytest
pytest --cov=langsmith_mcp
```

### Code Quality

```bash
ruff check langsmith_mcp/
ruff format langsmith_mcp/
```

> **Note:** `pyright langsmith_mcp/` is intentionally omitted — this repo does
> not configure `[tool.pyright]` or `pyrightconfig.json`, and running it
> against the source tree will fail with "no configuration found". Type
> checking is delegated to `mypy` (see `mypy.ini`) and `ty` (crackerjack
> gate), not `pyright`.

## Architecture

```
langsmith-mcp/
├── langsmith_mcp/
│   ├── __init__.py       # Package exports
│   ├── __main__.py       # Oneiric CLI entry point
│   ├── config.py         # LangSmithSettings (mcp-common)
│   ├── client.py         # LangSmith API client
│   └── main.py           # FastMCP server + tools
├── settings/
│   └── langsmith.yaml    # Oneiric configuration
├── pyproject.toml
└── README.md
```

## License

BSD-3-Clause
