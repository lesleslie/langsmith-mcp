# LangSmith MCP Server

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
| **Health** | `health_check` | Server health and connectivity status |

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Configuration

### Environment Variables

```bash
# Required
export LANGSMITH_API_KEY="your-api-key-here"

# Optional
export LANGSMITH_WORKSPACE_ID="your-workspace-id"
export LANGSMITH_API_ENDPOINT="https://api.smith.langchain.com"
```

### Configuration File

Edit `settings/langsmith.yaml` for persistent configuration:

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
pyright langsmith_mcp/
```

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
