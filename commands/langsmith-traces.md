---
description: Fetch LangSmith runs/traces for debugging and analysis of LLM calls and chains.
argument-hint: "[--project-id ID] [--trace-id ID] [--run-id ID] [--limit N] [--offset N]"
allowed-tools: mcp__langsmith__fetch_runs, mcp__langsmith__list_projects
---

# /langsmith-traces

Fetch runs and traces from a LangSmith workspace for debugging and analysis.

## Usage

`/langsmith-traces [--project-id ID] [--trace-id ID] [--run-id ID] [--limit N] [--offset N]`

Arguments:

- `--project-id ID`: optional project ID to scope the fetch. Omit to query across the workspace (paginate to discover project IDs first).
- `--trace-id ID`: optional trace ID to fetch all runs belonging to a single trace.
- `--run-id ID`: optional run ID to fetch one specific run.
- `--limit N`: optional cap on the number of returned runs (1..1000, default 100).
- `--offset N`: optional pagination offset.

The narrowest valid filter combination wins; passing `--run-id` alone returns just that run, while omitting all filters returns recent runs across the workspace.

## Workflow

1. If the caller does not know the `project_id`, call `mcp__langsmith__list_projects` first and surface the candidates.
2. Call `mcp__langsmith__fetch_runs` with the supplied filters (`project_id`, `trace_id`, `run_id`, `limit`, `offset`).
3. Summarize the returned runs: inputs, outputs, latency, token counts, error status. Highlight any runs with non-empty `error` fields.
4. If the caller asked about a specific trace, walk through the runs in causal order (root → children) and surface the failing step.

## Example

`/langsmith-traces --project-id proj_abc123 --limit 50`
