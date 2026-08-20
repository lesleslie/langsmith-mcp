---
description: List LangSmith experiments (A/B test results) and fetch a single experiment's results.
argument-hint: <action> [experiment-id] [--limit N] [--offset N]
allowed-tools: mcp__langsmith__list_experiments, mcp__langsmith__get_experiment
---

# /langsmith-experiments

Inspect LangSmith experiments: enumerate experiment results across the workspace, or drill into a single experiment for per-example scores.

## Usage

`/langsmith-experiments <action> [experiment-id] [--limit N] [--offset N]`

Actions:

- `list`: enumerate experiments. Optional `--limit N` (1..1000, default 100) and `--offset N` (default 0).
- `get <experiment-id>`: fetch one experiment's full results — aggregate metrics plus per-example scores.

## Workflow

1. Parse the action (`list` | `get`) and the trailing experiment id if present.
2. Dispatch:
   - `list` → `mcp__langsmith__list_experiments({"limit": ..., "offset": ...})`.
   - `get` → `mcp__langsmith__get_experiment({"dataset_id": <experiment-id>})`. (The server reuses the dataset identifier field for experiment lookup; surface whatever the tool returns verbatim.)
3. For `list`, group by dataset when the response includes dataset metadata, and call out the most recent experiment per dataset.
4. For `get`, summarize aggregate metrics (mean score, pass rate) and highlight the best/worst example per scorer.

## Examples

- `/langsmith-experiments list --limit 20`
- `/langsmith-experiments get exp_abc123`