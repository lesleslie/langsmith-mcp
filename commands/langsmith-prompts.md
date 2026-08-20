---
description: List, fetch, or push LangSmith prompt versions. Use list to discover, get to inspect content, push to publish a new version.
argument-hint: <action> [prompt-identifier] [--version V] [--content TEXT]
allowed-tools: mcp__langsmith__list_prompts, mcp__langsmith__get_prompt, mcp__langsmith__push_prompt
---

# /langsmith-prompts

Manage LangSmith prompts: list existing prompts, fetch a specific version, or push a new version.

## Usage

`/langsmith-prompts <action> [prompt-identifier] [--version V] [--content TEXT]`

Actions:

- `list`: enumerate prompts in the workspace. Optional `--limit N` (default 100) and `--offset N` (default 0).
- `get <prompt-identifier>`: fetch a prompt's content and metadata. Optional `--version V` to pin a version; defaults to the latest.
- `push <prompt-identifier>`: publish a new version. Requires `--content TEXT`; optional `--metadata '{"category":"system"}'` JSON inline.

## Workflow

1. Parse the requested action (`list` | `get` | `push`) and the trailing identifier / flags.
2. Dispatch to the matching MCP tool:
   - `list` → `mcp__langsmith__list_prompts(limit, offset)`.
   - `get` → `mcp__langsmith__get_prompt({"prompt_identifier": ..., "version": ...})`.
   - `push` → `mcp__langsmith__push_prompt({"prompt_identifier": ..., "content": ..., "metadata": ...})`.
3. For `get`, surface both the content body and the version metadata so the caller knows which revision they are looking at.
4. For `push`, return the new version number and the commit metadata so the caller can reference it in subsequent `get` calls.

## Examples

- `/langsmith-prompts list --limit 50`
- `/langsmith-prompts get customer-greeting`
- `/langsmith-prompts push customer-greeting --content "You are a friendly concierge..."`