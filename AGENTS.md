# Repository Guidelines

## Project Structure & Module Organization

- `langsmith_mcp/` contains the MCP server package, including API clients, tool definitions, config handling, and observability helpers.
- `settings/` stores server configuration and `tests/` should mirror the package structure for tool and client coverage.
- Keep examples and operator notes in `README.md` and root docs rather than embedding operational behavior in scripts.

## Build, Test, and Development Commands

- `uv sync --group dev` installs development dependencies.
- Use the documented local server command for stdio or HTTP MCP smoke tests.
- `uv run pytest` runs the test suite.
- `uv run ruff check langsmith_mcp tests` and `uv run ruff format langsmith_mcp tests` cover linting and formatting.

## Coding Style & Naming Conventions

- Use explicit type hints and small tool handlers that delegate to LangSmith client helpers.
- Keep modules snake_case, classes PascalCase, and request/response payloads structured and validated.

## Testing Guidelines

- Add tests for trace retrieval, prompt management, and error handling.
- Prefer mocked LangSmith responses over live-network tests unless the case explicitly needs end-to-end verification.

## Commit & Pull Request Guidelines

- Use focused commits such as `feat(traces): add paginated run lookup`.
- PRs should describe affected tools, commands run, and any auth or config changes.

## Security & Configuration Tips

- Keep LangSmith tokens and workspace identifiers out of version control.
- Scrub trace content and user data from shared logs or examples.

## Bodai integration

When installed alongside the [Bodai ecosystem](https://github.com/lesleslie/bodai),
langsmith-mcp follows the shared cross-repo conventions: Crackerjack for CI/CD
quality gates, the four mcp-common baseline tools (`discover_tools`,
`get_liveness`, `get_readiness`, `health_check_all`), and the MCP wiring
discipline documented in `mahavishnu/.claude/decisions/mcp-backend-wiring-discipline.md`.
No Bodai-specific code is imported at runtime — integration is purely via
shared conventions.
