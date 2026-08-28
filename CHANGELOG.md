# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-08-28

### Documentation

- readme: Bump Python badge from 3.13+ to 3.14+

### Internal

- Bump requires-python to >=3.14
- claude-md: Add oneiric action-kit discovery breadcrumb
- langsmith-mcp: Bump tool-config pins from 3.13 to 3.14
- langsmith-mcp: Uv python pin 3.14

## [0.3.0] - 2026-08-20

### Added

- langsmith: Bodai plugin conversion (manifest, mcp.json, slash commands)
- tools: Adopt W0 \_apply_tool_profile dispatch (Tier-B 2-tier)

### Changed

- langsmith-mcp: Migrate build backend setuptools → hatchling

### Fixed

- langsmith-mcp: Correct stale module-level docstring in test_tool_profile.py
- langsmith-mcp: Round 1 review fixes for W3.2 tool profile adoption
- langsmith-mcp: Untrack .pyscn/reports/ artifacts

### Internal

- Gitignore runtime artifacts + untrack user-authorized cache files (bodai cleanup 2026-08-17)
- gitignore: Untrack .pyscn/ (bodai 2026-08-20)
- langsmith-mcp: Bootstrap [tool.crackerjack] section + uv sync upgrade
- langsmith-mcp: Gitignore .lycheecache (file, not just dir)
- langsmith-mcp: Gitignore .lycheecache + .hypothesis
- langsmith-mcp: Untrack .lycheecache + .hypothesis runtime artifacts

## [0.2.1] - 2026-08-17

### Documentation

- Fix tool name drift, env-var coverage, and quickstart accuracy

### Internal

- Untrack backup files (.backup, .backup.json, .bak)

## [0.2.0] - 2026-08-12

### Changed

- Drop defensive suppress around get_args()
- Langsmith-mcp (quality: 62/100) - 2026-06-19 05:34:27
- Split load() to reduce C901 complexity

### Fixed

- Address remaining ty errors
- Drop unused type: ignore + correct ty: ignore code

### Internal

- Adopt register_http_health_route from mcp-common
- Bump oneiric dep to >=0.16.0
- Fix FastMCP 3.x test drift surfaced by pin bump
- langsmith-mcp: Fix or remove # type: ignore[no-any-return] straggler
- Migrate MCPBaseSettings → OneiricMCPConfig, bump fastmcp to >=3.4.0,\<4
- Use __version__ instead of hardcoded version literal

## [0.1.6] - 2026-06-19

### Fixed

- Resolve mypy strict errors in main.py and client.py

### Internal

- Add mypy.ini and .cache for quality tooling
- gitignore: Add backup file patterns to silence checkpoint tool artifacts
- Untrack and delete 2 historical *.backup/*.bak files

## [0.1.3] - 2026-02-26

### Fixed

- Use fake test key instead of real-looking key

## [0.1.2] - 2026-02-25

### Changed

- Langsmith-mcp (quality: 51/100) - 2026-02-25 21:16:52
- Update config, core, tests
