# CLAUDE.md

Personal ESPN fantasy manager for NFL (`ffl`) and NBA (`fba`). Read `docs/DESIGN.md` before changing anything; build
order and status live in `ROADMAP.md`.

## Commands

- `uv sync`: install
- `uv run pytest`: tests (single test: `uv run pytest tests/path/test_x.py::test_name`)
- `uv run ruff check . && uv run pyright`: lint + types
- `uv run fm <command>`: CLI

## Invariants

- **Workers propose, the executor acts.** Only `src/fm/executor/` (via `src/fm/browser/`) may write to ESPN. Never
  give `advisor/`, `decide/`, the MCP server, `fm.web` (the dashboard), `fm.setup`, `fm.claude`, or any LLM-facing
  tool a write path: they read, record human decisions, edit `config.toml` or store `proposed` rows. Trades are
  approval-only, and that must never become configurable.
- **No live ESPN writes during development.** Run executor flows with `--dry-run` unless the task explicitly says
  otherwise and the user approved it. A live write changes a real league.
- **Dev never touches the live installation.** Never read or write the real `~/.config/espn-fantasy/` or the separate
  clone the live `fm run` runs from; use `FM_CONFIG_DIR`/`FM_CACHE_DIR` temp dirs, and no real scheduler or `claude`
  CLI in tests.
- **Verify every write** by re-reading through the API. Unverified means failed.
- **League settings are data.** No hardcoded scoring, roster slots, lock times, waiver times, or deadlines.
- **Projections are stat lines.** Points are computed per league from its scoring items.
- **Claude adjustments are bounded, cited, and logged.** A Claude-only signal never triggers a drop or a trade.
- **Selectors** live only in `src/fm/browser/selectors.py`. Flows use role/text locators.

## Paths

- Config, state DB, browser profile, audit artifacts: `~/.config/espn-fantasy/` on Windows and macOS alike. Not
  `%APPDATA%`, because MSIX virtualizes it. Also there: `run.lock` (held by `fm run`), `dashboard.url` (the
  dashboard's one-time login link), `tick-state.json`, and `logs/run.log`.
- Runs on Windows and macOS. Platform-specific code stays behind a backend (`fm schedule` and `fm autostart`:
  `jobs/scheduler_windows.py` / `jobs/scheduler_macos.py` over `jobs/scheduler_base.py`) or in `jobs/runlock.py` and
  `jobs/proctree.py`; everything else must stay portable. CI runs the suite on both.
- Deletable cache: `~/.cache/espn-fantasy/`.
- Never commit secrets, cookies, browser profiles, or unscrubbed fixtures (real manager names).

## Testing

- Unit tests run offline against recorded ESPN fixtures in `tests/fixtures/`. No network in unit tests.
- `tests/test_journey.py` walks setup, doctor, `fm run`, a dashboard approval and the execution end to end, offline.
- Lineup outputs must be legal: no OUT, bye, or no-game starter while an alternative exists, and no moving locked
  players.

## Claude API

- Model `claude-opus-5-5`, with effort set per worker. Structured outputs via `client.messages.parse`; check
  `stop_reason` before reading. Load the `claude-api` skill before editing `src/fm/advisor/`.
