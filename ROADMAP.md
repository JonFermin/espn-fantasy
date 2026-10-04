# Roadmap

> Phase-based with dependency tracking. Tasks within a phase are independent.
> Statuses: `TODO` | `IN PROGRESS` | `DONE`
> Priorities: `P0` (must have) | `P1` (should have) | `P2` (nice to have)
> Sizes: `[S]` (small) | `[M]` (medium) | `[L]` (large — consider splitting)
> Flags: `[SPIKE]` — needs investigation before implementation; `[BLOCKED: reason]` — waiting on external dependency
> Dependencies: `depends: #x, #y` — all must be complete before task starts
> Scope: `scope: path/` — relevant files/directories for the task (required)
> Acceptance: `AC:` — machine-verifiable criteria (indented below task)
> Milestones: `MILESTONE:` — marks demo-able project states
> Completed phases are collapsed to one-line summaries.

## Tech Stack
- Language: Python 3.13 (uv-managed)
- Framework: typer CLI · httpx · pydantic · SQLite · polars/numpy/scipy · Playwright for Python (installed Edge/Chrome channel) · anthropic SDK · FastMCP
- Key conventions: follow `CLAUDE.md` invariants. Workers propose and only `executor/` writes to ESPN. **No live ESPN writes during development** (`--dry-run` only). League settings are data. Projections are stat lines. Selectors live only in `src/fm/browser/selectors.py`. Design reference: `docs/DESIGN.md`.
- Shared-file rules:
  - #1 declares every v1 dependency, so later tasks never edit `pyproject.toml` or `uv.lock`.
  - CLI command groups are auto-discovered from `src/fm/commands/` (one file per group), so tasks never edit `cli.py`.
  - #3 defines the full v1 schema.
  - Decision modules register via `src/fm/decide/registry.py` (#7); executor flows register via `src/fm/browser/flows/__init__.py` (#19).
- Build: `uv sync --locked`
- Test: `uv run pytest -q`
- Lint: `uv run ruff check . && uv run pyright`
- Dev: `uv run fm --help`

## Phase 1
- TODO [P0] [M] #1: Scaffold the uv project — scope: pyproject.toml, uv.lock, src/fm/__init__.py, src/fm/cli.py, src/fm/paths.py, src/fm/commands/__init__.py, tests/conftest.py
  - `src/fm` package and typer `fm` entry point that auto-discovers command modules in `src/fm/commands/`.
  - `paths.py`: config and cache dirs, overridable via `FM_CONFIG_DIR` / `FM_CACHE_DIR`.
  - ruff, pyright, and pytest config.
  - Declares all v1 dependencies (httpx, pydantic, typer, rich, playwright, polars, numpy, scipy, nflreadpy, nba_api, espn-api, feedparser, pdfplumber, anthropic, fastmcp, respx, hypothesis).
  - Pins Python 3.13, because `nba_api` supports ≤ 3.13; declares `license = "MIT"`.
  AC: build exits 0; `uv run fm --help` exits 0; test and lint commands pass

## Phase 2
- TODO [P0] [S] #2: Config loading — scope: src/fm/config.py, src/fm/commands/config_cmd.py, tests/test_config.py, tests/fixtures/config.sample.toml — depends: #1
  - Reads `config.toml` + `.env` into pydantic models (leagues, per-league policy, llm, notify).
  - `fm config check` validates the config.
  AC: test command passes for tests/test_config.py (valid, invalid, dir override); `uv run fm config check --path tests/fixtures/config.sample.toml` exits 0
- TODO [P0] [M] #3: SQLite store — scope: src/fm/store/, tests/store/ — depends: #1
  - Migration runner and the full v1 schema: leagues, settings, teams, players, player_ids, roster snapshots, projections (stat lines per source and period), availability, news items, news signals, market values, proposals, executions, decision evals, llm_usage, raw snapshot index.
  - Typed repositories over those tables.
  AC: test command passes for tests/store/ (fresh migrate, idempotent re-migrate, round-trip per table)
- TODO [P0] [M] #4: ESPN ID maps and settings parser — scope: src/fm/espn/ids.py, src/fm/espn/settings.py, tests/espn/test_settings.py, tests/fixtures/espn/ — depends: #1
  - `ids.py`: stat, lineup-slot, position, pro-team, and injury-status maps for `ffl` and `fba`.
  - `settings.py` parses into `LeagueSettings`: scoring items, slot counts, lock type, acquisition/FAAB/waiver timing, trade deadline, playoff weeks, points vs categories.
  AC: test command passes for tests/espn/test_settings.py on NFL PPR, NBA points, and NBA 9-cat fixture settings
- TODO [P0] [M] #5: Browser session and login — scope: src/fm/browser/session.py, src/fm/espn/auth.py, src/fm/commands/login.py, tests/browser/ — depends: #1
  - Persistent Playwright profile in the config dir, using the installed Edge/Chrome channel.
  - `fm login` opens a headed browser for a manual sign-in.
  - Harvests the `espn_s2` / `SWID` cookies and detects session expiry.
  AC: test command passes for tests/browser/ (cookie harvest from a fake context, expiry detection); `uv run fm login --help` exits 0
- TODO [P0] [M] #6: Source adapter base and NFL sources — scope: src/fm/sources/base.py, src/fm/sources/nflverse.py, src/fm/sources/sleeper.py, tests/sources/test_base.py, tests/sources/test_nflverse.py, tests/sources/test_sleeper.py, tests/fixtures/sources/ — depends: #1
  - `base.py`: TTL cache, `as_of` stamps, rate limiting, raw capture.
  - nflverse via `nflreadpy` (Polars; `nfl_data_py` is archived): weekly stats, snap counts, injuries/practice, depth charts, schedules with lines, `ff_playerids`, `ff_opportunity`, `ff_rankings` (FantasyPros ECR).
  - Sleeper: player DB, trending adds/drops, and projections + same-day snaps from the undocumented `api.sleeper.com` endpoints. Degrades gracefully when those break; the old `api.sleeper.app` projections endpoint broke in Sept 2026.
  AC: test command passes for tests/sources/ test_base, test_nflverse, test_sleeper using recorded responses (no network)

## Phase 3
- TODO [P0] [M] #7: Sport plugin interface, NFL plugin, decision registry — scope: src/fm/sports/base.py, src/fm/sports/nfl.py, src/fm/decide/registry.py, tests/sports/test_nfl.py — depends: #4
  - Sport protocol: stat schema, slot eligibility, scoring periods, per-game lock times from the pro schedule.
  - NFL implementation, covering FLEX and OP.
  - `register(sport, kind, fn)` registry for decision modules.
  AC: test command passes for tests/sports/test_nfl.py (FLEX/OP eligibility, lock time per game from a fixture schedule, registry lookup)
- TODO [P0] [M] #8: Proposals and policy — scope: src/fm/proposals/, src/fm/commands/proposals.py, tests/proposals/ — depends: #2, #3
  - Proposal lifecycle (proposed → approved/rejected/expired → executing → verified/failed).
  - Per-kind policy defaults; trade kinds are hard-coded approve-only.
  - Guardrails: untouchables, weekly caps, FAAB % cap, league allowlist, expiry at deadline.
  - Commands: `fm proposals list|approve|reject`, `fm pause|resume`.
  AC: test command passes for tests/proposals/ (trade kinds reject `auto`, expiry, guardrails); `uv run fm proposals list` exits 0 on an empty DB
- TODO [P0] [M] #9: ESPN read client — scope: src/fm/espn/client.py, src/fm/espn/models.py, tests/espn/test_client.py, tests/fixtures/espn/ — depends: #2, #4, #5
  - httpx client on `lm-api-reads.fantasy.espn.com/apis/v3/games/{ffl|fba}/…` for the views in DESIGN §6.1: settings, teams, rosters, matchups, free-agent pool (`X-Fantasy-Filter`), projections, transactions with bids, pending offers (both candidate views), and pro schedules.
  - Timeouts and 429 backoff.
  - Parses responses into typed models and writes raw responses to the cache.
  AC: test command passes for tests/espn/test_client.py with a mocked transport covering every view
- TODO [P0] [M] #10: NFL player ID crosswalk — scope: src/fm/model/ids.py, data/id_overrides.csv, tests/model/test_ids.py — depends: #3, #6
  - Maps ESPN ↔ gsis/sleeper IDs via `ff_playerids`, then applies the overrides file.
  - Reports unmapped players; raises when a rostered player is unmapped.
  AC: test command passes for tests/model/test_ids.py (mapping, overrides, unmapped-rostered gate)
- TODO [P1] [M] #11: NBA source adapters — scope: src/fm/sources/nba_stats.py, src/fm/sources/nba_schedule.py, src/fm/sources/nba_injuries.py, src/fm/sources/darko.py, tests/sources/test_nba_stats.py, tests/sources/test_nba_schedule.py, tests/sources/test_nba_injuries.py, tests/sources/test_darko.py — depends: #6
  - `nba_api` with the full header set and ~0.6 s pacing (home IP only): league game logs, Base/Advanced/Usage splits, on/off.
  - CDN `scheduleLeagueV2.json` → games per day/week and back-to-backs; flags not-yet-scheduled NBA Cup games.
  - ESPN injuries JSON for status; official PDFs for pregame detail (optional).
  - DARKO projections CSV, including projected minutes.
  AC: test command passes for the four NBA source tests using recorded responses
- TODO [P1] [S] #12: Market value adapter — scope: src/fm/sources/market.py, tests/sources/test_market.py — depends: #6
  - FantasyCalc redraft values (keyed by `espnId`) plus ESPN rank and ownership trends, used only for trade-acceptance modeling.
  AC: test command passes for tests/sources/test_market.py with recorded responses
- TODO [P2] [S] #13: Game environment adapters — scope: src/fm/sources/odds.py, src/fm/sources/weather.py, data/stadiums.csv, tests/sources/test_odds.py, tests/sources/test_weather.py — depends: #6
  - ESPN scoreboard first (DraftKings spread/O-U, AccuWeather, indoor flag) → implied team totals.
  - Open-Meteo forecasts at stadium coordinates (`greerreNFL/stadiums`) for outdoor games; The Odds API (free tier) optional for posted team totals.
  AC: test command passes for test_odds and test_weather (implied totals math; domes skip weather)

## Phase 4
- TODO [P0] [SPIKE] [M] #14: Real-league capture [BLOCKED: needs your ESPN league + team IDs in config.toml and a one-time `fm login` — Jon] — scope: scripts/capture/, docs/espn-api.md, tests/fixtures/espn/real/ — depends: #9
  - Fetches every read view for each configured league and saves fixtures with manager names scrubbed.
  - Verifies the community-documented write payloads (DESIGN §6.3) for `ffl` and `fba` by driving each UI flow with `page.route` interception, aborting each request before it reaches ESPN.
  - Settles the open unknowns: pending-offer view, lineup-lock-type key, `espn_s2` expiry, 429 behavior.
  AC: `docs/espn-api.md` has a section per read view and per write flow (both games) plus the resolved unknowns; the test command loads every fixture under tests/fixtures/espn/real/
- TODO [P0] [M] #15: League scoring, projection blend, basic availability — scope: src/fm/model/scoring.py, src/fm/model/projections.py, src/fm/model/availability.py, data/blend_weights.toml, tests/model/test_scoring.py, tests/model/test_projections.py, tests/model/test_availability.py — depends: #6, #7, #9, #10
  - Stat line → fantasy points from league scoring items; categories pass through.
  - Projection-source registry and a per-stat weighted blend of ESPN + Sleeper; weights read from `data/blend_weights.toml`.
  - SD by position.
  - Designation → `p_active`.
  AC: test command passes for the three model tests (PPR, half-PPR, and custom scoring; blend weights; designation mapping)
- TODO [P0] [M] #16: Sync job — scope: src/fm/jobs/sync.py, src/fm/commands/sync.py, tests/jobs/test_sync.py — depends: #3, #6, #9, #10
  - `fm sync` pulls league state and sources into the store, running the unmapped-rostered-player gate.
  AC: test command passes for tests/jobs/test_sync.py (fixture-backed sync populates the store; fails on an unmapped rostered player); `uv run fm sync --help` exits 0
- TODO [P1] [M] #17: NBA plugin and crosswalk — scope: src/fm/sports/nba.py, src/fm/model/ids_nba.py, data/id_overrides_nba.csv, tests/sports/test_nba.py, tests/model/test_ids_nba.py — depends: #7, #10, #11
  - Daily scoring periods, per-game lineup locks, add/drop cutoff at the day's first tip, PG/SG/SF/PF/C/G/F/UTIL eligibility.
  - ESPN ↔ NBA person ID matching by name, team, and position, with overrides and the unmapped gate.
  AC: test command passes for tests/sports/test_nba.py and tests/model/test_ids_nba.py
- TODO [P1] [M] #18: Phone notifications and approvals — scope: src/fm/notify/, src/fm/commands/notify.py, src/fm/commands/bot.py, tests/notify/ — depends: #8
  - Channel interface with two adapters, neither needing inbound ports:
    - Telegram: inline Approve/Reject over long polling; only your chat ID is accepted.
    - ntfy: `http` action buttons posting to a private reply topic; every button carries the proposal's single-use token; a confirmation push follows each decision.
  - `fm notify setup` captures the Telegram chat ID or generates ntfy topics; `fm bot` listens and records decisions.
  - Alert and report helpers.
  - Live use needs a Telegram account + @BotFather token, or the ntfy app; tests mock both.
  AC: test command passes for tests/notify/ with mocked Telegram and ntfy APIs (buttons rendered; callback records approval; other chats and bad tokens ignored; expired proposal can't be approved)
- TODO [P0] [M] #19: Executor framework — scope: src/fm/executor/, src/fm/browser/flows/__init__.py, src/fm/browser/fakes.py, src/fm/commands/execute.py, tests/executor/test_framework.py — depends: #5, #8, #9
  - Flow protocol (API mode + UI mode) and registry.
  - API precondition checks, then run, then verify by API re-read.
  - Write safety: single-use execution token per approved proposal; no automatic write retries; hard timeouts; a timeout marks the execution `UNKNOWN` and forces a re-read.
  - Request/response, screenshots, and Playwright trace written to `audit/`.
  - `--dry-run` stops before any write; fake page and fake transport for tests.
  - `fm execute <proposal> [--dry-run]`.
  AC: test command passes for tests/executor/test_framework.py (failed precondition blocks the run; token can't be reused; timeout → UNKNOWN with no retry; verify mismatch → failed; dry-run sends nothing); `uv run fm execute --help` exits 0

## Phase 5
- TODO [P0] [M] #20: Lineup optimizer — scope: src/fm/decide/lineup.py, tests/decide/test_lineup.py — depends: #7, #15
  - Assignment solve over slot eligibility, maximizing expected points; switches to a win-probability objective when the matchup is lopsided.
  - Respects locks; registers NFL lineup decisions.
  AC: test command passes for tests/decide/test_lineup.py incl. property tests: lineup always legal; no OUT/bye/no-game starter while a legal alternative exists; locked players never moved
- TODO [P0] [M] #21: Valuation and NFL waivers — scope: src/fm/model/valuation.py, src/fm/decide/waivers.py, tests/model/test_valuation.py, tests/decide/test_waivers.py — depends: #15, #16
  - Replacement level = best player on the league's actual wire; ROS value with playoff weeks weighted up.
  - Ranks (add, drop) pairs; FAAB bid heuristic within the policy cap; untouchables protected; timed to the league's waiver run.
  AC: test command passes for test_valuation and test_waivers (replacement level, untouchables never dropped, bid ≤ cap)
- TODO [P1] [M] #22: Full availability model — scope: src/fm/model/availability.py, tests/model/test_availability.py — depends: #11, #15
  - Practice-participation trends (NFL) and official injury report (NBA) shift `p_active`.
  - Claude news signals applied clamped to bounds and logged.
  - Late-game pivot awareness.
  AC: test command passes for tests/model/test_availability.py (trend shifts, clamping, logging)
- TODO [P1] [M] #23: News ingest and relevance — scope: src/fm/sources/news.py, src/fm/model/relevance.py, tests/sources/test_news.py, tests/model/test_relevance.py — depends: #6, #16
  - ESPN news API + RotoWire RSS (poll ≤ every 10 min), deduped.
  - Relevance filter: your roster, this week's opponent, top free agents, trade targets.
  AC: test command passes for test_news and test_relevance
- TODO [P1] [L] #24: NBA valuation — scope: src/fm/model/categories.py, src/fm/model/value_nba.py, tests/model/test_categories.py, tests/model/test_value_nba.py — depends: #15, #17
  - Points: per-game projections from the blend (ESPN + DARKO) × games, with league weights; schedule-aware daily lineup value.
  - Categories: z-scores with volume-weighted FG%/FT% and TO negative; G-scores (`√(σ² + κτ²)`, Rosenof arXiv:2307.02188) for H2H; punt weights.
  AC: test command passes for test_categories and test_value_nba (G-score reduces to z-score when τ = 0; percentage categories are volume-weighted)
- TODO [P0] [M] #25: Transaction writer and lineup flow — scope: src/fm/browser/transactions.py, src/fm/browser/selectors.py, src/fm/browser/flows/lineup.py, tests/executor/test_transactions.py, tests/executor/test_lineup_flow.py — depends: #14, #19
  - `transactions.py`: builds ESPN transaction envelopes, sends them from the Playwright context, and maps ESPN error codes.
  - `set_lineup` in API mode, with a role/text-locator UI fallback; registered with the executor.
  AC: test command passes for both tests (envelopes match the captured payload fixtures; API failure falls back to UI mode on a fake page; dry-run sends no POST); `uv run fm execute --dry-run` on a fixture proposal exits 0

## Phase 6 — MILESTONE: NFL manager end-to-end (recommend → approve → execute → verify on a schedule); NBA daily lineups + streaming proposals
- TODO [P0] [S] #26: Advisor CLI — scope: src/fm/commands/advise.py, src/fm/render/, tests/test_advise_cli.py — depends: #16, #20, #21
  - `fm status`, `fm lineup`, `fm waivers` render recommendations with engine numbers.
  AC: the three commands exit 0 against a fixture-backed store (`FM_CONFIG_DIR=tests/fixtures/home`); snapshot tests pass
- TODO [P0] [M] #27: Add/drop and waiver flows — scope: src/fm/browser/flows/add_drop.py, src/fm/browser/flows/waiver.py, src/fm/browser/selectors.py, tests/executor/test_add_drop.py, tests/executor/test_waiver.py — depends: #25
  - Free-agent add/drop and waiver claim with bid, plus cancel. API mode with UI fallback, preconditions (NBA adds before the day's first tip; no duplicate claims), and API verification.
  AC: test command passes for test_add_drop and test_waiver against a fake page (preconditions, verification, dry-run)
- TODO [P1] [S] #28: Selector canary — scope: src/fm/browser/canary.py, src/fm/commands/canary.py, tests/browser/test_canary.py — depends: #25
  - Read-only: asserts every registered selector resolves on the roster, free-agent, and trade pages, and that every read view still parses. Alerts on drift.
  AC: test command passes for tests/browser/test_canary.py (missing selector → alert payload); `uv run fm canary --help` exits 0
- TODO [P0] [M] #29: Tick and Windows scheduler — scope: src/fm/jobs/tick.py, src/fm/jobs/deadlines.py, src/fm/jobs/scheduler_windows.py, src/fm/commands/schedule.py, tests/jobs/test_tick.py, tests/jobs/test_scheduler_windows.py — depends: #8, #16, #19, #20, #21
  - Each tick: computes deadlines from pro schedules and league settings (international and holiday slots, the NBA first-tip add cutoff, the waiver run), runs due registered decisions once, and executes approved/auto proposals at their time.
  - Auto-bench fires only when unanswered at T-15; missed-window alerts.
  - `fm schedule install|uninstall|show` via schtasks + `.cmd` wrapper + wake/battery settings (port from software-factory).
  AC: test command passes for test_tick and test_scheduler_windows (rendered commands only, nothing installed); `uv run fm schedule show` exits 0
- TODO [P1] [M] #30: Advisor client and news triage — scope: src/fm/advisor/client.py, src/fm/advisor/news_triage.py, src/fm/advisor/prompts/, tests/advisor/test_client.py, tests/advisor/test_news_triage.py — depends: #2, #23
  - anthropic SDK client (`claude-opus-5-5`, effort per worker, structured outputs via `messages.parse`, `stop_reason` checks, refusal fallback, prompt caching, Batches for overnight work, `llm_usage` tracking, daily budget cap).
  - News triage → stored signals.
  AC: test command passes for tests/advisor/ with a stubbed client (parsed output stored; refusal/max_tokens handled; budget cap blocks calls)
- TODO [P1] [L] #31: NBA daily lineups and streaming — scope: src/fm/decide/lineup_daily.py, src/fm/decide/streaming.py, tests/decide/test_lineup_daily.py, tests/decide/test_streaming.py — depends: #20, #24
  - Daily lineups across the matchup week, honoring any games-played limit.
  - Open slot-day detection.
  - Add/drop sequence under the acquisition limit as a daily knapsack/DP; adds land before the day's first tip; core players protected.
  - Category leagues rank streamers by Σ ∂P(win cat)/∂stat × production × open-slot games, with P(win cat) ≈ Φ(Δμ/σ).
  AC: test command passes for test_lineup_daily and test_streaming (fills open slot-days; respects acquisition and games-played limits; never drops protected players; adds scheduled before first tip)
- TODO [P1] [M] #32: Season simulator — scope: src/fm/model/simulate.py, tests/model/test_simulate.py — depends: #15, #21
  - Monte Carlo over the remaining schedule → P(win week), P(playoffs), P(bye), P(title); per-category win probabilities for NBA.
  AC: test command passes for tests/model/test_simulate.py (seeded runs reproducible; playoff probabilities sum to playoff spots within tolerance)
- TODO [P1] [M] #33: Backtest harness — scope: src/fm/eval/backtest.py, src/fm/commands/backtest.py, tests/eval/test_backtest.py, tests/fixtures/backtest/ — depends: #15, #20
  - Replays past weeks → projection MAE by position, lineup efficiency (actual ÷ hindsight-optimal), start/sit regret.
  - Computes your historical lineup-efficiency baseline.
  AC: `uv run fm backtest --sport nfl --fixtures tests/fixtures/backtest` exits 0 and prints the three metrics; test command passes
- TODO [P2] [S] #34: FAAB bid model — scope: src/fm/decide/faab.py, src/fm/decide/waivers.py, tests/decide/test_faab.py — depends: #14, #21
  - Fits winning-bid behavior from league transaction history and replaces the heuristic.
  AC: test command passes for tests/decide/test_faab.py (bid monotonic in value; cap respected)
- TODO [P2] [S] #35: Rest-of-season rankings sheet — scope: src/fm/decide/rankings.py, src/fm/commands/rankings.py, tests/decide/test_rankings.py — depends: #21, #24
  - League-tuned ROS values for rostered players and free agents in both leagues (points, or G-scores for categories) → CSV.
  - A sanity check for waiver and trade calls; replaces the draft sheet, since both drafts are done.
  AC: `uv run fm rankings --fixtures tests/fixtures/espn` writes a CSV per league; test command passes

## Phase 7
- TODO [P1] [M] #36: Close-call and explain workers — scope: src/fm/advisor/close_call.py, src/fm/advisor/explain.py, tests/advisor/test_close_call.py, tests/advisor/test_explain.py — depends: #20, #30
  - Near-tie tie-break with the web search tool over a domain allowlist and cited sources.
  - Rationale per non-trivial proposal; trivial moves use templates.
  AC: test command passes for both tests with a stubbed client (allowlist configured; templated path makes no API call)
- TODO [P1] [M] #37: NBA category planner — scope: src/fm/decide/weekly.py, tests/decide/test_weekly.py — depends: #24, #32
  - Weekly per-category win probabilities → targets, punts (H-score-style roster-aware re-weighting, arXiv:2409.09884), and streamer stat targets.
  AC: test command passes for tests/decide/test_weekly.py (punt recommended below the threshold win probability)
- TODO [P1] [L] #38: Trade evaluator and finder — scope: src/fm/decide/trades.py, src/fm/commands/trade.py, tests/decide/test_trades.py — depends: #12, #32
  - `fm trade eval|find`: Δ ROS value for both sides, Δ title odds, legality.
  - Enumerates 1:1, 2:1, and 2:2 deals per opponent; screens, then re-scores; ranks by Δ title odds × P(accept), where P(accept) uses market values + their needs.
  AC: test command passes for tests/decide/test_trades.py (symmetric trade ≈ 0 Δ; finder returns only legal trades in ranked order); `uv run fm trade eval --help` exits 0
- TODO [P2] [S] #39: Blend weight tuning — scope: src/fm/eval/tune.py, data/blend_weights.toml, tests/eval/test_tune.py — depends: #33
  - Fits per-(source, position) weights on held-out weeks.
  AC: test command passes for tests/eval/test_tune.py (held-out MAE ≤ equal-weight MAE on fixtures)
- TODO [P2] [M] #40: MCP server — scope: src/fm/mcp_server.py, tests/test_mcp_server.py — depends: #8, #26
  - FastMCP read tools (status, lineup, waivers, trade eval) plus `create_proposal`; no execute tool.
  AC: test command passes for tests/test_mcp_server.py (tool list has no write/execute tool; `create_proposal` stores a proposal)
- TODO [P2] [M] #41: UI fallback drills — scope: src/fm/browser/drills.py, src/fm/commands/drill.py, tests/browser/test_drills.py — depends: #14, #27
  - Weekly dry-run of each UI-mode flow against the live site, stopping before the final confirm, so the fallback is known-good when API mode breaks. Alerts on failure.
  AC: test command passes for tests/browser/test_drills.py against a fake page; `uv run fm drill --help` exits 0
- TODO [P2] [M] #42: NFL opportunity baseline — scope: src/fm/model/baseline_nfl.py, tests/model/test_baseline_nfl.py — depends: #13, #15, #33
  - Shares × implied team total × regressed efficiency, registered as a projection source.
  AC: test command passes; the backtest on fixtures shows the blend with the baseline no worse than without it
- TODO [P2] [M] #43: NBA minutes baseline — scope: src/fm/model/baseline_nba.py, tests/model/test_baseline_nba.py — depends: #11, #24, #33
  - Minutes × per-minute rates, registered as a projection source.
  - Teammates-out redistribution from without-player splits + on/off data, with DARKO minutes as the prior; blowout risk and back-to-back rest risk.
  AC: test command passes; backtest on fixtures no worse than without

## Phase 8 — MILESTONE: trades end-to-end + weekly report
- TODO [P1] [M] #44: Trade flows and offer handling — scope: src/fm/browser/flows/trade.py, src/fm/browser/selectors.py, src/fm/decide/offers.py, tests/executor/test_trade_flow.py, tests/decide/test_offers.py — depends: #25, #29, #38
  - Propose/respond/cancel flows (approval-only), API mode with UI fallback.
  - Never duplicates an open offer; checks lock status for every player involved.
  - Pending incoming offers become evaluation proposals registered for the tick.
  AC: test command passes for both tests (fake page; `auto` policy rejected for trade kinds; duplicate offers and locked-player trades blocked)
- TODO [P1] [M] #45: Trade pitch and weekly strategist workers — scope: src/fm/advisor/pitch.py, src/fm/advisor/strategist.py, tests/advisor/test_pitch.py, tests/advisor/test_strategist.py — depends: #30, #32, #38
  - Pitch draft per approved trade idea.
  - Weekly priorities, punts, and targets → proposals.
  AC: test command passes for both tests with a stubbed client
- TODO [P1] [S] #46: Weekly report — scope: src/fm/jobs/report.py, src/fm/commands/report.py, tests/jobs/test_report.py — depends: #16, #18, #36
  - `fm report`: matchup outlook, playoff odds, moves made, upcoming deadlines; markdown + phone channel.
  AC: `uv run fm report` against fixtures writes a markdown report; test command passes

## Phase 9
- TODO [P2] [S] #47: Live report card — scope: src/fm/eval/report_card.py, tests/eval/test_report_card.py — depends: #33, #46
  - Weekly lineup efficiency, bench points, and pickup value on real decisions, appended to the report.
  AC: test command passes for tests/eval/test_report_card.py
