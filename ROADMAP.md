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

## Phase 1 — DONE
Built: uv project (`espn-fantasy`, package `fm`, Python 3.13 pinned, MIT) declaring every v1 dependency, with the typer `fm` root that auto-discovers command modules, `fm version` as the reference command, `fm.paths` for config/cache/state/profile/audit dirs with `FM_CONFIG_DIR`/`FM_CACHE_DIR` overrides, pytest/ruff/pyright config, and a test harness (18 tests). Patterns: one module per command group in `src/fm/commands/` exposing `register(root)` — nobody edits `cli.py`; all mutable paths come from `fm.paths`; `tests/conftest.py` autouse fixtures isolate dirs per test, drop inherited API keys, and block non-loopback sockets (sync and asyncio) so unit tests stay offline. Key files: pyproject.toml, uv.lock, src/fm/cli.py, src/fm/paths.py, src/fm/commands/__init__.py, src/fm/commands/version.py, tests/conftest.py. Skipped: none. (1/1 tasks completed, 0 skipped)

## Phase 2 — DONE
Built: `fm.config` loads `config.toml` + the `.env` beside it into frozen pydantic models (`Config` → `League` with `Policy`, `Llm`, `Notify`, `Secrets`), rejecting unknown keys, `auto` outside `bench_inactive`/`lineup`, any `trade*` policy key, duplicate leagues and non-positive IDs with one located line per problem, plus `fm config check [--path]`; `fm.store` with `Database` (autocommit sqlite3, WAL, foreign keys, `BEGIN IMMEDIATE` transactions nesting as savepoints), numbered `NNNN_name.sql` migrations recorded in `schema_migrations` (`0001_v1_schema.sql`: every v1 table, all STRICT with CHECK constraints and cascading FKs), frozen row models with fixed-width UTC timestamps and self-parsing JSON columns, one typed repository per table, and the `Store.open()` facade that migrates on open; `fm.espn.ids` read-only stat/slot/position/pro-team/injury maps for `ffl` and `fba` with a `Game` enum, and `fm.espn.settings` parsing `mSettings` into `LeagueSettings` (scoring items, points vs categories, slots, lock type, acquisition/FAAB/waiver timing, trade deadline, playoff periods); `fm.browser.session` persistent profile on the installed Edge-then-Chrome channel, `fm.espn.auth` cookie harvest / expiry / renewal (clears espn.com cookies and waits for a new `espn_s2`), and `fm login [--check] [--channel] [--timeout]`; `fm.sources.base` `Source`/`HttpSource` with TTL cache, `as_of` stamps, rate limiting, raw capture under `cache/sources/`, bounded 429/5xx retries and `Fetched[T]` (cached/stale/degraded/warnings), with nflverse (`nflreadpy`, 8 datasets) and Sleeper (documented + undocumented endpoints, graceful degradation) adapters over recorded fixtures. 292 tests. Patterns: secrets never echo (SecretStr, `.env` errors name the path only); every table has a frozen model whose field names equal its columns; ESPN ids and league settings are data, never literals; `SessionStatus` is OK/EXPIRING/EXPIRED and a missing session is `NotLoggedInError`; adapters never raise on a degraded upstream — they serve the last good copy as stale or an empty degraded result, and rejected payloads never overwrite a good copy; `tests/browser/conftest.py` fails any test that would launch a real browser; `tests/conftest.py` drops every `.env` secret. Key files: src/fm/config.py, src/fm/commands/config_cmd.py, src/fm/store/db.py, src/fm/store/models.py, src/fm/store/repos.py, src/fm/store/migrations/0001_v1_schema.sql, src/fm/espn/ids.py, src/fm/espn/settings.py, src/fm/browser/session.py, src/fm/espn/auth.py, src/fm/commands/login.py, src/fm/sources/base.py, src/fm/sources/nflverse.py, src/fm/sources/sleeper.py, tests/browser/conftest.py. Skipped: none. (5/5 tasks completed, 0 skipped)

## Phase 3 — DONE
Built: `fm.sports.base` `SportPlugin` (stat schema, slot eligibility, scoring-period windows and per-game or first-game lock times computed from a pro schedule through the structural `ScheduleLike`/`GameLike` protocol that `fm.espn.models.ProSchedule` satisfies, `plugin_for(sport)` discovery) with the NFL plugin (`fm.sports.nfl`: FLEX and OP, weekly periods, kickoff locks) and the `fm.decide.registry` `(sport, kind) -> callable` decision registry (`register`/`decision`/`lookup`); `fm.proposals` (typed per-kind payloads, `ProposalKind` + `KindSpec` with the four trade kinds hard-coded approve-only, `evaluate()` returning a `Verdict` that lists every guardrail reason at once: league allowlist, kind off, auto without deadline, deadline passed, pause, untouchables, weekly cap, FAAB cap; the propose → approve/reject → expire → executing → verified/failed lifecycle with single-use execution tokens, lazy expiry, T-15 auto-approval and the `paused.json` kill switch) plus `fm proposals list|approve|reject`, `fm pause`, `fm resume`; `fm.espn.client.EspnClient` (httpx over `lm-api-reads`, every DESIGN §6.1 read view incl. both pending-offer candidates, timeouts, pacing, bounded 429/5xx retries honoring Retry-After, 401 → `EspnAuthError`, every body captured under `cache/espn/` with a `.meta.json`, never cookies) and `fm.espn.models` (frozen camelCase-mirroring models: teams, rosters, pool entries, stat lines, matchups, transactions, pro schedules); the NFL crosswalk `fm.model.ids` (ff_playerids → `data/id_overrides.csv` → derived D/ST ids, O(1) both ways, `check_rostered` raising `UnmappedPlayersError`, saved by atomic `PlayerIdRepo.replace_sport`); NBA adapters `fm.sources.nba_stats` (nba_api with the full header set, 0.6 s pacing), `nba_schedule` (CDN `scheduleLeagueV2.json`: games per ET day and week, back-to-backs, unscheduled NBA Cup knockouts flagged), `nba_injuries` (ESPN JSON plus optional official PDFs), `darko` (public Google Sheet exports, per-game lines from per-100 talent × pace × minutes); `fm.sources.market` (FantasyCalc redraft values snapped to its published grid plus ESPN rank/ownership for ffl and fba, joined by `market_values`); `fm.sources.odds` (ESPN scoreboard lines, AccuWeather and indoor flag → implied team totals; optional The Odds API, key never cached) and `fm.sources.weather` (`data/stadiums.csv`, Open-Meteo hourly, domes skip the request). 654 tests. Patterns: lock rules come only from `LeagueSettings.lineup_lock_type` and `LockType.UNKNOWN` raises instead of guessing (`FIRSTGAME_SCORINGPERIOD` is ESPN's own first-game value; `FIRST_GAME_OF_WEEK` stays until #14); sport plugins are found by `plugin_for(sport)` → `fm.sports.<sport>.PLUGIN` and decision modules register with `fm.decide.registry.register(sport, kind, fn)`; trade kinds have no policy field anywhere; only `begin_execution(store, id, token)` moves a proposal to executing and `PausedError` is a refusal; the ESPN client never writes to the store (`EspnRead.capture` is what the sync job indexes); committed data files resolve through `fm.paths.data_file()`; no SQL outside `fm.store` (`PlayerIdRepo.for_sport`/`replace_sport`); adapters keep the Phase 2 degradation contract and their fixtures live under `tests/fixtures/sources/<source>/` (`tests/fixtures/sports/` for pro schedules); `tests/test_public_names.py` spans `fm.sports`, `fm.decide`, `fm.proposals` and the ESPN client/models. Key files: src/fm/sports/base.py, src/fm/sports/nfl.py, src/fm/decide/registry.py, src/fm/proposals/payloads.py, src/fm/proposals/policy.py, src/fm/proposals/queue.py, src/fm/proposals/pause.py, src/fm/commands/proposals.py, src/fm/espn/client.py, src/fm/espn/models.py, src/fm/model/ids.py, data/id_overrides.csv, src/fm/sources/nba_stats.py, src/fm/sources/nba_schedule.py, src/fm/sources/nba_injuries.py, src/fm/sources/darko.py, src/fm/sources/market.py, src/fm/sources/odds.py, src/fm/sources/weather.py, data/stadiums.csv, src/fm/paths.py, src/fm/store/repos.py. Skipped: none. (7/7 tasks completed, 0 skipped)

## Phase 4
- TODO [P0] [SPIKE] [M] #14: Real-league capture [BLOCKED: needs a manual web sign-in inside `capture.py writes` (`fm login` clears ESPN cookies and closes as soon as espn_s2/SWID land, leaving no web session) — Jon] — scope: scripts/capture/, docs/espn-api.md, tests/fixtures/espn/real/, tests/espn/test_settings.py — depends: #9 ✓
  - Fetches every read view for each configured league and saves fixtures with manager names scrubbed.
  - Verifies the community-documented write payloads (DESIGN §6.3) for `ffl` and `fba` by driving each UI flow with `page.route` interception, aborting each request before it reaches ESPN.
  - Settles the open unknowns: pending-offer view, lineup-lock-type key, `espn_s2` expiry, 429 behavior.
  - Replaces the hand-built `tests/fixtures/espn/*_settings_*.json` stand-ins from #4 with scrubbed `mSettings` captures and confirms what `fm.espn.settings` parses tolerantly: the `rosterSettings.lineupLocktimeType` values (ESPN's own `settings.typeNames.locktimeTypes` lists `INDIVIDUAL_GAME` and `FIRSTGAME_SCORINGPERIOD`, so `LockType` carries that plus the earlier `FIRST_GAME_OF_WEEK` guess; drop the one no league carries), the `lineupSlotStatLimits` shape for NBA games-played caps, and the `acquisitionSettings.waiverHours` key.
  - Settles the read-client unknowns from #9: the `mPendingTransactions` list key (`transactions` vs `pendingTransactions`; `TransactionsView` reads both), the `filterStatsForTopScoringPeriodIds` semantics (the value must be > 0: 0 is HTTP 400) and the composite stat-entry ids `{source}{split}{season}[{period}]` behind `stat_entry_id`, the NBA `statSplitTypeId` values for rolling windows (`Player.stat_entry` matches splits 0/1 only), whether `kona_player_info` accepts an omitted `filterSlotIds` (the client omits it; espn-api sends `[]`), and ESPN's real 429 behavior to tune `DEFAULT_MIN_INTERVAL_S` (0.25 s). The cookie-free `leaguedefaults/3?view=kona_player_info` pool (#12) served identical data for fba ids 1 and 3.
  - Partial (phase 4): the reads, probes, calendars and ESPN's own write code are captured and scrubbed (`scripts/capture/capture.py reads|webclient|snapshot|verify|fixtures`; `tests/fixtures/espn/real/webclient.json` pins the payload rules of docs/espn-api.md §4), every unknown above is settled in docs/espn-api.md §1, and a snapshot plus two verify runs found both leagues unchanged. The phase 4 integration applied the client fixes the capture found: `LockType.FIRST_GAME_OF_WEEK` is gone (ESPN's weekly values parse as `UNKNOWN`), an open offer is `PENDING`, not expired and not named by a CANCEL record (`Transaction.pending`, `TransactionsView.open`, `EspnClient.pending_offers`), and `Player.stat_entry` matches each game's "Game" split (1 in `ffl`, 5 in `fba`).
  - Left: (1) the guarded UI capture, which needs Jon's web sign-in: `capture.py snapshot`, then `capture.py writes` (sign in inside its window, drive a bench swap, a free-agent add/drop, a waiver claim and a trade proposal in each league; the guard aborts and saves each request), `capture.py fixtures` and `capture.py verify` (`scripts/capture/README.md`), which write `tests/fixtures/espn/real/{ffl,fba}/write_*.json`; (2) moving `tests/espn/test_settings.py` off the hand-built `tests/fixtures/espn/*_settings_*.json` stand-ins onto the real `mSettings` captures wherever a real league covers the case (keep the 9-cat, FAAB and games-played-cap stand-ins: neither real league exercises them).
  AC: `docs/espn-api.md` has a section per read view and per write flow (both games) plus the resolved unknowns; the test command loads every fixture under tests/fixtures/espn/real/, including `{ffl,fba}/write_*.json`
- DONE [P0] [M] #15: League scoring, projection blend, basic availability — scope: src/fm/model/scoring.py, src/fm/model/projections.py, src/fm/model/availability.py, data/blend_weights.toml, tests/model/test_scoring.py, tests/model/test_projections.py, tests/model/test_availability.py — depends: #6 ✓, #7 ✓, #9 ✓, #10 ✓
  - Stat line → fantasy points from league scoring items; categories pass through.
  - Projection-source registry and a per-stat weighted blend of ESPN + Sleeper; weights read from `data/blend_weights.toml`.
  - SD by position.
  - Designation → `p_active`.
  - `data/blend_weights.toml` resolves through `fm.paths.data_file("blend_weights.toml")`, like `id_overrides.csv` and `stadiums.csv`; the crosswalk (`fm.model.ids.Crosswalk`) joins the Sleeper lines to ESPN ids.
  AC: test command passes for the three model tests (PPR, half-PPR, and custom scoring; blend weights; designation mapping)
- DONE [P0] [M] #16: Sync job — scope: src/fm/jobs/sync.py, src/fm/commands/sync.py, tests/jobs/test_sync.py — depends: #3 ✓, #6 ✓, #9 ✓, #10 ✓
  - `fm sync` pulls league state and sources into the store, running the unmapped-rostered-player gate.
  - Surfaces `Fetched.stale` / `Fetched.degraded` and `warnings` from the source adapters (#6) in its output, since Sleeper's undocumented endpoints fail as HTTP 200 with junk rather than an error, so a silent break is noticed.
  - Builds `fm.store.RawSnapshotRow` from each `EspnRead.capture` (relative_path, sha256, size_bytes, status_code, url, params, fetched_at); `fm.espn.client` never writes to the store.
  - The gate is `fm.model.ids.check_rostered(store, league_id)` (raises `UnmappedPlayersError` naming every unmapped rostered player). It gates NFL leagues only and raises `CrosswalkError` for a league of another sport (`LookupError` for an unknown league id), so dispatch on `LeagueRow.sport`: #17's NBA crosswalk brings the NBA gate. `fetch_crosswalk` returns `Fetched[Crosswalk]` whose warnings (the adapter's, then the build's) belong in the output next to the other sources'.
  - Phase 4 integration: the sync also stores Sleeper's week projections as `sleeper` stat lines (`fm.model.projections.sleeper_rows`) and runs the NBA crosswalk (`fetch_nba_crosswalk`) and `check_nba_rostered` for every NBA league; a degraded NBA crosswalk does not replace a saved one.
  AC: test command passes for tests/jobs/test_sync.py (fixture-backed sync populates the store; fails on an unmapped rostered player); `uv run fm sync --help` exits 0
- DONE [P1] [M] #17: NBA plugin and crosswalk — scope: src/fm/sports/nba.py, src/fm/model/ids_nba.py, data/id_overrides_nba.csv, tests/sports/test_nba.py, tests/model/test_ids_nba.py — depends: #7 ✓, #10 ✓, #11 ✓
  - Daily scoring periods, per-game lineup locks, add/drop cutoff at the day's first tip, PG/SG/SF/PF/C/G/F/UTIL eligibility.
  - ESPN ↔ NBA person ID matching by name, team, and position, with overrides and the unmapped gate.
  - Subclass `fm.sports.base.SportPlugin` with `period_kind=PeriodKind.DAY`, overriding `scoring_period_at` (calendar day in US Eastern) and `transaction_cutoff` (the day's first tip, DESIGN §9.3); `plugin_for("nba")` finds `fm.sports.nba.PLUGIN` with no edit to base.py, and `fm.espn.models.ProSchedule` already serves `first_game`/`idle_teams` per day. Mirror `fm.model.ids` for the crosswalk (layered build, line-numbered overrides validation, `PlayerIdRepo.replace_sport`, and a gate that refuses a league of another sport the way `fm.model.ids.check_rostered` refuses a non-NFL one).
  - Before `fm.sources.nba_stats` feeds anything, smoke-test it live from the home PC (`NbaStatsSource().game_logs("2025-26")`): its fixtures are hand-built because stats.nba.com answered only Akamai timeouts while #11 was built.
  - Phase 4 review: days turn at 03:00 ET, ESPN's calendar boundary, not midnight (`fm.sports.base.fantasy_day`/`PERIOD_TURN`; the NFL weeks' turn moved from the last game's end to 03:00 ET after it as well), and the NBA `transaction_cutoff` override is gone: the base class computes the add/drop/trade cutoff from the league's `LeagueSettings.roster_lock_type` (`rosterLocktimeType`), which the real league sets to `FIRSTGAME_SCORINGPERIOD`, the day's first tip.
  AC: test command passes for tests/sports/test_nba.py and tests/model/test_ids_nba.py
- DONE [P1] [M] #18: Phone notifications and approvals — scope: src/fm/notify/, src/fm/commands/notify.py, src/fm/commands/bot.py, tests/notify/ — depends: #8 ✓
  - Channel interface with two adapters, neither needing inbound ports:
    - Telegram: inline Approve/Reject over long polling; only your chat ID is accepted.
    - ntfy: `http` action buttons posting to a private reply topic; every button carries a single-use approval nonce; a confirmation push follows each decision.
  - `fm notify setup` captures the Telegram chat ID or generates ntfy topics; `fm bot` listens and records decisions.
  - Alert and report helpers.
  - Live use needs a Telegram account + @BotFather token, or the ntfy app; tests mock both.
  - Decisions go through `fm.proposals.approve` / `reject(store, id, decided_by=...)`; an expired proposal raises `LifecycleError` (expiry is also swept lazily, so an old button can never approve one). Buttons (Telegram and ntfy alike) carry a single-use approval nonce from a file store under the config dir; the `execution_token` `approve` mints never leaves the machine (the #8 interface note in `src/fm/proposals/queue.py`).
  AC: test command passes for tests/notify/ with mocked Telegram and ntfy APIs (buttons rendered; callback records approval; other chats and bad tokens ignored; expired proposal can't be approved)
- DONE [P0] [M] #19: Executor framework — scope: src/fm/executor/, src/fm/browser/flows/__init__.py, src/fm/browser/fakes.py, src/fm/commands/execute.py, tests/executor/test_framework.py — depends: #5 ✓, #8 ✓, #9 ✓
  - Flow protocol (API mode + UI mode) and registry.
  - API precondition checks, then run, then verify by API re-read.
  - Write safety: single-use execution token per approved proposal; no automatic write retries; hard timeouts; a timeout marks the execution `UNKNOWN` and forces a re-read.
  - Request/response, screenshots, and Playwright trace written to `audit/`.
  - `--dry-run` stops before any write; fake page and fake transport for tests.
  - `fm execute <proposal> [--dry-run]`.
  - Enters through `fm.proposals.begin_execution(store, id, token)` (consumes the token atomically; a mismatched or used token is `LifecycleError`) and `finish_execution(store, id, outcome)`; `PausedError` is a refusal, never a retry. Payloads are the `fm.proposals.payloads` models (`LineupPayload`, `AddDropPayload`, `WaiverPayload`, `TradePayload`, `TradeResponsePayload`, `TransactionCancelPayload`) read with `parse_payload`.
  - Phase 4 review: a dry run's browser context aborts the write host and every non-GET to `espn.com` (`fm.executor.dry_run_block_reason`, tested through `open_live_runtime`), and `FakeOpener` hands dry runs a `RefusingTransport` as the live opener does; an interruption after the token is spent (Ctrl+C, `SystemExit`) is recorded before it propagates (attempt `unknown` if the write may have started, proposal `failed`), and `fm.executor.reconcile_executions` ends what a killed process left `executing`.
  AC: test command passes for tests/executor/test_framework.py (failed precondition blocks the run; token can't be reused; timeout → UNKNOWN with no retry; verify mismatch → failed; dry-run sends nothing); `uv run fm execute --help` exits 0

## Phase 5
- TODO [P0] [M] #20: Lineup optimizer — scope: src/fm/decide/lineup.py, tests/decide/test_lineup.py — depends: #7 ✓, #15 ✓
  - Assignment solve over slot eligibility, maximizing expected points; switches to a win-probability objective when the matchup is lopsided.
  - Respects locks; registers NFL lineup decisions.
  - Registers with `fm.decide.registry.register("nfl", <kind>, fn)` and emits a `LineupPayload` of `LineupMove`s; locks come from `NFL.locks(period, schedule, lock_type=settings.lineup_lock_type)`.
  AC: test command passes for tests/decide/test_lineup.py incl. property tests: lineup always legal; no OUT/bye/no-game starter while a legal alternative exists; locked players never moved
- TODO [P0] [M] #21: Valuation and NFL waivers — scope: src/fm/model/valuation.py, src/fm/decide/waivers.py, tests/model/test_valuation.py, tests/decide/test_waivers.py — depends: #15 ✓, #16 ✓
  - Replacement level = best player on the league's actual wire; ROS value with playoff weeks weighted up.
  - Ranks (add, drop) pairs; FAAB bid heuristic within the policy cap; untouchables protected; timed to the league's waiver run.
  - Emits `WaiverPayload` / `AddDropPayload`; the bid cap is `fm.proposals.policy.faab_bid_cap` from the synced season budget and untouchables are `find_untouchables`, both enforced again by `evaluate()` at propose time.
  - ESPN's season projection: `Player.projection(season, 0)` reads `102026` (split 0). ESPN also sends `122026` (split 2, labelled "Rest Of Season"), but in the real free-agent data `122026` covers about 16 games and `102026` about 12 (`appliedTotal / appliedAverage`, pinned by `test_nfl_sends_a_second_season_projection`): confirm which line is rest-of-season before either feeds ROS value or a blend (docs/espn-api.md §1 #10).
  AC: test command passes for test_valuation and test_waivers (replacement level, untouchables never dropped, bid ≤ cap)
- TODO [P1] [M] #22: Full availability model — scope: src/fm/model/availability.py, tests/model/test_availability.py — depends: #11 ✓, #15 ✓
  - Practice-participation trends (NFL) and official injury report (NBA) shift `p_active`.
  - Claude news signals applied clamped to bounds and logged.
  - Late-game pivot awareness.
  AC: test command passes for tests/model/test_availability.py (trend shifts, clamping, logging)
- TODO [P1] [M] #23: News ingest and relevance — scope: src/fm/sources/news.py, src/fm/model/relevance.py, tests/sources/test_news.py, tests/model/test_relevance.py — depends: #6 ✓, #16 ✓
  - ESPN news API + RotoWire RSS (poll ≤ every 10 min), deduped.
  - Relevance filter: your roster, this week's opponent, top free agents, trade targets.
  AC: test command passes for test_news and test_relevance
- TODO [P1] [L] #24: NBA valuation — scope: src/fm/model/categories.py, src/fm/model/value_nba.py, tests/model/test_categories.py, tests/model/test_value_nba.py — depends: #15 ✓, #17 ✓
  - Points: per-game projections from the blend (ESPN + DARKO) × games, with league weights; schedule-aware daily lineup value.
  - Categories: z-scores with volume-weighted FG%/FT% and TO negative; G-scores (`√(σ² + κτ²)`, Rosenof arXiv:2307.02188) for H2H; punt weights.
  - The `darko` projection source is registered (#15) but has no loader: map DARKO's per-game keys onto ESPN abbreviations, join them to ESPN ids through `fm.model.ids_nba.NbaCrosswalk.espn_id` (`fm sync` saves it), and re-register `darko` with that loader, which the #15 registry allows. ESPN publishes no NBA daily projections (`hasGameStatProjections: false`): a day's ESPN projection is the season line's per-game rate (docs/espn-api.md §1 #12). NBA daily actual lines are `fba`'s "Game" split 5 (`fm.espn.models.STAT_SPLIT_GAME`).
  AC: test command passes for test_categories and test_value_nba (G-score reduces to z-score when τ = 0; percentage categories are volume-weighted)
- TODO [P0] [M] #25: Transaction writer and lineup flow — scope: src/fm/browser/transactions.py, src/fm/browser/selectors.py, src/fm/browser/flows/lineup.py, src/fm/commands/execute.py, tests/executor/test_transactions.py, tests/executor/test_lineup_flow.py — depends: #14, #19 ✓
  - `transactions.py`: builds ESPN transaction envelopes, sends them from the Playwright context, and maps ESPN error codes.
  - `set_lineup` in API mode, with a role/text-locator UI fallback; registered with the executor.
  - Executes `LineupPayload` moves handed over by the executor framework (#19).
  - Until `tests/fixtures/espn/real/{ffl,fba}/write_*.json` exist (#14), the envelope tests check docs/espn-api.md §4, whose rules `tests/fixtures/espn/real/test_real_fixtures.py` pins against `tests/fixtures/espn/real/webclient.json` (ESPN's serializer, item builders and service code); switch to the write captures once they land.
  - `fm execute` always opens the browser profile and ESPN session for its precondition reads, so offline it refuses with "run fm login". The fixture dry run needs an offline read path in `src/fm/commands/execute.py`, e.g. a dry-run-only `--fixtures DIR` option serving recorded views (`tests/fixtures/espn/real/`) through an httpx `MockTransport`, with `RefusingTransport` and no browser. Reuse `fm.browser.fakes` (#19) in the tests.
  AC: test command passes for both tests (envelopes match docs/espn-api.md §4, whose rules tests/fixtures/espn/real/test_real_fixtures.py pins against tests/fixtures/espn/real/webclient.json, ESPN's serializer, item builders and service code; API failure falls back to UI mode on a fake page; dry-run sends no POST); `uv run fm execute --dry-run` on a fixture proposal exits 0

## Phase 6 — MILESTONE: NFL manager end-to-end (recommend → approve → execute → verify on a schedule); NBA daily lineups + streaming proposals
- TODO [P0] [S] #26: Advisor CLI — scope: src/fm/commands/advise.py, src/fm/render/, tests/test_advise_cli.py — depends: #16 ✓, #20, #21
  - `fm status`, `fm lineup`, `fm waivers` render recommendations with engine numbers.
  AC: the three commands exit 0 against a fixture-backed store (`FM_CONFIG_DIR=tests/fixtures/home`); snapshot tests pass
- TODO [P0] [M] #27: Add/drop and waiver flows — scope: src/fm/browser/flows/add_drop.py, src/fm/browser/flows/waiver.py, src/fm/browser/selectors.py, tests/executor/test_add_drop.py, tests/executor/test_waiver.py — depends: #25
  - Free-agent add/drop and waiver claim with bid, plus cancel. API mode with UI fallback, preconditions (NBA adds before the day's first tip; no duplicate claims), and API verification.
  - Executes `AddDropPayload`, `WaiverPayload` and `TransactionCancelPayload` proposals.
  - Duplicate claims: `EspnClient.pending_offers` returns only open claims and offers (`PENDING`, not expired, not named by a CANCEL record; docs/espn-api.md §1 #2). Adds and drops close at the league's roster lock: `plugin.transaction_cutoff(pro_team_id, period, schedule, lock_type=settings.roster_lock_type)` per player moved (the real NBA league's `FIRSTGAME_SCORINGPERIOD` is the day's first tip, the NFL league's `INDIVIDUAL_GAME` each player's kickoff; `UNKNOWN` and ESPN's weekly types raise, so refuse the move rather than guess).
  AC: test command passes for test_add_drop and test_waiver against a fake page (preconditions, verification, dry-run)
- TODO [P1] [S] #28: Selector canary — scope: src/fm/browser/canary.py, src/fm/commands/canary.py, tests/browser/test_canary.py — depends: #25
  - Read-only: asserts every registered selector resolves on the roster, free-agent, and trade pages, and that every read view still parses. Alerts on drift.
  AC: test command passes for tests/browser/test_canary.py (missing selector → alert payload); `uv run fm canary --help` exits 0
- TODO [P0] [M] #29: Tick and Windows scheduler — scope: src/fm/jobs/tick.py, src/fm/jobs/deadlines.py, src/fm/jobs/scheduler_windows.py, src/fm/commands/schedule.py, tests/jobs/test_tick.py, tests/jobs/test_scheduler_windows.py — depends: #8 ✓, #16 ✓, #18 ✓, #19 ✓, #20, #21
  - Each tick: computes deadlines from pro schedules and league settings (international and holiday slots, the NBA first-tip add cutoff, the waiver run), runs due registered decisions once, and executes approved/auto proposals at their time.
  - Auto-bench fires only when unanswered at T-15; missed-window alerts.
  - Session check before any execution: `fm.espn.auth.load_session()` raises `NotLoggedInError` when no session is saved and `EspnSession.status()` is only OK/EXPIRING/EXPIRED, so the tick maps the exception to its own "missing" verdict and alerts.
  - `fm schedule install|uninstall|show` via schtasks + `.cmd` wrapper + wake/battery settings (port from software-factory).
  - Each tick calls `fm.proposals.expire_due` before `auto_approve_due` (which fires only inside `AUTO_LEAD` = T-15 and never while paused) and checks `is_paused()` before any execution.
  - Each tick calls `fm.executor.reconcile_executions(store)` before it executes anything: it ends proposals a killed process left `executing` (attempts still `running` become `unknown`, the proposal `failed`) once they are `STALE_EXECUTION` (30 min) old, so a run in progress in `fm bot` is never touched. A Ctrl+C or `SystemExit` inside `fm.executor.execute` is already recorded before it propagates.
  - The current period between syncs is `plugin.scoring_period_at(now, schedule)`: ESPN's periods turn at 03:00 ET (`fm.sports.base.PERIOD_TURN`; NBA days, and NFL weeks on the Tuesday after Monday night), not at midnight or a game's end.
  - Phone (DESIGN §13, #18): push each new proposal with `fm.notify.notify_proposal(channel, store, proposal_id)` on `fm.notify.open_channel(config)`, and missed-window alerts with `fm.notify.send_alert(..., link=espn_url)`. "An approval near a deadline executes immediately" hooks into `fm.notify.listen(on_result=...)`; the bot itself only records decisions.
  AC: test command passes for test_tick and test_scheduler_windows (rendered commands only, nothing installed); `uv run fm schedule show` exits 0
- TODO [P1] [M] #30: Advisor client and news triage — scope: src/fm/advisor/client.py, src/fm/advisor/news_triage.py, src/fm/advisor/prompts/, tests/advisor/test_client.py, tests/advisor/test_news_triage.py — depends: #2 ✓, #23
  - anthropic SDK client (`claude-opus-5-5`, effort per worker, structured outputs via `messages.parse`, `stop_reason` checks, refusal fallback, prompt caching, Batches for overnight work, `llm_usage` tracking, daily budget cap).
  - News triage → stored signals.
  AC: test command passes for tests/advisor/ with a stubbed client (parsed output stored; refusal/max_tokens handled; budget cap blocks calls)
- TODO [P1] [L] #31: NBA daily lineups and streaming — scope: src/fm/decide/lineup_daily.py, src/fm/decide/streaming.py, src/fm/espn/calendar.py, data/calendars/, tests/decide/test_lineup_daily.py, tests/decide/test_streaming.py — depends: #20, #24
  - Daily lineups across the matchup week, honoring any games-played limit.
  - Open slot-day detection.
  - Add/drop sequence under the acquisition limit as a daily knapsack/DP; adds land before the day's first tip; core players protected.
  - Category leagues rank streamers by Σ ∂P(win cat)/∂stat × production × open-slot games, with P(win cat) ≈ Φ(Δμ/σ).
  - Matchup weeks: `scheduleSettings.matchupPeriods` lists periods of the league's `periodTypeId` type (weeks), not days, and no read view maps them to days; ESPN's web client ships the calendar (real league: matchup 1 = days 1–6, 2–17 = Monday–Sunday weeks, 18 = days 119–132, 19–21 = days 133–153). Today it exists only in `tests/fixtures/espn/real/*/calendar.json` and `scripts/capture/webclient.Calendar`: ship it first, as per-season JSON under `data/calendars/` read through `fm.paths.data_file` (refreshed by `capture.py webclient`) or a `src/fm/espn/calendar.py` extractor. ESPN's weekly lock types parse as `UNKNOWN` and need the same calendar. Until then `ScheduleSettings.scoring_periods` and `matchup_period_for` answer `None` for the NBA league's week ids, so the weekly transaction cap (`fm.proposals.policy.week_filter`) counts the trailing seven days there; switch it to the calendar's matchup days.
  AC: test command passes for test_lineup_daily and test_streaming (fills open slot-days; respects acquisition and games-played limits; never drops protected players; adds scheduled before first tip)
- TODO [P1] [M] #32: Season simulator — scope: src/fm/model/simulate.py, tests/model/test_simulate.py — depends: #15 ✓, #21
  - Monte Carlo over the remaining schedule → P(win week), P(playoffs), P(bye), P(title); per-category win probabilities for NBA.
  AC: test command passes for tests/model/test_simulate.py (seeded runs reproducible; playoff probabilities sum to playoff spots within tolerance)
- TODO [P1] [M] #33: Backtest harness — scope: src/fm/eval/backtest.py, src/fm/commands/backtest.py, tests/eval/test_backtest.py, tests/fixtures/backtest/ — depends: #15 ✓, #20
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
- TODO [P1] [L] #38: Trade evaluator and finder — scope: src/fm/decide/trades.py, src/fm/commands/trade.py, tests/decide/test_trades.py — depends: #12 ✓, #32
  - `fm trade eval|find`: Δ ROS value for both sides, Δ title odds, legality.
  - Enumerates 1:1, 2:1, and 2:2 deals per opponent; screens, then re-scores; ranks by Δ title odds × P(accept), where P(accept) uses market values + their needs.
  - Market values: `MarketSource.market_values(settings, rank_type=<"PPR"/"STANDARD"/"SUPERFLEX" for ffl, "STANDARD"/"ROTO" for fba>)` takes the game, the season and FantasyCalc's league shape from the league's `LeagueSettings` (`LeagueShape.from_settings`: team count, REC scoring points, starting slots a QB can fill, so superflex/2-QB leagues ask for 2 QBs; there is no default shape); derive `rank_type` from `LeagueSettings` too, never a literal. FantasyCalc is skipped for fba and every rank type stays in `MarketValue.espn_ranks`.
  AC: test command passes for tests/decide/test_trades.py (symmetric trade ≈ 0 Δ; finder returns only legal trades in ranked order); `uv run fm trade eval --help` exits 0
- TODO [P2] [S] #39: Blend weight tuning — scope: src/fm/eval/tune.py, data/blend_weights.toml, tests/eval/test_tune.py — depends: #33
  - Fits per-(source, position) weights on held-out weeks.
  AC: test command passes for tests/eval/test_tune.py (held-out MAE ≤ equal-weight MAE on fixtures)
- TODO [P2] [M] #40: MCP server — scope: src/fm/mcp_server.py, tests/test_mcp_server.py — depends: #8 ✓, #26
  - FastMCP read tools (status, lineup, waivers, trade eval) plus `create_proposal`; no execute tool.
  - `create_proposal` is `fm.proposals.propose` (policy verdict and dedupe included); the server exposes no approve or execute path.
  AC: test command passes for tests/test_mcp_server.py (tool list has no write/execute tool; `create_proposal` stores a proposal)
- TODO [P2] [M] #41: UI fallback drills — scope: src/fm/browser/drills.py, src/fm/commands/drill.py, tests/browser/test_drills.py — depends: #14, #27
  - Weekly dry-run of each UI-mode flow against the live site, stopping before the final confirm, so the fallback is known-good when API mode breaks. Alerts on failure.
  AC: test command passes for tests/browser/test_drills.py against a fake page; `uv run fm drill --help` exits 0
- TODO [P2] [M] #42: NFL opportunity baseline — scope: src/fm/model/baseline_nfl.py, tests/model/test_baseline_nfl.py — depends: #13 ✓, #15 ✓, #33
  - Shares × implied team total × regressed efficiency, registered as a projection source.
  - ESPN removes the scoreboard `odds` block at kickoff, so implied team totals must be captured while `ScoreboardGame.state == "pre"` (the scoreboard TTL is 10 minutes); have the sync job (#16) snapshot pregame lines if the baseline needs them after the fact.
  AC: test command passes; the backtest on fixtures shows the blend with the baseline no worse than without it
- TODO [P2] [M] #43: NBA minutes baseline — scope: src/fm/model/baseline_nba.py, tests/model/test_baseline_nba.py — depends: #11 ✓, #24, #33
  - Minutes × per-minute rates, registered as a projection source.
  - Teammates-out redistribution from without-player splits + on/off data, with DARKO minutes as the prior; blowout risk and back-to-back rest risk.
  - `fm.sources.nba_stats` works live (#17 smoke test, 2026-10-05: `game_logs("2025-26")` 26,651 rows, `player_splits("2025-26", "Base")` 582 rows). Its `leaguedashplayerstats` fixtures are hand-built from nba_api's expected_data (`CFID`, `CFPARAMS`); live stats.nba.com sends `NICKNAME`, `WNBA_FANTASY_PTS`, `FP_HIGH_SCORE`, their ranks and `TEAM_COUNT` instead. The adapter's required columns are present either way, and live game logs match their fixture.
  AC: test command passes; backtest on fixtures no worse than without

## Phase 8 — MILESTONE: trades end-to-end + weekly report
- TODO [P1] [M] #44: Trade flows and offer handling — scope: src/fm/browser/flows/trade.py, src/fm/browser/selectors.py, src/fm/decide/offers.py, tests/executor/test_trade_flow.py, tests/decide/test_offers.py — depends: #25, #29, #38
  - Propose/respond/cancel flows (approval-only), API mode with UI fallback.
  - Never duplicates an open offer; checks lock status for every player involved.
  - Pending incoming offers become evaluation proposals registered for the tick.
  - Trade kinds have no policy field in `fm.proposals.policy.KINDS` (`effective_setting` is always `approve`); incoming offers become `TradeResponsePayload` proposals through `propose`, outgoing ones `TradePayload`.
  - Open offers: ESPN leaves an expired offer `PENDING` and closes it with a separate CANCEL record that still says `isPending`; `EspnClient.pending_offers` keeps only open ones (docs/espn-api.md §1 #2). Offers come from `mTransactions2`, not `mPendingTransactions`, and expire 48 h after `proposedDate`.
  AC: test command passes for both tests (fake page; `auto` policy rejected for trade kinds; duplicate offers and locked-player trades blocked)
- TODO [P1] [M] #45: Trade pitch and weekly strategist workers — scope: src/fm/advisor/pitch.py, src/fm/advisor/strategist.py, tests/advisor/test_pitch.py, tests/advisor/test_strategist.py — depends: #30, #32, #38
  - Pitch draft per approved trade idea.
  - Weekly priorities, punts, and targets → proposals.
  AC: test command passes for both tests with a stubbed client
- TODO [P1] [S] #46: Weekly report — scope: src/fm/jobs/report.py, src/fm/commands/report.py, tests/jobs/test_report.py — depends: #16 ✓, #18 ✓, #36
  - `fm report`: matchup outlook, playoff odds, moves made, upcoming deadlines; markdown + phone channel.
  - Phone: `fm.notify.send_report(channel, title, body)` on `fm.notify.open_channel(config)`; long reports are split automatically.
  AC: `uv run fm report` against fixtures writes a markdown report; test command passes

## Phase 9
- TODO [P2] [S] #47: Live report card — scope: src/fm/eval/report_card.py, tests/eval/test_report_card.py — depends: #33, #46
  - Weekly lineup efficiency, bench points, and pickup value on real decisions, appended to the report.
  AC: test command passes for tests/eval/test_report_card.py
