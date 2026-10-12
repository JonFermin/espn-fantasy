# Plug and play: design for roadmap phases 10 to 12

Phases 1 to 9 built the engine. Using it still takes a hand-written `config.toml`, league and team ids copied out of
ESPN URLs, about ten policy knobs per league, a `.env`, `fm notify setup`, `fm schedule install`, a separately run
`fm bot`, and knowledge of 20 commands. These phases make it plug and play without loosening any invariant in
`CLAUDE.md`:

- **A. `fm setup` and `fm doctor`.** One command signs in, discovers your leagues and teams, picks a policy preset
  per league, collects keys, pairs the phone, writes the config, syncs and installs autostart. `fm doctor` says what is
  broken and how to fix it.
- **B. `fm run`.** One long-running process ticks on a timer and listens for approvals between ticks. It replaces the
  `fm tick` schedule plus a separately started `fm bot`, and it starts at login.
- **D. `fm ui`.** A local dashboard: overview, proposal queue, lineup and waiver views, reports, settings, health.
- **E. `fm claude install`.** Registers the MCP server with Claude Code and Claude Desktop and installs a skill and
  slash commands, so "what should I do this week?" works from Claude.

Read `CLAUDE.md` first. Everything below is subordinate to its invariants. The one that matters most here: **only
`fm.executor` (through `fm.browser`) writes to ESPN.** The setup wizard, the dashboard and the Claude surfaces only
read, record human decisions, or store `proposed` rows.

## 1. Policy presets (#48)

`[[league]]` gains an optional `preset = "hands_off" | "balanced" | "ask_first"`. The preset fills every
`[league.policy]` and `[league.trade_finder]` key the file does not set, and explicit keys always win. With no
`preset`, loading behaves exactly as it does today, so existing configs (including `tests/fixtures/home/config.toml`)
load unchanged. Presets live in `src/fm/presets.py` as plain data per sport:

| key | hands_off | balanced | ask_first |
|---|---|---|---|
| bench_inactive | auto | auto | approve |
| lineup | auto | auto | approve |
| add | NFL `auto` (auto_add_min_gain 15), NBA `approve` | approve | approve |
| add_drop / waiver | approve | approve | approve |
| max_transactions_per_week | NFL 3, NBA 7 | NFL 3, NBA 7 | NFL 3, NBA 7 |
| trade_finder.enabled | true | true | false |
| trade_finder thresholds | min_accept 0.5, min_title_gain 0.015, min_ros_gain 0.3, max_offers 1 | same | defaults |

- A test asserts that no preset contains a `trade*` policy key or `auto` for `add_drop` or `waiver`.
- NBA `add` stays `approve` everywhere, because NBA streaming reports no news-free gain yet.
- `League.preset` is kept on the model so `fm config check`, the dashboard and `fm doctor` can show it.

**Config editing (`src/fm/config_edit.py`, tomlkit).** One module writes `config.toml`, and every writer goes through
it:

- **API.**
  - `render_config(leagues, *, llm=None, notify=None) -> str` renders a fresh, commented file.
  - `add_leagues(path, entries)` appends `[[league]]` tables and skips any `(sport, espn_league_id)` already present.
  - `set_value(path, league_key, dotted_key, value)` handles values such as `preset`, `trade_finder.enabled` and
    `policy.lineup`.
  - `set_notify_channel(path, channel)`.
- **Write rules.**
  - Comments and key order survive an edit.
  - The edited text is validated with `load_config` before it replaces the file. An invalid edit raises `ConfigError`
    and leaves the file untouched.
  - The replace is atomic (temp file plus `os.replace`), and the previous file is kept as `config.toml.bak`.
- **CLI.** `fm config set-preset <league> <preset>` is the CLI front of `set_value`.

## 2. League discovery (#49)

`src/fm/espn/discover.py`, read-only:

- **`current_season(sport, now)`.** NFL is the year of the September kickoff (January to July belongs to the previous
  year). NBA is labelled by its end year: from August onward it is `year + 1`.
- **`discover_leagues(session, *, client=None, now=None) -> list[DiscoveredLeague]`.**
  - It sends one GET to `https://fan.api.espn.com/apis/v2/fans/{SWID}` with the `espn_s2`/`SWID` cookies, explicit
    timeouts and a single retry on transport errors. `DiscoveredLeague` carries `sport, espn_league_id, season,
    team_id, league_name, team_name`.
  - **The response shape is not verified against a live account**, so the parser is defensive. It tries the
    structured path `preferences[].metaData.entry`: `groups[].groupId` (league), `entryId` (team), `seasonId`, plus
    `abbrev`/`gameId` or the sport from `entryURL`. If that finds nothing, it deep-walks the JSON for fantasy URLs that
    carry `leagueId=` (and `teamId=` / `seasonId=` when present) under `/football/` or `/basketball/`.
  - It keeps ffl/fba only, deduplicates, and keeps the current season per sport unless `all_seasons=True`.
  - A failure raises `DiscoveryError`. Callers fall back to pasted URLs.
- **`parse_league_url(text) -> LeagueRef(sport, espn_league_id, season | None, team_id | None)`** accepts any
  `fantasy.espn.com/{football|basketball}/...` URL with `leagueId=`.
- **`resolve_team(teams: TeamsView, swid) -> int`** returns the team whose `owners` contains the SWID (braces
  normalized with `normalize_swid`). It raises `DiscoveryError` when none or several match.
- **`confirm_league(ref, session, *, client_factory=EspnClient) -> DiscoveredLeague`** reads `mTeam` and fills
  `team_id` (via `resolve_team` when missing), `season` (via `current_season` when missing) and the league and team
  names.
- **Fixtures.** A hand-written synthetic `tests/fixtures/espn/fan_api/*.json` covers the structured shape and the
  URL-only shape. It has no real names. The real `mTeam` fixture covers `resolve_team`. Never capture a real fan-API
  response into the repo: it holds personal data.

## 3. Health checks (#50)

`src/fm/health.py`: `run_checks(*, online=False, now=None, **injectables) -> list[Check]`.

- **`Check`.** `Check(name, status: OK|WARN|FAIL|SKIP, detail, fix)`. `fix` is the exact command to run, such as
  `fm login` or `fm setup`.
- **Offline checks** (the default; only local state is read):
  - config present and valid, plus missing required secrets;
  - the ESPN session: the profile exists and the cookie expiry gives ok, expiring or expired. The cookie is read
    through an injectable loader, and a browser that will not start is WARN;
  - a supported browser (Edge or Chrome) is installed;
  - the notify channel is configured;
  - the last tick, from `tick-state.json`: WARN after 30 min, FAIL after 6 h or when never run;
  - the last sync per league from the store;
  - pause state;
  - open proposals, and approvals whose deadline is under 2 h away;
  - today's LLM spend against the budget.
- **`--online`** adds an `mStatus` read per league and a channel ping (Telegram `getMe`, or a HEAD to the ntfy server).
- **`fm doctor [--online] [--json]`** prints one line per check plus the fix, and exits 1 on any FAIL.
- `run_checks` is pure data, consumed by the dashboard (#54) and the MCP `health` tool (#55). #53 adds the autostart
  check.

## 4. `fm run` and autostart (#51)

**Lock (`src/fm/jobs/runlock.py`).**

- `run.lock` in the config dir is an OS-level exclusive lock: `msvcrt.locking` on Windows, `fcntl.flock` on macOS. It
  is the one platform branch here, isolated in this module.
- `RunLock.acquire(blocking=False)` works as a context manager. The holder writes `{pid, command, started_at}` into the
  file, and `holder()` reads it.
- `fm run` holds the lock for its lifetime, and `fm tick` holds it for one tick.
- A tick that cannot get the lock prints `fm run is active (pid N); skipping` and exits 0. The scheduled tick task can
  therefore stay installed purely as the Windows wake timer, and the woken `fm run` loop catches up on its own.
- `fm run` waits for the lock (a scheduled tick may be mid-run), printing what holds it.

**Loop (`src/fm/jobs/runner.py`).** `run_forever(...)`, with every collaborator injectable: config loader, store
opener, channel opener, tick function, approval hook, `clock`, `sleep` and a `stop` event.

- **Ticks.** A tick is due by **wall clock** (`next_tick_at`), so after the machine sleeps the first iteration ticks at
  once. The config is reloaded each tick, so edits from `fm setup`, the dashboard or `fm config set-preset` apply
  without a restart.
- **Approvals.** Between ticks the loop polls the channel, waiting at most `min(seconds until next tick, 25)`. Each
  press goes through `fm.notify.handle_reply`, with `fm.jobs.tick.execute_on_approval` as the result hook, so a phone
  approval close to a deadline executes immediately. `fm bot` never wired that hook in.
- **No channel configured.** The loop just sleeps until the next tick.
- **Errors.** An exception in a tick or a poll is logged and alerted at most once an hour. It never ends the loop, and
  channel errors back off as in `fm.notify.listen`.
- **Exit.** Ctrl+C or `stop` exits cleanly and releases the lock.
- **Logs.** `logs/run.log`, rotating 5 x 1 MB, plus the console when attached.

**Commands.**

- `fm run [--every 10] [--no-execute]` runs the loop in the foreground.
- `fm autostart install|uninstall|show [--dry-run]` registers `fm run` to start at logon, through the scheduler
  backends (extend `scheduler_base.Backend` with logon-job functions, or add a parallel `AutostartBackend`; either way
  every command goes through an injectable `Runner`, and no unit test touches a real scheduler):
  - **Windows.** `Register-ScheduledTask` with `New-ScheduledTaskTrigger -AtLogOn -User <current user>` (no admin
    needed) and restart-on-failure. The action runs `uvw.exe run --project <repo> fm run` (uv's windowless launcher,
    beside `uv.exe`), so no console window stays open. Without `uvw.exe` it falls back to `uv.exe` with a note.
  - **macOS.** A LaunchAgent `local.espn-fantasy-run.plist` with `RunAtLoad` and `KeepAlive`.
  - The tick task from `fm schedule install` stays as the wake timer, and the docs say so.

## 5. Shared read views (#52)

Move the JSON-able builders out of `src/fm/mcp_server.py` into `src/fm/views.py`: status, lineup, waivers,
trade_eval, the proposal list and their helpers (`_status_block`, `_lineup_dict`, `_waiver_dict`, `_trade_dict`,
`_proposal`, the cached schedule and matchup readers). `fm.views` must not import fastmcp, so the dashboard can use
it. The MCP tools become thin wrappers, and the MCP tool behaviour and every existing MCP test stay unchanged.
`fm.views` keeps the MCP module's no-write-path rule (no `fm.executor` or `fm.browser` import, no approve, execute or
pause import). `create_proposal` stays in `fm.mcp_server`.

## 6. `fm setup` (#53)

`src/fm/setup/wizard.py` holds the logic. `src/fm/commands/setup.py` holds the prompts. Every external effect is
injectable: session loader and login, discovery, ESPN client, prompt functions, notify setup, sync, autostart
install.

1. **Session.** No session, or an expired one, runs the `fm login` flow. An expiring one offers renewal.
2. **Leagues.** It runs `discover_leagues` and shows every current-season league as `NFL  <league name>  -  team 7
   "<team name>"`. All are selected by default. If discovery fails, it asks for league URLs (`--league-url`,
   repeatable) and confirms each with `confirm_league`.
3. **Preset** per league (default `balanced`), with a one-line description of each.
4. **Keys.** `ANTHROPIC_API_KEY` uses a hidden prompt and can be skipped (the advisor degrades to templates).
   `ODDS_API_KEY` is optional. Both are written to `.env` with `dotenv.set_key`.
5. **Phone.** The choices are ntfy (default, no account), Telegram (asks for the BotFather token) or skip. Move the
   channel setup out of `src/fm/commands/notify.py` into `src/fm/notify/setup.py`, so the wizard and `fm notify setup`
   share it.
6. **Config.** A new file comes from `render_config`. With an existing file, `add_leagues` adds only the new leagues
   and never rewrites the existing ones, and `set_notify_channel` sets the channel.
7. **First sync** of each new league, with progress (`--no-sync` skips).
8. **Autostart.** `fm autostart install` plus `fm schedule install` (the wake timer). `--no-autostart` skips.
9. **Summary.** `fm doctor` output, then "Dashboard: fm ui. Ask Claude: fm claude install."

**Flags.** `--yes` accepts every default with no prompts, `--league-url`, `--preset`, `--channel ntfy|telegram|none`,
`--no-sync` and `--no-autostart`. Re-running is idempotent: nothing changes when nothing is new.

`fm.health` gains the autostart check: autostart installed and `fm run` holding the lock.

## 7. Dashboard (#54)

`src/fm/web/`:

- **Stack.** A Starlette app with a JSON API and one static page (`src/fm/web/static/index.html`, `app.js`,
  `app.css`). It is vanilla JS with no CDN and no build step, works offline, has light and dark themes, and keeps
  usable at phone width. `pyproject.toml` declares `starlette` and `uvicorn` explicitly; both are already in the lock
  through fastmcp.
- **API.**

  | endpoint | serves |
  |---|---|
  | `GET /api/overview` | per league: team, record, standing, sync age, next deadlines, open proposals; plus session status and pause state |
  | `GET /api/proposals?all=0` | the proposal list |
  | `POST /api/proposals/{id}/approve`, `.../reject` | `fm.proposals.approve` / `reject` with `decided_by="dashboard"` |
  | `POST /api/pause`, `POST /api/resume` | the kill switch |
  | `GET /api/lineup/{league}`, `GET /api/waivers/{league}` | through `fm.views`; nothing stored |
  | `GET /api/reports`, `GET /api/reports/{name}` | the report list and raw markdown, rendered client-side as safe text |
  | `GET /api/health` | `fm.health.run_checks()`, offline |
  | `GET /api/settings`, `POST /api/settings` | preset per league, `trade_finder.enabled`, the notify channel, through `fm.config_edit` |

- **No write path to ESPN.** `fm.web` imports nothing from `fm.executor` or `fm.browser`, enforced by an AST import
  test like the MCP one. Approving here records the decision only, and `fm run` or the tick executes it.
- **Security.**
  - It binds `127.0.0.1` only.
  - It rejects requests whose `Host` is not `127.0.0.1:<port>` or `localhost:<port>`, to stop DNS rebinding.
  - Every POST needs the per-process random token (injected into the page as a meta tag and sent as an
    `X-FM-Token` header) and a same-origin `Origin` header when one is present.
  - It sends no CORS headers, and HTML is never built from data on the server.
- **Commands.**
  - `fm ui [--port 8711] [--no-open]` serves the dashboard and opens the browser.
  - `fm run --ui/--no-ui` (default on) serves it in a daemon thread alongside the loop, through `uvicorn.Server`.
    Signal handlers are installed only in the main thread.

## 8. Claude integration (#55)

**`fm claude install|uninstall|show [--code/--no-code] [--desktop/--no-desktop] [--dry-run]`**, implemented in
`src/fm/claude/`:

- **Claude Code.** `claude mcp add-json --scope user espn-fantasy '<json>'`, with the server command
  `<uv> run --project <repo> fm mcp` and `FM_CONFIG_DIR`/`FM_CACHE_DIR` when they are set. It also copies the skill to
  `~/.claude/skills/espn-fantasy/SKILL.md` and the slash commands to `~/.claude/commands/espn-fantasy/*.md`. If
  `claude` is not on PATH, it says so and prints the command.
- **Claude Desktop.** It merges an `mcpServers.espn-fantasy` entry into `claude_desktop_config.json` and backs the file
  up first. The Windows paths are `%APPDATA%\Claude\` and the MSIX path
  `%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\Claude\`, preferring whichever exists. The macOS path is
  `~/Library/Application Support/Claude/`. Other servers' entries are never touched.
- **Packaged assets.** The skill and commands ship inside the package (`src/fm/claude/assets/`). The skill tells
  Claude to:
  - start with `status`;
  - use `lineup`, `waivers`, `trade_eval`, `report` and `health`;
  - store a proposal with `create_proposal` only when asked;
  - never claim a move was made, since approvals happen on the phone or in the dashboard and the executor acts;
  - cite engine numbers.

  The commands are `week` (weekly briefing), `lineup`, `waivers` and `trade <text>`.
- **Every effect is injectable.** Tests use a temp home and a fake runner.

**MCP additions.** Add `health` (`fm.health`, offline) and `report` (the latest weekly report's markdown per league,
from the reports dir). Set the server's `instructions` string. Update `TOOL_NAMES` and the no-write tests. Still no
approve, execute or pause tool.

## 9. Journey (#56)

- **Docs.** The README quickstart becomes `uv tool install --editable .` → `fm setup` → done, with `fm ui`, `fm doctor`
  and `fm claude install` next. DESIGN sections 12 to 14 are updated and this file folds into them. `CLAUDE.md`'s
  write-path invariant names `fm.web` and `fm.claude` beside `advisor/`, `decide/` and the MCP server.
- **Offline end-to-end test** in a temp `FM_CONFIG_DIR`/`FM_CACHE_DIR` with fixtures and fakes:
  1. `fm setup --yes` with a fake discovery and sync;
  2. `fm config check` passes;
  3. `fm doctor --json`;
  4. one `run_forever` iteration;
  5. a dashboard approve through the Starlette test client;
  6. the next iteration executes it with `--no-execute`, or a dry-run opener, and nothing reaches ESPN.
