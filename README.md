# espn-fantasy

A personal assistant GM for ESPN fantasy football (NFL) and basketball (NBA).

- It reads league state from ESPN, blends projections from several public data sources, and proposes lineups,
  waiver and streaming moves, and trades.
- Claude triages news and explains each decision.
- Approved moves are carried out through a logged-in browser session and verified afterward.

**Status:** early build. Roadmap Phases 1 to 3 are done: a uv-managed Python 3.13 project with the `fm` CLI scaffold
(auto-discovered command modules, runtime paths, offline test harness), config loading with `fm config check`, the
SQLite store and v1 schema, ESPN ID maps and the league-settings parser, a persistent browser profile with `fm login`
for the manual ESPN sign-in, cached source adapters (nflverse, Sleeper, stats.nba.com, the NBA schedule CDN, ESPN
injuries, DARKO, FantasyCalc and ESPN ownership, the ESPN scoreboard, Open-Meteo and The Odds API), the sport plugin
interface with the NFL plugin and decision registry, the proposal queue with policy guardrails and `fm proposals` /
`fm pause` / `fm resume`, the ESPN read client, and the NFL player ID crosswalk. Phase 4 is done except the guarded
capture of ESPN's write requests from the web UI (#14, which needs a manual web sign-in): scrubbed real-league read
fixtures and `docs/espn-api.md`, league scoring and the ESPN + Sleeper projection blend, `fm sync` (league state,
projections and the NFL and NBA id-crosswalk gates), the NBA plugin and crosswalk, phone approvals over Telegram or
ntfy (`fm notify`, `fm bot`), and the executor framework behind `fm execute --dry-run`. Phase 5 is done: the NFL
lineup optimizer, player valuation with NFL waiver and add/drop proposals, the full availability model (practice
trends, the NBA official injury report, bounded Claude news signals, late-game pivots), ESPN and RotoWire news ingest
with a relevance filter, NBA valuation (points and category G-scores, DARKO in the blend) and the transaction writer
with the `set_lineup` executor flow. Only `fm execute` writes to ESPN, and during development it runs with `--dry-run`
only (`--fixtures` serves recorded views offline).

- [`docs/DESIGN.md`](docs/DESIGN.md): architecture, data sources, safety rules, phase plan
- [`ROADMAP.md`](ROADMAP.md): 47 dependency-ordered build tasks
- [`CLAUDE.md`](CLAUDE.md): invariants for coding agents working in this repo

> **Note:** ESPN has no public write API, and automating it goes against the letter of Disney's Terms of Use (see
> DESIGN §6.4). This is a single-user tool. It acts only on its owner's team, at low volume, and every trade needs
> human approval.

## Running

It runs on Windows and macOS. You need [uv](https://docs.astral.sh/uv/) and Microsoft Edge or Google Chrome installed
(`fm login` drives the installed browser, never a bundled one). State lives in `~/.config/espn-fantasy/` on both.

`fm` is installed into the project's virtual environment, not onto your PATH, so a bare `fm` in PowerShell says it is
not recognized. From the repo root, prefix commands with `uv run`:

```powershell
uv sync
uv run fm --help
uv run fm config check
```

To type `fm` directly, activate the environment first (`.\.venv\Scripts\Activate.ps1`), or install it as a tool once
with `uv tool install --editable .` (then `uv tool update-shell` if `fm` is still not found) and open a new shell. A
PowerShell window opened before that install keeps its old `PATH`, so `fm` stays unrecognized there until you open a
new window (or, in the current one, run
`$env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')`).

### macOS

```bash
brew install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run fm --help
uv run fm login            # opens Chrome or Edge once for the manual ESPN sign-in
```

To type `fm` directly, `source .venv/bin/activate`, or `uv tool install --editable .` once.

### Scheduling the tick

`uv run fm schedule install` registers `fm tick` every 10 minutes with the platform's scheduler (`--dry-run` prints
what it would do, `show` and `uninstall` inspect and remove it):

- **Windows:** a Task Scheduler task that may wake the PC and run on battery.
- **macOS:** a launchd LaunchAgent, `~/Library/LaunchAgents/local.espn-fantasy-tick.plist`. launchd cannot wake a
  sleeping Mac, so keep it awake around lineup locks (System Settings > Battery > Options, or `sudo pmset`). A tick
  missed during sleep runs on wake and reports what it missed.

The log is `~/.config/espn-fantasy/logs/tick.log` on both. The browser profile and `state.db` are per machine: run the
tick on one machine at a time, and run `fm login` on each.

## License

MIT. See [LICENSE](LICENSE).
