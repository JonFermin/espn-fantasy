# espn-fantasy

A personal assistant GM for ESPN fantasy football (NFL) and basketball (NBA).

- It reads league state from ESPN, blends projections from several public data sources, and proposes lineups,
  waiver and streaming moves, and trades.
- Claude triages news and explains each decision.
- You approve on your phone, in a local dashboard or from Claude. Approved moves are carried out through a logged-in
  browser session and verified afterward.

**Status:** roadmap phases 1 to 12 are built. Phases 1 to 9 are the engine: ESPN reads and the write executor,
projections and availability, NFL and NBA lineups, waivers and streaming, trades, phone approvals, the weekly report
and backtests. Phases 10 to 12 make it plug and play: `fm setup`, `fm doctor`, the always-on `fm run`, the dashboard
and the Claude integration, with an offline end-to-end test of that journey. Open follow-ups are listed per phase in
`ROADMAP.md`.

- [`docs/DESIGN.md`](docs/DESIGN.md): architecture, data sources, safety rules, runtime, config
- [`ROADMAP.md`](ROADMAP.md): the dependency-ordered build tasks and what each phase built
- [`CLAUDE.md`](CLAUDE.md): invariants for coding agents working in this repo

> **Note:** ESPN has no public write API, and automating it goes against the letter of Disney's Terms of Use (see
> DESIGN §6.4). This is a single-user tool. It acts only on its owner's team, at low volume, and every trade needs
> human approval.

## Quickstart

It runs on Windows and macOS. You need [uv](https://docs.astral.sh/uv/) and Microsoft Edge or Google Chrome
(`fm login` drives the installed browser, never a bundled one).

```powershell
git clone https://github.com/JonFermin/espn-fantasy.git
cd espn-fantasy
uv tool install --editable .
fm setup
```

That is all. `fm setup` signs you in to ESPN (a browser window opens once), finds your current NFL and NBA leagues
and your team in each, asks for a preset per league, offers to save your API keys to `.env`, pairs your phone, writes
`~/.config/espn-fantasy/config.toml`, runs the first sync, starts `fm run` at logon, and ends with `fm doctor`'s
checks. Run it again whenever you like: it changes nothing that is already set up.

- **Why `--editable`:** `fm` reads committed data (`data/`: id overrides, blend weights, calendars, stadiums) from
  the checkout, and a plain install would leave it behind. Keep the clone where it is: the logon job and the Claude
  MCP server run `uv run --project <this clone>`.
- **`fm` not found:** run `uv tool update-shell` and open a new terminal. A PowerShell window opened before the
  install keeps its old `PATH`. From the clone, `uv run fm ...` always works.
- **Flags:** `--yes` takes every default and skips secrets; `--preset hands_off|balanced|ask_first` and
  `--channel ntfy|telegram|none` answer those questions; `--no-sync` and `--no-autostart` skip those steps.
- **Discovery failed:** pass your team page instead, `fm setup --league-url '<team page URL>'` (repeatable).

**Run live from a separate, pinned clone (recommended).** The jobs `fm setup` installs run the clone they were
installed from, so a pull or a branch switch there changes what acts on your real leagues. Keep one clone for the live
installation, develop in another, and upgrade the live one on purpose:

```powershell
git pull --ff-only
uv sync --locked
```

Each tick is a fresh child process, so the next tick runs the new code; the `fm run` loop itself reloads at the next
logon. If `pyproject.toml` changed, also run `uv tool install --editable . --reinstall` there for the `fm` on your
`PATH`.

## Running

- **`fm run`** is the one long-running process. It ticks every 10 minutes (sync what is stale, run the decisions that
  are due, push proposals, execute what is approved) and listens for phone approvals in between. `fm setup` starts it
  at logon; `fm autostart install|uninstall|show` manages that job (Windows Task Scheduler, or a macOS LaunchAgent).
  The tick task from `fm schedule install` stays installed as the wake timer and skips while `fm run` is active. Log:
  `~/.config/espn-fantasy/logs/run.log`.
- **`fm ui`** opens the dashboard: overview, proposals to approve or reject, lineup and waiver views, reports,
  settings and health. It is served on `127.0.0.1:8711` by `fm run` (or by `fm ui` itself), and `fm ui` signs the
  browser in through a one-time link; another account on the machine cannot use it.
- **`fm doctor`** says what is broken and the command that fixes it (`--json` for scripts, `--online` adds an ESPN
  read and a phone ping). It exits 1 on any failure.
- **`fm claude install`** registers the MCP server with Claude Code and Claude Desktop and installs a skill and the
  `/espn-fantasy:week`, `lineup`, `waivers` and `trade` commands. Claude reads and drafts; it never approves or acts.
- **`fm config set-preset <league> <preset>`** changes a league's preset (`--reset-overrides` drops policy keys you
  wrote that would override it). `fm config check` validates the file. Edits apply at the next tick, no restart.
- **`fm pause` / `fm resume`** is the kill switch: no proposals, auto-approvals or executions until resumed.

### Presets

A preset fills every policy key a league leaves unset; a key written in `config.toml` wins.

| | `hands_off` | `balanced` (default) | `ask_first` |
|---|---|---|---|
| Lineups: bench an inactive starter, other changes | auto | auto | approve |
| Free-agent add into an open spot | NFL auto when the gain is at least 15 points; NBA approve | approve | approve |
| Add/drop, waiver claim | approve | approve | approve |
| Trade finder | on: drafts offers for approval | on | off |

"auto" for a lineup fires at T-15 when you have not answered; an auto NFL add fires at once. NBA adds stay approve in
every preset, because NBA streaming reports no news-free gain yet. Every trade waits for you in every preset, and that
is not configurable. The weekly cap is 3 transactions in NFL and 7 in NBA.

### Phone

- **ntfy** (default): no account. Install the ntfy app and subscribe to the topic `fm setup` prints.
- **Telegram**: create a bot with @BotFather; `fm setup` asks for its token and captures your chat.
- **none**: no phone. Proposals wait in the dashboard or `fm proposals`, and alerts go to the log.

`fm notify setup --channel <ntfy|telegram>` adds or replaces a channel later. Approvals are Approve/Reject buttons
that work over outbound connections only, so no port is opened on the PC.

### macOS

```bash
brew install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/JonFermin/espn-fantasy.git && cd espn-fantasy
uv tool install --editable .
fm setup                   # opens Chrome or Edge once for the manual ESPN sign-in
```

`fm run` starts at login as the LaunchAgent `local.espn-fantasy-run`. launchd cannot wake a sleeping Mac, so keep it
awake around lineup locks (System Settings > Battery > Options, or `sudo pmset`). A tick missed during sleep runs on
wake and reports what it missed.

The browser profile and `state.db` are per machine: run the tool on one machine at a time, and sign in on each.

## License

MIT. See [LICENSE](LICENSE).
