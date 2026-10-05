# espn-fantasy

A personal assistant GM for ESPN fantasy football (NFL) and basketball (NBA).

- It reads league state from ESPN, blends projections from several public data sources, and proposes lineups,
  waiver and streaming moves, and trades.
- Claude triages news and explains each decision.
- Approved moves are carried out through a logged-in browser session and verified afterward.

**Status:** early build. Roadmap Phases 1 and 2 are done: a uv-managed Python 3.13 project with the `fm` CLI scaffold
(auto-discovered command modules, runtime paths, offline test harness), config loading with `fm config check`, the
SQLite store and v1 schema, ESPN ID maps and the league-settings parser, a persistent browser profile with `fm login`
for the manual ESPN sign-in, and cached source adapters for nflverse and Sleeper. Nothing reads or writes ESPN league
data yet.

- [`docs/DESIGN.md`](docs/DESIGN.md): architecture, data sources, safety rules, phase plan
- [`ROADMAP.md`](ROADMAP.md): 47 dependency-ordered build tasks
- [`CLAUDE.md`](CLAUDE.md): invariants for coding agents working in this repo

> **Note:** ESPN has no public write API, and automating it goes against the letter of Disney's Terms of Use (see
> DESIGN §6.4). This is a single-user tool. It acts only on its owner's team, at low volume, and every trade needs
> human approval.

## Running

`fm` is installed into the project's virtual environment, not onto your PATH, so a bare `fm` in PowerShell says it is
not recognized. From the repo root, prefix commands with `uv run`:

```powershell
uv sync
uv run fm --help
uv run fm config check
```

To type `fm` directly, activate the environment first (`.\.venv\Scripts\Activate.ps1`), or install it as a tool once
with `uv tool install --editable .` (then `uv tool update-shell` if `fm` is still not found) and open a new shell.

## License

MIT. See [LICENSE](LICENSE).
