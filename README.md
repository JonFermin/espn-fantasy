# espn-fantasy

A personal assistant GM for ESPN fantasy football (NFL) and basketball (NBA).

- It reads league state from ESPN, blends projections from several public data sources, and proposes lineups,
  waiver and streaming moves, and trades.
- Claude triages news and explains each decision.
- Approved moves are carried out through a logged-in browser session and verified afterward.

**Status:** early build. Roadmap Phase 1 is done: a uv-managed Python 3.13 project with the `fm` CLI scaffold
(auto-discovered command modules, runtime paths, offline test harness). Nothing talks to ESPN yet.

- [`docs/DESIGN.md`](docs/DESIGN.md): architecture, data sources, safety rules, phase plan
- [`ROADMAP.md`](ROADMAP.md): 47 dependency-ordered build tasks
- [`CLAUDE.md`](CLAUDE.md): invariants for coding agents working in this repo

> **Note:** ESPN has no public write API, and automating it goes against the letter of Disney's Terms of Use (see
> DESIGN §6.4). This is a single-user tool. It acts only on its owner's team, at low volume, and every trade needs
> human approval.

## License

MIT. See [LICENSE](LICENSE).
