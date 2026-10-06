# Fixture home

A complete `~/.config/espn-fantasy` + `~/.cache/espn-fantasy` pair for the advisor CLI (ROADMAP #26), built from the
hand-built ESPN fixtures only. No real league, manager, cookie or token: the league is `Fixture League (PPR)`
(`tests/fixtures/espn/ffl_*.json`, ESPN id `1234567`, week 4 of 2026), the teams are `Fixture Team N`, and player
ids and names are public NFL data.

```sh
FM_CONFIG_DIR=tests/fixtures/home FM_CACHE_DIR=tests/fixtures/home/cache uv run fm status
FM_CONFIG_DIR=tests/fixtures/home FM_CACHE_DIR=tests/fixtures/home/cache uv run fm lineup --as-of 2026-10-04T15:00Z --opponent 2 --dry-run
FM_CONFIG_DIR=tests/fixtures/home FM_CACHE_DIR=tests/fixtures/home/cache uv run fm waivers --as-of 2026-10-04T15:00Z --dry-run
```

| Path | What |
|---|---|
| `config.toml` | The league (`nfl`, team 1) with the default policy and one untouchable (Josh Allen). |
| `state.db` | The store after `fm.jobs.sync.sync_league` read the fixtures (settings, teams, rosters, players, ESPN projections, the indexed captures). No proposals. Committed although `*.db` is gitignored: `git add -f`. |
| `cache/espn/ffl/2026/...` | The sync's captures: the pool page `fm waivers` reads the wire from and the pro schedule (`tests/fixtures/sports/ffl_pro_schedule_2026.json`, weeks 4 and 5) `fm lineup` reads lock times and byes from. |
| `build.py` | Rebuilds `state.db` and `cache/` (`uv run python tests/fixtures/home/build.py`; with a directory argument, builds there). |
| `snapshots/*.txt` | Expected output of the three commands, compared by `tests/test_advise_cli.py`; rewrite with `FM_UPDATE_SNAPSHOTS=1 uv run pytest tests/test_advise_cli.py`. |

Both environment variables matter: the store indexes its captures by paths under the cache dir. With `FM_CONFIG_DIR`
alone the commands still exit 0 but `fm waivers` sees an empty wire and `fm lineup` plans nothing, each with a warning
naming the fix.

The sync is stamped Sunday 2026-10-04 15:00Z (11 a.m. ET), after Thursday night and before the early games, so
`--as-of 2026-10-04T15:00Z` replays the week from there: Gibbs is locked, everyone else is movable, the waiver run
is Monday 16:40Z and Carolina's Sunday-night kickoff is the free-agent deadline. Without `--as-of` the commands use
the wall clock; once week 4 is over everyone is locked and every wire candidate is skipped (with the reason), so the
output is a report of a finished week rather than an error.

`fm lineup` and `fm waivers` without `--dry-run` store proposals in `state.db`; the tests copy the home into a temp
dir first. If you run them against the committed home, `git checkout -- tests/fixtures/home/state.db` restores it.
