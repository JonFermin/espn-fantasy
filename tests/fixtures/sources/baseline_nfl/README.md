# NFL opportunity baseline inputs

Hand-built stand-ins, not captures, for `tests/model/test_baseline_nfl.py` (ROADMAP #42). `build.py` writes them
(`uv run python tests/fixtures/sources/baseline_nfl/build.py`; seeded, so the output is stable). The players are the
backtest fixture's twelve QB/RB/WR/TE on ten teams (BUF, BAL, DET, ATL, CAR, PHI, CIN, NO, ARI, KC); their GSIS ids
are real for Allen, Gibbs and Chase (as in `tests/fixtures/sources/nflverse/db_playerids.csv`) and made up for the
rest, PFR ids made up. Filler players (`00-0090xxx`, no ESPN id) complete each team's pass attempts, targets, carries
and air yards so team totals sum the way nflverse's do.

Nothing here reads the backtest fixture's actual or projected lines. Each player has a role profile written from
public knowledge of the 2025 season (shares of his team's volume, efficiency per opportunity, snap share) and every
game draws around it with noise, scaled by that game's implied total. So the history is independent of the lines the
backtest replays, and the two are not one world: the backtest fixture's generated lines run lower than real NFL
levels (its quarterbacks throw for about 170 yards), which is why the equal-weight acceptance test in
`test_baseline_nfl.py` is a strict `xfail`.

| File | Shape | Content |
|---|---|---|
| `player_stats.json` | nflverse `player_stats` columns (subset) | 2025 weeks 1-17 and 2026 weeks 1-4, regular season. Chris Olave has no 2026 rows in weeks 1-2 (IR, as in the backtest fixture); BUF has no 2026 week 2 game and CAR none in week 3 (the fixture's byes) |
| `snap_counts.json` | nflverse `snap_counts` columns (subset) | `offense_pct` per named player-game, keyed by PFR id |
| `schedules.json` | nflverse `schedules` columns (subset) | every game of those teams (`spread_line` is the home margin, positive when home is favored; `total_line`). 2026 week 4 has ATL @ NO (spread and total as in the recorded scoreboard) and DET @ CAR |
| `player_ids.json` | `ff_playerids` columns (subset) | `espn_id`, `gsis_id`, `pfr_id` for the twelve |
| `scoreboards.json` | week to games | the 2026 slates as ESPN's scoreboard would carry them before kickoff: `home`, `away`, `spread` (home spread, negative when home is favored) and `over_under`, nflverse team codes |

The two backtest fixtures answer different questions. Scored on `tests/fixtures/backtest/nfl/` (made-up levels) this
history hurts the equal-weight blend; `tests/fixtures/backtest/nfl_world/` generates history, actuals and the other
sources' projections from one world, and its README pre-registers the protocol and records the result.
