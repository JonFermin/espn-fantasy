# Backtest fixtures

Replay data for `fm backtest` and `fm.eval.backtest` (ROADMAP #33). `fm backtest --sport nfl --fixtures
tests/fixtures/backtest` reads `<dir>/<sport>/backtest.json`, with no network and no state database.

Hand-built stand-ins, not captures: the players are real, public NFL players (ids as in `tests/fixtures/espn/`) on a
made-up roster, the league is the fixture PPR league (`nfl/settings.json` is a copy of
`tests/fixtures/espn/ffl_settings_ppr.json`: QB, 2 RB, 2 WR, TE, FLEX, D/ST, K, 7 bench, IR), and every stat line is
generated: each player has a season-level stat line, each week a random form factor scales it, actuals are noisy draws
around that, and the two projection sources see half the form plus their own error (ESPN is tighter on RB and D/ST,
Sleeper on QB and TE). No manager names, no league ids beyond the fixture league.

## `nfl/backtest.json` (format 1)

| Key | Meaning |
|---|---|
| `format` | `1`; the loader rejects anything else |
| `sport`, `season` | `nfl`, `2026` |
| `settings` | League settings file (raw ESPN `mSettings`), relative to the JSON |
| `as_of` | Stamp for the rows built from the file (the harness ignores it) |
| `players[]` | `espn_id`, `name`, `position` (ESPN label), `eligible_slots` (ESPN's `eligibleSlots`, which includes slots the league does not use) |
| `weeks[].period` | Scoring period (NFL week); weeks are unique and ascending |
| `weeks[].lineup` | The manager's roster that week: ESPN id to the slot he held (`0` QB, `2` RB, `4` WR, `6` TE, `16` D/ST, `17` K, `23` FLEX, `20` bench, `21` IR). Its keys are the roster the hindsight lineup draws from, its starters the baseline lineup |
| `weeks[].projections` | Source name to ESPN id to projected stat line (ESPN stat abbreviations). Any number of sources. An empty line `{}` is ESPN's projection for a player on bye; a source may also simply omit a player. Players need not be on the roster (Barkley is not) |
| `weeks[].actuals` | ESPN id to actual stat line. A player with no entry did not play and scored 0 |

Every id in `lineup`, `projections` and `actuals` must be in `players`; a slot must be one the league has.

## What the fixture contains

- Four weeks, 14 players, sources `espn` and `sleeper` (the command adds `blend`).
- Chuba Hubbard (RB) is on bye in week 3 and the manager starts him at FLEX; Josh Allen is on bye in week 2 and the
  manager starts Lamar Jackson; both sources project the bye player as an empty (ESPN) or missing (Sleeper) line.
- Harrison Butker (K) is inactive in week 4: both sources project him, there is no actual line.
- Chris Olave is on IR in weeks 1 and 2 and returns in week 3, and the week 4 lineup swaps him in for Rashid Shaheed.
- The manager's lineups are chalk by preseason rank; the lineup efficiency baseline of this fixture is about 88%.

To make a new fixture: write `<sport>/backtest.json` and a settings file in this format. There is no generator in the
repo; the loader's error messages name the offending key.
