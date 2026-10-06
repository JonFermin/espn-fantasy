# Consistent-world NFL backtest (ROADMAP #42)

A second NFL backtest fixture, in the format documented in `tests/fixtures/backtest/README.md` (`backtest.json`,
format 1, plus `settings.json`, a copy of `../nfl/settings.json`: the fixture PPR league). It exists because the first
fixture cannot test the NFL opportunity baseline fairly: that fixture's actuals and projections are generated around
made-up season levels, while the baseline's inputs (hand-built nflverse history, `tests/fixtures/sources/baseline_nfl/`)
carry real-world levels, so the baseline learns in one world and is scored in another (its equal-weight blend is about
3% worse there). Here **one world** produces the history the baseline learns from, the actuals it is scored against
and the lines the competing sources project.

`build.py` writes every file in this directory and is the only way they change. It is deterministic.

## Protocol (written before any baseline score was computed)

1. **The world.** The roles, shares, team volumes, efficiencies, snap shares and noise of
   `tests/fixtures/sources/baseline_nfl/build.py` (imported, not copied), with a new seed (`WORLD_SEED = 20261101`):
   ten teams, 2025 weeks 1-17 as history and 2026 weeks 1-10 as the replayed season, BUF on bye in week 2 and CAR in
   week 3, Chris Olave on IR in weeks 1-2. Each team-game's implied total comes from a generated schedule line
   (`spread_line`, `total_line`) and scales volume (elasticities 0.4 pass, 0.2 rush) and touchdown rates (0.6, 0.8),
   plus noise on volume, efficiency and touchdown counts (Poisson). The true weekly stat lines are the world's
   realised lines. The actuals of `backtest.json` are those lines for 2026 (the nine scored stats `PY PTD INTT RY RTD
   REC REY RETD FUML`).
2. **The baseline's inputs.** The same world's nflverse-shaped files in this directory (`player_stats.json`,
   `snap_counts.json`, `schedules.json`, `player_ids.json`) and the pregame lines in `scoreboards.json`. For week `w`
   the baseline reads only games before `w` (the module enforces it; a test proves it). The baseline's constants are
   those committed in `fm.model.baseline_nfl` at the time of writing; they are not changed after seeing a result.
3. **The competing sources.** ESPN- and Sleeper-like projections are noisy, slightly biased observations of each
   week's **true expectation**: the world's mean line for that player-week given his role, the team's volume and the
   game's implied total, before game noise. Per stat, `expectation x (1 + bias) x lognormal(mean 1, sigma)`, drawn
   from a separate seed (`SOURCE_SEED = 20261102`):

   | Source | sigma (relative, per stat) | bias | Rationale |
   |---|---|---|---|
   | `espn` | 0.22 | +3% | tighter source |
   | `sleeper` | 0.30 | +6% | noisier, runs higher, as in the first fixture (MAE about 3.9 vs 4.2) |

   The first fixture's per-source MAE (espn about 3.9, sleeper about 4.2) cannot be the target here: its game noise is
   lower than a real NFL week's, and any projection of this world is bounded below by the game noise. The sigmas come
   from published weekly-projection accuracy instead (expert projections miss individual stats by roughly 20-30%) and
   keep the first fixture's ordering and ratio of the two sources. A player on bye gets an empty line from `espn`
   and no line from `sleeper`; a player on IR is projected by every source (his actual is missing, which scores 0).
4. **The players.** The twelve named QB/RB/WR/TE of the first fixture, as the manager's roster (fixed lineup: Allen,
   Gibbs, Bijan Robinson, Chase, Shaheed, McBride and Hubbard at FLEX), plus the world's filler players (a team's
   aggregate remaining RB/WR/TE volume, and a quarterback for teams without a named one, ESPN ids from 9,000,000)
   so the position MAEs have a sample of about 500 player-weeks rather than 100. No kickers or defenses.
5. **The evaluation.** `with_source("opportunity", rows)` and `with_blend` at equal weights (`espn = sleeper =
   opportunity = 1`) against `espn = sleeper = 1`, then `run_backtest(restrict_to_common=True)`. The criterion is the
   overall MAE of the two blends. If the blend with the baseline is no worse, the test asserts it; if not, the test is a
   strict `xfail` and the numbers are reported. Nothing in the world, the noise, the sample or the baseline is changed
   after seeing the result.

Caveats stated up front: this world generates stats with the same structural form the baseline models (volume times
share times efficiency, scaled by the implied total), and the sources see the true expectation, including each team's
actual implied total, so the test checks that the baseline can recover a consistent world from noisy history and
whether doing so adds anything beside two sources that observe the truth with error. It says nothing about real NFL
weeks, which the hand-built fixture cannot.

## Files

| File | Content |
|---|---|
| `backtest.json`, `settings.json` | the replay (format 1): `espn`, `sleeper` projections and the actuals of 2026 weeks 1-10 |
| `player_stats.json`, `snap_counts.json`, `schedules.json`, `player_ids.json` | nflverse-shaped history of the same world (2025 weeks 1-17, 2026 weeks 1-10), in the shapes described in `tests/fixtures/sources/baseline_nfl/README.md`; filler players have a GSIS id and an ESPN id but no PFR id |
| `scoreboards.json` | the 2026 pregame lines by week: `home`, `away`, `spread` (home, ESPN's sign), `over_under` |
