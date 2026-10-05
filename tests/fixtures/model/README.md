# Scoring and projection fixtures (ROADMAP #15)

Read offline by `tests/model/test_scoring.py`, `tests/model/test_projections.py` and `tests/model/test_availability.py`.
No league, team or manager data, no cookies or tokens; player names are public figures.

| File | Source | Contents |
|---|---|---|
| `espn_ffl_pool_week4.json` | ESPN's league-less default player pool (`lm-api-reads` `leaguedefaults/3?view=kona_player_info`, the cookie-free pool `tests/fixtures/sources/market/` also comes from), captured 2026-10-05 | Eight players, one per line: Josh Allen and Patrick Mahomes (QB), Derrick Henry and Jahmyr Gibbs (RB), Ja'Marr Chase (WR, listed `QUESTIONABLE`), Trey McBride (TE), Harrison Butker (K) and the Eagles D/ST. Each keeps its 2026 week-4 actual line (`statSourceId` 0) and ESPN's week-4 projection (`statSourceId` 1) whole, with ESPN's `appliedTotal`; every other stat entry was dropped |

`appliedTotal` is ESPN's own points for a line under the default league's scoring, which is the PPR fixture
(`tests/fixtures/espn/ffl_settings_ppr.json`) minus its missed-PAT penalty: the scoring tests reproduce every total
from the stat lines alone. Every entry has `onTeamId: 0`. The game ids in the actual lines (`externalId`) match the
week-4 games in `tests/fixtures/sports/ffl_pro_schedule_2026.json`, and five of the players (Allen, Mahomes, Henry,
Gibbs and the Eagles) also appear in Sleeper's week-4 projections under `tests/fixtures/sources/sleeper/`, which is what
the ESPN + Sleeper blend tests join through the crosswalk built from `tests/fixtures/sources/nflverse/db_playerids.csv`.
