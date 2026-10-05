# Sport plugin fixtures

ESPN pro schedules in the shape of the public `proTeamSchedules_wl` view
(`/apis/v3/games/{game}/seasons/{season}?view=proTeamSchedules_wl`). The ESPN read client (ROADMAP #9,
`fm.espn.models.ProSchedule`) parses this view in production; `tests/sports/test_nfl.py` reads the NFL file through a
small test double with the same surface (`fm.sports.base.ScheduleLike`) and `tests/sports/test_nba.py` parses the NBA
file with `ProSchedule` itself, to check each sport plugin's lock and period semantics offline. No league, manager,
cookie or token data: the view is public and carries only pro teams and games.

| File | Game | Contents |
|---|---|---|
| `ffl_pro_schedule_2026.json` | `ffl` | Real capture from 2026-10-04, trimmed to scoring periods 4 and 5 and with the per-team `teamPlayersByPosition` lists removed. All 32 pro teams plus ESPN's `FA` pseudo-team (id 0). Week 4: 16 games, Thursday night, the 9:30 a.m. ET international slot, the 1:00 / 4:05 / 4:25 Sunday windows, Sunday night and Monday night, no byes. Week 5: 15 games, CAR and KC on bye (matching their `byeWeek`), Monday night BUF at LAR |
| `fba_pro_schedule_2027.json` | `fba` | Real capture (public, no cookies) from 2026-10-05, trimmed to scoring periods 1-3 (Oct 20-22, 2026: 3, 11 and 2 games) and 67 (Christmas Day: 5 games from a noon ET tip). All 30 pro teams plus ESPN's `FA` pseudo-team (id 0) in ESPN's order; `settings` keeps only `proTeams` (ROADMAP #17) |

Each game appears under both teams, exactly as ESPN sends it: `{awayProTeamId, date (epoch ms), homeProTeamId, id,
scoringPeriodId, startTimeTBD, statsOfficial, validForLocking}`. Weeks 16-18 of the live NFL payload carry
`startTimeTBD: true` / `validForLocking: false` with a 03:01 ET placeholder for flex-scheduled games; the tests
exercise that by editing a game in the fixture rather than carrying those weeks. ESPN spells six NBA teams its own way
here (`NY`, `SA`, `GS`, `NO`, `UTAH`, `WSH`); `fm.espn.ids.FBA_PRO_TEAMS` maps the same ids to nba.com tricodes.

What the full NBA capture showed (1,200 games, periods 1-174):

- Period 1 is opening night, Tue Oct 20, 2026, and every game tips on the US Eastern day its period number implies,
  10:30 p.m. ET tips (02:30 UTC the next day) included. No game was `startTimeTBD` or not `validForLocking`.
- Days without games keep their numbers and are absent: 15 (Election Day), 38 (Thanksgiving), 46-53 (the NBA Cup
  knockout window, not yet scheduled), 66 (Christmas Eve), 123-128 (the All-Star break) and 173. The schedule runs to
  period 174 (Apr 11, 2027), past the real league's `finalScoringPeriod` of 153.

`settings.typeNames.locktimeTypes`, the value set behind `rosterSettings.lineupLocktimeType`, was
`["INDIVIDUAL_GAME", "FIRSTGAME_SCORINGPERIOD"]` in the NFL capture and adds `INDIVIDUAL_FIRSTGAME_WEEKLY` and
`FIRSTGAME_WEEKLY` for `fba`. Both real leagues lock lineups per game (`INDIVIDUAL_GAME`, ROADMAP #14,
`docs/espn-api.md`); `fm.espn.settings.LockType` carries the two per-period values, and the weekly ones parse as
`UNKNOWN`, which no plugin computes a lock for.
