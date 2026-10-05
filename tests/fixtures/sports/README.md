# Sport plugin fixtures

ESPN pro schedules in the shape of the public `proTeamSchedules_wl` view
(`/apis/v3/games/{game}/seasons/{season}?view=proTeamSchedules_wl`). The ESPN read client (ROADMAP #9,
`fm.espn.models.ProSchedule`) parses this view in production; `tests/sports/test_nfl.py` reads it through a small
test double with the same surface (`fm.sports.base.ScheduleLike`) to check the sport plugin's lock and period
semantics offline. No league, manager, cookie or token data: the view is public and carries only pro teams and games.

| File | Game | Contents |
|---|---|---|
| `ffl_pro_schedule_2026.json` | `ffl` | Real capture from 2026-10-04, trimmed to scoring periods 4 and 5 and with the per-team `teamPlayersByPosition` lists removed. All 32 pro teams plus ESPN's `FA` pseudo-team (id 0). Week 4: 16 games, Thursday night, the 9:30 a.m. ET international slot, the 1:00 / 4:05 / 4:25 Sunday windows, Sunday night and Monday night, no byes. Week 5: 15 games, CAR and KC on bye (matching their `byeWeek`), Monday night BUF at LAR |

Each game appears under both teams, exactly as ESPN sends it: `{awayProTeamId, date (epoch ms), homeProTeamId, id,
scoringPeriodId, startTimeTBD, statsOfficial, validForLocking}`. Weeks 16-18 of the live payload carry
`startTimeTBD: true` / `validForLocking: false` with a 03:01 ET placeholder for flex-scheduled games; the tests
exercise that by editing a game in this fixture rather than carrying those weeks.

The same capture's `settings.typeNames.locktimeTypes` was `["INDIVIDUAL_GAME", "FIRSTGAME_SCORINGPERIOD"]`, the value
set behind `rosterSettings.lineupLocktimeType`; `fm.espn.settings.LockType` carries both names (ROADMAP #14 confirms
against a league capture).
