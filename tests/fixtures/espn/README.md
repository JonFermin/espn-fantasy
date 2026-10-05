# ESPN fixtures

ESPN JSON in the shape the unit tests parse offline. No real league names, manager names, cookies or tokens: members
are `Manager N` with all-zero SWIDs, teams are `Fixture Team N`. Player ids and names are real NFL/NBA players (public
data).

All files are hand-built stand-ins in the shape of ESPN's responses (field catalogue from `cwendt94/espn-api` and the
ESPN web app), not live captures. ROADMAP #14 replaces them with scrubbed real-league captures and settles the open
unknowns (the lineup-lock key and value set, the pending-offer view and its list key, the player-card filter).

## Settings (`mSettings`, parsed by `fm.espn.settings`)

| File | League |
|---|---|
| `ffl_settings_ppr.json` | 10-team NFL league `1234567`, 2026, full PPR (REC 1.0) with a D/ST points override on INT, FAAB ($100), Wednesday 3 a.m. ET waivers, 14 regular-season weeks with weeks 15–17 playoffs (6 teams), individual-game locks, Nov 25 2026 trade deadline |
| `fba_settings_points.json` | 10-team NBA league `2345678`, 2027, H2H points with ESPN's default scoring, PG/SG/SF/PF/C/G/F + 3 UTIL + 3 BE + IR, 4 adds per matchup, 19 matchup weeks of daily scoring periods, matchups 20–22 playoffs, Feb 4 2027 trade deadline |
| `fba_settings_9cat.json` | NBA league `3456789`, 2027, same roster, H2H most-categories 9-cat (TO reversed), FAAB ($200, $1 minimum), 60 adds per season, per-slot games-played limits, no trade deadline |

## Read views (`fm.espn.client` / `fm.espn.models`)

The NFL files describe league `1234567` in week 4 of 2026 (matchup period 4, our team is 1, opponent 2); the NBA
files describe the 9-cat league `3456789` on its opening day (scoring period 1 of 2027).

| File | View | Contents |
|---|---|---|
| `ffl_teams.json` | `mTeam` + `mStandings` | 10 teams with records (home/away/division splits on team 1 only), seeds, waiver ranks, FAAB spent, `moveToIR`, `valuesByStat` with a junk key; team 10 has `location`/`nickname` instead of `name`; 10 scrubbed members, member 3 is league manager; `status` with the next waiver run |
| `ffl_rosters_week4.json` | `mRoster` (`scoringPeriodId=4`) | Team 1's 11-player roster (Gibbs locked after Thursday night with season + week stat lines, McBride likewise, Allgeier added off waivers, Shaheed with a pending claim id, Olave on IR) and 3 players of team 2 without stat lines |
| `ffl_matchups.json` | `mMatchup` | Two decided week-3 matchups, five undecided week-4 matchups, a week-15 playoff bye (home only) and a winners-bracket game; `scoreByStat` is `null` as in points leagues |
| `ffl_scoreboard_week4.json` | `mMatchupScore` + `mScoreboard` | Week 4 with live and projected totals; matchup 16 carries `rosterForCurrentScoringPeriod` for both sides |
| `ffl_free_agents_week4.json` | `kona_player_info` | Four pool players sorted by `percentOwned`: free agents and one on waivers with `waiverProcessDate`; ownership and ratings on the first two |
| `ffl_player_cards_week4.json` | `kona_playercard` | Gibbs and Chase with season actual/projected and week-4 projected/actual stat lines, ownership and ratings |
| `ffl_transactions_week4.json` | `mTransactions2` (`scoringPeriodId=4`) | An executed waiver claim ($17), the losing bid for the same player (`FAILED_INVALIDPLAYERSOURCE`, $9 kept), a free-agent add/drop, a pending trade proposal from team 2 and our pending $12 claim |
| `ffl_pending_transactions.json` | `mPendingTransactions` | The two pending rows above plus an incoming trade offer from team 6, under a `pendingTransactions` key (unconfirmed; the parser also reads `transactions`) |
| `ffl_pro_schedule_2026.json` | `proTeamSchedules_wl` (season level) | 14 NFL teams plus the free-agent pseudo team, weeks 4 and 5 (Thursday through Monday night), PIT/SF on bye in week 4 and ATL/PHI in week 5; both teams list each game |
| `fba_scoreboard_9cat_day1.json` | `mMatchupScore` + `mScoreboard` | One category matchup with `cumulativeScore.scoreByStat` for the nine categories (WIN/LOSS/TIE) and two-player lineups with day and season stat lines (the day lines in `fba`'s "Game" split 5, keyed by pro game id as ESPN keys them) |
| `fba_pro_schedule_2027.json` | `proTeamSchedules_wl` (season level) | Six NBA teams over the first two days of 2026-27; the 7:30 p.m. ET tip on day 1 is the add/drop cutoff the tests check |

Timestamps are epoch milliseconds computed from US Eastern wall-clock times (waiver run 3 a.m. ET, kickoffs and
tips).
