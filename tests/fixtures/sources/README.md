# Source fixtures

Recorded (or, where noted, hand-built) responses for the adapters in `fm.sources`, one directory per source, read
offline by `tests/sources/test_<source>.py`. No cookies, tokens, API keys or manager names; player names are public
figures. Captured 2026-10-04 (NFL week 4, NBA preseason) unless noted.

## `nflverse/` (`fm.sources.nflverse`)

Trimmed real nflreadpy downloads keyed by file name: `db_playerids.csv`, `fp_latest_weekly.csv`, `games.parquet`,
`depth_charts_2026.parquet`, `ep_weekly_2026.parquet`, `injuries_2026.parquet`, `snap_counts_2026.parquet` and
`stats_player_week_2026.parquet`. The test replaces `NflverseDownloader._download_file` with a server for these
files, so the real loaders (season filters, roof cleanup, URL building) still run. The parquet files stay parquet: a
CSV conversion would widen the dtypes the crosswalk and projection blend depend on.

## `sleeper/` (`fm.sources.sleeper`)

Sleeper responses trimmed to a dozen players: `players_nfl.json`, `state_nfl.json`, `projections_2026_4.json`,
`stats_2026_4.json`, `trending_add.json` and `trending_drop.json`. `projections_legacy_junk.json` is what the broken
`api.sleeper.app/v1/projections` endpoint answered that day.

## `nba_stats/`, `nba_schedule/`, `nba_injuries/`, `darko/` (the NBA adapters)

| File | Source | Provenance |
|---|---|---|
| `nba_schedule/scheduleLeagueV2.json` | `cdn.nba.com/static/json/staticData/scheduleLeagueV2.json` (2026-27) | Real capture cut to the finished preseason game, opening-week games of ATL/CHA/OKC/IND/PHI (back-to-backs included), one NBA Cup group game and the seven knockout placeholders (`TBD` teams); `broadcasters` and `pointsLeaders` dropped |
| `nba_injuries/espn_injuries.json` | `site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries` | Real capture cut to three teams and eight players; athlete `links` trimmed to the player card, `notes` and team `logos`/`links` dropped. Micah Peavy is filed under Memphis with a New Orleans athlete record, as ESPN served it |
| `nba_injuries/Injury-Report_2026-10-21_05PM.pdf` | `ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<time>.pdf` | Hand-built one-page PDF in the official report's row layout (the 2026-27 season had not started, so no report existed yet) |
| `darko/talent.csv`, `darko/daily.csv` | DARKO Google Sheet tabs `gid=284274620` (per-100 talent with minutes and pace) and `gid=951008984` (daily box projections) | Real exports (June 2026 end-of-season run) cut to a dozen players each, columns intact. DARKO's Shiny app closed in June 2026 and its successor `darko.app` offers only a client-side CSV button, so the adapter reads the public sheet exports |
| `nba_stats/leaguegamelog_2025-26.json`, `leaguedashplayerstats_{Base,Advanced,Usage}_2025-26.json`, `teamplayeronoffdetails_OKC_2025-26.json` | `stats.nba.com/stats/*` | Hand-built in the documented `resultSets` shape using `nba_api`'s own header lists (`expected_data`); stats.nba.com answered only Akamai read timeouts from this network when the fixtures were built. Values are plausible, not real. Smoke-test the endpoint live from the home PC before anything relies on it (ROADMAP #17, #43) |
| `nba_stats/access_denied.html` | Akamai block page | Hand-built stand-in for the non-JSON body stats.nba.com returns to a blocked client |

## `market/` (`fm.sources.market`)

Recorded from the two public, no-auth APIs the adapter reads, then trimmed. ESPN's pool is the league-less default
(`leaguedefaults/3`), so every entry has `onTeamId: 0` and no manager, league or team name appears.

| File | Request | Trimming |
|---|---|---|
| `fantasycalc_redraft_12t_1qb_ppr1.json` | `GET https://api.fantasycalc.com/values/current?isDynasty=false&numQbs=1&numTeams=12&ppr=1` (197 entries) | 12 entries kept whole: the top 8, the first QB and TE, a mid-tier WR (rank 93) and the tail (rank 197, value 5) |
| `espn_ffl_kona_player_info.json` | `GET https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2026/segments/0/leaguedefaults/3?view=kona_player_info` with `X-Fantasy-Filter: {"players":{"limit":30,"sortPercOwned":{"sortAsc":false,"sortPriority":1}}}` | 10 of 30 players (QB, RB, WR, TE, K); `player.stats`, `outlooks`, `seasonOutlook`, `rankings` and `lastVideoDate` removed |
| `espn_fba_kona_player_info.json` | the same view for `fba/seasons/2027` | 7 of 30 players (PG, SG, SF, PF, C); the same keys removed |

The adapter adds `"filterStatsForTopScoringPeriodIds": {"value": 1, "additionalValue": []}` to its filter so ESPN
returns one stat line per player instead of about sixty (the value must be greater than 0: 0 is an HTTP 400); the
fixtures carry none because the parser never reads them.

## `odds/` and `weather/` (`fm.sources.odds`, `fm.sources.weather`)

| File | Source | Provenance |
|---|---|---|
| `odds/espn_scoreboard_nfl_2026_w4.json` | ESPN public scoreboard (`site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard`) | Recorded 2026-10-04 (NFL week 4) and trimmed to three events: ATL @ NO (scheduled, DraftKings line, dome), DET @ CAR (in progress, so no `odds`; the AccuWeather `displayValue`/`conditionId` pair arrived swapped) and IND vs WSH (final, neutral site in London). Deep links, logos and the duplicated competition status were dropped |
| `odds/odds_api_events.json` | The Odds API v4 `/sports/americanfootball_nfl/events` | Hand-built in the documented shape (two events; ids are made up) |
| `odds/odds_api_event_team_totals.json` | The Odds API v4 `/sports/americanfootball_nfl/events/{id}/odds?markets=team_totals` | Hand-built in the documented shape: two books, main team-total line per team with over/under prices |
| `weather/open_meteo_hourly_metlife.json` | Open-Meteo `v1/forecast` | Recorded 2026-10-04 for MetLife Stadium's coordinates: 48 hours from 2026-10-05T00:00 UTC in °F, mph and inches (`timezone=UTC`, which Open-Meteo echoes as `GMT`) |

Stadium coordinates for the weather lookups live in the repo's `data/stadiums.csv` (hand-entered from public stadium
locations, accurate well inside the 0.01-degree cell Open-Meteo is queried at; the `greerreNFL/stadiums` CSV named in
DESIGN section 7 was not reachable when the file was built).
