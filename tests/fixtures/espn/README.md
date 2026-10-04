# ESPN fixtures

ESPN JSON in the shape the unit tests parse offline. No real league names, manager names, cookies or tokens.

| File | View | League |
|---|---|---|
| `ffl_settings_ppr.json` | `mSettings` | 10-team NFL, full PPR (REC 1.0) with a D/ST points override on INT, FAAB ($100), Wednesday 3 a.m. ET waivers, 14 regular-season weeks with weeks 15–17 playoffs (6 teams), individual-game locks, Nov 25 2026 trade deadline |
| `fba_settings_points.json` | `mSettings` | 10-team NBA H2H points with ESPN's default scoring, PG/SG/SF/PF/C/G/F + 3 UTIL + 3 BE + IR, 4 adds per matchup, 19 matchup weeks of daily scoring periods, matchups 20–22 playoffs, Feb 4 2027 trade deadline |
| `fba_settings_9cat.json` | `mSettings` | Same roster, H2H most-categories 9-cat (TO reversed), FAAB ($200, $1 minimum), 60 adds per season, per-slot games-played limits, no trade deadline |

The `*_settings_*.json` files are hand-built stand-ins in the shape of ESPN's `mSettings` response (field catalogue
from `cwendt94/espn-api`), not live captures. ROADMAP #14 replaces them with scrubbed real-league captures and settles
the open unknowns (the lineup-lock key and value set, the pending-offer view).
