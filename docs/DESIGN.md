# ESPN Fantasy Manager: Design

> Status: draft v0.2, 2026-10-04; your answers are recorded in §18 · Sports: NFL (`ffl`), NBA (`fba`) · Platform: ESPN Fantasy
> Planning doc; there is no code yet. `ROADMAP.md` is generated from this file.

## 1. Summary

A personal, local-first "assistant GM" for your ESPN fantasy football and basketball teams. It reads league state
from ESPN's JSON API and blends player data from several public sources into league-specific projections. A
deterministic engine decides lineups, waiver/free-agent moves, streaming, and trades. Claude reads the news, breaks
close calls, and explains every decision. Approved moves are carried out through a logged-in Playwright browser
session, then verified against the API. The session sends the ESPN web app's own transaction calls, with UI
click-through as the fallback.

**Decisions it owns**

| | NFL | NBA |
|---|---|---|
| Lineups | Weekly, locks per game; finalize after inactives (~90 min before kickoff) | Daily, locks per game; re-check late scratches before each tip window |
| Adds / drops | Waiver claims (FAAB or priority) before the league's waiver run; free agents after | Streaming under the league's acquisition limits |
| Trades | Evaluate incoming offers; find and propose outgoing ones | Same, category-aware |
| Strategy | Matchup outlook, bye coverage, playoff odds | Weekly category targets and punts, playoff-week game counts |

**Non-goals for v1:** drafting (both 2026 drafts are done; revisit next season), Yahoo/Sleeper leagues (keep the seam), DFS or betting, hosting for other
people, commissioner tools.

## 2. Where the edge actually comes from

Ordered by value per unit of effort. The phase plan follows this order.

1. **Never start a zero.** Benching OUT, inactive, bye-week, and no-game players is the biggest and most automatable
   gain. NBA has roughly 25 lineup decisions per week against NFL's one, so it matters most there.
2. **Use every slot.** NBA streaming (games played per week) and NFL bye and injury coverage.
3. **React to news before league-mates.** Injury replacements and role changes are won on waivers within hours.
4. **Trade where the model disagrees with the market**, and where rosters complement each other.
5. **Better projections.** This is a real gain, but the smallest at the margin over ESPN's own numbers. Blend sources
   first, and build an in-house model only where backtests show it helps.

## 3. Timing (as of 2026-10-04)

- **NFL is in Week 4.** Every week of delay costs real decisions, so a read-only NFL advisor ships first.
- **NBA 2026-27 opens Tue Oct 20, 16 days out.** The NBA plugin must be live by opening week.
- **NBA Cup:** the Dec 4–11 knockout window leaves most teams with only 1–3 games in the fantasy week of Dec 7–13. The 24
  consolation games aren't scheduled until group play ends, so re-pull the schedule in December.
- **Trade deadlines** are league settings (read from ESPN) and are commonly mid-to-late November. The trade engine has
  to land by early November to matter for NFL.

## 4. Principles

1. **The engine computes; Claude judges and explains.** Projections, scoring, optimization, and simulation are
   deterministic, tested Python. Claude handles unstructured input (news, injury notes, beat reports). It proposes
   *bounded, logged* adjustments, breaks near-ties with cited research, and writes rationales. It never does
   arithmetic the engine can do.
2. **Workers propose, the executor acts** (the same rule as software-factory). Everything the engine or Claude
   produces is a `Proposal`. Only `executor/` writes to ESPN, and only for proposals that cleared policy. No
   LLM-facing tool can write to ESPN.
3. **Read via API, write through the browser session, verify via API.** All reads use the JSON API that ESPN's own
   web app uses. Writes go through the logged-in Playwright session: the web app's own transaction request first, UI
   click-through as the fallback. A write counts as successful only after a re-read of the roster confirms it.
   Anything else is a failure and an alert.
4. **League settings are data.** Scoring items, roster slots, lock behavior, acquisition/FAAB rules, the trade
   deadline, and playoff weeks all come from ESPN's settings. No format is hardcoded: PPR, half-PPR, and custom
   scoring, plus NBA points, H2H categories, and roto, all go through one path.
5. **Every decision is replayable.** Each decision stores its inputs (with `as_of` timestamps), the engine's numbers,
   Claude's rationale, and the outcome. This powers backtests, the weekly report card, and tuning.
6. **Fail loud and safe near a lock.** If anything breaks close to a lock, send the exact manual fix (a deep link plus
   "start X over Y") instead of failing silently.

## 5. Architecture

```
 sources/ ──► store/ ──► model/ ──► decide/ ──► proposals/ ──► executor/ ──► ESPN website
                            ▲          │            ▲    policy:     │
                            │          ▼            │    auto |      ▼
                         advisor/ (Claude) ─────────┘    approve |   verify via API
                    news → bounded adjustments           off         audit log + notify
                    close calls · rationale · pitches
```

| Module | Responsibility |
|---|---|
| `espn/` | Read client for the ESPN API, cookie harvest from the browser profile, settings parser, stat/slot/position ID maps |
| `sources/` | One adapter per external source; caching, rate limits, `as_of` stamps |
| `store/` | SQLite state (leagues, rosters, projections, news, proposals, executions) plus parquet cache for bulk stats |
| `sports/` | Plugin per sport: stat and slot maps, schedule and lock semantics, which adapters and decision modules apply |
| `model/` | Player ID crosswalk → projection blend → availability → league scoring → valuation → season simulation |
| `decide/` | Lineup, waivers, streaming, trades, weekly plan; emits candidate moves with numbers |
| `advisor/` | Claude workers: news triage, close calls, rationale, trade pitches, weekly strategy |
| `proposals/` | Queue, policy evaluation, approvals, expiry |
| `browser/` + `executor/` | Playwright session, in-browser transaction writer plus UI fallback flows, selector registry, verification, audit artifacts, canary |
| `notify/` | Phone channel, Telegram or ntfy (approvals with buttons, alerts, reports) |
| `cli.py`, `mcp_server.py` | `fm` CLI; MCP server exposing read tools plus `create_proposal` (never execute) |

## 6. ESPN integration

### 6.1 Reads

- **Endpoint:** `https://lm-api-reads.fantasy.espn.com/apis/v3/games/{ffl|fba}/seasons/{season}/segments/0/leagues/{leagueId}?view=…`.
  The host moved off `fantasy.espn.com` in April 2024. Pro schedules come from
  `/apis/v3/games/{game}/seasons/{season}?view=proTeamSchedules_wl`.
- **Auth:** the `espn_s2` + `SWID` cookies (private leagues). Seasons before 2018 sit behind `leagueHistory`, which has
  also required `espn_s2` since Aug 2025.
- **Scoring period:** a week in `ffl`, a **day** in `fba`.

| Need | View / mechanism |
|---|---|
| Scoring items, slot counts, position limits, acquisition/FAAB/waiver timing, trade deadline, playoffs | `mSettings` |
| Teams, rosters (per `scoringPeriodId`), standings | `mTeam`, `mRoster`, `mStandings` |
| Matchups and live scores | `mMatchup`, `mMatchupScore`, `mScoreboard` |
| Free agents and waivers | `kona_player_info` with an `X-Fantasy-Filter` header (`filterStatus: FREEAGENT/WAIVERS`, slot filter, sort by % owned, `limit`) |
| Player detail, projections, news | `kona_playercard`; each stat entry has `statSourceId` (0 actual, 1 projected) and `statSplitTypeId` (0 season, 1 single period) |
| Transactions with FAAB bids | `mTransactions2` per `scoringPeriodId`; `bidAmount` is present on winning *and* failed bids, which feeds the bid model |
| Pending offers and claims | `mPendingTransactions`, or `mTransactions2` filtered by status. Reports conflict, so the spike settles it |

- **Client:** a thin `httpx` client over these views. Reuse the stat/slot/position maps from `cwendt94/espn-api`
  (v1.0.0, Oct 2026; maintained, read-only).
- **Pacing:** explicit timeouts, polite pacing, and backoff on 429. ESPN's rate limits are undocumented, so measure them
  in the spike.
- **Raw capture:** every response is saved, for fixtures and debugging.

### 6.2 Session and auth

- Use a dedicated persistent browser profile at `~/.config/espn-fantasy/browser-profile/`. Don't put it under
  `%APPDATA%`: MSIX virtualization already caused problems for software-factory there.
- `fm login` opens a headed browser and you sign in once by hand, which covers any one-time code or captcha. The
  session persists in the profile.
- At runtime, `espn_s2` and `SWID` are read from the profile's cookie jar for API calls. That gives one source of truth,
  and nothing ESPN-related goes in `.env`.
- Every run starts with a session check. On expiry, a phone notification says "run `fm login`".
- Drive the installed Edge or Chrome channel (`channel="msedge"`) rather than bundled Chromium. Run headless by
  default; if challenged, fall back to a headed, minimized window.
- The Disney/MyDisney login takes email + password or an emailed one-time code. No captcha has been observed and 2FA
  isn't forced. Every recent ESPN automation project skips scripted login, and so does this one.
- Session lifetime is unclear: reports range from ~30 days to ~1 year. Logout or a password reset invalidates it, and
  expiry shows up as a 401. Each tick checks the session and warns early based on the cookie's expiry date.
- Headless Playwright from home IPs was working in Sept 2026. There is one report of datacenter-IP blocks, which is
  another reason to run locally.

### 6.3 Writes

ESPN's web app writes through one undocumented endpoint:
`POST https://lm-api-writes.fantasy.espn.com/apis/v3/games/{game}/seasons/{season}/segments/0/leagues/{id}/transactions/`.
Several community projects ran lineups, adds, waivers, and trades through it live in Sept 2026. Meanwhile, the
Selenium click-bots in prior art broke with every ESPN UI change. So the executor has two modes, and both run in the
logged-in browser session:

1. **API mode (primary).** Send the same transaction request the web app sends, from inside the Playwright context
   (`context.request`: same cookies, same client). No DOM clicking on the happy path.
2. **UI mode (fallback).** A role/text-locator click-through of the same action, with selectors in
   `browser/selectors.py`. Used when API mode fails, or for any action not yet captured. A weekly dry-run drill keeps
   it from rotting.

Every request uses the envelope `{isLeagueManager: false, teamId, type, memberId: SWID, scoringPeriodId,
executionType: "EXECUTE", items: [...]}`:

| Action | `type` and items | Notes |
|---|---|---|
| Lineup | `ROSTER`; `LINEUP {playerId, fromLineupSlotId, toLineupSlotId}` | Include both sides of a swap; current period only (`FUTURE_ROSTER` exists for later periods) |
| Free-agent add/drop | `FREEAGENT`; `ADD {toTeamId}`, `DROP {fromTeamId}` | |
| Waiver claim | `WAIVER` plus a top-level `bidAmount` in FAAB leagues | Cancel: `executionType: "CANCEL"` + `relatedTransactionId` |
| Trade | `TRADE_PROPOSAL`; `TRADE {playerId, fromTeamId, toTeamId}` | Respond: `TRADE_ACCEPT` / `TRADE_DECLINE` + `relatedTransactionId` |

ESPN returns error codes such as `TRAN_LINEUP_LOCKED`, `TRAN_ROSTER_SAME_SLOT`, `TRAN_ROSTER_LIMIT_EXCEEDED`,
`FAILED_ROSTERLOCK`, and 401 `AUTH_MISSING_CREDENTIALS`. All of this is community-verified for `ffl` only; **`fba`
writes are unverified**, and the spike covers them.

| Flow | Preconditions (checked via API) | Verified via API |
|---|---|---|
| `set_lineup(moves)` | no affected player locked; slot eligibility | slots match the requested lineup |
| `add_drop(add, drop)` | add is a free agent (not on waivers); drop not protected; roster space; before the league's roster lock (`rosterLocktimeType`: the day's first tip in the NBA league) | roster has `add`, lacks `drop` |
| `claim_waiver(add, drop, bid)` | claim window open; budget ≥ bid; no duplicate claim | pending claim exists with the bid |
| `propose_trade(team, give, get)` | before deadline; **no player locked**; both rosters legal afterward; no open offer to that team | pending offer exists |
| `respond_trade(offer, accept\|decline)` | offer still pending; no player locked | status changed |
| `cancel(claim\|offer)` | still pending | gone |

**Write safety** (lessons from prior art, §6.5):

- **No automatic retries on writes.** There's no idempotency key, so a retry double-submits. A timeout marks the
  execution `UNKNOWN`, and the executor re-reads state before doing anything else.
- **Single-use execution token** per approved proposal: preview → approve → execute exactly once.
- **Check preconditions ourselves.** ESPN has accepted a trade that included a locked player, so lock status, roster
  limits, and existing pending offers/claims are all checked before every write.
- **Hard timeouts on every request.** Hung sockets were a failure mode in prior art.
- **`isLeagueManager` is always `false`.** Never use commissioner powers, even if the account has them.
- **Audit:** every execution saves the request and response, plus screenshots and a Playwright trace in UI mode, under
  `audit/`.
- `--dry-run` builds and logs the request (API mode) or walks the UI up to the final confirm button (UI mode), then
  stops.
- **Canary** (daily, read-only): assert that every selector resolves and that the read views parse. This alerts on
  ESPN drift before a real move depends on them.
- **Pacing:** jittered delays, a hard cap on writes per run, and one league at a time.
- **Spike method:** confirm payloads by driving each UI flow with `page.route` intercepting the transaction POST. Log
  the payload and **abort the request before it reaches ESPN**. This verifies shapes (including `fba`) with no side
  effects. The Playwright MCP server in Claude Code is handy for exploring the pages.

### 6.4 Terms of service and account risk

- Disney's Terms of Use (updated 2024-05-24) prohibit accessing or extracting the products "using a robot, spider,
  script, or other automated means" (§2.B.x). They also prohibit software that "allows automated gameplay" (§3.I), and
  they allow suspension or termination (§1.H). **This tool breaks the letter of those terms.**
- ESPN's Fair Play policy covers collusion and roster dumping. No documented enforcement against personal automation
  turned up, and community tools have run openly for years.
- ESPN ships its own "Auto Control AI" team management (toggled by the league manager; no trades). That suggests
  automated lineup management is tolerated, but it is not a license for this tool.
- **Footprint:** reads are the same calls the web app makes; writes are a handful per day; one account; own team only.
  Whether to run it is your call.

### 6.5 Prior art worth copying (research, 2026-10-04)

- **`gagandaroach/fantasy-yolo`** (MCP + CLI, reads and writes): preview → single-use token → execute, retries
  disabled, `UNKNOWN` on timeout. The executor copies this.
- **`kieran-venieris/fantasy-bot`** (Claude-run manager with permission tiers, dry-run by default). Things that broke:
  duplicate trade proposals, hung sockets, and ESPN accepting a trade that included a locked player.
- **`cwendt94/espn-api`** (Python, read-only, maintained): view usage plus stat/slot/position maps.
- **Read-only ESPN MCP servers** exist for football and basketball. Ours adds proposals to MCP, never writes.
- **Selenium lineup bots** (`fntsylu`, `fantasyLineup`, `rosterdriver`) all died after UI changes. Lesson: the UI
  can't be the primary write path.

## 7. Data sources

### NFL (verified 2026-10-04)

| Source | What we use | Access | Cost / limits | Freshness | Role |
|---|---|---|---|---|---|
| ESPN league API | League state, ESPN projections, free-agent pool, transactions | `espn_s2`/`SWID` cookies | none stated | live | core |
| ESPN site APIs | Scoreboard (DraftKings spread/O-U, AccuWeather, indoor flag), injuries, news, depth charts | no key; send a browser UA | none stated | live | core |
| nflverse via `nflreadpy` (Polars) | Weekly stats, snap counts, injuries + practice status, depth charts, schedules with lines, `ff_playerids`, `ff_opportunity` (expected FP), `ff_rankings` (FantasyPros ECR) | pip, no key | free | stats nightly; snaps ~6 h; injuries/depth daily; schedules ~5 min | core |
| Sleeper | Trending adds/drops; player DB (injury, practice, news timestamps); RotoWire-based projections and same-day snaps via undocumented endpoints | no auth | free; < 1,000 calls/min | real time | core (needs a fallback) |
| FantasyCalc | Redraft trade values keyed by `espnId` | no key | free; no published terms | ~daily | market value |
| RotoWire RSS | Player news | RSS | free; poll ≤ every 10 min | minutes | news |
| Open-Meteo | Hourly forecasts at stadium coordinates (`data/stadiums.csv`, hand-entered from public venue locations since `greerreNFL/stadiums` was unreachable) for outdoor games | no key | free non-commercial; 10k calls/day | hourly | P2 |
| The Odds API | Posted team totals | key | free tier: 500 credits/month | near-live | optional |
| FantasyPros API | ECR and projections | key | a personal production key needs the HOF tier (~$108/yr) | live | optional; ECR is already free via nflverse |

**Avoid:** the Reddit API (pre-approval required since Nov 2025), scraping FantasyPros or KTC, and
Sportradar/SportsDataIO (cost).

**Gotchas:**

- `nfl_data_py` was archived in Sept 2025; use `nflreadpy`.
- nflverse fills in schedule temperature and wind *after* games, so use forecasts.
- Sleeper's `espn_id` coverage is poor (zero for rookies), so the crosswalk uses `ff_playerids` plus overrides.
- Sleeper's old projections endpoint broke in Sept 2026, and the current one is undocumented.

### NBA

| Source | What we use | Access | Cost / limits | Freshness | Role |
|---|---|---|---|---|---|
| ESPN league API | League state, ESPN projections, free agents, transactions | `espn_s2`/`SWID` cookies | none stated | live | core |
| ESPN site APIs | Injuries (status + comments), news, scoreboard with DraftKings spread/O-U, team rosters | no key; send a browser UA | none stated | live | core |
| `nba_api` (stats.nba.com) | League-wide game logs, Base/Advanced/Usage splits, V3 box scores, on/off splits | full nba.com header set; **home IP only** (cloud IPs hang) | free; throttling undocumented, so ~0.6 s between calls and nightly cached pulls | nightly | core |
| NBA CDN `scheduleLeagueV2.json` | 2026-27 schedule → games per fantasy day/week, back-to-backs | full Chrome header set (a bare UA gets 403) | free | weekly; re-pull after Cup group play | core |
| DARKO | Projections CSV including projected minutes, with ESPN-points and 9-cat presets | public Google Sheet exports (the Shiny app closed June 2026; `darko.app` only has a client-side CSV button) | free | daily | core projection source |
| Official injury report PDFs | Pregame "reason" detail beyond ESPN's status | `ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<time>.pdf` | free; updated every 15 min on game days | 15 min | optional |
| The Odds API | Backup lines | key | free tier: 500 credits/month | near-live | optional |

**Avoid:** automating Basketball-Reference (its data-use terms ban scraping, and >20 requests/min gets the IP
jailed), Hashtag Basketball (paywalled, no redistribution), and balldontlie unless a cloud-friendly fallback is needed
($9.99–39.99/month).

**Gotchas:**

- stats.nba.com and the NBA CDN sit behind Akamai checks that block cloud IPs and inconsistent headers. This settles
  the local-runtime choice (§13).
- `nba_api` supports Python 3.10–3.13, so the project pins 3.13.
- `nba_api.live` endpoints currently return 403 (an outdated UA; the upstream issue is open), so live data comes from
  ESPN's scoreboard.
- ESPN's NBA default is H2H points (PTS 1, 3PM 1, FGM 2, FGA −1, FTM 1, FTA −1, REB 1, AST 2, STL 4, BLK 4, TO −2).
  Its default categories league is 8-cat, or 9-cat with TO. The default roster is PG/SG/SF/PF/C/G/F + 3 UTIL + 3 bench.
  All of this is read from settings; these defaults are only the test fixtures.

**Player identity.** Each player has one canonical row keyed by ESPN ID, plus a crosswalk to every other source:

- **NFL:** the DynastyProcess ID map shipped with nflverse (`ff_playerids`: ESPN, Sleeper, GSIS, FantasyPros IDs),
  plus a reviewed overrides file for gaps. Sleeper's own `espn_id` field is too sparse to rely on.
- **NBA:** match ESPN ID to NBA person ID on (normalized name, team, position), with a reviewed `overrides.csv`.
- **Gate:** sync fails loudly if any *rostered* player in a managed league is unmapped. Silent ID mismatches are the
  most likely source of wrong decisions.

**Freshness.** Each adapter declares a TTL per dataset. A job asks for data "fresh as of T", and every decision records
the `as_of` of each input.

## 8. Model core

### 8.1 Projections

- Store projections per (player, scoring period, source) as **stat lines, not points**. Each league's scoring items
  turn stats into points; category leagues use the stats directly.
- **v1 sources:** NFL uses ESPN (from the player pool) and Sleeper (RotoWire-based); NBA uses ESPN and DARKO.
  **v2 in-house baselines:**
  - *NFL:* an opportunity model. Snap, target, and carry shares (nflverse weekly stats and snap counts; Sleeper has
    same-day snaps) × team environment (implied team total from betting lines) × regressed efficiency. Expected
    fantasy points (`ff_opportunity`) are a feature. In-season route participation isn't public: nflverse's FTN data
    lands after the season.
  - *NBA:* minutes × per-minute rates (recency-weighted) × context. Context means teammates out (usage
    redistribution), the spread (blowout risk), and back-to-backs (rest risk). Teammates-out effects come from
    without-player splits in game logs plus stats.nba.com on/off data, with DARKO's projected minutes as the prior.
- **Blend:** a weighted average per stat, with weights per (source, position). Fit the weights on backtests with
  held-out weeks; start equal-weighted.
- **Uncertainty:** an SD per player-period from historical residuals, by position (NFL) or minutes bucket (NBA). This
  feeds floors, ceilings, and the simulator.

### 8.2 Availability

- Compute `p_active` from official designations. NFL uses Q/D/O plus the practice-participation trend; NBA uses
  probable/questionable/doubtful/out. Claude-classified news adjusts it within bounds.
- Expected value = `p_active` × projection.
- **Lock awareness:** when a questionable player has a late game, prefer lineups that keep a same-slot pivot who plays
  even later (the FLEX/UTIL trick).

### 8.3 Valuation

- **Points leagues:** value over replacement, where replacement is the best player *actually on your league's wire*
  for that slot. Rest-of-season (ROS) value is the sum over remaining weeks of start-value, with fantasy playoff weeks
  weighted up.
- **NBA points leagues:** no z-scores needed. Value = per-game projection × games in the period, with league weights
  applied.
- **NBA categories:** volume-weighted z-scores (FG% and FT% as `attempts/avg_attempts × (pct − pool_pct)`; TO
  negative), upgraded to **G-scores** for H2H. G-scores divide by `√(σ² + κτ²)`; the added term is week-to-week
  variance, which discounts noisy categories like STL and BLK (Rosenof, arXiv:2307.02188).
- For roster-aware trade and punt decisions, use **H-scores**. They re-weight categories as the roster takes shape, so
  punts emerge on their own (arXiv:2409.09884; reference code in `zer2/fantasy-basketball-optimizer`).
- **Market value** (what league-mates believe) is tracked separately. Sources are redraft trade values (NFL) and ESPN
  rank and ownership trends (both sports). It's used only to model whether a trade gets accepted.

### 8.4 Simulation

- Run a Monte Carlo (numpy, ~10k runs) over the remaining ESPN schedule, using each team's expected-optimal lineups.
  Outputs: P(win this week), P(playoffs), P(bye), P(title). For NBA categories it also gives per-category win
  probabilities.
- Consumers: trades (Δ title odds), weekly strategy, and variance-aware lineups.

## 9. Decision modules

### 9.1 Lineups

- A single-period lineup is an assignment problem over slot eligibility (`scipy.optimize.linear_sum_assignment`).
  Multi-day NBA planning with adds is a small knapsack/DP, or a MILP (`scipy.optimize.milp`, HiGHS) when needed (§9.3).
- **Objective:** expected points by default. When the matchup simulation says we're a heavy favorite (underdog),
  maximize P(win) instead, which favors lower (higher) variance.
- **Hard rules:** never start an OUT, bye, or no-game player while any legal alternative exists; never move a locked
  player.
- **When it runs:** deadlines are computed from the pro schedule, never hardcoded.
  - NFL: before TNF, after Sunday inactives (international and holiday slots included), before each later window,
    and before MNF.
  - NBA: a morning plan, then a re-check ~30 min before each tip window for late scratches.

### 9.2 NFL waivers and free agents

- **Candidates:** the best free agents by Δ roster value, trending adds, and news-flagged openings (Claude: "starter
  out → backup up").
- Score each (add, drop) pair by Δ roster value. Untouchables are never dropped.
- **FAAB bid** = f(Δ value, budget left, weeks left, the league's historical winning bids), capped by policy.
- Timed to the league's waiver run, read from settings. ESPN's default clears waivers Wednesday around 3–5 a.m. ET,
  and dropped players sit on waivers for 24 hours. Free-agent pickups happen after waivers clear.

### 9.3 NBA streaming

- **Timing:** adds, drops, and trades lock by the league's `rosterLocktimeType`, a setting of its own next to the
  lineup lock. The real NBA league's `FIRSTGAME_SCORINGPERIOD` locks them at the day's *first* tip even though its
  lineups lock per game, so a streamer must be added before the first game of the day to play that day
  (`SportPlugin.transaction_cutoff` with `LeagueSettings.roster_lock_type`; an unknown or weekly type is refused).
- **Limits:** respect the league's matchup acquisition limit (often about one per game day) and any games-played
  limit, both read from settings.
- For the current matchup week, find **open slot-days**: days where active slots outnumber players with games.
- Choose the add/drop sequence that maximizes expected category wins (or points) under the acquisition limit, as a
  small daily knapsack/DP. Schedule density is the main driver, and core players are protected.
- **Category scoring:** P(win category) ≈ Φ(Δμ/σ). "Swing" categories sit near 50%, where ∂P/∂stat is largest. Rank
  streamers by Σ ∂P/∂stat × projected production × games on open slot-days.
- **Prior art to read before building:** `TylerGrossi/Fantasy-Basketball-Simulation-Model` (ESPN 9-cat, streamers,
  acquisition-limit aware, MIT) and `giasemidis/espn-nba-fantasy` (weekly Monte Carlo on the ESPN API).
- A Monday plan sets which categories are winnable, which to concede, and target stats for streamers.
- **Facts from the real leagues (read 2026-10-05; confirmed by the real fixtures of #14, `docs/espn-api.md`):**
  - NBA scoring periods are **days**. Day 1 is opening night (Tue Oct 20, 2026), and `status.finalScoringPeriod` is the
    last fantasy day (153, Sun Mar 21, 2027 in the real league). NBA games in `proTeamSchedules_wl` are keyed by
    that day number. ESPN's days (and NFL weeks) run from 3 a.m. to 3 a.m. ET, not midnight to midnight: a late West
    Coast game is still in its day after midnight, and so is a lineup set before 3 a.m.
  - `scheduleSettings.matchupPeriods` maps a matchup to schedule-period IDs (`{"1": [1], …}`), **not to days**: the
    IDs are periods of the type `scheduleSettings.periodTypeId` names (weeks in the real league), and ESPN's web client
    calendar maps those to days. Matchup 1 = days 1–6 (Tue Oct 20 – Sun Oct 25), matchups 2–17 = 7-day Monday–Sunday
    weeks, matchup 18 = days 119–132 (14 days around the All-Star break), matchups 19–21 (playoffs) = days 133–153:
    6 + 16×7 + 14 + 3×7 = 153 (`tests/fixtures/espn/real/fba/calendar.json`). No read view carries that calendar.
    `fm.espn.settings` reads `matchupPeriods` as scoring periods only under `periodTypeId` 1 (ESPN's per-scoring-period
    type, the NFL league's) and otherwise answers `None`. The calendar ships as `data/calendars/<game>_<season>.json`
    (`fm.espn.calendar`; refresh with `capture.py webclient`), and `matchup_period_of` / `matchup_scoring_periods`
    resolve NBA matchup weeks through it, so the weekly transaction cap counts the matchup's days (the trailing seven
    days only when a season has no calendar file).
  - `matchupAcquisitionLimit` is a per-day rate when `matchupLimitPerScoringPeriod` is true: "3 adds per weekly
    matchup" arrives as 3/7. Use `AcquisitionSettings.matchup_limit_for(days)`, never the raw number.
  - Both real leagues are H2H points with traditional waivers (no FAAB, so there are no bids). The NFL league seeds its
    4-team playoff by total points scored, not record, so raw points matter beyond weekly wins.

### 9.4 Trades

- **Evaluate** an incoming offer or a manual `fm trade eval "give A,B get C"`. Report Δ ROS starting value for both
  sides, Δ playoff/title odds (simulation), roster legality, and fit (byes, positional depth, categories). Then
  recommend accept, decline, or counter.
- **Find:** for each opponent, enumerate 1-for-1, 2-for-1, and 2-for-2 deals within a value window and screen them with
  fast greedy lineup values. Re-score the finalists with exact lineups plus simulation, then rank by our Δ title odds ×
  P(accept).
- **P(accept)** models *their* view, not ours: market values plus their roster needs (positional holes, bye
  problems, weak categories). Calibrate it as offers resolve.
- **Signals:**
  - NFL: actual vs expected fantasy points (buy-low / sell-high), plus snap and target-share trends.
  - NBA: minutes and usage trends.
  - Both: the playoff-week schedule.
- **Etiquette:** every trade is approval-gated, with at most one open offer per team and a weekly cap. Claude drafts a
  short pitch you can send.

### 9.5 Weekly strategy

Each league gets a report on Monday/Tuesday. It covers the matchup outlook, the playoff-odds trend, priorities (needs,
punts, stash candidates), and trade targets, which become proposals.

## 10. Advisor (Claude)

Claude runs as narrow workers. Each is one Messages API call, or a short tool loop, with structured output and
read-only inputs. None can write to ESPN.

| Worker | Trigger | Output | Effort |
|---|---|---|---|
| `news_triage` | New items about relevant players: your roster, this week's opponent, top free agents, trade targets | `{player_id, kind: injury\|role\|rest\|suspension\|other, severity, games_out, p_active_delta, confidence, source_url, published_at}` | low; batched; overnight work via the Batches API |
| `close_call` | Top two options within a margin, near a deadline | pick + cited reason (web search over an allowlist of domains) | medium |
| `explain` | Each non-trivial proposal (trivial ones like "X has no game" are templated) | 2–4 sentence rationale | low |
| `trade_pitch` | An approved trade idea | short message to the other manager | medium |
| `weekly_strategist` | Mon/Tue per league | priorities, punts, targets → proposals | high |

- **Model:** `claude-opus-5-5`, with effort set per worker. One model means one prompt-cache namespace. Measure a
  cheaper model only if spend becomes a problem.
- **Structured outputs** use `client.messages.parse` with Pydantic models. Check `stop_reason` (`refusal`,
  `max_tokens`) before reading the output, and keep the server-side refusal fallback enabled.
- **Web search** uses the server tool `web_search_20260209` with `allowed_domains` (espn.com, nfl.com, nba.com,
  rotowire.com, cbssports.com, …) and a `max_uses` cap.
- **Guardrails:**
  - Adjustments are bounded (e.g. `p_active` ±0.3, projection ±25%), must cite a source and timestamp, and are logged.
  - A Claude-only signal can never trigger a drop or a trade by itself.
- **Cost:**
  - Prompt caching on the stable prefix (system prompt + league context).
  - Batches API (50% off) for anything not urgent; triage only relevant players.
  - A daily spend cap in config.
  - Rough estimate: **~$15–30/month** with both seasons running.
- **No Claude Agent SDK.** The workers need no filesystem or shell, so the plain Messages API is the smaller, safer
  surface.

## 11. Proposals, policy, approvals

A proposal records `league, kind, payload, engine numbers, rationale, deadline, policy, status, decided_by,
verification, artifacts`. Its status runs `proposed → approved | rejected | expired → executing → verified | failed`.

| Kind | Default policy | Allowed settings |
|---|---|---|
| Bench an OUT / bye / no-game starter | approve; **auto if unanswered by T-15 min** (you confirmed this) | off · approve · auto |
| Other lineup optimizations | auto (T-15 if unanswered) | off · approve · auto |
| Free-agent add into an open spot (`add`) | approve; **auto fires at once** when the engine's news-free gain is at least `auto_add_min_gain` (NFL waivers only; NBA streaming reports no such gain, so it waits) | off · approve · auto |
| Free-agent add that drops a player (`add_drop`), waiver claim | approve | off · approve |
| Trade propose / accept / decline | approve | approve only (hard-coded) |

**Guardrails:**

- Untouchables list.
- Maximum transactions per week, and maximum FAAB % per bid.
- League allowlist.
- `fm pause` kill switch.
- Approvals expire at their deadline: a lineup approval is void after lock.
- Never use league-manager (commissioner) actions.

**Approval channels:** the CLI (`fm proposals`) first, then the phone. Neither phone option needs open ports on the PC.

- **Telegram (default).** Needs a free Telegram account (sign-up uses a phone number).
  - Setup: create a bot with @BotFather (`/newbot`) and put its token in `.env`. `fm notify setup` captures your chat
    ID from your first message to the bot.
  - Use: proposals arrive with inline Approve/Reject buttons, and the bot replies with the result.
  - Security: the bot ignores every other chat, because bot usernames are public.
- **ntfy (no account).** Install the ntfy app and subscribe to a long random topic.
  - Use: Approve/Reject are `http` action buttons that post to a private reply topic the PC listens on, over an
    outbound connection only.
  - Security: topic names act as secrets, and every button carries the proposal's single-use token.
  - Known iOS bug: the app doesn't clear the notification after a tap, so the PC sends a confirmation push.

## 12. Interfaces

- **CLI** (`typer`):
  - Setup and data: `fm login`, `config check`, `sync`, `status`
  - Decisions: `fm lineup`, `waivers`, `stream`, `trade eval|find`, `rankings`
  - Approval and execution: `fm proposals list|approve|reject`, `execute [--dry-run]`
  - Runtime: `fm tick`, `schedule install|show`, `bot`, `pause|resume`
  - Upkeep: `fm report`, `backtest`, `canary`, `drill`
- **Notifications:** Telegram or ntfy for proposals (with buttons), executions, failures, and the weekly report.
- **MCP server** (FastMCP, like `qgis_mcp` / `blender-mcp`): read tools plus `create_proposal`, which never executes.
  This lets you ask "who should I start at FLEX?" from Claude Code or Desktop, including from your phone via Remote
  Control.
- **Later (optional):** a small local dashboard.

## 13. Runtime and scheduling

- Runs on the local machine (Windows PC or Mac) that holds the browser profile. This isn't optional: stats.nba.com
  and the NBA CDN block cloud IPs, and ESPN may block datacenter IPs too. Each machine has its own profile and
  `state.db`; run the tick on one of them at a time.
- **One scheduled job: `fm tick` every 10 minutes.** This is software-factory's model: a cheap, idempotent tick that
  works out what is due. `fm schedule install` picks the backend by platform:
  - **Windows:** `schtasks` + a `.cmd` wrapper and the wake/battery settings from software-factory's
    `scheduler/windows.py` (`jobs/scheduler_windows.py`).
  - **macOS:** a per-user launchd LaunchAgent (`~/Library/LaunchAgents/local.espn-fantasy-tick.plist`,
    `StartInterval`, loaded into `gui/<uid>`) + a `.sh` wrapper (`jobs/scheduler_macos.py`). launchd runs on battery
    but cannot wake a sleeping Mac; a tick missed during sleep runs on wake and reports the missed windows, so keep
    the Mac awake at locks with the energy settings or `pmset`.
- Each tick:
  1. Refresh stale data.
  2. Compute upcoming deadlines (kickoffs and tips from the pro schedule; the waiver run from league settings).
  3. Run the decisions that are due.
  4. Execute approved or auto proposals whose time has come.
  5. Run health checks (session, canary staleness).
- A weekly `fm report` job runs alongside the tick.
- `fm bot`, started at logon, listens for approvals (Telegram long polling or the ntfy reply topic), so they land
  within seconds. An approval close to a deadline
  triggers execution immediately instead of waiting for the next tick.
- The PC must be awake at locks: enable wake timers. The first tick after a missed window reports what was missed.
- If the PC proves unreliable, move to an always-on box later. The tradeoff is that the ESPN session then lives there.

## 14. Storage, config, layout

- `~/.config/espn-fantasy/`: `config.toml`, `.env` (`ANTHROPIC_API_KEY`, `ODDS_API_KEY`, and either
  `TELEGRAM_BOT_TOKEN` or the ntfy topic names), `browser-profile/`, `state.db` (SQLite), and `audit/` (traces,
  screenshots).
- `~/.cache/espn-fantasy/`: raw source responses and parquet snapshots. Safe to delete.
- In the repo: code plus scrubbed fixtures only. No secrets and no real manager names.

```toml
# ~/.config/espn-fantasy/config.toml
[[league]]
key = "nfl"
sport = "nfl"                 # nfl | nba
espn_league_id = 0            # leagueId=… in the league URL
season = 2026
team_id = 0                   # teamId=… on your team page

[league.policy]
bench_inactive = "auto"       # off | approve | auto (auto only fires at T-15 when unanswered)
lineup = "approve"
add_drop = "approve"
waiver = "approve"
max_transactions_per_week = 3
max_faab_pct_per_bid = 0.35
untouchables = []             # player names or ESPN IDs; resolved at sync

[[league]]
key = "nba"
sport = "nba"
espn_league_id = 0
season = 2027                 # ESPN labels NBA seasons by end year; confirm with seasonId= in the URL
team_id = 0

[league.policy]
bench_inactive = "auto"
lineup = "approve"
add_drop = "approve"          # streaming adds must be approved before the day's first tip
max_transactions_per_week = 7 # our cap; ESPN's own acquisition limit is read from settings
untouchables = []

[llm]
model = "claude-opus-5-5"
daily_budget_usd = 2.0

[notify]
channel = "telegram"          # telegram | ntfy
```

```
espn-fantasy/
  pyproject.toml  CLAUDE.md  ROADMAP.md  docs/DESIGN.md
  data/           blend_weights.toml, id_overrides*.csv, stadiums.csv
  src/fm/
    cli.py  paths.py  config.py  mcp_server.py
    commands/     one module per CLI group, auto-discovered
    store/        db.py, models.py, migrations/
    espn/         client.py, models.py, auth.py, settings.py, ids.py
    browser/      session.py, transactions.py, selectors.py, canary.py, drills.py, fakes.py,
                  flows/{lineup,add_drop,waiver,trade}.py
    sources/      base.py, nflverse.py, sleeper.py, market.py, odds.py, weather.py, news.py,
                  nba_stats.py, nba_schedule.py, nba_injuries.py, darko.py
    sports/       base.py, nfl.py, nba.py
    model/        ids.py, ids_nba.py, scoring.py, projections.py, availability.py, relevance.py, valuation.py,
                  categories.py, value_nba.py, simulate.py, baseline_nfl.py, baseline_nba.py
    decide/       registry.py, lineup.py, lineup_daily.py, waivers.py, faab.py, streaming.py, weekly.py,
                  trades.py, offers.py, draft.py
    advisor/      client.py, news_triage.py, close_call.py, explain.py, pitch.py, strategist.py, prompts/
    proposals/    queue.py, policy.py
    executor/     run.py, verify.py
    notify/       base.py, telegram.py, ntfy.py
    jobs/         sync.py, tick.py, deadlines.py, report.py, scheduler_base.py, scheduler_windows.py,
                  scheduler_macos.py
    eval/         backtest.py, tune.py, report_card.py
    render/       CLI tables
  tests/          mirrors src/; fixtures/ holds scrubbed ESPN JSON and recorded source responses
```

## 15. Testing and evaluation

- **Unit and property tests** run against recorded ESPN JSON from real leagues, with manager names scrubbed. They cover
  settings → scoring, the ID crosswalk, lineup legality (never start an OUT player while an alternative exists),
  valuation, trade math, and policy.
- **Executor:** dry-runs against the real league, plus the daily canary. A spike checks whether a private sandbox
  league is feasible for full write tests.
- **Backtest harness:** replay past weeks (ESPN keeps prior seasons for your league; nflverse and NBA game logs supply
  actuals). Measure projection MAE by position, lineup efficiency (actual ÷ hindsight-optimal), start/sit regret, and
  waiver pickup value. Changes to blending or valuation must not regress these numbers, the same gate pattern as
  shardfall's feel tests.
- **Live report card:** each week, the same metrics on real decisions, plus "points left on the bench".
- **Baseline first:** before automating anything, compute your own lineup efficiency from last season's history. That
  is the number to beat.

## 16. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ESPN changes the undocumented write API or the UI | medium over a season | missed move | API mode with UI fallback (each covers the other), canary, error-code mapping, manual-fix notification |
| Duplicate submission on retry or timeout | medium | double add/drop or a duplicate trade offer | no write retries, `UNKNOWN` status + re-read, single-use tokens, pending-offer check |
| Bot detection or session expiry | medium | executor down | persistent profile, real browser channel, headed fallback, login reminder |
| ID mismatches across sources | high at first | silently wrong decisions | crosswalk + unmapped-rostered-player gate |
| Claude misreads news (wrong player, stale item) | medium | bad start/sit | structured output with source and time, bounded adjustments, no Claude-only drops or trades |
| PC asleep at a lock | medium | missed lineup | wake timers, missed-window alert |
| ESPN ToS / account action | low | lose account access | personal use, own team only, low volume (§6.4) |
| League trust (trade spam, automation norms) | low–medium | social | approval-gated trades, rate limits; tell the league if its norms call for it |
| Overfitting blend weights | medium | worse projections | held-out weeks and seasons in the backtest |

## 17. Phase plan

| Phase | Target | Ships | Exit criteria |
|---|---|---|---|
| 0. Spikes | 1–2 days | ESPN reads for both leagues; login + profile; write payloads verified by intercept-and-abort; source viability | Fixtures saved; `ffl` + `fba` payloads and open unknowns documented |
| 1. NFL advisor (read-only) | before Week 6 waivers | config, store, ESPN client, settings → scoring, crosswalk, ESPN + Sleeper blend, lineup optimizer, waiver ranker, `fm status/lineup/waivers` | Recommendations for your real team; lineup legality tests; zero unmapped rostered players |
| 2. Act (NFL) | +1 week | proposals + policy, executor (API mode + UI fallback: lineup → add/drop → waiver), verify, audit, dry-run, canary, phone approvals (Telegram or ntfy) | One real lineup change and one add/drop approved, executed, and verified |
| 3. Intelligence | +1 week | availability model, news ingest + Claude triage, close calls, rationale, `fm tick` scheduler, weekly report | One NFL Sunday run end-to-end, with your only input being approvals |
| 4. NBA | by opening night | NBA plugin: schedule and locks, stats + injury report, points and category valuation, daily lineups, streaming, category planner | Daily lineups proposed and executed in opening week |
| 5. Trades | before your NFL trade deadline | simulator, evaluator, finder + acceptance model, pitches, trade flows | Incoming offers auto-evaluated; weekly trade ideas with Δ title odds |
| 6. Learn and harden | ongoing | backtests + report card, blend tuning, FAAB model, MCP server, UI-fallback drills, in-house baselines, dashboard | Backtest gate in CI; weekly report card |

Phases 4 and 5 can swap if your NFL trade deadline is early (it's in league settings).

These are product phases with calendar targets. `ROADMAP.md` breaks them into 47 tasks. Its "Phase N" headings are
dependency layers, not these product phases, so NBA work runs in parallel with NFL work there. Its Phase 6 milestone
covers product phases 1–3 plus NBA lineups and streaming; its Phase 8 milestone is trades.

## 18. Decisions and open questions

Answered 2026-10-04:

1. **Leagues:** one NFL league and one NBA league. Their formats aren't known yet; the first `fm sync` reads them from
   ESPN settings, so nothing waits on them. Both NBA paths (points and categories) stay in scope until then. ESPN's
   NBA default is H2H points.
2. **Drafts:** both are done, so there's no draft tooling this season. The draft-sheet task became a rest-of-season
   rankings sheet.
3. **Autonomy:** auto-benching an OUT/inactive starter and other lineup changes at T-15 when unanswered is approved,
   and so is a free-agent add into an open spot whose engine gain clears `auto_add_min_gain` (it fires at once). Any
   move that drops a player, waiver claims, and every trade wait for approval.
4. **Phone approvals:** Telegram by default, which needs a free account. ntfy is the no-account alternative (§11).
5. **Runtime:** your own Windows PC or Mac. This is effectively forced, because the NBA data sources block cloud IPs.

Nothing blocking is left open. The setup items on your side are the ESPN league and team IDs, a one-time `fm login`,
and the phone channel.

## 19. Decisions and alternatives

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Language | Python 3.13 + uv | TypeScript | The data ecosystem is Python (espn-api, nflreadpy, nba_api, numpy/scipy, Playwright, anthropic SDK). Matches software-factory. Pinned to 3.13 because `nba_api` supports ≤ 3.13 |
| ESPN reads | JSON API | DOM scraping | Structured, fast, and the same calls the web app makes |
| ESPN writes | The web app's own transaction calls sent from the logged-in Playwright session, with UI click-through as fallback | UI click-through only; plain HTTP outside the browser | The API path was verified live by several projects in Sept 2026, while UI-only bots broke on every redesign. Keeping it in the browser keeps one session and a real client |
| LLM role | Narrow workers on a deterministic engine | One agent deciding everything | Testable, auditable, predictable cost |
| Runtime | Local PC or Mac + `schtasks` / launchd tick | Cloud VPS, scheduled cloud agents | The ESPN session and browser profile live locally, and NBA data sources block cloud IPs |
| Storage | SQLite + parquet cache | Postgres, DuckDB | Single user, zero ops |
| Approvals | Telegram bot (default) or ntfy | Discord, Pushover, openclaw | Both give tap-to-approve with no inbound ports; ntfy needs no account |
