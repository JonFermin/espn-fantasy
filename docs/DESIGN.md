# ESPN Fantasy Manager: Design

> Status: v0.4, 2026-10-24; your answers are recorded in §18 · Sports: NFL (`ffl`), NBA (`fba`) · Platform: ESPN Fantasy
> Started as the planning doc `ROADMAP.md` was generated from; it now also describes the plug-and-play layer as built
> (§11 presets, §12 setup, health, dashboard and Claude, §13 `fm run`, §14 config) and where each Claude worker runs
> in the tick (§10, §13.4). `ROADMAP.md` records what each phase built.

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
| `jobs/` | Sync, the tick, the `fm run` loop, the run lock, process-tree kills, scheduler backends (§13) |
| `cli.py`, `mcp_server.py` | `fm` CLI; MCP server exposing read tools plus `create_proposal` (never execute) |
| `setup/`, `health.py`, `views.py`, `web/`, `claude/` | `fm setup`, health checks, the read views the dashboard and MCP share, the local dashboard, the Claude Code and Desktop integration (§12). None has a write path to ESPN |
| `presets.py`, `config_edit.py` | Policy presets (§11) and the one writer of `config.toml` (§14) |

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
- **NBA:** match ESPN ID to NBA person ID on (normalized name, team, position), with a reviewed
  `data/id_overrides_nba.csv` (`src/fm/model/ids_nba.py`). The persons come from stats.nba.com's splits, DARKO's
  sheet and `nba_api`'s active players.
- **Gate:** sync fails loudly if any *rostered* player in a managed league is unmapped in a way that could be a wrong
  ID. Silent ID mismatches are the most likely source of wrong decisions. The NFL gate stops on any unmapped rostered
  player. The NBA gate (`nba_gate`) stops only where the identity is in doubt, and says why:
  - **Maps, with a warning.** `NAME_ONLY`: the only nba.com person with his name, but team or position disagree with
    ESPN (an offseason move nba.com has not caught up with). A partial name on his team (`Herb Jones` -> `Herbert
    Jones`) maps the same way. A match on the name alone is taken only to a person in this season's data (splits,
    DARKO, the active list), so a rookie never inherits a departed namesake's history.
  - **Passes as no history.** `NO_HISTORY`: nba.com lists nobody who could plausibly be him: no free person with his
    surname on his ESPN team, and none anywhere with his surname and first initial (a rookie). He gets no nba.com id
    and is valued from ESPN's projection alone; with no line at all the daily lineup holds him at 0 and streaming never
    drops him. The verdict is saved as a `player_ids` row (source `nba_no_history`), so a sync that keeps the saved
    crosswalk when a source is down still passes him.
  - **Stops the sync.** `AMBIGUOUS` (several persons fit), a namesake already mapped to another player, only a
    namesake from an earlier season, or `NO_MATCH` with a plausible candidate (a veteran ESPN spells differently). The
    error names the candidates and prints the overrides line to complete.
  - **Overrides** (`espn_id,source,source_id,name,note`). A row pins the player to that nba.com id whatever the match
    says, and takes the id out of play for everyone else. An empty `source_id` runs him as a no-history player (and
    undoes a wrong match on the name alone). A row also confirms a partial or name-only match and silences its
    warning.

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
- **Scheduled finder (opt-in):** with `[league.trade_finder] enabled = true`, the tick runs the finder at each period's
  opening and drafts the deals whose P(accept) is at least `min_accept` and whose gain clears `min_title_gain` (title
  odds) or, without a simulation, `min_ros_gain` (starter seasons). It drafts at most `max_offers` a run under the same
  etiquette, never redrafts a deal you rejected, and lets a draft expire after 3 days unanswered. Drafts are ordinary
  `trade_propose` proposals: they reach your phone and wait for your approval.

### 9.5 Weekly strategy

Each league gets a report in its weekly window: from the matchup period's Monday 09:00 ET until the period turns, once
per period, run by the tick (§13.4). It covers the matchup outlook, the playoff-odds trend, priorities (needs, punts,
stash candidates), and trade targets, which become approve-only proposals. `fm report` writes the same report on
demand.

## 10. Advisor (Claude)

Claude runs as narrow workers. Each is one Messages API call, or a short tool loop, with structured output and
read-only inputs. None can write to ESPN.

| Worker | Trigger | Output | Effort |
|---|---|---|---|
| `news_triage` | Each news poll in the tick (NBA every 30 min, NFL hourly): the untriaged items about relevant players (your roster, this week's opponent, top free agents, trade targets), a chunk of items per call | `{player_id, kind: injury\|role\|rest\|suspension\|other, severity, games_out, p_active_delta, confidence, source_url, published_at}` | low |
| `close_call` | Inside the NFL and NBA lineup decisions: a slot whose top engine options are near-ties, when its lock is under 6 h away; at most 3 per run | pick among those options + cited reason (web search over an allowlist of domains) | medium |
| `explain` | After a league's decisions, before the push: each new non-trivial proposal (trivial ones like "X has no game" are templated) | 2–4 sentence rationale | low |
| `trade_pitch` | Each new `trade_propose` (the trade finder's or the strategist's), in the same pass as `explain` | short message to the other manager, stored on the proposal for you to paste | medium |
| `weekly_strategist` | The weekly window per league: Monday 09:00 ET until the matchup period turns, once per period | priorities, punts, targets → approve-only proposals, and the weekly report's strategy section | high |

**Where each worker runs** (ROADMAP Phase 13; `src/fm/jobs/tick.py`, §13.4). A tick builds at most one advisor,
through `fm.jobs.advice.advisor_for`: a client when `ANTHROPIC_API_KEY` is in `.env` and `[llm] daily_budget_usd` is
above 0, else `None` with the reason (`advisor_off`), which the tick notes per league and `fm doctor` shows (§12.3).
A tick without one never imports the SDK.

- `news_triage` runs in each league's news refresh. The poll (`fm.sources.news`: ESPN and RotoWire) and the store
  (`fm.model.relevance.ingest_news`) need no key; the triage of that sport's untriaged items does, and stores bounded
  `news_signals`. New signals refresh the availability inputs at once. Without an advisor the items wait for the
  first poll that has one.
- `close_call` runs inside the lineup decisions (`fm.decide.lineup.close_call_lineup`, both sports), which take the
  advisor as an optional argument. It picks among the engine's near-tie options only, with a source from the
  allowlist; an answer outside them, uncited or with no time left keeps the engine's lineup. The pick and its sources
  go into the proposal's `engine_numbers`. Legality rules are unchanged.
- `explain` and `trade_pitch` run after a league's decisions and before the push. Claude's rationale replaces the
  template as the stored `rationale`, once per proposal (`tick-state.json` remembers which), and each new
  `trade_propose` gets its pitch in `engine_numbers["pitch"]` with its `source`; the phone, the dashboard and
  `fm proposals` show it.
- `weekly_strategist` runs in the weekly window (`fm.jobs.deadlines.weekly_window`), once per matchup period. The
  engine's inputs (`fm.jobs.report.strategy_inputs`: the category plan, our season odds, the trade finder's deals) go
  to `fm.advisor.strategist.run_strategist`; its targets become approve-only `trade_propose` rows under
  `[league.trade_finder]`'s `max_offers` and the weekly cap (none while the finder is off); then the league's weekly
  report is written and pushed, the strategy as its last section. Without an advisor `fm.decide.strategy` gives the
  template note and the same drafts. A window the tick slept through is alerted, never re-run.
- **Fallbacks.** Every worker has a template answer, used when there is no advisor, the daily budget is spent, the
  call fails, is refused or cut off, or the answer does not parse. A failure is a warning in the tick report, never a
  skipped league and never a retry that asks Claude again.

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
  - Batches API (50% off) in the client for anything not urgent; the tick itself triages synchronously, a chunk of
    items per call, and only relevant players.
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

**Presets** (`src/fm/presets.py`). A `[[league]]` may set `preset = "hands_off" | "balanced" | "ask_first"`. The
preset fills every `[league.policy]` and `[league.trade_finder]` key the file leaves unset, and a key written in the
file always wins. A league without a preset loads with the defaults in the table above. `fm setup` writes `balanced`
unless told otherwise; `fm config set-preset` and the dashboard change it.

| key | hands_off | balanced | ask_first |
|---|---|---|---|
| bench_inactive | auto | auto | approve |
| lineup | auto | auto | approve |
| add | NFL `auto` (`auto_add_min_gain` 15), NBA `approve` | approve | approve |
| add_drop / waiver | approve | approve | approve |
| max_transactions_per_week | NFL 3, NBA 7 | NFL 3, NBA 7 | NFL 3, NBA 7 |
| trade_finder.enabled | true | true | false |
| trade_finder thresholds | min_accept 0.5, min_title_gain 0.015, min_ros_gain 0.3, max_offers 1 | same | defaults |

- No preset contains a `trade*` policy key, or `auto` for `add_drop` or `waiver`; `tests/test_presets.py` holds the
  table to that, so a new preset cannot loosen an invariant unnoticed.
- NBA `add` stays `approve` everywhere, because NBA streaming reports no news-free gain yet.
- `League.preset` stays on the model, so `fm config check`, the dashboard and `fm doctor` show it.

**Approval channels:** the CLI (`fm proposals`), the dashboard (§12.4) and the phone. Neither phone option needs open
ports on the PC. `fm setup` offers ntfy first, because it needs no account. `notify.channel = "none"` runs without a
phone: proposals wait in the dashboard or `fm proposals`, and alerts go to the log.

- **Telegram.** Needs a free Telegram account (sign-up uses a phone number).
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

Phases 10 to 12 of `ROADMAP.md` made the tool plug and play: one command sets it up, one process runs it, a dashboard
and Claude sit on top. None of these surfaces loosens an invariant in `CLAUDE.md`. Only `fm.executor` (through
`fm.browser`) writes to ESPN; the setup wizard, the dashboard and the Claude surfaces only read, record human
decisions, edit `config.toml`, or store `proposed` rows.

### 12.1 CLI

`typer`, with one module per command group in `src/fm/commands/`, auto-discovered.

- Getting started: `fm setup`, `doctor`, `login`, `config check|set-preset`, `notify setup`
- Always on: `fm run`, `autostart install|uninstall|show`, `schedule install|uninstall|show` (the wake timer), `tick`,
  `pause|resume`
- Looking and deciding: `fm ui`, `status`, `lineup`, `waivers`, `trade eval|find`, `rankings`, `report`,
  `proposals list|approve|reject`
- Claude: `fm claude install|uninstall|show`, `mcp`
- Execution and upkeep: `fm sync`, `execute [--dry-run]`, `backtest`, `tune`, `canary`, `drill`, `version`

`fm bot` (phone approvals only, started by hand) is superseded by `fm run` (§13.1). The two together would poll one
channel twice, so `fm setup` reports a running `fm bot` and never starts `fm run` beside it.

**Notifications:** Telegram or ntfy for proposals (with buttons), executions, failures, and the weekly report.

### 12.2 `fm setup`

`src/fm/setup/wizard.py` holds the steps and `src/fm/commands/setup.py` the prompts and the live collaborators. Every
external effect is injected (`SetupDeps`: session load and login, discovery, league confirmation, phone setup, sync,
the two scheduler jobs, the `fm bot` probe, the doctor), so the tests drive the whole flow offline.

1. **Session.** No session, or an expired one, runs the `fm login` flow. An expiring one offers renewal.
2. **Leagues.** `discover_leagues` lists the current-season leagues, each as `NFL  <league name>  -  team 7 "<team
   name>"`. On a fresh install every one is selected by default. A discovered league that an existing `config.toml`
   lacks defaults to no, so `--yes` never opts an installation into a new league's automation. When discovery fails
   or finds nothing, the `--league-url` values (repeatable) or typed URLs are confirmed with `confirm_league`; a league
   named by URL defaults to yes.
3. **Presets.** Each new league gets a short key (`nfl`, `nba`, then `nfl2`...) and a preset (default `balanced`;
   `--preset` sets it for all), shown with its one-line description.
4. **API keys.** `ANTHROPIC_API_KEY` (a hidden prompt, skippable: the advisor falls back to template rationales) and
   the optional `ODDS_API_KEY` go to `.env` through `dotenv.set_key`. A key `.env` already has is left alone. A key
   only this shell exports is offered for saving (default yes), because `fm run` at logon reads `.env`, not the shell.
   No value is ever printed.
5. **Phone.** ntfy (the default: no account), Telegram (asks for the @BotFather token, then captures the chat) or
   none, through `src/fm/notify/setup.py`, which `fm notify setup` shares. A channel whose secrets are already in
   `.env` is not set up again.
6. **Config.** A missing `config.toml` is written fresh by `render_config`. An existing one is never rewritten:
   `add_leagues` appends only the new leagues, and a present league whose season or team differs from ESPN's is shown
   and updated through `set_values` only once confirmed. The confirmation defaults to yes only for a move to the
   sport's current season with the same team; another team, or a season that is not the current one, defaults to no.
   `set_notify_channel` runs only when the channel changed.
7. **First sync** of each new or updated league, one at a time, with progress (`--no-sync` skips).
8. **Autostart.** `fm autostart install` and the wake timer (`fm schedule install`), each only when the scheduler
   does not already have it (`--no-autostart` skips). A running `fm bot` is reported, never stopped: the logon job is
   then registered without starting `fm run`, and setup says to stop `fm bot` first.
9. **Summary.** The `fm doctor` lines, then "Dashboard: fm ui. Ask Claude: fm claude install."

**Flags.** `--yes` (every default, no prompts, every secret skipped), `--league-url`, `--preset`,
`--channel ntfy|telegram|none`, `--no-sync`, `--no-autostart`. Re-running is idempotent: with nothing new no file is
written, and `config.toml` stays byte-identical.

**League discovery** (`src/fm/espn/discover.py`, read-only):

- **`current_season(sport, now)`.** NFL is the year of the September kickoff (January to July belong to the previous
  year). NBA is labelled by its end year: from August onward it is `year + 1`.
- **`discover_leagues(session, *, client, now, all_seasons)`.** One GET to `https://fan.api.espn.com/apis/v2/fans/{SWID}`
  with the `espn_s2`/`SWID` cookies, explicit timeouts and a single retry on transport errors.
  - **The response shape is not verified against a live account**, so the parser (`parse_fan_profile`) is defensive.
    It tries the structured path `preferences[].metaData.entry` (`groups[].groupId` for the league, `entryId` for the
    team, `seasonId`, and `abbrev`/`gameId` or the sport from `entryURL`). If that finds nothing, it deep-walks the
    JSON for fantasy URLs that carry `leagueId=` under `/football/` or `/basketball/`.
  - It keeps ffl/fba only, deduplicates, and keeps the newest season at or after the current one per league (every
    season with `all_seasons=True`).
  - Any failure raises `DiscoveryError` (a 401 says to run `fm login`), and setup falls back to URLs.
- **Teams.** Only the fan API's `entryId` sets `team_id`. A URL's `teamId` is a hint only (`team_hint`).
  `confirm_league(ref, session)` reads `mTeam`, `mStandings` and `mSettings`, fills in the team (`resolve_team`: the
  team whose `owners` hold the session's SWID), the season and the names, and refuses a team the SWID does not own.
  `parse_league_url(text)` accepts any `fantasy.espn.com/{football|basketball}/...` URL with `leagueId=`.
- **Fixtures.** Synthetic `tests/fixtures/espn/fan_api/*.json` cover the structured and the URL-only shapes, with no
  real names. Never capture a real fan-API response into the repo: it holds personal data.

### 12.3 Health checks and `fm doctor`

`src/fm/health.py`: `run_checks(*, online=False, now=None, launch_browser=False, ...) -> list[Check]`, with
`Check(name, status: ok|warn|fail|skip, detail, fix)` where `fix` is the exact command to run (`fm login`,
`fm setup`). It is pure data, shared by `fm doctor`, the setup summary, the dashboard and the MCP `health` tool, and
import-light: no executor, flows, tick, advisor, fastmcp or anthropic (a test pins that).

- **Offline checks** (the default; only local state is read):
  - `config`: the file and `.env` load, and the required secrets are there. With `notify.channel = "none"` only
    `ANTHROPIC_API_KEY` is required, and its absence is a WARN: the advisor writes template rationales.
  - `session`: the profile exists and the cookie expiry gives ok, expiring or expired.
  - `browser`: Edge or Chrome is installed, found without launching anything.
  - `notify`: the channel's secrets are in `.env`; `none` is a WARN.
  - `tick`: the last tick in `tick-state.json`, WARN after 30 min, FAIL after 6 h or when none ever ran.
  - `autostart`: `fm run` registered at logon and holding the run lock.
  - `sync:<league>`: WARN after 2 h, FAIL after 24 h or before the first sync.
  - `pause`, open `proposals`, `deadlines` (open proposals due within 2 h).
  - `llm`: the advisor's state by the tick's own rule (`fm.jobs.advice.advisor_off`, which loads no SDK): on, with
    today's spend against `[llm] daily_budget_usd`; off because the key is missing or the budget is 0 (OK with the
    reason, since `config` carries the WARN); or off until 00:00 UTC because today's budget is spent (WARN).
  - `news`: when each league's news was last polled (`tick-state.json`), when an item was last triaged
    (`news_items`) and how many wait. Items the last poll left untriaged while the advisor is on are a WARN pointing
    at `logs/run.log`.
- **The session without a second browser.** Every tick records its session verdict in `tick-state.json`, and the
  check reads that record. Only `fm doctor` (`launch_browser=True`) may launch the browser to read the cookie, only
  when the record is older than 30 minutes, and only while it can take the run lock, so it never starts a browser
  beside a tick. The dashboard and MCP never launch one. Likewise only `fm doctor` asks the scheduler about the logon
  job; the others judge from the run lock alone. Health never creates `state.db`.
- **`--online`** adds an `mStatus` read per league and a channel ping (Telegram `getMe`, or the ntfy server's
  `/v1/health`).
- **`fm doctor [--online] [--json]`** prints one line per check plus its fix, and exits 1 on any FAIL.

### 12.4 Dashboard

`src/fm/web/`: a Starlette app (`fm.web.app.create_app`) with a JSON API and one static page
(`src/fm/web/static/index.html`, `app.js`, `app.css`, plus `signin.html` shown without a session). It is vanilla JS with no CDN and no build step, works offline,
has light and dark themes and stays usable at phone width. `fm.web.__init__` holds constants only, so the CLI never
imports Starlette.

| endpoint | serves |
|---|---|
| `GET /auth?t=<secret>`, `GET /auth/ping` | sign-in with a one-time link; `fm ui`'s probe for a running dashboard |
| `GET /api/session` | the session's CSRF token |
| `GET /api/overview` | per league: team, record, standing, sync age, next deadlines, open proposals; plus the session check, health counts and pause state |
| `GET /api/proposals?all=0&league=` | the proposal list, with the moves in player names and a trade offer's pitch |
| `POST /api/proposals/{id}/approve`, `.../reject` | `fm.proposals.approve` / `reject` with `decided_by="dashboard"` (404 unknown, 409 not allowed) |
| `POST /api/pause`, `POST /api/resume` | the kill switch |
| `GET /api/lineup/{league}`, `GET /api/waivers/{league}` | through `fm.views`; nothing stored |
| `GET /api/reports`, `GET /api/reports/{name}` | the report list and one report's markdown, rendered client-side as text |
| `GET /api/health` | `fm.health.run_checks()`, offline, never launching a browser |
| `GET /api/settings`, `POST /api/settings` | preset per league (`reset_overrides`), `trade_finder.enabled`, the notify channel, through `fm.config_edit` |

- **No write path to ESPN.** `fm.web` imports nothing from `fm.executor` or the browser flows (an AST import test
  enforces it, like the MCP one). Approving records the decision only; `fm run` or the scheduled tick executes it, and
  an approval near its deadline pulls `fm run`'s next tick forward (§13.1).
- **Security.** The dashboard binds `127.0.0.1` only (exclusively on Windows), which every account and process on the
  machine can reach, so it also needs a sign-in only this user can complete:
  - The server keeps one unused login secret and writes its link, `http://127.0.0.1:<port>/auth?t=<secret>`, to
    `<config dir>/dashboard.url`, readable by this user only. The link works once, within 10 minutes, and is replaced
    in the file as soon as it is used. It answers with an `HttpOnly`, `SameSite=Strict` session cookie,
    `fm_session_<port>`. Every `/api/*` request needs that session (401 otherwise).
  - Requests whose `Host` is not `127.0.0.1:<port>` or `localhost:<port>` are refused (400), against DNS rebinding.
  - Every request other than GET and HEAD needs `X-FM-Token` equal to the session's CSRF token, which the page reads
    from `GET /api/session`, and a same-origin `Origin` when one is sent (403 otherwise).
  - No CORS headers; a strict CSP, `nosniff`, `DENY` framing and `no-store` on every response. HTML is never built
    from data on the server, and the page builds the DOM with `textContent` only.
- **Commands.** `fm ui [--port 8711] [--no-open]` opens the dashboard signed in: it asks a running one (`fm run`'s,
  found through the URL file and `/auth/ping`) for a fresh link, or serves its own until Ctrl+C. `fm run --ui/--no-ui`
  (default on) serves it in a daemon thread beside the loop, through `uvicorn.Server`.

### 12.5 MCP server and Claude

**MCP server** (`src/fm/mcp_server.py`, FastMCP, run with `fm mcp` over stdio). Its tools are `status`, `lineup`,
`waivers`, `trade_eval`, `list_proposals`, `health` (offline checks), `report` (the latest weekly report's markdown per
league) and `create_proposal`. `create_proposal` is `fm.proposals.propose` capped at `approve`, so nothing from MCP is
ever auto-approved, with the trade caps. There is no approve, execute or pause tool. The server's `instructions`
string says the same. An allowlist, a description scan, an AST import scan and a subprocess `sys.modules` test keep
it free of any write path. This lets you ask "who should I start at FLEX?" from Claude Code or Desktop, including from
your phone via Remote Control.

**Shared read views** (`src/fm/views.py`). The JSON-able builders behind the MCP read tools and the dashboard:
`status`, `lineup`, `waivers`, `trade_eval`, `list_proposals`, `proposal_dict`. They read the store and the sync's
captures, raise `ViewError` (`ToolError` in MCP, 400 in the dashboard), import no fastmcp, executor or browser flows,
and store nothing.

**`fm claude install|uninstall|show [--code/--no-code] [--desktop/--no-desktop] [--dry-run]`** (`src/fm/claude/`):

- **Claude Code.** `claude mcp add-json --scope user espn-fantasy '<json>'` (by the full path to `claude`, re-adding
  when present), with the server command `<uv> run --project <repo> fm mcp` plus `FM_CONFIG_DIR`/`FM_CACHE_DIR` when
  they are set. It copies the skill to `~/.claude/skills/espn-fantasy/SKILL.md` and the slash commands to
  `~/.claude/commands/espn-fantasy/{week,lineup,waivers,trade}.md` (`/espn-fantasy:week` and so on). Without `claude`
  on PATH it says so and prints the command.
- **Claude Desktop.** It merges an `mcpServers.espn-fantasy` entry into every existing `claude_desktop_config.json`:
  on Windows `%APPDATA%\Claude\` and the MSIX path
  `%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\Claude\`, on macOS `~/Library/Application Support/Claude/`.
  The original is kept once as `.fm-orig.bak` and each edit backs the file up first. Other servers' entries are never
  touched.
- **Your edits survive.** A shipped asset the user edited (its hash is not a shipped one) is backed up as `.user.bak`
  before a re-install overwrites it, and `uninstall` keeps edited files.
- **The skill** tells Claude to start with `status`; use `lineup`, `waivers`, `trade_eval`, `report` and `health`;
  store a proposal with `create_proposal` only when asked; never claim a move was made, since approvals happen on the
  phone or in the dashboard and the executor acts; and cite engine numbers. The commands are `week` (weekly briefing),
  `lineup`, `waivers` and `trade <text>`.
- Every effect is injectable: tests use a temp home and a fake runner.

## 13. Runtime and scheduling

- Runs on the local machine (Windows PC or Mac) that holds the browser profile. This isn't optional: stats.nba.com
  and the NBA CDN block cloud IPs, and ESPN may block datacenter IPs too. Each machine has its own profile and
  `state.db`; run it on one of them at a time.
- The scheduled jobs and the MCP server run `uv run --project <repo>` from the clone they were installed from. The
  recommended practice is a separate, pinned clone for the live installation, upgraded on purpose
  (`git pull --ff-only && uv sync --locked`), while development happens in another clone.

### 13.1 `fm run`: the one process

`src/fm/jobs/runner.py`, `run_forever(...)`. One long-running process ticks on a timer and records approvals between
ticks. It replaces the scheduled `fm tick` plus a separately started `fm bot`, and it starts at logon (§13.3). Every
collaborator is injectable: config loader, store opener, channel opener, tick runner, reply handler, tree killer,
clock, sleep, a `stop` event and the lock.

- **Ticks are due by wall clock** (`next_tick_at`, every 10 minutes by default), so after the machine sleeps the first
  iteration ticks at once and reports what was missed. The config is reloaded for every tick, so edits from `fm setup`,
  the dashboard or `fm config set-preset` apply without a restart. A config that will not load is logged and alerted,
  and that tick is skipped.
- **Each tick is a child process with a hard timeout.** A real scheduled tick once hung for five days inside a
  headless browser. So the loop runs every tick as `python -m fm.jobs.runner tick`, appends its output to
  `logs/run.log` and waits at most the tick timeout (20 minutes by default). A tick that overstays is stopped with its
  whole process tree, the browser included (`src/fm/jobs/proctree.py`: `taskkill /T /F` on Windows, the process group
  on macOS), alerted, and the loop carries on. `in_process_tick` runs the tick in the loop's own process, without a
  timeout; only tests and the offline journey (`tests/test_journey.py`) use it.
- **Approvals are recorded here and executed by a tick.** Between ticks the loop polls the phone channel (Telegram's
  long poll, the ntfy reply topic), and each press goes through `fm.notify.handle_reply`: the nonce, then
  `fm.proposals`. Nothing executes in the `fm run` process. An approval whose deadline is within 30 minutes pulls the
  next tick forward to now, and that tick child executes it under the timeout and tree kill. An approval recorded
  elsewhere (the dashboard, `fm proposals approve`) is found by a cheap store query between polls and pulls the tick
  forward the same way, once per proposal. With no channel configured the loop just sleeps until the next tick.
- **Errors never end the loop.** A tick, poll, press or config load that raises is logged and alerted at most once an
  hour per kind; channel errors back off from 2 s, doubling to 60 s. Ctrl+C or `stop` exits cleanly, stopping a tick
  child that is running, and releases the lock.
- **Logs.** `logs/run.log` under the config dir, rotating 5 × 1 MB, plus the console when one is attached.
- **Command.** `fm run [--every 10] [--no-execute] [--tick-timeout 20] [--ui/--no-ui] [--port 8711]` runs the loop
  in the foreground; `--no-execute` never writes, and its ticks list what they would have executed. The dashboard
  (§12.4) is served beside the loop unless `--no-ui`.

### 13.2 The run lock

`src/fm/jobs/runlock.py`: `run.lock` in the config dir, an OS-level exclusive lock (`msvcrt.locking` on Windows,
`fcntl.flock` on macOS). Platform branches live only here, in `jobs/proctree.py` and in the scheduler backends. The
holder writes `{pid, command, started_at}` into the file.

- `fm run` holds the lock for its lifetime. At start it waits for a holder (a scheduled tick mid-run), but a
  `fm tick` holder older than the tick timeout plus 2 minutes is stuck: `fm run` stops its whole tree, alerts once and
  takes the lock.
- A scheduled `fm tick` that cannot get the lock prints `fm run is active (pid N); skipping` and exits 0. The tick
  children of `fm run` carry `FM_RUN_LOCK_HELD_BY` naming their parent, so they run under its lock instead.
- The scheduled `fm tick` runs its own work the same way: as a child with `--timeout` (default 20 minutes) and the
  tree kill (`--in-process` for debugging).
- `fm doctor` holds the lock while it launches the browser to read the session (§12.3).

### 13.3 Autostart and the wake timer

`fm autostart install|uninstall|show [--dry-run]` registers `fm run` to start at logon, through the scheduler
backends behind `jobs/scheduler_base.py`. Every command goes through an injectable `Runner`, so no unit test touches a
real scheduler. Neither platform needs admin.

- **Windows.** `Register-ScheduledTask` task `espn-fantasy-run`: an at-logon trigger for the current user, restart on
  failure, no run-time limit, `-MultipleInstances IgnoreNew`. The action is
  `uvw.exe run --project <repo> --env-file <env> fm run`; `uvw.exe` is uv's windowless launcher, so no console window
  stays open (without it the task runs `uv.exe`, with a note). An at-logon trigger first fires at the next logon, so
  install starts the task at once with `Start-ScheduledTask`; `--no-start` only registers it.
- **macOS.** A LaunchAgent, `local.espn-fantasy-run.plist`, with `RunAtLoad` and `KeepAlive`, loaded into the user's
  GUI domain.

**The tick schedule stays installed as the wake timer.** `fm schedule install` keeps registering `fm tick` every 10
minutes:

- **Windows:** `schtasks` + a `.cmd` wrapper and the wake/battery settings from software-factory's
  `scheduler/windows.py` (`jobs/scheduler_windows.py`). It may wake the PC and run on battery.
- **macOS:** a per-user launchd LaunchAgent (`~/Library/LaunchAgents/local.espn-fantasy-tick.plist`,
  `StartInterval`, loaded into `gui/<uid>`) + a `.sh` wrapper (`jobs/scheduler_macos.py`). launchd runs on battery
  but cannot wake a sleeping Mac, so keep the Mac awake at locks with the energy settings or `pmset`.

A tick that fires while `fm run` holds the lock skips at once; the woken loop ticks on its own, because its next tick
is due by wall clock. Without `fm run`, the scheduled tick does everything by itself, as it did before Phase 10, but
approvals wait for the next tick. The scheduled tick's log is `logs/tick.log`.

### 13.4 Each tick

1. Reconcile what a killed process left executing, and expire every proposal whose deadline has passed (an approved
   or auto proposal that expired unexecuted is alerted as missed).
2. Check the session, and record the verdict in `tick-state.json` for the health checks.
3. Per league: refresh stale data (sync when older than an hour); compute upcoming deadlines (kickoffs and tips from
   the pro schedule, the waiver run from league settings); alert the windows missed since the last tick; poll and
   store the news on its cadence (NBA 30 min, NFL 1 h) and have the advisor triage it (§10); refresh the availability
   inputs (a few times a day, and at once when the triage stored new signals); run the decisions that are due, once
   per window, the advisor breaking cited near-ties in lineups under 6 h from a lock; in the weekly window (Monday
   09:00 ET until the matchup period turns, once per period) run the weekly strategist and write and push the weekly
   report; have the advisor explain the run's new non-trivial proposals and draft the pitch of each new trade offer;
   push new proposals to the phone. News and advisor failures are warnings, never a skipped league.
4. Auto-approve at T-15, then execute approved proposals whose time has come, soonest deadline first, through the one
   write path. Nothing executes while paused.
5. Record the tick in `tick-state.json`: last run, windows run, proposals explained and pushed, inputs refreshed, news
   polled, alerts sent, and the session verdict.

- The weekly report is the tick's (step 3); `fm report` writes the same report on demand. A weekly window the tick
  slept through is alerted ("run fm report by hand"), never re-run, since a retry would ask Claude again.
- The PC must be awake at locks: enable wake timers. The first tick after a missed window reports what was missed.
- If the PC proves unreliable, move to an always-on box later. The tradeoff is that the ESPN session then lives there.

## 14. Storage, config, layout

- `~/.config/espn-fantasy/` (not `%APPDATA%`, which MSIX virtualizes):
  - `config.toml` (and `config.toml.bak`, the file before the last edit);
  - `.env`: `ANTHROPIC_API_KEY`, `ODDS_API_KEY`, and either `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` or
    `NTFY_TOPIC` + `NTFY_REPLY_TOPIC`;
  - `browser-profile/`, `state.db` (SQLite), `audit/` (requests, responses, traces, screenshots);
  - `tick-state.json` (what earlier ticks did: windows run, proposals explained and pushed, inputs refreshed, news
    polled, alerts sent, and the last session verdict), `run.lock` (§13.2), `dashboard.url`
    (the dashboard's one-time login link, owner-only), `logs/run.log` (`fm run`) and `logs/tick.log` (the scheduled
    tick), `paused.json` while `fm pause` holds.
- `~/.cache/espn-fantasy/`: raw source responses and ESPN captures, parquet snapshots, reports. Safe to delete.
- `FM_CONFIG_DIR` / `FM_CACHE_DIR` override both (tests point them at temp dirs). Development never uses the real
  config dir or the live clone.
- In the repo: code plus scrubbed fixtures only. No secrets and no real manager names.

**`config.toml`.** `fm setup` writes it; this template is the same file by hand. A `preset` (§11) fills every policy
and trade-finder key a league leaves unset, so the `[league.policy]` and `[league.trade_finder]` tables are optional,
and any key written there wins over the preset. `notify.channel` is `telegram`, `ntfy` or `none`.

```toml
# ~/.config/espn-fantasy/config.toml
[[league]]
key = "nfl"                   # short name used on the command line (fm sync --league <key>)
sport = "nfl"                 # nfl | nba
espn_league_id = 1234567      # leagueId=… in the league URL
season = 2026
team_id = 1                   # teamId=… on your team page
preset = "balanced"           # hands_off | balanced | ask_first

[league.policy]               # optional; each key written here overrides the preset
untouchables = []             # player names or ESPN IDs; resolved at sync

[[league]]
key = "nba"
sport = "nba"
espn_league_id = 7654321
season = 2027                 # ESPN labels NBA seasons by end year; confirm with seasonId= in the URL
team_id = 1
preset = "balanced"

[llm]
model = "claude-opus-5-5"
daily_budget_usd = 2.0

[notify]
channel = "ntfy"              # telegram | ntfy | none; the credentials live in .env
```

Without a preset, these are the keys and their model defaults:

```toml
[league.policy]
bench_inactive = "auto"       # off | approve | auto (auto only fires at T-15 when unanswered)
lineup = "auto"               # off | approve | auto
add = "approve"               # off | approve | auto (auto needs auto_add_min_gain)
add_drop = "approve"          # off | approve
waiver = "approve"            # off | approve
max_transactions_per_week = 3
max_faab_pct_per_bid = 0.35
untouchables = []

[league.trade_finder]         # drafts outgoing offers for approval, never sends one
enabled = false
min_accept = 0.4              # least P(accept)
min_title_gain = 0.01         # least gain in title odds (0.01 = 1 point)
min_ros_gain = 0.25           # without a simulation: least gain in starter seasons
max_offers = 1                # drafts a run; the weekly cap of 3 still holds
```

**Editing `config.toml`** (`src/fm/config_edit.py`, tomlkit). One module writes the file, and every writer (`fm setup`,
`fm config set-preset`, the dashboard's settings) goes through it:

- **API.** `render_config(leagues, *, llm, notify)` renders a fresh, commented file. `add_leagues(path, entries)`
  appends `[[league]]` tables, skips any `(sport, espn_league_id)` already present and reports the leagues whose
  season or team differ (`AddLeaguesResult(added, unchanged, changed)`). `set_value` / `set_values` change values
  such as `preset`, `trade_finder.enabled` and `policy.lineup`; `remove_keys` and `preset_overrides` back
  `fm config set-preset --reset-overrides`; `set_notify_channel(path, channel)`.
- **Write rules.** Comments and key order survive an edit. The edited text is validated with `load_config` before it
  replaces the file; an invalid edit raises `ConfigError` and leaves the file untouched. Each write is atomic (a
  `mkstemp` temp file plus `os.replace`) under `config.toml.lock`, and the previous file is kept as `config.toml.bak`.

```
espn-fantasy/
  pyproject.toml  CLAUDE.md  ROADMAP.md  docs/DESIGN.md  docs/espn-api.md
  data/           blend_weights.toml, id_overrides*.csv, stadiums.csv, calendars/
  src/fm/
    cli.py  paths.py  config.py  config_edit.py  presets.py  health.py  views.py  mcp_server.py
    commands/     one module per CLI group, auto-discovered
    setup/        wizard.py (fm setup)
    web/          app.py, login.py, server.py, static/ (the dashboard)
    claude/       install.py, assets/ (fm claude install)
    store/        db.py, models.py, repos.py, migrations/
    espn/         client.py, models.py, auth.py, settings.py, ids.py, discover.py, calendar.py
    browser/      session.py, transactions.py, selectors.py, canary.py, drills.py, fakes.py,
                  flows/{lineup,add_drop,waiver,trade}.py
    sources/      base.py, nflverse.py, sleeper.py, market.py, odds.py, weather.py, news.py,
                  nba_stats.py, nba_schedule.py, nba_injuries.py, darko.py
    sports/       base.py, nfl.py, nba.py
    model/        ids.py, ids_nba.py, scoring.py, projections.py, availability.py, relevance.py, valuation.py,
                  categories.py, value_nba.py, simulate.py, baseline_nfl.py, baseline_nba.py
    decide/       registry.py, lineup.py, lineup_daily.py, waivers.py, faab.py, streaming.py, weekly.py,
                  trades.py, offers.py, rankings.py
    advisor/      client.py, news_triage.py, close_call.py, explain.py, pitch.py, strategist.py, prompts/
    proposals/    queue.py, policy.py, payloads.py, pause.py
    executor/     run.py, runtime.py, transport.py, ui.py, verify.py, audit.py
    notify/       base.py, telegram.py, ntfy.py, nonces.py, messages.py, send.py, bot.py, setup.py
    jobs/         sync.py, tick.py, runner.py, runlock.py, proctree.py, deadlines.py, report.py,
                  scheduler_base.py, scheduler_windows.py, scheduler_macos.py
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
- **The offline journey** (`tests/test_journey.py`) walks the plug-and-play path end to end in temp config and cache
  dirs built from `tests/fixtures/home`, with fakes for every external effect: `fm setup --yes`, `fm config check`,
  `fm doctor --json`, one `fm run` iteration with `--no-execute` (the real lineup decision drafts proposals), a
  dashboard sign-in and approval, then the next iteration executing it through the real flow over a fake runtime. The
  network guard is on and any real HTTP request fails the test.
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

These are product phases with calendar targets. `ROADMAP.md` breaks them into 63 tasks. Its "Phase N" headings are
dependency layers, not these product phases, so NBA work runs in parallel with NFL work there. Its Phase 6 milestone
covers product phases 1–3 plus NBA lineups and streaming; its Phase 8 milestone is trades. Its Phases 10 to 12 came
after this plan and make the tool plug and play: presets (§11), `fm setup`, `fm doctor`, the dashboard and the Claude
integration (§12), and `fm run` (§13). Its Phase 13 wires the Claude workers into the tick (§10, §13.4).

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
   Since Phase 11, `fm setup` offers ntfy first, because it needs no account.
5. **Runtime:** your own Windows PC or Mac. This is effectively forced, because the NBA data sources block cloud IPs.

Nothing blocking is left open. Since Phase 11 the setup on your side is `fm setup`: one ESPN sign-in, then it finds
the league and team IDs and pairs the phone.

## 19. Decisions and alternatives

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Language | Python 3.13 + uv | TypeScript | The data ecosystem is Python (espn-api, nflreadpy, nba_api, numpy/scipy, Playwright, anthropic SDK). Matches software-factory. Pinned to 3.13 because `nba_api` supports ≤ 3.13 |
| ESPN reads | JSON API | DOM scraping | Structured, fast, and the same calls the web app makes |
| ESPN writes | The web app's own transaction calls sent from the logged-in Playwright session, with UI click-through as fallback | UI click-through only; plain HTTP outside the browser | The API path was verified live by several projects in Sept 2026, while UI-only bots broke on every redesign. Keeping it in the browser keeps one session and a real client |
| LLM role | Narrow workers on a deterministic engine | One agent deciding everything | Testable, auditable, predictable cost |
| Runtime | Local PC or Mac: `fm run` at logon, with the `schtasks` / launchd tick kept as the wake timer | Cloud VPS, scheduled cloud agents | The ESPN session and browser profile live locally, and NBA data sources block cloud IPs. One process ticks and listens for approvals; each tick is a child process with a hard timeout |
| Storage | SQLite + parquet cache | Postgres, DuckDB | Single user, zero ops |
| Approvals | ntfy (offered first) or a Telegram bot, plus the local dashboard | Discord, Pushover, openclaw | Both give tap-to-approve with no inbound ports; ntfy needs no account |
