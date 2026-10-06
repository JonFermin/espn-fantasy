# ESPN fantasy API: what the real leagues show

> ROADMAP #14 spike, captured 2026-10-05 and 2026-10-06 from the two configured leagues: NFL (`ffl`, season 2026,
> week 4 on the Monday, week 5 on the Tuesday) and NBA (`fba`, season 2027, preseason, scoring period 1). Read views
> come from real responses, scrubbed into `tests/fixtures/espn/real/` (`index.json` names the view and request behind
> each file). Write payloads come from three sources that agree: the requests ESPN's web app sent while Jon drove the
> flows under the write guard on 2026-10-06 (aborted before they reached ESPN, scrubbed into
> `tests/fixtures/espn/real/{ffl,fba}/write_*.json`; section 4), ESPN's own web client code (the `kona` build
> `35f16ade75be-1.504`, shared `commons/main` bundle, saved verbatim in `tests/fixtures/espn/real/webclient.json` and
> pinned by the fixture tests), and the transactions both leagues have on record. A snapshot before and a verification
> after the guarded session found both leagues unchanged. `scripts/capture/README.md` explains how to reproduce all of
> it.
>
> Claims marked *inferred* rest on reasoning or on less data than a fixture; everything else cites a fixture.

## 1. Resolved unknowns

| # | Question (DESIGN §6.1/§6.2/§6.3/§9.3, ROADMAP #9) | Answer from real data | Evidence |
|---|---|---|---|
| 1 | NBA matchup → days | `scheduleSettings.matchupPeriods` lists period ids of the type named by `scheduleSettings.periodTypeId` (2 = weeks in this league). Weeks come from the web client's calendar, not from any read view. Matchup 1 = days 1–6 (Tue Oct 20 – Sun Oct 25), matchups 2–17 = 7-day Monday–Sunday weeks, matchup 18 = days 119–132 (14 days, the All-Star break), matchups 19–21 (playoffs) = days 133–153. Total 6 + 16×7 + 14 + 3×7 = 153 = `status.finalScoringPeriod`. **DESIGN §9.3's "days 1–13" inference is wrong.** | `fba/calendar.json`, `fba/mSettings.json`; every one of 1,200 NBA games starts inside its day's window |
| 2 | Pending-offer view | Trade offers come from `mTransactions2` (types `TRADE_PROPOSAL` …), not `mPendingTransactions`. `mPendingTransactions` holds the viewing team's own pending moves (the web client reorders waiver claims by `subOrder` and edits their `bidAmount` through it); it listed none of the NBA league's offers. **An expired offer keeps `status: PENDING` and `isPending: true`, and so does the CANCEL record of its expiry:** ESPN records the expiry as a separate `TRADE_PROPOSAL` with `status: CANCELED`, `executionType: CANCEL`, `isPending: true` and `relatedTransactionId` = the offer. Offers expired 48 h after `proposedDate`. "Open" therefore means PENDING, `expirationDate` in the future, and no CANCEL record pointing at it: six records said pending, none was open. `Transaction.pending` (status `PENDING`, not a CANCEL record), `TransactionsView.open(now)` and `EspnClient.pending_offers` apply this rule since the phase 4 integration. | `fba/mTransactions2_waiver_trade.json` (three expired offers, three CANCEL records; `test_no_offer_was_open_although_six_records_say_pending`), `*/mPendingTransactions.json` |
| 3 | `mPendingTransactions` list key | `pendingTransactions` (also what the web client reads). With nothing pending, `ffl` sent `"pendingTransactions": []` and `fba` sent no key at all. | `ffl/` and `fba/mPendingTransactions.json` |
| 4 | Lineup-lock key and values | `rosterSettings.lineupLocktimeType` = `INDIVIDUAL_GAME` in both leagues. `rosterSettings.rosterLocktimeType` = `INDIVIDUAL_GAME` (`ffl`) and `FIRSTGAME_SCORINGPERIOD` (`fba`): NBA adds, drops and trades lock at the day's first tip while lineups lock per game. No league carries `FIRST_GAME_OF_WEEK`. | `*/mSettings.json` |
| 5 | `espn_s2` expiry | The cookie was set at the sign-in (2026-10-05 00:50 UTC) and expires 2027-11-09: exactly 400 days, which is Chromium's cap on cookie lifetime, so ESPN asks for at least that. It is not `HttpOnly`. *Inferred:* page visits and API reads do not refresh it; that rests on about 20 hours of observation (no `Set-Cookie` on any API response, the expiry unchanged), not on a renewal cycle. Revocation (logout, password reset) still shows up as a 401. The web UI's OneID login is separate: its `ESPN-ONESITE.WEB-PROD.api` cookie lasts 1 day, and this profile holds no OneID token in any cookie or storage key (section 4). | profile cookie store, read-only (names, flags and expiry dates, never values); `reads.json` request log |
| 6 | 429 behavior | None seen in 104 scripted reads at ≥ 1.0 s spacing (two read passes of 42, a snapshot and two verifications of 6 each, 2 smoke reads) or in 15 guarded page loads. Everything answered 200 except the intentional `top0` probes (400). ESPN sends no `RateLimit`/`Retry-After` headers. The threshold stays unmeasured: the spike's protocol forbade faster probing. Keep `DEFAULT_MIN_INTERVAL_S` (0.25 s): the web app itself fires about four league reads at once on every page load, a sync is gentler than that, and the client already honors `Retry-After`. | `reads.json` (`min_start_gap_s: 1.0`, statuses) |
| 7 | `lineupSlotStatLimits` (NBA games-played caps) | `{}` in both leagues: no caps, so the shape of a real cap is still unseen. The parser's bare-integer reading stays an assumption. | `*/mSettings.json` |
| 8 | `acquisitionSettings.waiverHours` | Present, `24` in both leagues. `waiverProcessHour` is **not** an ET hour: it is 11 (`ffl`) and 8 (`fba`), but claims processed at 03:47 ET (NFL, Wed Sep 30) and 03:50 ET (NBA, Mon Oct 5). Use `status.waiverLastExecutionDate` and `waiverProcessStatus` (run time → count) for timing. | `*/mSettings.json`, `*/mTransactions2.json` `processDate` |
| 9 | `filterStatsForTopScoringPeriodIds` | `value` must be > 0: 0 returns HTTP 400 `FILTER_INVALID_VALUE: Filter: Value is invalid, must be > 0`. `value: N` returns each player's actual lines for the last N scoring periods he played; `additionalValue` adds named entries. | `*/probes.json`, `*/kona_playercard_top2.json` |
| 10 | Composite stat-entry ids | `{source}{split}{season}` for season lines (`002026`, `102026`) and `{source}{split}{season}{period}` for period **projections** (`1120264`). **Actual single-period lines are keyed by the pro game id** with the game's "Game" split (#11): `01` + game id in `ffl` (`01401872969`), `05` + game id in `fba` (`05401811041`). In `ffl`, asking for `0120264` (week 4, partly played) returned nothing while week 4's game-keyed lines came back. In `fba` the fixtures alone leave it *inferred*: their `0120271` probe asked for a day not yet played. A one-off run of the new `played_period` probe (2026-10-05, not in the fixtures yet; the next `capture.py reads` records it in `probes.json`) asked for a played day, 2026 day 174, as `012026174` and `052026174`: neither came back, only `05401811043`. Match entries on `seasonId`/`scoringPeriodId`/`statSourceId`/`statSplitTypeId`, never on `id`. `ffl` also sends `122026`: projected, split 2, which ESPN labels "Rest Of Season". The label misleads: `102026` (split 0, "Season") is the rest-of-season line, its `GP` being exactly each healthy player's games left (13 for Loveland, Coker, Chase Brown and Rice in week 4: weeks 5–18 less the bye) and fewer only for the injured (DeVonta Smith 12, A.J. Brown 10 on IR), while `122026` gives Smith 16 games with 13 left; `Player.projection(season, 0)` reads `102026` and nothing reads `122026` (ROADMAP #21; `tests/model/test_valuation.py::test_the_split_0_season_line_is_espns_rest_of_season_projection`). | `*/kona_playercard_stat_entries.json`, `*/probes.json`, `ffl/kona_player_info.json` (first player: every stat line), `webclient.json` |
| 11 | `statSplitTypeId` | ESPN's web client labels the splits per game (`webclient.json` `statSettings`): basketball 0 Season, 1/2/3 Last 7/15/30 Days, 4 Average, **5 Game**; football 0 Season, **1 Game**, 2 Rest Of Season. The "Game" split (`gameSplit: true`) is the single-period actual line: `scoringPeriodId` = the week in `ffl`, the day in `fba`. The real `fba` window entries (`012027`, `022027`, `032027`, `scoringPeriodId: 0`) are preseason and empty, so the labels, not the data, carry their meaning. `Player.stat_entry` matched split 1 for a period, which found nothing in `fba`; since the phase 4 integration it matches each game's "Game" split (`fm.espn.models.STAT_SPLIT_GAME`) on the entry's fields, and `EspnClient.player_cards` no longer asks for period-keyed actual ids. | `webclient.json`, `fba/probes.json` |
| 12 | NBA daily projections | ESPN publishes none: `kona_game_state` says `hasGameStatProjections: false, showSeasonProjections: true` for `fba` (`true/false` for `ffl`), and `1120271` returns nothing. A day's projection is the season projection's per-game rate. | `*/kona_game_state.json` |
| 13 | `kona_player_info` without `filterSlotIds` | Fine: omitted and `[]` returned the same 50 players in the same order, in both games. | `*/probes.json` |
| 14 | `fba` writes (DESIGN §6.3: "unverified") | One code path serves every game: the transaction model, serializer, item builders and the service that picks a type per flow live in the shared bundle with no per-game branch, and the request path takes the game from `config.uri_nextgen_api`, so only `games/{ffl\|fba}` in the URL differs. Real NBA records match: `ROSTER` for day 1, `FUTURE_ROSTER` for days 2–3. **Confirmed by the guarded UI capture:** the NBA app sent `ROSTER` for a bench swap and `FREEAGENT` for an add, with the same keys in the same order as the NFL app's `ROSTER` and `WAIVER` (section 4). | section 4; `fba/write_*.json`; `webclient.json`; `fba/mTransactions2.json` |
| 15 | Re-reads right after a write | Not cached: every league-scoped view answered `cache-control: must-revalidate` and `x-cache: Miss from cloudfront`. Only the season-level views (`proTeamSchedules_wl`, `kona_game_state`) are cached, `max-age=300` (hits up to 297 s old). | `reads.json` request log |
| 16 | Does `mRoster` for a later `scoringPeriodId` echo that period? (#25's `set_lineup` refuses an answer for another period before spending the token) | **Yes, in both games.** One read-only GET per league on 2026-10-06 with `scoringPeriodId` well past the current one (NFL week 7 while `status.latestScoringPeriod` was 5; NBA day 3 while it was 1) answered with the top-level `scoringPeriodId` equal to the period asked for, `status.latestScoringPeriod` still the current one, and every team's roster filled for that period (16 and 13 entries for ours). The committed `mRoster.json` fixtures show the current period only. | read-only probe, not a fixture (it would carry nothing new but the period numbers) |
| 17 | Do writes carry `X-Fantasy-Source` / `X-Fantasy-Platform`, and `platformVersion`? | **Yes.** Every captured write carried exactly four headers beyond the browser's own: `accept: application/json`, `content-type: application/json`, `x-fantasy-platform: espn-fantasy-web`, `x-fantasy-source: kona`, which is what `fm.browser.transactions.WEB_CLIENT_HEADERS` plus the executor's transport sends. Every captured URL ended in `?platformVersion=<40-hex build sha>`, the same value on all five. We cannot know the current build, so our requests omit it; whether ESPN requires it is untested, since no request of ours has been sent. | `*/write_*.json`; `tests/executor/test_transactions.py` |

DESIGN §9.3's other "facts from the real leagues" hold: NBA scoring periods are days, day 1 is Tue Oct 20 2026 and day
153 is Sun Mar 21 2027, NBA games in `proTeamSchedules_wl` are keyed by day, `matchupAcquisitionLimit` arrives as
`0.42857142857142855` (3/7) with `matchupLimitPerScoringPeriod: true`, both leagues are H2H points with traditional
waivers and no FAAB, and the NFL league seeds its 4-team playoff by `TOTAL_POINTS_SCORED`. How ESPN rounds the NBA
acquisition limit for the 6-day and 14-day matchups (2.57 and 6 adds) is not visible until matchup 1 is played.

## 2. Hosts, auth and transport

- **Reads:** `GET https://lm-api-reads.fantasy.espn.com/apis/v3/games/{ffl|fba}/seasons/{season}/segments/0/leagues/{id}?view=…`
  (repeat `view=` for several). Season-level data drops `/segments/0/leagues/{id}`. Filters travel in the
  `X-Fantasy-Filter` header as JSON.
- **Writes:** `https://lm-api-writes.fantasy.espn.com/apis/v3/…`, the web client's `HOST_TYPE_FANTASY_WRITE_API`
  (section 4).
- **Auth:** the `espn_s2` and `SWID` cookies, for reads and writes alike. The web client sends requests with
  `withCredentials: true` and no bearer token, so the API needs nothing from the OneID web login.
- **Headers the web client adds:** `Accept: application/json`, `X-Fantasy-Source: kona`,
  `X-Fantasy-Platform: espn-fantasy-web`, a `platformVersion=<build sha>` query parameter, `Content-Type:
  application/json` on writes, and a 50 s timeout. Our reads send only `Accept` (and our own `User-Agent`) and work.
  The captured writes confirm the write-side set (§1 #17): those four headers, nothing else of ESPN's, cookies from the
  session.
- **Response headers worth reading:** `x-fantasy-filter-player-count`, `x-fantasy-filter-transaction-count` and
  `x-fantasy-filter-schedule-count` give the total behind a filtered page, which is what paging needs: 889 (`ffl`) and
  964 (`fba`) free agents and waiver players stood behind a 50-player `kona_player_info` page. Unfiltered views report
  the whole player universe (1,051 and 1,095). `x-fantasy-server-time` is the server clock in epoch ms.
  `x-fantasy-role` was `NONE` on every read.
- **CDN:** CloudFront (`via`, `x-amz-cf-pop`). League views are never served from cache; season views are cached for
  5 minutes. Headers arrived in 18–110 ms.
- **Errors:** `{"details": [{"type": "…", "message": "…", "metaData": {"teamid": …}, "resolution": "…"}]}`, which
  `fm.espn.client._error_detail` already reads.

## 3. Read views

Each subsection covers both games. Fixture paths are under `tests/fixtures/espn/real/`; the test command loads every
one (`tests/fixtures/espn/real/test_real_fixtures.py`).

### `mSettings` → `fm.espn.settings.LeagueSettings`

- `settings.scoringSettings`: `H2H_POINTS` in both leagues; 46 scoring items (`ffl`), 11 (`fba`, ESPN's default NBA
  points). `matchupTieRule` and `playoffMatchupTieRule` are `NONE`.
- `settings.rosterSettings`: `lineupSlotCounts` (`ffl`: QB 1, RB 2, WR 2, TE 1, FLEX 1, D/ST 1, K 1, BE 7, IR 1;
  `fba`: PG, SG, SF, PF, C, G, F 1 each, UTIL 3, BE 3, IR 3), `positionLimits` (−1 = none), the two lock types (§1
  #4; `LeagueSettings.lineup_lock_type` and `roster_lock_type`), `lineupSlotStatLimits: {}`, `isBenchUnlimited: true`,
  `isUsingUndroppableList: true`, `moveLimit: -1`.
- `settings.acquisitionSettings`: `WAIVERS_TRADITIONAL`, `isUsingAcquisitionBudget: false` (budget 100 unused),
  `waiverHours: 24`, `waiverProcessDays` (`ffl`: every day but Tuesday; `fba`: Sunday), `waiverProcessHour` (§1 #8),
  `matchupAcquisitionLimit` (`ffl`: −1; `fba`: 3/7 per day), `transactionLockingEnabled: false`.
- `settings.scheduleSettings`: `periodTypeId` (1 in `ffl`, 2 in `fba`), `matchupPeriodCount` (13 and 18 regular-season
  matchups), `matchupPeriods`, `playoffTeamCount` (4 and 6), `playoffMatchupPeriodLength` (2 weeks and 1 week),
  `playoffSeedingRule` (`TOTAL_POINTS_SCORED`, `H2H_RECORD`). NFL playoffs: matchup 14 = weeks 14–15, 15 = weeks 16–17.
  `ScheduleSettings` reads `matchupPeriods` as scoring periods only under `periodTypeId` 1 and answers `None` for the
  NBA league's week ids (§1 #1).
- `settings.tradeSettings`: deadline Wed Dec 2 2026 noon ET (`ffl`) and Fri Feb 26 2027 noon ET (`fba`),
  `revisionHours: 24`, `vetoVotesRequired` 4 and 3, `max: -1`.
- `status`: `currentMatchupPeriod`, `latestScoringPeriod`, `firstScoringPeriod`, `finalScoringPeriod` (17 and 153),
  `transactionScoringPeriod`, `waiverLastExecutionDate`, `waiverProcessStatus`.
- Fixtures: `ffl/mSettings.json`, `fba/mSettings.json` (whole responses). `LockType.FIRST_GAME_OF_WEEK` (no league
  carries it) was dropped at the phase 4 integration; ESPN's weekly values (`FIRSTGAME_WEEKLY`,
  `INDIVIDUAL_FIRSTGAME_WEEKLY`) parse as `UNKNOWN`. `tests/espn/test_settings.py` reads these captures for everything
  the real leagues show (scoring items and the D/ST overrides, slots, position limits, lock types, traditional waivers,
  deadlines, both schedule shapes) and mutates copies of them for the variant and error cases; the hand-built
  `tests/fixtures/espn/fba_settings_9cat.json` and `ffl_settings_ppr.json` stand-ins stay for categories, FAAB, a
  season acquisition limit and games-played caps, which neither real league exercises.

### `mTeam` + `mStandings` → `TeamsView`

- `teams[]`: `id`, `name`, `abbrev`, `logo`, `owners`/`primaryOwner` (SWIDs), `record` (overall/home/away/division),
  `points`, `playoffSeed`, `waiverRank`, `transactionCounter` (acquisitions, drops, `matchupAcquisitionTotals` by
  matchup period), `valuesByStat`, `tradeBlock`, `isTransactionLocked`.
- `members[]`: `id` (SWID), `displayName`, `firstName`, `lastName`, `notificationSettings`. The response also carries
  the whole season `schedule` (home/away team ids per matchup period).
- **Team ids are not contiguous:** the NBA league's ten teams have ids 1–16 with gaps (the fixture keeps that shape).
- Fixtures: `*/mTeam+mStandings.json` (notification settings emptied, schedule cut to matchup 1).

### `mRoster` → `RostersView`

- `teams[].roster.entries[]`: `playerId`, `lineupSlotId`, `acquisitionType`, `acquisitionDate`, `injuryStatus`,
  `status`, `pendingTransactionIds`, and `playerPoolEntry` with `lineupLocked`/`rosterLocked`/`tradeLocked`, `onTeamId`,
  `ratings` and the full `player` (eligibility, injury, ownership, stat lines). `roster.tradeReservedEntries` lists
  players held by a pending trade.
- `scoringPeriodId` picks the period; the NFL league's untrimmed response is 2.5 MB (every player's season, period and
  per-game lines), so sync code should not request it per period without need. A later period's answer echoes that
  period at the top level and carries that period's lineups, while `status.latestScoringPeriod` stays the current one
  (§1 #16), which is what `set_lineup` reads before and after a `FUTURE_ROSTER` write.
- Fixtures: `*/mRoster.json` (our whole roster, 3 opponent players, one stat line each; the current period).

### `mMatchup` → `MatchupsView`

- `schedule[]`: `id`, `matchupPeriodId`, `winner`, `home`/`away` with `teamId`, `totalPoints`, `pointsByScoringPeriod`,
  `cumulativeScore` (`scoreByStat` is filled even in points leagues). Current-period entries also carry
  `rosterForCurrentScoringPeriod`/`rosterForMatchupPeriod`, and the response includes `teams` with rosters: 2.3 MB for
  the NFL league.
- Playoff matchups are not scheduled yet (the last scheduled period is 13 in `ffl`).
- Fixtures: `*/mMatchup.json` (our matchups in the first, current and last periods plus one other).

### `mMatchupScore` + `mScoreboard` → `MatchupsView`

- One matchup period (`filterMatchupPeriodIds`; the web app uses `filterCurrentMatchupPeriod: true`) with live and
  projected totals (`totalPointsLive`, `totalProjectedPointsLive`, `winProbability`) and each side's lineups.
- Fixtures: `*/mMatchupScore+mScoreboard.json` (our matchup; our lineup whole, 4 opponent players).

### `kona_player_info` → `PlayersView` (free agents and waivers)

- `players[]` are pool entries: `status` `FREEAGENT`/`WAIVERS`, `waiverProcessDate`, `onTeamId: 0`, `draftAuctionValue`,
  `keeperValue`, `droppedByEliminatedTeam`, `ratings` and `player`.
- `x-fantasy-filter-player-count` gives the pool size: 889 free agents and waiver players in `ffl`, 964 in `fba`. On
  that Monday 496 of the NFL ones were on waivers (a player whose game has started stays on waivers until the next
  run); the NBA league had nobody on waivers (`{"players": []}`). `filterSlotIds` may be omitted (§1 #13).
- Fixtures: `*/kona_player_info.json` (4 players; the first keeps every stat line, the others 3),
  `*/kona_player_info_waivers.json`.

### `kona_playercard` → `PlayersView` (player detail and stat lines)

- `filterIds` + `filterStatsForTopScoringPeriodIds` (`value` > 0, plus `additionalValue` ids). Stat ids and splits:
  §1 #9–#12.
- Fixtures: `*/kona_playercard.json`, `*/kona_playercard_stat_entries.json` and `*/kona_playercard_top2.json` (one card
  with every entry; the probes behind §1 #9–#11).

### `mTransactions2` → `TransactionsView`

- `filterType` picks types. Records carry `id`, `type`, `status`, `executionType` (`EXECUTE` for user moves, `PROCESS`
  for waiver runs, `CANCEL`), `teamId`, `memberId`, `scoringPeriodId`, `bidAmount` (0 without FAAB),
  `proposedDate`/`processDate`/`expirationDate`, `relatedTransactionId`, `isPending`, `isActingAsTeamOwner`,
  `isLeagueManager`, `rating`, `teamActions` (trades: `{"<teamId>": "ACCEPTED"}`) and `items[]` with `type`,
  `playerId`, `fromTeamId`/`toTeamId` (0 = free agency) and `fromLineupSlotId`/`toLineupSlotId` (−1 = none).
- On record: `ROSTER` (lineup swaps list both sides), `FUTURE_ROSTER` (NBA moves for later days), `FREEAGENT` (`ffl`:
  ADD into the bench slot plus DROP from it), `WAIVER` `EXECUTED`/`FAILED_ROSTERLIMIT`/`FAILED_PLAYERALREADYDROPPED`
  (`executionType: PROCESS`), `TRADE_PROPOSAL` `PENDING`/`CANCELED`, `DRAFT` (130 picks in `fba`, scoring period 1).
- Fixtures: `*/mTransactions2.json` (every type, at most 3 per type/status), `*/mTransactions2_waiver_trade.json`.

### `mPendingTransactions` → `TransactionsView`

- §1 #2 and #3. The web app requests it on the team page next to `mRoster`.
- Fixtures: `*/mPendingTransactions.json`.

### `proTeamSchedules_wl` (season level) → `ProSchedule`

- `settings.proTeams[]` with `proGamesByScoringPeriod` (each game listed under both teams), byes for `ffl`. NBA games
  are keyed by day: 1,200 games over days 1–174 (games on 156 days). Each game's start falls inside its period's window
  in the web client's calendar.
- Fixtures: `*/proTeamSchedules_wl.json` (the current period only; `tests/fixtures/sports/` keeps the NFL weeks 4–5).

### `kona_game_state` (season level)

- `currentScoringPeriod.id` and `settings` (`statSettings`, `firstMatchupOverrides: {}`, `proScheduleAvailable`,
  `teamAutoPilotSettings`). The web app loads it on every page. Fixtures: `*/kona_game_state.json`.

### `mStatus`

- The `status` block alone, with `creationInfo` (the creating member's SWID, scrubbed) and `previousSeasons`.
  Fixtures: `*/mStatus.json`.

### Not a view: the season calendar

- The web client ships each game's calendar as constants: `scoringPeriods[]` (`id`, `startDate`, `endDate`,
  `preSeason`, `postSeason`) and `periodTypes[]` (0 season-long, 1 daily, 2 weekly; `periods[]` with
  `scoringPeriodStart`/`scoringPeriodEnd`). Periods run 3 a.m. to 3 a.m. ET; period 1's stored start is a preseason
  placeholder, so the client uses `endDate` minus one period (NFL weeks run Tuesday to Tuesday). Both sport plugins
  turn their periods at that hour (`fm.sports.base.PERIOD_TURN`). The "daily" type (1) is one scoring period per
  period in both games (a week each in `ffl`); the weekly type (2) groups `fba` days into weeks.
- `scripts/capture/capture.py webclient` re-extracts it each season and checks it against the pro schedule.
  Fixtures: `*/calendar.json` (the league's own period type plus the season-long one). The same bundle labels each
  game's stat splits (§1 #11) and holds the write code (section 4), both in `webclient.json`.

### What the web app itself reads

| Page | Requests |
|---|---|
| Team | `kona_game_state`; one league read with `rosterForTeamId` and `mDraftDetail`, `mLiveScoring`, `mMatchupScore`, `mPendingTransactions`, `mPositionalRatings`, `mRoster`, `mSettings`, `mTeam`, `modular`, `mNav`; `proTeamSchedules_wl`; `GET /seasons/{season}/players?view=players_wl` with `{"filterActive": {"value": true}}` |
| Scoreboard | `modular`, `mNav`, `mMatchupScore`, `mScoreboard`, `mSettings`, `mTopPerformers`, `mTeam` with `{"schedule": {"filterCurrentMatchupPeriod": {"value": true}}}` |
| Schedule | `mMatchupScore`, `mStatus`, `mSettings`, `mTeam`, `modular`, `mNav` |

## 4. Write flows

All writes are `POST https://lm-api-writes.fantasy.espn.com/apis/v3/games/{ffl|fba}/seasons/{season}/segments/0/leagues/{id}/transactions/`
with the cookies and JSON body below, the same for both games. The reference is what the web app actually sent: the
five requests the guard aborted on 2026-10-06 (`tests/fixtures/espn/real/{ffl,fba}/write_*.json`, each with method,
URL, headers and body as captured; "Captured flows" below). Behind them is ESPN's own code, saved verbatim in
`tests/fixtures/espn/real/webclient.json` (`capture.py webclient` re-extracts it from a new build): `model` holds the
item builders and the serializer `get()`, `service` the method behind each flow (`movePlayers`, `addPlayers`,
`dropPlayers`, `proposeTrade`, `acceptTrade`, `declineTrade`, `cancelTrade`, `cancelWaiverClaim`), and
`saveTransaction`, `post`, `writeHost`, `requestDefaults` and `requestConfig` the request. `typeNames` spells out the
constants the code reads (`r["N"]` is `"WAIVER"`). Each rule below is pinned by a test in
`tests/fixtures/espn/real/test_real_fixtures.py` (the "write payloads" section), and `tests/executor/test_transactions.py`
builds every captured body from `fm.browser.transactions` the way its flow would. The body comes from the serializer:

```
{isLeagueManager, teamId, type}                       always
memberId                                              the signed-in SWID
scoringPeriodId                                       the target period, else status.latestScoringPeriod
executionType                                         "EXECUTE" unless set ("CANCEL" to cancel)
items                                                 when there are any
bidAmount (+ relatedTransactionId when set)           type WAIVER only
expirationDate, comment (+ relatedTransactionId when set)   type TRADE_PROPOSAL
comment + relatedTransactionId                        TRADE_DECLINE, TRADE_VETO
relatedTransactionId                                  TRADE_ACCEPT, TRADE_UPHOLD
isActingAsTeamOwner, skipTransactionCounters          only when isLeagueManager (never for us)
```

Items: `ADD {playerId, type, toTeamId}`, `DROP {playerId, type, fromTeamId}`, `LINEUP {playerId, type,
fromLineupSlotId, toLineupSlotId}` (no team ids), `TRADE {playerId, type, fromTeamId, toTeamId}`, FAAB in trades as
`{acquisitionBudget, fromTeamId, toTeamId, type: "ACQUISITION_BUDGET_TRADE"}`. A decline may carry `{playerId}` items
without a `type` and draft-pick items. The request adds `Accept: application/json`, `X-Fantasy-Source: kona`,
`X-Fantasy-Platform: espn-fantasy-web`, `Content-Type: application/json`, a `platformVersion` query parameter and
`withCredentials: true` (cookies, no bearer token). A failure returns
`details[{type, message, metaData.teamid, resolution}]`; the client knows `TRAN_ROSTER_LIMIT_EXCEEDED_*`,
`TRAN_ROSTER_LIMIT_EXCEEDED_TRADE_RESERVED_*`, `TRAN_ROSTER_POSITION_LIMIT_EXCEEDED*` and `TRAN_ROSTER_SLOT_LIMIT_EXCEEDED*`.
Records come back with statuses such as `FAILED_LINEUPLOCK`, `FAILED_ROSTERLOCK`, `FAILED_ROSTERLIMIT`,
`FAILED_TRANSACTIONLOCKED`, `FAILED_UNDROPPABLEPLAYER`, `FAILED_MATCHUPACQUISITIONLIMIT`, `FAILED_TRADELOCK`,
`FAILED_TRADE_RESERVED`, `FAILED_NOTCLEAREDWAIVERS`, `FAILED_INVALIDPLAYERSOURCE` (42 codes in the bundle; the
calendar fixtures list them in `errorCodes`).

Two things the code alone did not show, and the captures did: **`bidAmount` is sent as `null`** on a waiver claim in a
league without FAAB (the serializer writes it unconditionally for `WAIVER`, and the UI's model holds `null`; a cancel's
model never sets it, so a cancel carries none), and **`memberId` is optional**: the player list's one-click Add sent
no `memberId` at all, while every other captured flow (both bench swaps, the claim, the roster-fix add) sent the
signed-in SWID. `fm.browser.transactions.Envelope` does both.

Two other write paths exist: `POST …/teams/{teamId}/pendingTransactions` with `[{id, subOrder}]` reorders our waiver
claims, and `POST …/teams/{teamId}/pendingTransactions/{id}` with `{bidAmount}` changes a claim's bid.

### Captured flows (2026-10-06)

Jon signed in by hand inside the guarded `capture.py writes` window (the guard aborted nothing the sign-in needed) and
drove each flow in each league until the app sent its transaction, which the guard aborted and saved. `capture.py
fixtures` scrubbed them (league ids, team ids and SWIDs replaced; player ids are ESPN's). Before the session,
`capture.py snapshot` recorded every roster for the rest of each matchup, our transaction counters and trade block,
every transaction record and every pending item; `verify` afterwards found both leagues unchanged. Every request was
`POST` to the league's `transactions/` path on the write host with `?platformVersion=<build sha>` and the four headers
of §1 #17.

| Fixture | Flow driven | Body |
|---|---|---|
| `ffl/write_ROSTER_1.json` | bench swap on the team page (FLEX ↔ Bench), week 5 | `{isLeagueManager: false, teamId, type: "ROSTER", memberId, scoringPeriodId: 5, executionType: "EXECUTE", items: [LINEUP 23→20, LINEUP 20→23]}` |
| `ffl/write_WAIVER_1.json` | waiver claim with a drop (player list → roster-fix page → Confirm) | `{…, type: "WAIVER", memberId, scoringPeriodId: 5, executionType: "EXECUTE", items: [ADD {toTeamId}, DROP {fromTeamId}], bidAmount: null}` |
| `fba/write_ROSTER_1.json` | bench swap on the team page (UTIL ↔ Bench), day 1 | `{…, type: "ROSTER", memberId, scoringPeriodId: 1, executionType: "EXECUTE", items: [LINEUP 11→12, LINEUP 12→11]}` |
| `fba/write_FREEAGENT_1.json` | one-click **Add** from the player list (roster had room) | `{…, type: "FREEAGENT", scoringPeriodId: 1, executionType: "EXECUTE", items: [ADD {toTeamId}]}`, **no `memberId`** |
| `fba/write_FREEAGENT_2.json` | add with a drop through the roster-fix page → Confirm | `{…, type: "FREEAGENT", memberId, scoringPeriodId: 1, executionType: "EXECUTE", items: [ADD, DROP]}` |

**Not captured, and why:** a trade proposal in either game (the trade builder was opened and read, but an offer is a
message to a real manager, and the capture stopped short of driving one to its request even under the guard); an NBA
waiver claim (no NBA player was on waivers: the player list's WAIVERS filter was empty, as
`fba/kona_player_info_waivers.json` already showed); and an NFL free-agent add (every NFL player was on waivers until
Wednesday, so the list offered only claims). Their payloads rest on the web client's code and the real records
(`fba/mTransactions2_waiver_trade.json` holds three real offers; both leagues hold real claims), which the captured
flows match key for key.

### What the pages look like (for `fm.browser.selectors`)

Seen in both games during the capture; accessible names, not visible text, are what the role locators match.

- **Team page** `/{football|basketball}/team?leagueId&teamId&seasonId[&scoringPeriodId]`: a `table` of `row`s. The
  slot column is a `cell` named by the slot label (`QB`, `RB`, `WR`, `TE`, `FLEX`, `D/ST`, `K`, `Bench`, `IR`; NBA
  `PG`, `SG`, `SF`, `PF`, `C`, `G`, `F`, `UTIL`, `Bench`, `IR`). An empty IR row's player column reads `Empty`. The
  move control reads `MOVE` but is a `button` named **`Select <Player> to move`**; while his move is open, the same
  row's button becomes **`Cancel Move of <Player>`**. The destination control reads `HERE` and is a `button` named
  **`Confirm move of <Player in that row> to <Slot full name>`** (`… to Bench`, `… to Tight End`, `… to Forward`); on
  an empty slot's row it was named just **`Move`**. Clicking it saves the move at once (the `ROSTER` request above).
  Without a web sign-in the page shows the heading `Log in Required` instead of the table.
- **Player list** `/{sport}/players/add?leagueId&teamId&seasonId`: a player on waivers has a `button` named
  `Claim <Name> <Position words> for <Team>`; a free agent has `Add <Name> <Position words> for <Team>`. A status
  filter `select` offers `ALL`, `AVAILABLE`, `WAIVERS`, `FREEAGENT`, `ONTEAM`. With roster room, `Add` sends the
  `FREEAGENT` request at once (the one-click capture).
- **Roster-fix page** `/{sport}/rosterfix?leagueId&seasonId&teamId&players=<playerId>&type=claim|add`, where an add
  or claim lands when the roster is full: a `button` `Drop Player <Name>` (text `DROP`) per droppable player, an
  undroppable one reading `Can't drop <Name>`, plus `Cancel` and `Continue`, which becomes `Continue to add <Name> and
  drop <Name>` once a drop is picked. `Continue` opens a "Confirm Transaction" dialog with `Confirm add <Name> and drop
  <Name>` (text `Confirm`) and `Cancel`; `Confirm` sends the `WAIVER` or `FREEAGENT` request.
- **Failed save:** the app shows `Oops! Looks like something went wrong. Please try again` (what a guarded abort
  looks like to the UI, and what a rejected write would look like to the drill).
- **Trade builder** `/{sport}/team/trade?leagueId&teamId=<them>&fromTeamId=<us>&seasonId`, reached from an opponent's
  team page link `Propose Trade`: `checkbox`es named `Trade <Player>` on both rosters and a `Continue` button, then a
  review with an expiry `select` (`1`–`7 Days`, default `2`, which matches the 48 h expiries on record), a `textarea`
  for the message and a `Send Trade Proposal` button (not clicked).

### `set_lineup`: `ROSTER` / `FUTURE_ROSTER`

- **Payload:** `{isLeagueManager: false, teamId, type: "ROSTER", memberId, scoringPeriodId: <latest>, executionType:
  "EXECUTE", items: [LINEUP…]}`. A move for a later period is `FUTURE_ROSTER` with that `scoringPeriodId`. A swap lists
  both players, the moved player first.
- **Captured: both games** (`ffl/write_ROSTER_1.json`, `fba/write_ROSTER_1.json`), each a bench swap for the current
  period, byte for byte what `lineup_envelope` builds from two `LineupMove`s. `FUTURE_ROSTER` rests on the real NBA
  records for days 2–5 (`fba/mTransactions2.json`) and on `movePlayers` in the saved code.
- **NFL:** a period is a week; locks are per game (`INDIVIDUAL_GAME`), so Thursday players lock and Sunday ones can
  still move. **NBA:** a period is a day; tomorrow's lineup is `FUTURE_ROSTER` with tomorrow's day number. Lineups lock
  per game (`INDIVIDUAL_GAME`).
- **Verify:** `mRoster` for that period (not cached; a later period echoes its id, §1 #16).
- **Mode: API.** Cookie-only auth works for this profile, the payload is fixed by ESPN's code and matched by the
  captures in both games. The UI fallback (MOVE then HERE, the names above) needs a web sign-in on the profile.

### `add_drop`: `FREEAGENT`

- **Payload:** `type: "FREEAGENT"`, items `ADD {playerId, toTeamId: ours}` plus `DROP {playerId, fromTeamId: ours}`
  when the roster is full; no `bidAmount` (the serializer sends it for `WAIVER` only). `memberId` only when the app
  went through the roster-fix page. A bare drop is a `ROSTER` transaction with a `DROP` item (saved code; not
  captured).
- **Captured: NBA only**, both ways (`fba/write_FREEAGENT_1.json` one-click from the player list, no `memberId`;
  `fba/write_FREEAGENT_2.json` add plus drop through the roster-fix page). **NFL: not captured**, because every NFL
  player was on waivers until Wednesday (the list offered only claims); the code path is the same `addPlayers` with
  `"add"` instead of `"claim"`, and the NFL league's real `FREEAGENT` records (`ffl/mTransactions2.json`) have the same
  item shape.
- **NFL:** free agents after waivers clear; a dropped player sits on waivers 24 h (`waiverHours`). **NBA:**
  `rosterLocktimeType: FIRSTGAME_SCORINGPERIOD`: adds and drops lock at the day's first tip, and the per-matchup
  limit is `rate × days in the matchup`. Which day an add lands on after the first tip (the client sends
  `latestScoringPeriod`) is unverified. The cutoff follows each league's roster lock type:
  `SportPlugin.transaction_cutoff(team, period, schedule, lock_type=settings.roster_lock_type)`.
- **Verify:** `mRoster`: roster has `add`, lacks `drop`; `transactionCounter` moved.
- **Mode: API**, as for lineups. The UI fallback is the player list's `Add <Name> …` button, then the roster-fix page's
  `Drop Player <Name>`, `Continue …` and `Confirm add … and drop …` when the roster is full (ROADMAP #27).

### `claim_waiver`: `WAIVER`

- **Payload:** `type: "WAIVER"`, items `ADD` (+ `DROP`), `bidAmount` (the FAAB bid; **`null` in a league without
  FAAB**, as captured). **Cancel:** `{type: "WAIVER", executionType: "CANCEL", relatedTransactionId: <claim>}`, no
  `bidAmount`. **Change a bid:** the `pendingTransactions/{id}` path above.
- **Captured: NFL only** (`ffl/write_WAIVER_1.json`, a claim with a drop through the roster-fix page). **NBA: not
  captured**, because no NBA player was on waivers (preseason; `fba/kona_player_info_waivers.json` is empty). The NBA
  league's real `WAIVER` records (`fba/mTransactions2.json`, `executionType: PROCESS` after the run) have the same
  items, and the code path is shared.
- **NFL:** runs on record were at 03:03 and 03:47 ET (Wednesdays) and 03:02 ET (a Monday); process days are every
  day but Tuesday in this league. **NBA:** one run on record, 03:50 ET on a Monday, although `waiverProcessDays` says
  Sunday.
- **Verify:** the claim appears in `mPendingTransactions` (our team) with its bid; after the run, `mTransactions2`
  shows `EXECUTED` or a `FAILED_*` status, `executionType: PROCESS`.
- **Mode: API.** Cancelling is documented, not captured. The UI fallback is the player list's `Claim <Name> …` button
  and the roster-fix page (ROADMAP #27).

### `propose_trade`: `TRADE_PROPOSAL`

- **Payload:** `type: "TRADE_PROPOSAL"`, items `TRADE {playerId, fromTeamId, toTeamId}` for both sides, optional
  `ACQUISITION_BUDGET_TRADE` items, `expirationDate`, `comment`, and `DROP` items when our roster must make room.
  Real offers expired 48 h after proposal, the trade builder's default expiry is `2 Days`; whether ESPN fills
  `expirationDate` when it is omitted is unknown, so the executor should send it.
- **Not captured in either game.** The trade builder was opened and read (names above) but no offer was driven to its
  request: an offer is a message to a real manager, and the capture stopped short of that. The payload rests on
  `proposeTrade` in the saved code and the three real offers in `fba/mTransactions2_waiver_trade.json`.
- **Both games:** players with `tradeLocked` cannot move; `tradeReservedEntries` hold players already in an offer;
  `settings.tradeSettings.deadlineDate` closes trading; NBA trades lock at the day's first tip like adds.
- **Verify:** a `TRADE_PROPOSAL` with `status: PENDING` and a future `expirationDate` in `mTransactions2`, with no
  CANCEL record pointing at it.
- **Mode: API, approval-only** (CLAUDE.md). Never automatic.

### `respond_trade`: `TRADE_ACCEPT` / `TRADE_DECLINE`

- **Payload:** accept: `{type: "TRADE_ACCEPT", relatedTransactionId: <offer>, teamId: ours}` plus `DROP` items when the
  roster must make room. Decline: `{type: "TRADE_DECLINE", relatedTransactionId, comment}`.
- **Status: documented, not captured** (DESIGN §6.3 plus `acceptTrade`/`declineTrade` in the saved code). Real pending
  offers were never touched and must not be: no capture drives a response to a real offer, guarded or not.
- **Verify:** the offer's record changes, and `teamActions` gains our team's answer.
- **Mode: API, approval-only.**

### `cancel`: offer or claim

- **Payload:** offer: `{type: "TRADE_PROPOSAL", executionType: "CANCEL", relatedTransactionId: <offer>}`. Claim:
  `{type: "WAIVER", executionType: "CANCEL", relatedTransactionId: <claim>}`. ESPN records an offer's expiry in exactly
  this shape (§1 #2).
- **Status: documented, not captured** (`cancelTrade`/`cancelWaiverClaim` in the saved code).
- **Verify:** a CANCEL record pointing at the offer or claim; the claim leaves `mPendingTransactions`.
- **Mode: API.**

### Decision summary

| Flow | Type | Mode | Payload source | Captured |
|---|---|---|---|---|
| `set_lineup` | `ROSTER` / `FUTURE_ROSTER` | API, UI fallback with a web login | guarded UI capture + saved client code + real records (both games) | both games (`ROSTER`); `FUTURE_ROSTER` from real NBA records |
| `add_drop` | `FREEAGENT` (+ `ROSTER` for a bare drop) | API, UI fallback | guarded UI capture (`fba`) + saved client code + real records (`ffl`) | `fba` (one-click and add+drop); `ffl` not captured (everyone was on waivers) |
| `claim_waiver` | `WAIVER` | API, UI fallback | guarded UI capture (`ffl`) + saved client code + real records (both games) | `ffl`; `fba` not captured (nobody was on waivers) |
| `propose_trade` | `TRADE_PROPOSAL` | API, approval-only | saved client code + real records (`fba`) | neither game (an offer reaches a real manager) |
| `respond_trade` | `TRADE_ACCEPT` / `TRADE_DECLINE` | API, approval-only | DESIGN §6.3 + saved client code | documented, not captured |
| `cancel` | `TRADE_PROPOSAL`/`WAIVER` + `CANCEL` | API | saved client code + real expiry records | documented, not captured |

The captures are the reference: `tests/executor/test_transactions.py` builds each one from `fm.browser.transactions`
and compares the body key for key, in order, and the request's method, path and headers (the client's
`platformVersion` aside). Where a flow is not captured, its envelope is held to the saved code and the real records.

API mode wins every flow for the same three reasons: the API needs only the long-lived cookies this profile has, while
the UI needs a OneID web session (the profile had one after Jon's sign-in in the capture window; whether it outlives a
browser restart is what `capture.py web-login` reports); ESPN's own code fixes the payloads, and the captures show both
games sending them; and API mode has no DOM to drift. The UI fallback (and the weekly drill) therefore needs a check
that the web login is alive: "Log in Required" on the team page means it is not.
