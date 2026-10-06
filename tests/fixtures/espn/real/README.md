# Real-league ESPN fixtures

Scrubbed captures of the two configured leagues from 2026-10-05 (reads) and 2026-10-06 (writes) (ROADMAP #14): `ffl/`
is the NFL league (season 2026, week 4; the writes on the Tuesday of week 5) and `fba/` the NBA league (season 2027,
preseason day 1). Written by
`scripts/capture/capture.py fixtures`; `docs/espn-api.md` explains every view and the facts the tests pin down.

- `index.json` lists every file with its game, ESPN view, request (views, scoring period, `X-Fantasy-Filter`), capture
  time, the model kind that parses it, and how it was trimmed. `test_real_fixtures.py` loads every file listed there and
  fails if a JSON file is missing from the index.
- **Scrubbed:** league ids are `1010101` (`ffl`) and `2020202` (`fba`); team ids are permuted within each league's own
  id set, so our team is the smallest id (1 in both); members are `Manager N` with SWIDs
  `{00000000-0000-0000-0000-00000000000N}` (ours is 1); teams are `Team N` / `TN` with a placeholder logo; league names
  are `Fixture <game> League`. Player ids, names and stats are real public data. No cookies, no `espn_s2`.
- **Trimmed, not edited:** fewer teams, players, matchups and stat entries, and no player rankings, outlooks or
  notification settings; ESPN's structure and field names are unchanged. Files are compact JSON, as ESPN sends them.
- `calendar.json` is not a view: it is the season calendar ESPN's web client ships as constants (scoring-period dates
  and the period types that map matchups to scoring periods), plus the transaction error codes the client knows.
- `webclient.json` (shared by both games) is not a view either: verbatim excerpts of the web client bundle with the code
  that builds and sends every write (type constants, item builders and serializer, the service method per flow, the
  request to the write host), plus each game's stat source and split labels. The write-payload tests in
  `test_real_fixtures.py` read the payload rules of `docs/espn-api.md` section 4 out of it.
- `probes.json` holds the outcomes of the read-pass experiments (`filterSlotIds`, `filterStatsForTopScoringPeriodIds`,
  stat-entry ids and splits); `kona_playercard_stat_entries.json` and `kona_playercard_top2.json` are the cards behind
  them, with every stat entry kept, and the first player of `kona_player_info.json` keeps every stat line too.
- `write_*.json` files hold the write requests the guard aborted during `capture.py writes` on 2026-10-06 (method,
  URL with the web client's `platformVersion`, headers and body, as the web app sent them; none reached ESPN):
  `ffl/write_ROSTER_1.json` and `fba/write_ROSTER_1.json` (bench swaps), `ffl/write_WAIVER_1.json` (a claim with a drop,
  `bidAmount: null`), `fba/write_FREEAGENT_1.json` (one-click Add, no `memberId`) and `fba/write_FREEAGENT_2.json` (add
  plus drop through the roster-fix page). `tests/executor/test_transactions.py` rebuilds each from
  `fm.browser.transactions`; docs/espn-api.md section 4 says what was not captured and why.
- `test_capture_tools.py` covers the capture scripts themselves on synthetic inputs: the write guard, the snapshot
  comparison and the web-client extractors.
