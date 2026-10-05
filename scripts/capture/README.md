# Real-league capture (ROADMAP #14)

Tools that read the configured ESPN leagues, capture what the web app sends for each write flow without letting it
reach ESPN, prove the leagues did not change, and turn the captures into the scrubbed fixtures under
`tests/fixtures/espn/real/`. Findings live in `docs/espn-api.md`.

## Safety protocol

- **The write guard comes first** (`guard.py`). Every browser session installs one context-wide route before any ESPN
  page opens. It aborts every request to `lm-api-writes.fantasy.espn.com`, every non-GET request to `espn.com` or a
  subdomain, and every URL containing `transactions`, and records each one (method, URL, JSON body, resource type).
  The route fails closed, WebSockets to ESPN hosts are closed, and GETs to ESPN APIs are paced at least 1 s apart.
  A response to anything it should have aborted is a leak, and the run stops.
- **The guard is proven** on `about:blank` and again on every ESPN page the tools load before driving it. The probes
  are a POST to the write host for league 0, which does not exist, with an empty body and no credentials; a POST to
  another ESPN host; and a GET whose URL contains `transactions`. Each would be harmless even if it got through.
- **Reads** go through `fm.espn.client.EspnClient` with request starts at least 1 s apart, across leagues too.
- **Before and after** (`reads.py`): `snapshot` records, for each league, every team's roster in every remaining
  scoring period of the current matchup (an NBA lineup move can target a later day), our team's raw
  `transactionCounter` and trade block, every `mTransactions2` record a write could create (lineup moves included) and
  every pending item. `verify`, and `writes` when it ends, re-read the same periods and fail on anything touching our
  team that is new, gone or changed: a roster slot, the counter, a transaction record or a pending item. That catches
  an answered or cancelled offer, a cancelled claim and a future-day lineup move, not just new moves. Other teams'
  changes are printed as notes.
- **Scrubbing** (`scrub.py`): one stable mapping per league for the league id, team ids (permuted within the league's id
  set, ours first), member SWIDs, manager names, team names, abbreviations and logos, and any other braced GUID. Then a
  leak check: `fixtures` refuses to write a file that still contains a real value. Cookies and `espn_s2` are never
  recorded.
- Never clear cookies, never touch a real pending trade offer (not even to rehearse a decline under the guard), and
  never write anything under `~/.config/espn-fantasy`. Only a person signs in, by hand. Raw captures (real names and
  ids) stay in `--raw`, which defaults to the system temp dir and may not be inside the repo.

## Commands

Run from the repo root with `uv run python scripts/capture/capture.py [--raw DIR] [--league KEY] <command>`:

| Command | What it does |
|---|---|
| `reads` | Every read view of DESIGN §6.1 plus `kona_game_state` and `mStatus`, and the probes behind docs/espn-api.md §1 (`filterSlotIds`, `filterStatsForTopScoringPeriodIds` 0 and 2, stat-entry ids and splits, a played period's ids), into `reads.json` and `cache/` |
| `webclient` (alias `calendar`) | One guarded page load. From the bundle it fetches anyway, it saves each league's season calendar, matched to the pro schedule from `reads` (`calendar_<game>.json`), and `webclient.json`: the transaction code verbatim and each game's stat split labels |
| `snapshot` | The league state above, into `snapshot_before.json`. It needs a calendar for the matchup's periods: `calendar_<game>.json`, else the committed fixture |
| `explore URL [STEPS]` | One guarded page visit: screenshot, aria snapshot, the guard's report and the app's own API reads. URLs may use `{league}`, `{team}`, `{season}`, `{sport}` |
| `writes` | Assisted capture in a headed, guarded window: a person drives each flow, and the guard aborts and saves every transaction request into `writes/`. If the app shows "Log in Required", it waits (`--login-minutes`) while the person signs in in that window, printing every request the guard aborts meanwhile. When the window closes, the sign-in fails or the guard fails, it verifies every league given (all by default) |
| `web-login` | Fallback when the guard blocks the sign-in inside `writes`: a headed window that stays open until closed, with the guard in sign-in mode (only non-GET requests to ESPN hosts outside `fantasy.espn.com` go through; the write host and every league API stay blocked). Afterwards it reports whether the web session survived a browser restart |
| `verify` | Re-reads the leagues and compares them with the snapshot |
| `fixtures [--dest]` | Scrubs and trims the raw captures into `tests/fixtures/espn/real/` with `index.json` |

Typical order: `reads`, `webclient`, `snapshot`, `writes` (a person signs in and drives the flows), `fixtures`, then
`uv run pytest -q tests/fixtures/espn/real`. `verify` repeats the check `writes` ends with.

## Status (2026-10-05)

Reads, probes, calendars, the web-client code, fixtures and the verification are done; the last `snapshot`/`verify`
pair found both leagues unchanged. The UI write captures are not done. The browser profile holds valid API cookies
(`espn_s2`/`SWID`) but no OneID web session, so every ESPN page shows "Log in Required" and the UI cannot reach a
transaction. `fm login` cannot create that session: it clears ESPN's cookies first and closes its window by itself as
soon as the API cookies land. To finish:

1. `capture.py snapshot`
2. `capture.py writes`. Sign in inside its window when "Log in Required" shows. Once the team page shows the team,
   drive a bench swap, a free-agent add/drop, a waiver claim and a trade proposal in each league, then close the
   window. If the sign-in fails because the guard aborted a request it needed (the command lists them), run
   `capture.py web-login`, sign in, close its window, and rerun `writes` if it reports that the session survived.
3. `capture.py fixtures`, then `uv run pytest -q tests/fixtures/espn/real`.

## Modules

- `guard.py`: the write guard (normal and sign-in mode), its proof and its records.
- `reads.py`: read-pass captures, probes, league snapshots and their comparison, and the paced, logging HTTP client.
- `webclient.py`: calendars, stat split labels, error codes and the transaction code from the web client bundle.
- `scrub.py`: stable placeholders and the leak check.
- `fixture_writer.py`: per-view trims, scrubbing and `index.json`.
- `capture.py`: the command line.

The offline tests for these modules are `tests/fixtures/espn/real/test_capture_tools.py` (synthetic inputs) and
`tests/fixtures/espn/real/test_real_fixtures.py` (the captured fixtures and the scrubber).
