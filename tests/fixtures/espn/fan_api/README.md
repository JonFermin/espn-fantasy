# Fan API fixtures (synthetic)

Hand-written stand-ins for `GET https://fan.api.espn.com/apis/v2/fans/{SWID}`, the fan profile that
`fm.espn.discover.discover_leagues` reads to list an account's leagues. **None of these is a capture.** A real fan
profile holds personal data (the account's SWID, every favourite team, every league and team name), so one must never
be committed; these carry placeholder names only (`League A`, `Fixture Team 7`) and the all-zero SWID the scrubbed
`real/` captures use. The shape follows the ESPN web app's fantasy navigation (`preferences[].metaData.entry` with
`groups[].groupId`, `entryId`, `seasonId`, `abbrev`, `gameId`, `entryURL`) but is **not verified against a live
account**, which is why the parser also has a deep-walk fallback; `url_only.json` exercises it.

| File | Shape | What the tests expect (`now` = 2026-10-12, so NFL 2026 and NBA 2027 are current) |
|---|---|---|
| `structured.json` | `preferences[].metaData.entry` | NFL `1010101` team 7 "Fixture Team 7" in "League A" and NBA `2020202` team 3 "Fixture Team Three" (location + nickname, no `entryMetadata`) in "League B"; a 2025 NFL league `3030303` ("League C") only with `all_seasons`; a hockey entry (`FHL`, `gameId` 4), a favourite-team preference and a sparse duplicate of league A are dropped or merged |
| `url_only.json` | no entries; fantasy URLs scattered under `fan.links[]` and `modules[].cards[]` | the same two current leagues, found by the deep walk: team 7 from the team URL merged with the league URL, NBA team 3 from a URL inside card text (no `seasonId`, so the current season); no names; league `3030303` (2025) only with `all_seasons`; the hockey URL is dropped |
| `other_sport.json` | entries for hockey (`gameId` 4) and baseball (`gameId` 2) plus a favourite-team preference | no leagues (an empty list, not an error) |

A malformed body (HTML, a JSON list, `null`) is inlined in `tests/espn/test_discover.py` and raises `DiscoveryError`.
