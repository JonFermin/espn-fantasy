# ESPN season calendars

One JSON file per game and season, `<game>_<season>.json` (`fba_2027.json`, `ffl_2026.json`), read through
`fm.paths.data_file` by `fm.espn.calendar` (`load_calendar`, `find_calendar`).

ESPN's `scheduleSettings.matchupPeriods` lists schedule-period ids of the type `scheduleSettings.periodTypeId` names.
The real NBA league's type is weekly (2), so matchup 1 is `[1]` and only this calendar says it is days 1-6 (Tue Oct 20 -
Sun Oct 25, 2026); matchups 2-17 are Monday-Sunday weeks, matchup 18 is the 14 days around the All-Star break (days
119-132) and matchups 19-21 (the playoffs) are days 133-153. No ESPN read view carries this table: the fantasy web app
bundles it as a constant.

A file holds `game`, `season`, `webClientBuild`, `scoringPeriods` (`{id, startDate, endDate, preSeason, postSeason}`,
epoch milliseconds) and `periodTypes` (`{id, daily, weekly, seasonLong, periods: [{id, scoringPeriodStart,
scoringPeriodEnd}]}`): `capture.py webclient`'s `calendar_<game>.json` without its `errorCodes`.

## Refreshing a season

`capture.py webclient` extracts the calendar from the web client bundle and matches it against the league's pro schedule
(it is out of the engine's scope; run it as in `scripts/capture/README.md`). It writes `calendar_<game>.json` into the
capture's raw directory. Turn that into the data file with the extractor, which validates it:

```python
import json
from fm.espn.calendar import calendar_file, extract_calendar

capture = json.load(open("calendar_fba.json", encoding="utf-8"))
data = extract_calendar(capture)
with calendar_file(capture["game"], capture["season"]).open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(data, handle, separators=(",", ":"), sort_keys=True)
    handle.write("\n")
```

A season with no file has no calendar: `fm.espn.calendar.find_calendar` answers `None` and the weekly transaction cap
counts the trailing seven days; `fm.decide.lineup_daily` and `fm.decide.streaming` refuse to plan a matchup they cannot
place.
