"""Builds the consistent-world NFL backtest fixture (ROADMAP #42); the pre-registered protocol is in ``README.md``.

``uv run python tests/fixtures/backtest/nfl_world/build.py`` rewrites every file beside this one; the output is
deterministic (two seeds, below). One world produces three things: the true weekly stat lines (the backtest's actuals),
the nflverse-shaped history the NFL opportunity baseline learns from, and the ESPN- and Sleeper-like projections, which
are noisy, slightly biased observations of each week's true expectation. The noise constants below were fixed, and
written into the README, before any baseline score was computed; they are not to be changed after seeing a result.
"""

from __future__ import annotations

import importlib.util
import json
import random
import shutil
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent.parent / "sources" / "baseline_nfl" / "build.py"

WORLD_SEED = 20261101
"""Seeds the schedule, team volumes, shares and every realised stat line."""
SOURCE_SEED = 20261102
"""Seeds the projection sources' noise, a stream apart from the world's."""

SEASON = 2026
WEEKS = {2025: 17, 2026: 10}
"""Regular-season weeks generated per season. 2026 is the replayed season; 2025 is history only."""

# Pre-registered projection-source noise: per stat, a lognormal factor with this sigma and mean 1 + bias.
# ESPN is the tighter source, Sleeper (RotoWire-based) noisier and higher, as in the first fixture, where the sources'
# MAE was about 3.9 and 4.2. The first fixture's game noise is lower than a real NFL week's, so the sigmas were set
# from published weekly-projection accuracy (per-stat relative error of roughly 20-30% for expert projections), not
# fitted to those MAEs.
SOURCE_NOISE: dict[str, tuple[float, float]] = {
    "espn": (0.22, 0.03),
    "sleeper": (0.30, 0.06),
}
SCORED_STATS = ("PY", "PTD", "INTT", "RY", "RTD", "REC", "REY", "RETD", "FUML")
MIN_EXPECTED = 0.005
FILLER_ESPN_BASE = 9_000_000
POSITIONS = ("QB", "RB", "WR", "TE")
ELIGIBLE_SLOTS: dict[str, list[int]] = {
    "QB": [0, 7, 20, 21],
    "RB": [2, 3, 23, 7, 20, 21],
    "WR": [3, 4, 5, 23, 7, 20, 21],
    "TE": [5, 6, 23, 7, 20, 21],
}
STARTERS = {"Josh Allen": 0, "Jahmyr Gibbs": 2, "Bijan Robinson": 2, "Ja'Marr Chase": 4, "Rashid Shaheed": 4}
STARTERS |= {"Trey McBride": 6, "Chuba Hubbard": 23}
BENCH_SLOT, IR_SLOT = 20, 21


def legacy() -> ModuleType:
    """The first fixture's build script: its role profiles, team contexts and noise helpers are this world's too."""
    spec = importlib.util.spec_from_file_location("baseline_nfl_legacy_build", LEGACY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = legacy()
    rng = random.Random(WORLD_SEED)
    teams: dict[str, tuple[float, float, float]] = base.TEAMS
    schedule: list[dict[str, object]] = []
    for season, weeks in WEEKS.items():
        for week in range(1, weeks + 1):
            playing = [t for t in teams if base.BYES_2026.get(t) != week or season != SEASON]
            if season == 2025:
                playing = [t for t in playing if (list(teams).index(t) + 6) % 11 != week % 11]
            outside = list(base.OUTSIDE)
            rng.shuffle(outside)
            for team in playing:
                opponent = outside.pop()
                home_game = rng.random() < 0.5
                total = round(max(36.0, min(56.0, base.noisy(rng, 2 * teams[team][2], 0.06))) * 2) / 2
                margin = round(rng.gauss(0, 4.5) * 2) / 2  # expected home margin, positive when home is favored
                home, away = (team, opponent) if home_game else (opponent, team)
                schedule.append(
                    {
                        "season": season,
                        "week": week,
                        "game_type": "REG",
                        "home_team": home,
                        "away_team": away,
                        "spread_line": margin,
                        "total_line": total,
                    }
                )
    implied: dict[tuple[int, int, str], float] = {}
    for game in schedule:
        spread, total = float(game["spread_line"]), float(game["total_line"])  # type: ignore[arg-type]
        implied[(game["season"], game["week"], game["home_team"])] = (total + spread) / 2  # type: ignore[index]
        implied[(game["season"], game["week"], game["away_team"])] = (total - spread) / 2  # type: ignore[index]
    average = sum(implied.values()) / len(implied)

    stats: list[dict[str, object]] = []
    snaps: list[dict[str, object]] = []
    expected: dict[tuple[int, str], dict[int, dict[str, float]]] = {}  # (week, gsis) -> 2026 true expectation
    players: dict[str, tuple[int, str, str]] = {}  # gsis -> (espn id, name, position)
    team_of: dict[str, str] = {}
    named = {team: [p for p in base.PLAYERS if p[4] == team] for team in teams}
    for (season, week, team), total in sorted(implied.items()):
        if team not in teams:
            continue
        env = max(0.6, min(1.6, total / average))
        pass_mean = teams[team][0] * env**0.4
        carry_mean = teams[team][1] * env**0.2
        pass_att = max(15.0, base.noisy(rng, pass_mean, 0.12))
        carries = max(12.0, base.noisy(rng, carry_mean, 0.14))
        targets = pass_att * 0.99
        air = targets * base.noisy(rng, base.ADOT, 0.08)
        target_mean = pass_mean * 0.99
        air_mean = target_mean * base.ADOT
        rows: list[tuple[str, str, str, str | None, tuple[float, ...], tuple[float, ...], float, int]] = []
        taken = [0.0, 0.0, 0.0, 0.0]
        for name, gsis, pfr, espn_id, _team, position, shares, rates, snap in named[team]:
            taken = [a + b for a, b in zip(taken, shares, strict=True)]
            on_ir = name == "Chris Olave" and season == SEASON and week in base.IR_OLAVE
            players[gsis] = (espn_id, name, position)
            team_of[gsis] = team
            rows.append((name, gsis, position, None if on_ir else pfr, shares, rates, snap, 1 if on_ir else 0))
        if not any(p[5] == "QB" for p in named[team]):
            rows.append(
                (
                    f"{team} QB",
                    base.filler_id(team, "QB"),
                    "QB",
                    None,
                    (0.99, 0.05, 0, 0),
                    base.FILLER_RATES["QB"],
                    1.0,
                    0,
                )
            )
            taken[1] += 0.05
        left = [max(0.0, 1 - share) for share in taken]
        for position, targets_part, air_part, carries_part in (
            ("RB", 0.15, 0.05, 1.0),
            ("WR", 0.60, 0.80, 0.0),
            ("TE", 0.25, 0.15, 0.0),
        ):
            shares = (0.0, left[1] * carries_part, left[2] * targets_part, left[3] * air_part)
            rows.append(
                (
                    f"{team} {position}",
                    base.filler_id(team, position),
                    position,
                    None,
                    shares,
                    base.FILLER_RATES[position],
                    0.5,
                    0,
                )
            )
        game_id = f"{season}_{week:02d}_{team}"
        for name, gsis, position, snap_id, shares, rates, snap, on_ir in rows:
            if gsis not in players:
                gsis_index = int(gsis[-5:]) - 90000
                players[gsis] = (FILLER_ESPN_BASE + gsis_index, name, position)
                team_of[gsis] = team
            share_pass, share_rush, share_target, share_air = shares
            cmp_rate, ypa, ptd, intr, ypc, rtd, catch, ypt, retd, ypay = rates
            if season == SEASON:
                attempts_e = pass_mean * share_pass
                carries_e = carry_mean * share_rush
                targets_e = target_mean * share_target
                air_e = air_mean * share_air
                line = {
                    "PY": attempts_e * ypa,
                    "PTD": attempts_e * ptd * env**0.6,
                    "INTT": attempts_e * intr,
                    "RY": carries_e * ypc,
                    "RTD": carries_e * rtd * env**0.8,
                    "REC": targets_e * catch,
                    "REY": 0.5 * air_e * ypay + 0.5 * targets_e * ypt,
                    "RETD": targets_e * retd * env**0.6,
                    "FUML": attempts_e * 0.004,
                }
                expected.setdefault((week, gsis), {})[0] = {k: v for k, v in line.items() if v >= MIN_EXPECTED}
            if on_ir:
                continue
            attempts = round(base.noisy(rng, pass_att * share_pass, 0.07)) if share_pass else 0
            completions = round(attempts * max(0.0, min(1.0, base.noisy(rng, cmp_rate, 0.08)))) if attempts else 0
            carried = round(base.noisy(rng, carries * share_rush, 0.25)) if share_rush else 0
            tgt = round(base.noisy(rng, targets * share_target, 0.3)) if share_target else 0
            rec = min(tgt, round(tgt * max(0.0, min(1.0, base.noisy(rng, catch, 0.15))))) if tgt else 0
            air_yards = round(base.noisy(rng, air * share_air, 0.35)) if share_air else 0
            if not (attempts or carried or tgt):
                continue
            stats.append(
                {
                    "player_id": gsis,
                    "player_name": name,
                    "position": position,
                    "season": season,
                    "week": week,
                    "season_type": "REG",
                    "team": team,
                    "completions": completions,
                    "attempts": attempts,
                    "passing_yards": round(attempts * base.noisy(rng, ypa, 0.2)) if attempts else 0,
                    "passing_tds": base.poisson(rng, attempts * ptd * env**0.6),
                    "passing_interceptions": base.poisson(rng, attempts * intr),
                    "carries": carried,
                    "rushing_yards": round(carried * base.noisy(rng, ypc, 0.3)) if carried else 0,
                    "rushing_tds": base.poisson(rng, carried * rtd * env**0.8),
                    "targets": tgt,
                    "receptions": rec,
                    "receiving_yards": round(
                        max(0.0, air_yards * ypay * base.noisy(rng, 1.0, 0.3)) * 0.5
                        + 0.5 * tgt * ypt * base.noisy(rng, 1.0, 0.3)
                    )
                    if tgt
                    else 0,
                    "receiving_tds": base.poisson(rng, tgt * retd * env**0.6),
                    "receiving_air_yards": air_yards,
                    "rushing_fumbles_lost": 0,
                    "receiving_fumbles_lost": 0,
                    "sack_fumbles_lost": base.poisson(rng, attempts * 0.004),
                }
            )
            if snap_id:
                snaps.append(
                    {
                        "season": season,
                        "week": week,
                        "pfr_player_id": snap_id,
                        "team": team,
                        "position": position,
                        "offense_pct": round(max(0.2, min(1.0, rng.gauss(snap, 0.06))), 2),
                        "game_id": game_id,
                    }
                )

    espn_of = {gsis: values[0] for gsis, values in players.items()}
    actual_lines: dict[tuple[int, str], dict[str, float]] = {}
    for row in stats:
        if row["season"] != SEASON:
            continue
        line = {
            "PY": row["passing_yards"],
            "PTD": row["passing_tds"],
            "INTT": row["passing_interceptions"],
            "RY": row["rushing_yards"],
            "RTD": row["rushing_tds"],
            "REC": row["receptions"],
            "REY": row["receiving_yards"],
            "RETD": row["receiving_tds"],
            "FUML": float(row["sack_fumbles_lost"]),  # type: ignore[arg-type]
        }
        actual_lines[(int(row["week"]), str(row["player_id"]))] = {k: float(v) for k, v in line.items() if v}  # type: ignore[arg-type]

    source_rng = random.Random(SOURCE_SEED)
    on_bye = {(week, team) for team, week in base.BYES_2026.items()}
    weeks_out: list[dict[str, object]] = []
    roster = [p for p in base.PLAYERS]
    for week in range(1, WEEKS[SEASON] + 1):
        lineup: dict[str, int] = {}
        for name, _gsis, _pfr, espn_id, _team, _position, _shares, _rates, _snap in roster:
            slot = STARTERS.get(name, IR_SLOT if name == "Chris Olave" and week in base.IR_OLAVE else BENCH_SLOT)
            lineup[str(espn_id)] = slot
        projections: dict[str, dict[str, dict[str, float]]] = {"espn": {}, "sleeper": {}}
        for gsis in sorted(players, key=lambda g: players[g][0]):
            true_line = expected.get((week, gsis), {}).get(0)
            if true_line is None:
                continue
            for source, (sigma, bias) in SOURCE_NOISE.items():
                noisy_line = {
                    stat: round(value * (1 + bias) * base.noisy(source_rng, 1.0, sigma), 2)
                    for stat, value in true_line.items()
                    if stat in SCORED_STATS
                }
                projections[source][str(espn_of[gsis])] = {k: v for k, v in noisy_line.items() if v > 0}
        for _name, _gsis, _pfr, espn_id, team, *_rest in roster:
            if (week, team) in on_bye:
                projections["espn"][str(espn_id)] = {}  # ESPN projects a player on bye as an empty line
        actuals = {str(espn_of[gsis]): line for (w, gsis), line in sorted(actual_lines.items()) if w == week and line}
        weeks_out.append({"period": week, "lineup": lineup, "projections": projections, "actuals": actuals})

    fixture = {
        "format": 1,
        "sport": "nfl",
        "season": SEASON,
        "settings": "settings.json",
        "as_of": "2026-09-01T12:00:00Z",
        "players": [
            {
                "espn_id": espn_id,
                "name": name,
                "position": position,
                "eligible_slots": ELIGIBLE_SLOTS[position],
            }
            for espn_id, name, position in sorted(players.values())
        ],
        "weeks": weeks_out,
    }
    ids = [
        {
            "espn_id": str(espn_id),
            "gsis_id": gsis,
            "pfr_id": next((p[2] for p in base.PLAYERS if p[1] == gsis), ""),
            "name": name,
            "position": position,
        }
        for gsis, (espn_id, name, position) in sorted(players.items(), key=lambda item: item[1][0])
    ]
    scoreboards: dict[str, list[dict[str, object]]] = {}
    for game in schedule:
        if game["season"] == SEASON:
            scoreboards.setdefault(str(game["week"]), []).append(
                {
                    "home": game["home_team"],
                    "away": game["away_team"],
                    "spread": -float(game["spread_line"]),  # type: ignore[arg-type]
                    "over_under": game["total_line"],
                }
            )
    shutil.copyfile(HERE.parent / "nfl" / "settings.json", HERE / "settings.json")
    compact = {"separators": (",", ":")}
    (HERE / "backtest.json").write_text(json.dumps(fixture, **compact) + "\n", encoding="utf-8")  # type: ignore[arg-type]
    for name, payload in (
        ("player_stats.json", stats),
        ("snap_counts.json", snaps),
        ("schedules.json", schedule),
        ("player_ids.json", ids),
        ("scoreboards.json", scoreboards),
    ):
        (HERE / name).write_text(json.dumps(payload, **compact) + "\n", encoding="utf-8")  # type: ignore[arg-type]
    print(f"{len(stats)} stat rows, {len(players)} players, {len(weeks_out)} replayed weeks, league avg {average:.2f}")


if __name__ == "__main__":
    main()
