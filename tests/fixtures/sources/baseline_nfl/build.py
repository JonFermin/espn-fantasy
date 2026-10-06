"""Rebuilds the hand-built nflverse and odds inputs of the NFL opportunity baseline tests (ROADMAP #42).

``uv run python tests/fixtures/sources/baseline_nfl/build.py`` rewrites the JSON files beside it; the output is
deterministic (seeded). Nothing here reads the backtest fixture's actual or projected lines: each player gets a role
profile written from public knowledge of the 2025 season (shares of his team's volume and per-opportunity efficiency),
and every game draws around it with noise, so the history the baseline learns from is independent of the lines the
backtest replays.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261004

# team: (pass attempts, carries, typical implied total) per game
TEAMS: dict[str, tuple[float, float, float]] = {
    "BUF": (29, 28, 26.0),
    "BAL": (27, 31, 26.0),
    "DET": (31, 28, 27.0),
    "ATL": (31, 27, 22.0),
    "CAR": (31, 26, 21.0),
    "PHI": (30, 30, 24.0),
    "CIN": (37, 21, 23.0),
    "NO": (34, 24, 19.5),
    "ARI": (35, 25, 22.0),
    "KC": (36, 24, 25.0),
}
OUTSIDE = ["MIA", "NYJ", "NE", "PIT", "CLE", "HOU", "IND", "JAX", "TEN", "DEN", "LV", "LAC"]

# name, gsis, pfr, espn, team, position, then shares of team (pass_att, carries, targets, air yards),
# efficiency (completion, yards per attempt, pass TD rate, INT rate, yards per carry, rush TD rate, catch rate,
# yards per target, receiving TD rate, yards per air yard), offensive snap share
type Profile = tuple[str, str, str, int, str, str, tuple[float, float, float, float], tuple[float, ...], float]
type Row = tuple[str, str, str, str | None, tuple[float, float, float, float], tuple[float, ...], float]
PLAYERS: list[Profile] = [
    (
        "Josh Allen",
        "00-0034857",
        "AlleJo02",
        3918298,
        "BUF",
        "QB",
        (0.99, 0.19, 0, 0),
        (0.64, 7.4, 0.060, 0.017, 5.0, 0.10, 0, 0, 0, 1),
        1.0,
    ),
    (
        "Lamar Jackson",
        "00-0034796",
        "JackLa00",
        3916387,
        "BAL",
        "QB",
        (0.99, 0.21, 0, 0),
        (0.66, 8.2, 0.068, 0.014, 6.0, 0.055, 0, 0, 0, 1),
        1.0,
    ),
    (
        "Jahmyr Gibbs",
        "00-0039139",
        "GibbJa01",
        4429795,
        "DET",
        "RB",
        (0, 0.50, 0.14, 0.03),
        (0, 0, 0, 0, 4.9, 0.055, 0.80, 7.6, 0.04, 2.0),
        0.62,
    ),
    (
        "Bijan Robinson",
        "00-0039163",
        "RobiBi00",
        4430807,
        "ATL",
        "RB",
        (0, 0.58, 0.17, 0.03),
        (0, 0, 0, 0, 4.8, 0.035, 0.79, 6.8, 0.03, 2.0),
        0.70,
    ),
    (
        "Tyler Allgeier",
        "00-0037259",
        "AllgTy00",
        4373626,
        "ATL",
        "RB",
        (0, 0.22, 0.04, 0.01),
        (0, 0, 0, 0, 4.2, 0.030, 0.75, 5.8, 0.02, 2.0),
        0.30,
    ),
    (
        "Chuba Hubbard",
        "00-0036885",
        "HubbCh00",
        4241416,
        "CAR",
        "RB",
        (0, 0.58, 0.09, 0.02),
        (0, 0, 0, 0, 4.4, 0.045, 0.76, 5.9, 0.03, 2.0),
        0.62,
    ),
    (
        "Saquon Barkley",
        "00-0034844",
        "BarkSa00",
        3929630,
        "PHI",
        "RB",
        (0, 0.60, 0.10, 0.02),
        (0, 0, 0, 0, 4.9, 0.040, 0.78, 6.5, 0.03, 2.0),
        0.66,
    ),
    (
        "Ja'Marr Chase",
        "00-0036900",
        "ChasJa00",
        4362628,
        "CIN",
        "WR",
        (0, 0.0, 0.28, 0.36),
        (0, 0, 0, 0, 6.0, 0.03, 0.70, 9.8, 0.080, 1.05),
        0.93,
    ),
    (
        "Rashid Shaheed",
        "00-0036196",
        "ShahRa00",
        4249836,
        "NO",
        "WR",
        (0, 0.01, 0.15, 0.24),
        (0, 0, 0, 0, 6.0, 0.03, 0.60, 9.6, 0.050, 1.05),
        0.78,
    ),
    (
        "Chris Olave",
        "00-0036961",
        "OlavCh00",
        4361370,
        "NO",
        "WR",
        (0, 0.0, 0.22, 0.28),
        (0, 0, 0, 0, 6.0, 0.03, 0.64, 8.7, 0.045, 1.05),
        0.88,
    ),
    (
        "Trey McBride",
        "00-0037744",
        "McBrTr00",
        4361307,
        "ARI",
        "TE",
        (0, 0.0, 0.26, 0.17),
        (0, 0, 0, 0, 3.0, 0.03, 0.74, 7.4, 0.045, 1.2),
        0.86,
    ),
    (
        "Travis Kelce",
        "00-0030506",
        "KelcTr00",
        15847,
        "KC",
        "TE",
        (0, 0.0, 0.19, 0.16),
        (0, 0, 0, 0, 3.0, 0.03, 0.72, 7.0, 0.060, 1.2),
        0.80,
    ),
]
# Filler players complete each team's volume: a quarterback for every team without one above, and the WR, TE and RB
# rows that hold what the named players do not. Their efficiency is the position's typical rate.
FILLER_RATES = {
    "QB": (0.645, 6.9, 0.045, 0.024, 5.2, 0.035, 0, 0, 0, 1),
    "RB": (0, 0, 0, 0, 4.3, 0.032, 0.78, 6.2, 0.03, 2.0),
    "WR": (0, 0, 0, 0, 6.5, 0.03, 0.65, 8.4, 0.055, 1.05),
    "TE": (0, 0, 0, 0, 3.0, 0.03, 0.70, 7.4, 0.06, 1.2),
}
ADOT = 8.0  # team air yards per target
IR_OLAVE = {1, 2}  # Chris Olave is on IR in 2026 weeks 1-2 (the backtest fixture's story): no rows
BYES_2026 = {"BUF": 2, "CAR": 3}
WEEKS_2026 = 4
SEASONS = {2025: 17, 2026: WEEKS_2026}


def filler_id(team: str, position: str) -> str:
    """A stable stand-in GSIS id per team and position for the players that complete a team's volume."""
    return f"00-00{90000 + 10 * list(TEAMS).index(team) + ['QB', 'RB', 'WR', 'TE'].index(position)}"


def poisson(rng: random.Random, mean: float) -> int:
    limit, k, p = math.exp(-mean), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def noisy(rng: random.Random, value: float, sigma: float) -> float:
    return value * math.exp(rng.gauss(0.0, sigma) - sigma * sigma / 2)


def main() -> None:
    rng = random.Random(SEED)
    schedule: list[dict[str, object]] = []
    for season, weeks in SEASONS.items():
        for week in range(1, weeks + 1):
            teams = [t for t in TEAMS if BYES_2026.get(t) != week or season != 2026]
            if season == 2025:
                teams = [t for t in teams if (list(TEAMS).index(t) + 6) % 11 != week % 11]
            paired: set[str] = set()
            if (season, week) == (2026, 4):
                forced = [("NO", "ATL", -1.5, 47.5), ("CAR", "DET", 7.0, 46.5)]
            else:
                forced = []
            for home, away, spread, total in forced:
                paired.update((home, away))
                schedule.append(
                    {
                        "season": season,
                        "week": week,
                        "game_type": "REG",
                        "home_team": home,
                        "away_team": away,
                        "spread_line": -spread,
                        "total_line": total,
                    }
                )
            outside = list(OUTSIDE)
            rng.shuffle(outside)
            for team in teams:
                if team in paired:
                    continue
                paired.add(team)
                opponent = outside.pop()
                home_game = rng.random() < 0.5
                mine = TEAMS[team][2]
                total = round(max(36.0, min(56.0, noisy(rng, 2 * mine, 0.06))) * 2) / 2
                margin = round(rng.gauss(0, 4.5) * 2) / 2  # expected home margin
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
    totals: dict[tuple[int, int, str], float] = {}
    for game in schedule:
        spread, total = float(game["spread_line"]), float(game["total_line"])  # type: ignore[arg-type]
        totals[(game["season"], game["week"], game["home_team"])] = (total + spread) / 2  # type: ignore[index]
        totals[(game["season"], game["week"], game["away_team"])] = (total - spread) / 2  # type: ignore[index]
    avg = sum(totals.values()) / len(totals)

    stats: list[dict[str, object]] = []
    snaps: list[dict[str, object]] = []
    named = {team: [p for p in PLAYERS if p[4] == team] for team in TEAMS}
    for (season, week, team), implied in sorted(totals.items()):
        if team not in TEAMS:
            continue
        env = max(0.6, min(1.6, implied / avg))
        pass_att = max(15.0, noisy(rng, TEAMS[team][0] * env**0.4, 0.12))
        carries = max(12.0, noisy(rng, TEAMS[team][1] * env**0.2, 0.14))
        targets = pass_att * 0.99
        air = targets * noisy(rng, ADOT, 0.08)
        roster = [p for p in named[team] if not (p[0] == "Chris Olave" and season == 2026 and week in IR_OLAVE)]
        rows: list[Row] = []
        taken = [0.0, 0.0, 0.0, 0.0]
        for name, gsis, pfr, _espn, _team, position, shares, rates, snap in named[team]:
            taken = [a + b for a, b in zip(taken, shares, strict=True)]  # the role exists even while he is on IR
            if (name, gsis, pfr, _espn, _team, position, shares, rates, snap) in roster:
                rows.append((name, gsis, position, pfr, shares, rates, snap))
        if not any(p[5] == "QB" for p in named[team]):
            rows.append((f"{team} QB", filler_id(team, "QB"), "QB", None, (0.99, 0.05, 0, 0), FILLER_RATES["QB"], 1.0))
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
                    filler_id(team, position),
                    position,
                    None,
                    shares,
                    FILLER_RATES[position],
                    0.5,
                )
            )
        game_id = f"{season}_{week:02d}_{team}"
        for name, gsis, position, snap_id, shares, rates, snap in rows:
            share_pass, share_rush, share_target, share_air = shares
            cmp_rate, ypa, ptd, intr, ypc, rtd, catch, ypt, retd, ypay = rates
            attempts = round(noisy(rng, pass_att * share_pass, 0.07)) if share_pass else 0
            completions = round(attempts * max(0.0, min(1.0, noisy(rng, cmp_rate, 0.08)))) if attempts else 0
            carried = round(noisy(rng, carries * share_rush, 0.25)) if share_rush else 0
            tgt = round(noisy(rng, targets * share_target, 0.3)) if share_target else 0
            rec = min(tgt, round(tgt * max(0.0, min(1.0, noisy(rng, catch, 0.15))))) if tgt else 0
            air_yards = round(noisy(rng, air * share_air, 0.35)) if share_air else 0
            if not (attempts or carried or tgt):
                continue
            row = {
                "player_id": gsis,
                "player_name": name,
                "position": position,
                "season": season,
                "week": week,
                "season_type": "REG",
                "team": team,
                "completions": completions,
                "attempts": attempts,
                "passing_yards": round(attempts * noisy(rng, ypa, 0.2)) if attempts else 0,
                "passing_tds": poisson(rng, attempts * ptd * env**0.6),
                "passing_interceptions": poisson(rng, attempts * intr),
                "carries": carried,
                "rushing_yards": round(carried * noisy(rng, ypc, 0.3)) if carried else 0,
                "rushing_tds": poisson(rng, carried * rtd * env**0.8),
                "targets": tgt,
                "receptions": rec,
                "receiving_yards": round(
                    max(0.0, air_yards * ypay * noisy(rng, 1.0, 0.3)) * 0.5 + 0.5 * tgt * ypt * noisy(rng, 1.0, 0.3)
                )
                if tgt
                else 0,
                "receiving_tds": poisson(rng, tgt * retd * env**0.6),
                "receiving_air_yards": air_yards,
                "rushing_fumbles_lost": 0,
                "receiving_fumbles_lost": 0,
                "sack_fumbles_lost": poisson(rng, attempts * 0.004),
            }
            stats.append(row)
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
    ids = [
        {"espn_id": str(p[3]), "gsis_id": p[1], "pfr_id": p[2], "name": p[0], "position": p[5], "team": p[4]}
        for p in PLAYERS
    ]
    scoreboards: dict[str, list[dict[str, object]]] = {}
    for game in schedule:
        if game["season"] != 2026:
            continue
        scoreboards.setdefault(str(game["week"]), []).append(
            {
                "home": game["home_team"],
                "away": game["away_team"],
                "spread": -float(game["spread_line"]),  # type: ignore[arg-type]
                "over_under": game["total_line"],
            }
        )
    for name, payload in (
        ("player_stats.json", stats),
        ("snap_counts.json", snaps),
        ("schedules.json", schedule),
        ("player_ids.json", ids),
        ("scoreboards.json", scoreboards),
    ):
        (HERE / name).write_text(json.dumps(payload, indent=None, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{len(stats)} stat rows, {len(snaps)} snap rows, {len(schedule)} games, league average {avg:.2f}")


if __name__ == "__main__":
    main()
