"""Generates ``backtest.json`` and ``inputs.json`` of this directory: a synthetic NBA points-league replay with the
inputs of the ``baseline_nba`` source (ROADMAP #43). Run it from the repository root:

    uv run python tests/fixtures/backtest/nba/generate.py

Nothing here is a capture. Four made-up teams of ten made-up players (named ``BOS 1`` ... ``OKC 10``, ids in a private
range) play 54 days; the last 14 are the backtest's scoring periods 1-14 and the 40 before are game-log history. Every
player has a nominal minutes load and per-minute rates. The *truth* a game is drawn from has the effects the baseline
models, and the two projection sources ESPN and DARKO see none of them:

- teammates out (each day, 12% of a team's top five and 4% of the rest sit): the vacated minutes go to the others in
  proportion to their nominal minutes times a hidden role factor, and the usage of those players rises;
- the second night of a back-to-back: starters lose 7% of their minutes;
- blowouts (a margin of 18 or more, drawn around the spread): starters lose 12% of their minutes when their team wins
  and 8% when it loses.

ESPN projects the nominal minutes times the true rates, scaled by a fixed per-player error; DARKO projects its own
talent rates (a few percent of noise) at nominal minutes plus noise. Both leave out players who sit. Counts are Poisson
and binomial draws. The point is the mechanics of the comparison (does a source that models context help the blend?),
which a fixture built this way can show; it says nothing about how much real NBA context is worth.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from fm.model.value_nba import darko_line
from fm.sources.darko import DarkoProjection
from fm.sports.nba import NBA

HERE = Path(__file__).resolve().parent
SEED = 43
FIRST_DAY = date(2026, 10, 12)
HISTORY_DAYS = 40
BACKTEST_DAYS = 14
AS_OF = "2026-12-01T12:00:00Z"
TEAMS = {"BOS": 1610612738, "NYK": 1610612752, "DEN": 1610612743, "OKC": 1610612760}
MINUTES = (36, 34, 32, 30, 28, 24, 20, 16, 10, 10)
FGA_RATE = (0.62, 0.52, 0.46, 0.40, 0.34, 0.42, 0.36, 0.32, 0.30, 0.26)
POSITIONS = ("PG", "SG", "SF", "PF", "C", "PG", "SG", "SF", "PF", "C")
ROSTER = (("BOS", 0), ("NYK", 1), ("DEN", 2), ("OKC", 3), ("BOS", 4), ("NYK", 5), ("DEN", 6)) + (
    ("OKC", 7),
    ("BOS", 8),
    ("NYK", 9),
    ("DEN", 0),
    ("OKC", 1),
    ("BOS", 2),
)
LINEUP_SLOTS = (0, 1, 2, 3, 4, 5, 11, 6, 11, 11, 12, 12, 12)
"""The manager's slot for each ``ROSTER`` entry: PG SG SF PF C G UTIL F UTIL UTIL, then three bench players."""
USAGE = ("fga", "fg3a", "fta", "ast", "tov")
STATS = ("MIN", "PTS", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF")
COLUMNS = ("FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF")


@dataclass
class Player:
    espn_id: int
    nba_id: int
    name: str
    team: str
    role: int
    age: float
    minutes: float
    factor: float
    espn_bias: float
    darko_minutes: float
    rates: dict[str, float] = field(default_factory=dict)
    talent: dict[str, float] = field(default_factory=dict)

    @property
    def position(self) -> str:
        return POSITIONS[self.role]


def poisson(rng: random.Random, mean: float) -> int:
    limit, count, product = 2.718281828459045**-mean, 0, rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def binomial(rng: random.Random, trials: int, chance: float) -> int:
    return sum(rng.random() < chance for _ in range(trials))


def make_players(rng: random.Random) -> list[Player]:
    players: list[Player] = []
    for team_index, team in enumerate(TEAMS):
        for role in range(10):
            guard = POSITIONS[role] in ("PG", "SG")
            fga = FGA_RATE[role] * rng.uniform(0.92, 1.08)
            f3a_share = rng.uniform(0.2, 0.5) if guard else rng.uniform(0.05, 0.35)
            player = Player(
                espn_id=9_100_000 + team_index * 100 + role + 1,
                nba_id=9_900_000 + team_index * 100 + role + 1,
                name=f"{team} {role + 1}",
                team=team,
                role=role,
                age=float(rng.randint(21, 36)),
                minutes=float(MINUTES[role]),
                factor=rng.uniform(0.6, 1.4),
                espn_bias=rng.gauss(0.0, 0.06),
                darko_minutes=0.0,
            )
            player.darko_minutes = max(6.0, round(player.minutes + rng.gauss(0.0, 1.2), 1))
            player.rates = {
                "fga": fga,
                "fg3a": fga * f3a_share,
                "fta": fga * rng.uniform(0.25, 0.5),
                "oreb": rng.uniform(0.02, 0.09) * (0.5 if guard else 1.4),
                "dreb": rng.uniform(0.08, 0.2) * (0.6 if guard else 1.3),
                "ast": rng.uniform(0.05, 0.12) * (2.0 if guard else 0.9),
                "stl": rng.uniform(0.02, 0.045),
                "blk": rng.uniform(0.005, 0.03) * (0.5 if guard else 2.0),
                "tov": fga * rng.uniform(0.12, 0.25),
                "pf": rng.uniform(0.07, 0.11),
                "p2": rng.uniform(0.5, 0.6),
                "p3": rng.uniform(0.32, 0.4),
                "pft": rng.uniform(0.7, 0.88),
            }
            player.talent = {key: value * rng.gauss(1.0, 0.04) for key, value in player.rates.items()}
            players.append(player)
    return players


def made_per_minute(rates: dict[str, float]) -> tuple[float, float, float]:
    """Expected (FGM, 3PM, FTM) per minute."""
    three = rates["fg3a"] * rates["p3"]
    two = (rates["fga"] - rates["fg3a"]) * rates["p2"]
    return three + two, three, rates["fta"] * rates["pft"]


def talent_points(player: Player) -> float:
    """DARKO's points per minute of the player's talent rates."""
    fgm, three, ftm = made_per_minute(player.talent)
    return 2.0 * fgm + three + ftm


def expected_line(player: Player, minutes: float, scale: float = 1.0) -> dict[str, float]:
    rates = player.rates
    fgm, three, ftm = made_per_minute(rates)
    line = {
        "MIN": minutes,
        "FGM": fgm * minutes,
        "FGA": rates["fga"] * minutes,
        "3PM": three * minutes,
        "3PA": rates["fg3a"] * minutes,
        "FTM": ftm * minutes,
        "FTA": rates["fta"] * minutes,
        "OREB": rates["oreb"] * minutes,
        "DREB": rates["dreb"] * minutes,
        "AST": rates["ast"] * minutes,
        "STL": rates["stl"] * minutes,
        "BLK": rates["blk"] * minutes,
        "TO": rates["tov"] * minutes,
        "PF": rates["pf"] * minutes,
    }
    line = {stat: (value if stat == "MIN" else value * scale) for stat, value in line.items()}
    line["PTS"] = 2.0 * line["FGM"] + line["3PM"] + line["FTM"]
    line["REB"] = line["OREB"] + line["DREB"]
    line["GP"] = 1.0
    return {stat: round(line[stat], 3) for stat in (*STATS, "GP")}


def game_counts(rng: random.Random, player: Player, minutes: float, usage: float) -> dict[str, int]:
    rates = player.rates

    def mean(key: str) -> float:
        return rates[key] * (usage if key in USAGE else 1.0) * minutes

    fg3a = poisson(rng, mean("fg3a"))
    fga = fg3a + poisson(rng, max(0.0, mean("fga") - mean("fg3a")))
    fg3m = binomial(rng, fg3a, rates["p3"])
    fgm = fg3m + binomial(rng, fga - fg3a, rates["p2"])
    fta = poisson(rng, mean("fta"))
    return {
        "FGM": fgm,
        "FGA": fga,
        "FG3M": fg3m,
        "FG3A": fg3a,
        "FTM": binomial(rng, fta, rates["pft"]),
        "FTA": fta,
        "OREB": poisson(rng, mean("oreb")),
        "DREB": poisson(rng, mean("dreb")),
        "AST": poisson(rng, mean("ast")),
        "STL": poisson(rng, mean("stl")),
        "BLK": poisson(rng, mean("blk")),
        "TOV": poisson(rng, mean("tov")),
        "PF": poisson(rng, mean("pf")),
    }


def true_minutes(
    rng: random.Random, team: list[Player], absent: set[int], second_night: bool, margin: float
) -> dict[int, tuple[float, float]]:
    """Minutes and usage multiplier of each player who plays: nominal, plus the vacated minutes, less back-to-back
    and blowout rest, plus noise."""
    present = [player for player in team if player.nba_id not in absent]
    vacated = sum(player.minutes for player in team if player.nba_id in absent)
    weight = {player.nba_id: player.minutes * player.factor for player in present}
    total = sum(weight.values())
    minutes = {player.nba_id: player.minutes + vacated * weight[player.nba_id] / total for player in present}
    usage_gone = sum(player.minutes * player.rates["fga"] for player in team if player.nba_id in absent)
    usage_rest = sum(player.minutes * player.rates["fga"] for player in present)
    usage = {player.nba_id: 1.0 + 0.3 * usage_gone / usage_rest for player in present}
    for share, applies in ((0.07, second_night), (0.12 if margin >= 18 else 0.08, abs(margin) >= 18)):
        if not applies:
            continue
        starters = [player for player in present if minutes[player.nba_id] >= 28]
        lost = 0.0
        for player in starters:
            cut = share * minutes[player.nba_id]
            minutes[player.nba_id] -= cut
            lost += cut
        bench = [player for player in present if player not in starters]
        base = sum(minutes[player.nba_id] for player in bench)
        for player in bench:
            minutes[player.nba_id] += lost * minutes[player.nba_id] / base if base else 0.0
    return {
        player.nba_id: (min(42.0, max(2.0, minutes[player.nba_id] + rng.gauss(0.0, 2.2))), usage[player.nba_id])
        for player in present
    }


def main() -> None:
    rng = random.Random(SEED)
    players = make_players(rng)
    by_team = {team: [player for player in players if player.team == team] for team in TEAMS}
    logs: list[list[object]] = []
    contexts: dict[str, dict[str, object]] = {}
    espn: dict[int, dict[int, dict[str, float]]] = {}
    darko: dict[int, dict[int, dict[str, float]]] = {}
    actuals: dict[int, dict[int, dict[str, float]]] = {}
    talents = [
        DarkoProjection(
            nba_id=player.nba_id,
            name=player.name,
            position=player.position,
            team_id=TEAMS[player.team],
            team=player.team,
            age=player.age,
            minutes=player.darko_minutes,
            pace=98.0,
            per_100={
                key: player.talent[key] * 4800.0 / 98.0
                for key in ("fga", "fg3a", "fta", "ast", "stl", "blk", "tov", "pf")
            }
            | {
                "orb": player.talent["oreb"] * 4800.0 / 98.0,
                "drb": player.talent["dreb"] * 4800.0 / 98.0,
                "pts": talent_points(player) * 4800.0 / 98.0,
            },
            rates={
                "fg_pct": made_per_minute(player.talent)[0] / player.talent["fga"],
                "fg3_pct": player.talent["p3"],
                "ft_pct": player.talent["pft"],
            },
            dpm={},
        )
        for player in players
    ]
    darko_by_id = {talent.nba_id: talent for talent in talents}
    played_yesterday: set[str] = set()
    for index in range(HISTORY_DAYS + BACKTEST_DAYS):
        day = FIRST_DAY + timedelta(days=index)
        period = index - HISTORY_DAYS + 1
        slate = rng.random()
        names = list(TEAMS)
        rng.shuffle(names)
        pairs = (
            []
            if slate < 0.25
            else [(names[0], names[1])]
            if slate < 0.6
            else [(names[0], names[1]), (names[2], names[3])]
        )
        playing = {team for pair in pairs for team in pair}
        out: dict[int, float] = {}
        spreads: dict[str, float] = {}
        for number, (home, away) in enumerate(pairs):
            strength = rng.gauss(0.0, 5.0)
            spread = round(-strength * 2) / 2
            margin = {home: strength + rng.gauss(0.0, 12.0)}
            margin[away] = -margin[home]
            spreads |= {home: spread, away: -spread}
            for team in (home, away):
                roster = by_team[team]
                absent = {player.nba_id for player in roster if rng.random() < (0.12 if player.role < 5 else 0.04)}
                out |= dict.fromkeys(absent, 1.0)
                minutes = true_minutes(rng, roster, absent, team in played_yesterday, margin[team])
                game_id = f"00226{index:03d}{number}"
                for player in roster:
                    if player.nba_id not in minutes:
                        continue
                    played, usage = minutes[player.nba_id]
                    counts = game_counts(rng, player, played, usage)
                    logs.append([player.nba_id, team, game_id, day.isoformat(), round(played), *counts.values()])
                    if period >= 1:
                        points = 2 * counts["FGM"] + counts["FG3M"] + counts["FTM"]
                        actuals.setdefault(period, {})[player.espn_id] = {
                            "MIN": float(round(played)),
                            "PTS": float(points),
                            "FGM": float(counts["FGM"]),
                            "FGA": float(counts["FGA"]),
                            "3PM": float(counts["FG3M"]),
                            "3PA": float(counts["FG3A"]),
                            "FTM": float(counts["FTM"]),
                            "FTA": float(counts["FTA"]),
                            "OREB": float(counts["OREB"]),
                            "DREB": float(counts["DREB"]),
                            "REB": float(counts["OREB"] + counts["DREB"]),
                            "AST": float(counts["AST"]),
                            "STL": float(counts["STL"]),
                            "BLK": float(counts["BLK"]),
                            "TO": float(counts["TOV"]),
                            "PF": float(counts["PF"]),
                            "GP": 1.0,
                        }
                if period >= 1:
                    for player in roster:
                        if player.nba_id in absent:
                            continue
                        scale = (1.0 + player.espn_bias) * rng.gauss(1.0, 0.02)
                        espn.setdefault(period, {})[player.espn_id] = expected_line(player, player.minutes, scale)
                        talent = darko_by_id[player.nba_id]
                        darko.setdefault(period, {})[player.espn_id] = {
                            stat: round(value, 3) for stat, value in darko_line(talent).items()
                        }
        if period >= 1:
            contexts[str(period)] = {
                "day": day.isoformat(),
                "teams": sorted(playing),
                "out": {str(nba_id): probability for nba_id, probability in out.items()},
                "spreads": spreads,
                "back_to_back": sorted(playing & played_yesterday),
            }
        played_yesterday = playing

    write_backtest(players, espn, darko, actuals)
    write_inputs(players, talents, logs, contexts)


def row(value: object) -> str:
    return json.dumps(value, separators=(", ", ": "))


def write_backtest(
    players: list[Player],
    espn: dict[int, dict[int, dict[str, float]]],
    darko: dict[int, dict[int, dict[str, float]]],
    actuals: dict[int, dict[int, dict[str, float]]],
) -> None:
    ids = {(player.team, player.role): player for player in players}
    roster = [(ids[key], slot) for key, slot in zip(ROSTER, LINEUP_SLOTS, strict=True)]
    lines = ["{", '  "format": 1,', '  "sport": "nba",', '  "season": 2027,', '  "settings": "settings.json",']
    lines += [f'  "as_of": "{AS_OF}",', '  "players": [']
    entries = []
    for player in players:
        slots = sorted(NBA.eligible_slots(player.position, include_reserve=True))
        entries.append(
            "    "
            + row(
                {"espn_id": player.espn_id, "name": player.name, "position": player.position, "eligible_slots": slots}
            )
        )
    lines += [",\n".join(entries), "  ],", '  "weeks": [']
    weeks = []
    for period in sorted(actuals):
        lineup = {str(player.espn_id): slot for player, slot in roster}
        body = [f'      "period": {period},', f'      "lineup": {row(lineup)},', '      "projections": {']
        body.append(f'        "espn": {row({str(i): v for i, v in espn[period].items()})},')
        body.append(f'        "darko": {row({str(i): v for i, v in darko[period].items()})}')
        body += ["      },", f'      "actuals": {row({str(i): v for i, v in actuals[period].items()})}']
        weeks.append("    {\n" + "\n".join(body) + "\n    }")
    lines += [",\n".join(weeks), "  ]", "}"]
    (HERE / "backtest.json").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copyfile(HERE.parents[1] / "espn" / "fba_settings_points.json", HERE / "settings.json")


def write_inputs(
    players: list[Player],
    talents: list[DarkoProjection],
    logs: list[list[object]],
    contexts: dict[str, dict[str, object]],
) -> None:
    by_nba = {player.nba_id: player for player in players}
    lines = ["{", '  "format": 1,', '  "players": [']
    lines += [
        ",\n".join(
            "    "
            + row(
                {
                    "espn_id": by_nba[talent.nba_id].espn_id,
                    "talent": talent.model_dump(mode="json", exclude_none=True),
                }
            )
            for talent in talents
        ),
        "  ],",
        f'  "columns": {row(["PLAYER_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MIN", *COLUMNS])},',
        '  "logs": [',
        ",\n".join("    " + row(entry) for entry in logs),
        "  ],",
        '  "days": {',
        ",\n".join(f'    "{period}": {row(context)}' for period, context in contexts.items()),
        "  }",
        "}",
    ]
    (HERE / "inputs.json").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
