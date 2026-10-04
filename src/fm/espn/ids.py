"""ESPN numeric ID maps for ``ffl`` (NFL) and ``fba`` (NBA): stats, lineup slots, positions, pro teams, injury status.

ESPN's API speaks in integers: ``scoringItems[].statId``, ``lineupSlotCounts`` keys, ``player.defaultPositionId``,
``player.proTeamId``, and so on. These tables translate them to the labels the rest of ``fm`` uses. The values come
from ESPN's own responses as catalogued by the maintained ``cwendt94/espn-api`` project (DESIGN section 6.1 says to
reuse them); abbreviations follow ESPN's scoring-settings page so a parsed scoring item reads like the league page.

Conventions:

- Stat ids share one id space per game between ``scoringItems`` and a player's ``stats[].stats`` dict, so the same
  table serves both the settings parser and the projection parsers.
- Lineup slot ids (where a player sits) and position ids (``defaultPositionId``, what a player is) are different
  spaces. ``positionLimits`` and ``pointsOverrides`` are keyed by position id; ``lineupSlotCounts`` by slot id.
- NFL pro-team abbreviations are ESPN's (``LAR``, ``WSH``, ``JAX``); nflverse differs for the Rams (``LA``) and
  Washington (``WAS``), which the ID crosswalk handles. NBA pro teams use nba.com tricodes (``PHX``, ``GSW``, ``NOP``)
  because every NBA data source joins on those.
- Lookups never raise for an unknown id: they return a ``STAT_<id>``-style placeholder so a new ESPN id shows up in
  output instead of crashing a sync. Reverse lookups (label to id) do raise ``KeyError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Self


class Game(StrEnum):
    """ESPN game keys as they appear in API URLs (``/apis/v3/games/{game}/...``)."""

    FFL = "ffl"  # NFL
    FBA = "fba"  # NBA

    @property
    def game_id(self) -> int:
        """ESPN's numeric ``gameId`` for this game."""
        return GAME_IDS[self]

    @property
    def sport(self) -> str:
        """The ``sport`` value used in ``config.toml`` (``nfl`` / ``nba``)."""
        return _GAME_SPORTS[self]

    @classmethod
    def from_game_id(cls, game_id: int) -> Self:
        for game, candidate in GAME_IDS.items():
            if candidate == game_id:
                return cls(game)
        raise ValueError(f"unsupported ESPN gameId {game_id!r}; expected one of {sorted(GAME_IDS.values())}")

    @classmethod
    def from_sport(cls, sport: str) -> Self:
        for game, candidate in _GAME_SPORTS.items():
            if candidate == sport.lower():
                return cls(game)
        raise ValueError(f"unsupported sport {sport!r}; expected one of {sorted(_GAME_SPORTS.values())}")

    @classmethod
    def coerce(cls, value: Game | str) -> Self:
        """Accept a ``Game``, a game key (``ffl``), or a sport (``nfl``)."""
        if isinstance(value, Game):
            return cls(value)
        lowered = value.lower()
        if lowered in cls._value2member_map_:
            return cls(lowered)
        return cls.from_sport(lowered)


GAME_IDS: Mapping[Game, int] = MappingProxyType({Game.FFL: 1, Game.FBA: 3})
_GAME_SPORTS: Mapping[Game, str] = MappingProxyType({Game.FFL: "nfl", Game.FBA: "nba"})


class InjuryStatus(StrEnum):
    """Normalized ``player.injuryStatus``. ``UNKNOWN`` covers a missing field or a value ESPN has not used before."""

    ACTIVE = "ACTIVE"
    PROBABLE = "PROBABLE"
    QUESTIONABLE = "QUESTIONABLE"
    DOUBTFUL = "DOUBTFUL"
    OUT = "OUT"
    DAY_TO_DAY = "DAY_TO_DAY"
    INJURY_RESERVE = "INJURY_RESERVE"
    SUSPENSION = "SUSPENSION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class StatDef:
    """A stat id's abbreviation (as on ESPN's scoring page) and human label."""

    abbr: str
    label: str


def _stats(table: Mapping[int, tuple[str, str]]) -> Mapping[int, StatDef]:
    return MappingProxyType({stat_id: StatDef(abbr, label) for stat_id, (abbr, label) in table.items()})


@dataclass(frozen=True)
class IdMaps:
    """All ESPN id tables for one game, with forgiving forward lookups and strict reverse lookups."""

    game: Game
    stats: Mapping[int, StatDef]
    lineup_slots: Mapping[int, str]
    positions: Mapping[int, str]
    pro_teams: Mapping[int, str]
    injury_statuses: Mapping[str, InjuryStatus]
    bench_slot: int
    ir_slot: int
    slot_aliases: Mapping[str, int] = field(default_factory=dict)

    def stat_abbr(self, stat_id: int) -> str:
        stat = self.stats.get(stat_id)
        return stat.abbr if stat else f"STAT_{stat_id}"

    def stat_label(self, stat_id: int) -> str:
        stat = self.stats.get(stat_id)
        return stat.label if stat else f"Stat {stat_id}"

    def slot_label(self, slot_id: int) -> str:
        return self.lineup_slots.get(slot_id, f"SLOT_{slot_id}")

    def position_label(self, position_id: int) -> str:
        return self.positions.get(position_id, f"POS_{position_id}")

    def pro_team(self, team_id: int) -> str:
        return self.pro_teams.get(team_id, f"TEAM_{team_id}")

    def injury_status(self, raw: str | None) -> InjuryStatus:
        """Normalize a raw ``injuryStatus``; ``None``/empty and unrecognized values become ``UNKNOWN``."""
        if not raw:
            return InjuryStatus.UNKNOWN
        return self.injury_statuses.get(raw.strip().upper(), InjuryStatus.UNKNOWN)

    def is_active_slot(self, slot_id: int) -> bool:
        """True for slots that score (everything but bench and IR)."""
        return slot_id not in (self.bench_slot, self.ir_slot)

    def stat_id(self, abbr: str) -> int:
        """Reverse lookup by abbreviation (case-sensitive, e.g. ``REC``, ``FG%``). Raises ``KeyError``."""
        for stat_id, stat in self.stats.items():
            if stat.abbr == abbr:
                return stat_id
        raise KeyError(f"{self.game.value}: unknown stat abbreviation {abbr!r}")

    def slot_id(self, label: str) -> int:
        """Reverse lookup by slot label or alias (``FLEX`` for ffl 23, ``UT`` for fba 11). Raises ``KeyError``."""
        if label in self.slot_aliases:
            return self.slot_aliases[label]
        for slot_id, candidate in self.lineup_slots.items():
            if candidate == label:
                return slot_id
        raise KeyError(f"{self.game.value}: unknown lineup slot {label!r}")

    def position_id(self, label: str) -> int:
        """Reverse lookup by position label. Raises ``KeyError``."""
        for position_id, candidate in self.positions.items():
            if candidate == label:
                return position_id
        raise KeyError(f"{self.game.value}: unknown position {label!r}")


# --- shared -----------------------------------------------------------------------------------------------------------

_INJURY_STATUSES: Mapping[str, InjuryStatus] = MappingProxyType(
    {
        "ACTIVE": InjuryStatus.ACTIVE,
        "NORMAL": InjuryStatus.ACTIVE,
        "PROBABLE": InjuryStatus.PROBABLE,
        "QUESTIONABLE": InjuryStatus.QUESTIONABLE,
        "DOUBTFUL": InjuryStatus.DOUBTFUL,
        "OUT": InjuryStatus.OUT,
        "DAY_TO_DAY": InjuryStatus.DAY_TO_DAY,
        "INJURY_RESERVE": InjuryStatus.INJURY_RESERVE,
        "IR": InjuryStatus.INJURY_RESERVE,
        "SUSPENSION": InjuryStatus.SUSPENSION,
        "SUSPENDED": InjuryStatus.SUSPENSION,
    }
)

# --- ffl (NFL) --------------------------------------------------------------------------------------------------------

FFL_STATS: Mapping[int, StatDef] = _stats(
    {
        0: ("PA", "Each Pass Attempted"),
        1: ("PC", "Each Pass Completed"),
        2: ("INC", "Each Incomplete Pass"),
        3: ("PY", "Passing Yards"),
        4: ("PTD", "TD Pass"),
        5: ("PY5", "Every 5 passing yards"),
        6: ("PY10", "Every 10 passing yards"),
        7: ("PY20", "Every 20 passing yards"),
        8: ("PY25", "Every 25 passing yards"),
        9: ("PY50", "Every 50 passing yards"),
        10: ("PY100", "Every 100 passing yards"),
        11: ("PC5", "Every 5 pass completions"),
        12: ("PC10", "Every 10 pass completions"),
        13: ("IP5", "Every 5 pass incompletions"),
        14: ("IP10", "Every 10 pass incompletions"),
        15: ("PTD40", "40+ yard TD pass bonus"),
        16: ("PTD50", "50+ yard TD pass bonus"),
        17: ("P300", "300-399 yard passing game"),
        18: ("P400", "400+ yard passing game"),
        19: ("2PC", "2pt Passing Conversion"),
        20: ("INTT", "Interceptions Thrown"),
        21: ("CPCT", "Passing Completion Pct"),
        22: ("PYPG", "Passing Yards Per Game"),
        23: ("RA", "Rushing Attempts"),
        24: ("RY", "Rushing Yards"),
        25: ("RTD", "TD Rush"),
        26: ("2PR", "2pt Rushing Conversion"),
        27: ("RY5", "Every 5 rushing yards"),
        28: ("RY10", "Every 10 rushing yards"),
        29: ("RY20", "Every 20 rushing yards"),
        30: ("RY25", "Every 25 rushing yards"),
        31: ("RY50", "Every 50 rushing yards"),
        32: ("R100", "Every 100 rushing yards"),
        33: ("RA5", "Every 5 rush attempts"),
        34: ("RA10", "Every 10 rush attempts"),
        35: ("RTD40", "40+ yard TD rush bonus"),
        36: ("RTD50", "50+ yard TD rush bonus"),
        37: ("RY100", "100-199 yard rushing game"),
        38: ("RY200", "200+ yard rushing game"),
        39: ("RYPA", "Rushing Yards Per Attempt"),
        40: ("RYPG", "Rushing Yards Per Game"),
        41: ("RECS", "Receptions"),
        42: ("REY", "Receiving Yards"),
        43: ("RETD", "TD Reception"),
        44: ("2PRE", "2pt Receiving Conversion"),
        45: ("RETD40", "40+ yard TD rec bonus"),
        46: ("RETD50", "50+ yard TD rec bonus"),
        47: ("REY5", "Every 5 receiving yards"),
        48: ("REY10", "Every 10 receiving yards"),
        49: ("REY20", "Every 20 receiving yards"),
        50: ("REY25", "Every 25 receiving yards"),
        51: ("REY50", "Every 50 receiving yards"),
        52: ("RE100", "Every 100 receiving yards"),
        53: ("REC", "Each reception"),
        54: ("REC5", "Every 5 receptions"),
        55: ("REC10", "Every 10 receptions"),
        56: ("REY100", "100-199 yard receiving game"),
        57: ("REY200", "200+ yard receiving game"),
        58: ("RET", "Receiving Target"),
        59: ("YAC", "Receiving Yards After Catch"),
        60: ("YPC", "Receiving Yards Per Catch"),
        61: ("REYPG", "Receiving Yards Per Game"),
        62: ("PTL", "Total 2pt Conversions"),
        63: ("FTD", "Fumble Recovered for TD"),
        64: ("SKD", "Sacked"),
        65: ("PFUM", "Passing Fumbles"),
        66: ("RFUM", "Rushing Fumbles"),
        67: ("REFUM", "Receiving Fumbles"),
        68: ("FUM", "Total Fumbles"),
        69: ("PFUML", "Passing Fumbles Lost"),
        70: ("RFUML", "Rushing Fumbles Lost"),
        71: ("REFUML", "Receiving Fumbles Lost"),
        72: ("FUML", "Total Fumbles Lost"),
        73: ("TT", "Total Turnovers"),
        74: ("FG50P", "FG Made (50+ yards)"),
        75: ("FGA50P", "FG Attempted (50+ yards)"),
        76: ("FGM50P", "FG Missed (50+ yards)"),
        77: ("FG40", "FG Made (40-49 yards)"),
        78: ("FGA40", "FG Attempted (40-49 yards)"),
        79: ("FGM40", "FG Missed (40-49 yards)"),
        80: ("FG0", "FG Made (0-39 yards)"),
        81: ("FGA0", "FG Attempted (0-39 yards)"),
        82: ("FGM0", "FG Missed (0-39 yards)"),
        83: ("FG", "Total FG Made"),
        84: ("FGA", "Total FG Attempted"),
        85: ("FGM", "Total FG Missed"),
        86: ("PAT", "Each PAT Made"),
        87: ("PATA", "Each PAT Attempted"),
        88: ("PATM", "Each PAT Missed"),
        89: ("PA0", "0 points allowed"),
        90: ("PA1", "1-6 points allowed"),
        91: ("PA7", "7-13 points allowed"),
        92: ("PA14", "14-17 points allowed"),
        93: ("BLKKRTD", "Blocked Punt or FG return for TD"),
        94: ("DEFRETTD", "Fumble or INT Return for TD"),
        95: ("INT", "Each Interception"),
        96: ("FR", "Each Fumble Recovered"),
        97: ("BLKK", "Blocked Punt, PAT or FG"),
        98: ("SF", "Each Safety"),
        99: ("SK", "Each Sack"),
        100: ("HALFSK", "1/2 Sack"),
        101: ("KRTD", "Kickoff Return TD"),
        102: ("PRTD", "Punt Return TD"),
        103: ("INTTD", "Interception Return TD"),
        104: ("FRTD", "Fumble Return TD"),
        105: ("TRTD", "Total Return TD"),
        106: ("FF", "Each Fumble Forced"),
        107: ("TKA", "Assisted Tackles"),
        108: ("TKS", "Solo Tackles"),
        109: ("TK", "Total Tackles"),
        110: ("TK3", "Every 3 Total Tackles"),
        111: ("TK5", "Every 5 Total Tackles"),
        112: ("STF", "Stuffs"),
        113: ("PD", "Passes Defensed"),
        114: ("KR", "Kickoff Return Yards"),
        115: ("PR", "Punt Return Yards"),
        116: ("KR10", "Every 10 kickoff return yards"),
        117: ("KR25", "Every 25 kickoff return yards"),
        118: ("PR10", "Every 10 punt return yards"),
        119: ("PR25", "Every 25 punt return yards"),
        120: ("PTSA", "Points Allowed"),
        121: ("PA18", "18-21 points allowed"),
        122: ("PA22", "22-27 points allowed"),
        123: ("PA28", "28-34 points allowed"),
        124: ("PA35", "35-45 points allowed"),
        125: ("PA46", "46+ points allowed"),
        126: ("PAPG", "Points Allowed Per Game"),
        127: ("YA", "Yards Allowed"),
        128: ("YA100", "Less than 100 total yards allowed"),
        129: ("YA199", "100-199 total yards allowed"),
        130: ("YA299", "200-299 total yards allowed"),
        131: ("YA349", "300-349 total yards allowed"),
        132: ("YA399", "350-399 total yards allowed"),
        133: ("YA449", "400-449 total yards allowed"),
        134: ("YA499", "450-499 total yards allowed"),
        135: ("YA549", "500-549 total yards allowed"),
        136: ("YA550", "550+ total yards allowed"),
        137: ("YAPG", "Yards Allowed Per Game"),
        138: ("PT", "Net Punts"),
        139: ("PTY", "Punt Yards"),
        140: ("PT10", "Punts Inside the 10"),
        141: ("PT20", "Punts Inside the 20"),
        142: ("PTB", "Blocked Punts"),
        143: ("PTR", "Punts Returned"),
        144: ("PTRY", "Punt Return Yards"),
        145: ("PTTB", "Touchbacks"),
        146: ("PTFC", "Fair Catches"),
        147: ("PTAVG", "Punt Average"),
        148: ("PTA44", "Punt Average 44.0+"),
        149: ("PTA42", "Punt Average 42.0-43.9"),
        150: ("PTA40", "Punt Average 40.0-41.9"),
        151: ("PTA38", "Punt Average 38.0-39.9"),
        152: ("PTA36", "Punt Average 36.0-37.9"),
        153: ("PTA34", "Punt Average 34.0-35.9"),
        154: ("PTA33", "Punt Average 33.9 or less"),
        155: ("TW", "Team Win"),
        156: ("TL", "Team Loss"),
        157: ("TIE", "Team Tie"),
        158: ("PTS", "Points Scored"),
        159: ("PPG", "Points Scored Per Game"),
        160: ("MGN", "Margin of Victory"),
        161: ("WM25", "25+ point Win Margin"),
        162: ("WM20", "20-24 point Win Margin"),
        163: ("WM15", "15-19 point Win Margin"),
        164: ("WM10", "10-14 point Win Margin"),
        165: ("WM5", "5-9 point Win Margin"),
        166: ("WM1", "1-4 point Win Margin"),
        167: ("LM1", "1-4 point Loss Margin"),
        168: ("LM5", "5-9 point Loss Margin"),
        169: ("LM10", "10-14 point Loss Margin"),
        170: ("LM15", "15-19 point Loss Margin"),
        171: ("LM20", "20-24 point Loss Margin"),
        172: ("LM25", "25+ point Loss Margin"),
        173: ("MGNPG", "Margin of Victory Per Game"),
        174: ("WINPCT", "Winning Pct"),
        175: ("PTD0", "0-9 yd TD pass bonus"),
        176: ("PTD10", "10-19 yd TD pass bonus"),
        177: ("PTD20", "20-29 yd TD pass bonus"),
        178: ("PTD30", "30-39 yd TD pass bonus"),
        179: ("RTD0", "0-9 yd TD rush bonus"),
        180: ("RTD10", "10-19 yd TD rush bonus"),
        181: ("RTD20", "20-29 yd TD rush bonus"),
        182: ("RTD30", "30-39 yd TD rush bonus"),
        183: ("RETD0", "0-9 yd TD rec bonus"),
        184: ("RETD10", "10-19 yd TD rec bonus"),
        185: ("RETD20", "20-29 yd TD rec bonus"),
        186: ("RETD30", "30-39 yd TD rec bonus"),
        187: ("DPTSA", "D/ST Points Allowed"),
        188: ("DPA0", "D/ST 0 points allowed"),
        189: ("DPA1", "D/ST 1-6 points allowed"),
        190: ("DPA7", "D/ST 7-13 points allowed"),
        191: ("DPA14", "D/ST 14-17 points allowed"),
        192: ("DPA18", "D/ST 18-21 points allowed"),
        193: ("DPA22", "D/ST 22-27 points allowed"),
        194: ("DPA28", "D/ST 28-34 points allowed"),
        195: ("DPA35", "D/ST 35-45 points allowed"),
        196: ("DPA46", "D/ST 46+ points allowed"),
        197: ("DPAPG", "D/ST Points Allowed Per Game"),
        198: ("FG50", "FG Made (50-59 yards)"),
        199: ("FGA50", "FG Attempted (50-59 yards)"),
        200: ("FGM50", "FG Missed (50-59 yards)"),
        201: ("FG60", "FG Made (60+ yards)"),
        202: ("FGA60", "FG Attempted (60+ yards)"),
        203: ("FGM60", "FG Missed (60+ yards)"),
        204: ("O2PRET", "Offensive 2pt Return"),
        205: ("D2PRET", "Defensive 2pt Return"),
        206: ("2PRET", "2pt Return"),
        207: ("O1PSF", "Offensive 1pt Safety"),
        208: ("D1PSF", "Defensive 1pt Safety"),
        209: ("1PSF", "1pt Safety"),
        210: ("GP", "Games Played"),
        211: ("PFD", "Passing First Down"),
        212: ("RFD", "Rushing First Down"),
        213: ("REFD", "Receiving First Down"),
        214: ("FGY", "FG Made Yards"),
        215: ("FGMY", "FG Missed Yards"),
        216: ("FGAY", "FG Attempt Yards"),
        217: ("FGY5", "Every 5 FG Made yards"),
        218: ("FGY10", "Every 10 FG Made yards"),
        219: ("FGY20", "Every 20 FG Made yards"),
        220: ("FGY25", "Every 25 FG Made yards"),
        221: ("FGY50", "Every 50 FG Made yards"),
        222: ("FGY100", "Every 100 FG Made yards"),
        223: ("FGMY5", "Every 5 FG Missed yards"),
        224: ("FGMY10", "Every 10 FG Missed yards"),
        225: ("FGMY20", "Every 20 FG Missed yards"),
        226: ("FGMY25", "Every 25 FG Missed yards"),
        227: ("FGMY50", "Every 50 FG Missed yards"),
        228: ("FGMY100", "Every 100 FG Missed yards"),
        229: ("FGAY5", "Every 5 FG Attempt yards"),
        230: ("FGAY10", "Every 10 FG Attempt yards"),
        231: ("FGAY20", "Every 20 FG Attempt yards"),
        232: ("FGAY25", "Every 25 FG Attempt yards"),
        233: ("FGAY50", "Every 50 FG Attempt yards"),
        234: ("FGAY100", "Every 100 FG Attempt yards"),
    }
)

# ``lineupSlotCounts`` keys / ``lineupSlotId``. 22 is unused by ESPN and intentionally absent.
FFL_LINEUP_SLOTS: Mapping[int, str] = MappingProxyType(
    {
        0: "QB",
        1: "TQB",
        2: "RB",
        3: "RB/WR",
        4: "WR",
        5: "WR/TE",
        6: "TE",
        7: "OP",
        8: "DT",
        9: "DE",
        10: "LB",
        11: "DL",
        12: "CB",
        13: "S",
        14: "DB",
        15: "DP",
        16: "D/ST",
        17: "K",
        18: "P",
        19: "HC",
        20: "BE",
        21: "IR",
        23: "RB/WR/TE",
        24: "ER",
        25: "Rookie",
    }
)

# ``player.defaultPositionId`` / ``positionLimits`` keys / ``pointsOverrides`` keys.
FFL_POSITIONS: Mapping[int, str] = MappingProxyType(
    {
        1: "QB",
        2: "RB",
        3: "WR",
        4: "TE",
        5: "K",
        7: "P",
        9: "DT",
        10: "DE",
        11: "LB",
        12: "CB",
        13: "S",
        14: "HC",
        16: "D/ST",
    }
)

FFL_PRO_TEAMS: Mapping[int, str] = MappingProxyType(
    {
        0: "FA",
        1: "ATL",
        2: "BUF",
        3: "CHI",
        4: "CIN",
        5: "CLE",
        6: "DAL",
        7: "DEN",
        8: "DET",
        9: "GB",
        10: "TEN",
        11: "IND",
        12: "KC",
        13: "LV",
        14: "LAR",
        15: "MIA",
        16: "MIN",
        17: "NE",
        18: "NO",
        19: "NYG",
        20: "NYJ",
        21: "PHI",
        22: "ARI",
        23: "PIT",
        24: "LAC",
        25: "SF",
        26: "SEA",
        27: "TB",
        28: "WSH",
        29: "CAR",
        30: "JAX",
        33: "BAL",
        34: "HOU",
    }
)

# --- fba (NBA) --------------------------------------------------------------------------------------------------------

FBA_STATS: Mapping[int, StatDef] = _stats(
    {
        0: ("PTS", "Points"),
        1: ("BLK", "Blocks"),
        2: ("STL", "Steals"),
        3: ("AST", "Assists"),
        4: ("OREB", "Offensive Rebounds"),
        5: ("DREB", "Defensive Rebounds"),
        6: ("REB", "Rebounds"),
        7: ("EJ", "Ejections"),
        8: ("FF", "Flagrant Fouls"),
        9: ("PF", "Personal Fouls"),
        10: ("TF", "Technical Fouls"),
        11: ("TO", "Turnovers"),
        12: ("DQ", "Disqualifications"),
        13: ("FGM", "Field Goals Made"),
        14: ("FGA", "Field Goals Attempted"),
        15: ("FTM", "Free Throws Made"),
        16: ("FTA", "Free Throws Attempted"),
        17: ("3PM", "Three Pointers Made"),
        18: ("3PA", "Three Pointers Attempted"),
        19: ("FG%", "Field Goal Percentage"),
        20: ("FT%", "Free Throw Percentage"),
        21: ("3PT%", "Three Point Percentage"),
        22: ("AFG%", "Adjusted Field Goal Percentage"),
        23: ("FGMI", "Field Goals Missed"),
        24: ("FTMI", "Free Throws Missed"),
        25: ("3PMI", "Three Pointers Missed"),
        26: ("APG", "Assists Per Game"),
        27: ("BPG", "Blocks Per Game"),
        28: ("MPG", "Minutes Per Game"),
        29: ("PPG", "Points Per Game"),
        30: ("RPG", "Rebounds Per Game"),
        31: ("SPG", "Steals Per Game"),
        32: ("TOPG", "Turnovers Per Game"),
        33: ("3PG", "Three Pointers Made Per Game"),
        34: ("PPM", "Points Per Minute"),
        35: ("A/TO", "Assist To Turnover Ratio"),
        36: ("STR", "Steal To Turnover Ratio"),
        37: ("DD", "Double Doubles"),
        38: ("TD", "Triple Doubles"),
        39: ("QD", "Quadruple Doubles"),
        40: ("MIN", "Minutes"),
        41: ("GS", "Games Started"),
        42: ("GP", "Games Played"),
        43: ("TW", "Team Wins"),
        44: ("FTR", "Free Throw Rate"),
    }
)

# ``lineupSlotCounts`` keys / ``lineupSlotId``. 14 is unused by ESPN and intentionally absent.
FBA_LINEUP_SLOTS: Mapping[int, str] = MappingProxyType(
    {
        0: "PG",
        1: "SG",
        2: "SF",
        3: "PF",
        4: "C",
        5: "G",
        6: "F",
        7: "SG/SF",
        8: "G/F",
        9: "PF/C",
        10: "F/C",
        11: "UTIL",
        12: "BE",
        13: "IR",
        15: "Rookie",
    }
)

# ``player.defaultPositionId`` (1-based, unlike the slot ids).
FBA_POSITIONS: Mapping[int, str] = MappingProxyType({1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C"})

FBA_PRO_TEAMS: Mapping[int, str] = MappingProxyType(
    {
        0: "FA",
        1: "ATL",
        2: "BOS",
        3: "NOP",
        4: "CHI",
        5: "CLE",
        6: "DAL",
        7: "DEN",
        8: "DET",
        9: "GSW",
        10: "HOU",
        11: "IND",
        12: "LAC",
        13: "LAL",
        14: "MIA",
        15: "MIL",
        16: "MIN",
        17: "BKN",
        18: "NYK",
        19: "ORL",
        20: "PHI",
        21: "PHX",
        22: "POR",
        23: "SAC",
        24: "SAS",
        25: "OKC",
        26: "UTA",
        27: "WAS",
        28: "TOR",
        29: "MEM",
        30: "CHA",
    }
)

# --- registry ---------------------------------------------------------------------------------------------------------

FFL = IdMaps(
    game=Game.FFL,
    stats=FFL_STATS,
    lineup_slots=FFL_LINEUP_SLOTS,
    positions=FFL_POSITIONS,
    pro_teams=FFL_PRO_TEAMS,
    injury_statuses=_INJURY_STATUSES,
    bench_slot=20,
    ir_slot=21,
    slot_aliases=MappingProxyType({"FLEX": 23, "DST": 16}),
)

FBA = IdMaps(
    game=Game.FBA,
    stats=FBA_STATS,
    lineup_slots=FBA_LINEUP_SLOTS,
    positions=FBA_POSITIONS,
    pro_teams=FBA_PRO_TEAMS,
    injury_statuses=_INJURY_STATUSES,
    bench_slot=12,
    ir_slot=13,
    slot_aliases=MappingProxyType({"UT": 11}),
)

_ID_MAPS: Mapping[Game, IdMaps] = MappingProxyType({Game.FFL: FFL, Game.FBA: FBA})


def ids_for(game: Game | str) -> IdMaps:
    """The id tables for a game, given a ``Game``, a game key (``ffl``) or a sport (``nfl``)."""
    return _ID_MAPS[Game.coerce(game)]
