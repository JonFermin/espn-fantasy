-- v1 schema (ROADMAP #3, DESIGN sections 8, 10, 11, 14).
--
-- Every table is STRICT. Timestamps are fixed-width UTC ISO-8601 text ("2026-10-04T13:00:00.000000Z"), so text order
-- is time order. JSON columns hold compact JSON text and are checked with json_valid. Booleans are 0/1 INTEGER.
-- Column names equal the field names of the row models in fm.store.models; fm.store.repos maps SELECT * onto them.

-- Configured leagues, one row per season. `key` is the config key (nfl, nba); `team_id` is our team.
CREATE TABLE leagues (
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL,
    sport TEXT NOT NULL CHECK (sport IN ('nfl', 'nba')),
    espn_league_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    name TEXT,
    as_of TEXT NOT NULL,
    UNIQUE (sport, espn_league_id, season),
    UNIQUE (key, season)
) STRICT;

-- Index of raw responses saved under the cache dir (DESIGN 6.1 "raw capture"). `path` is relative to cache_dir();
-- `params` is the query/filter set, never cookies or tokens.
CREATE TABLE raw_snapshots (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    kind TEXT NOT NULL,
    league_id INTEGER REFERENCES leagues (id) ON DELETE SET NULL,
    scoring_period_id INTEGER,
    url TEXT,
    params TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(params)),
    path TEXT NOT NULL UNIQUE,
    sha256 TEXT,
    size_bytes INTEGER,
    status_code INTEGER,
    fetched_at TEXT NOT NULL
) STRICT;

CREATE INDEX raw_snapshots_lookup ON raw_snapshots (source, kind, league_id, fetched_at);

-- Parsed mSettings per league: scoring items, slots, lock type, waiver/acquisition rules, trade deadline, playoffs.
-- League settings are data (CLAUDE.md), so nothing here is a column: the parser's model is stored whole.
CREATE TABLE league_settings (
    league_id INTEGER PRIMARY KEY REFERENCES leagues (id) ON DELETE CASCADE,
    settings TEXT NOT NULL CHECK (json_valid(settings)),
    raw_snapshot_id INTEGER REFERENCES raw_snapshots (id) ON DELETE SET NULL,
    as_of TEXT NOT NULL
) STRICT;

-- ESPN teams with standings and FAAB state.
CREATE TABLE teams (
    league_id INTEGER NOT NULL REFERENCES leagues (id) ON DELETE CASCADE,
    team_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    abbrev TEXT,
    division_id INTEGER,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    ties INTEGER NOT NULL DEFAULT 0,
    points_for REAL NOT NULL DEFAULT 0,
    points_against REAL NOT NULL DEFAULT 0,
    playoff_seed INTEGER,
    waiver_rank INTEGER,
    acquisition_budget_spent INTEGER NOT NULL DEFAULT 0,
    as_of TEXT NOT NULL,
    PRIMARY KEY (league_id, team_id)
) STRICT;

-- Canonical player rows keyed by ESPN id per sport. Not league-scoped: the same player appears in every league.
CREATE TABLE players (
    sport TEXT NOT NULL CHECK (sport IN ('nfl', 'nba')),
    espn_id INTEGER NOT NULL,
    full_name TEXT NOT NULL,
    default_position_id INTEGER,
    position TEXT,
    pro_team_id INTEGER,
    pro_team TEXT,
    eligible_slot_ids TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(eligible_slot_ids)),
    injury_status TEXT,
    injured INTEGER NOT NULL DEFAULT 0 CHECK (injured IN (0, 1)),
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    as_of TEXT NOT NULL,
    PRIMARY KEY (sport, espn_id)
) STRICT;

CREATE INDEX players_name ON players (sport, full_name);

-- Crosswalk ESPN id <-> other id systems (gsis, sleeper, nba person id, ...). A source id maps to one ESPN player.
-- No foreign key to players: the crosswalk covers players the ESPN pool has not been pulled for.
CREATE TABLE player_ids (
    sport TEXT NOT NULL CHECK (sport IN ('nfl', 'nba')),
    espn_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    origin TEXT NOT NULL,
    as_of TEXT NOT NULL,
    PRIMARY KEY (sport, espn_id, source),
    UNIQUE (sport, source, source_id)
) STRICT;

-- Latest read of each team's roster per scoring period. A team's rows for a period are replaced together.
CREATE TABLE roster_snapshots (
    league_id INTEGER NOT NULL,
    scoring_period_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    espn_id INTEGER NOT NULL,
    lineup_slot_id INTEGER NOT NULL,
    acquisition_type TEXT,
    acquisition_date TEXT,
    lineup_locked INTEGER NOT NULL DEFAULT 0 CHECK (lineup_locked IN (0, 1)),
    as_of TEXT NOT NULL,
    PRIMARY KEY (league_id, scoring_period_id, team_id, espn_id),
    FOREIGN KEY (league_id, team_id) REFERENCES teams (league_id, team_id) ON DELETE CASCADE
) STRICT;

CREATE INDEX roster_snapshots_player ON roster_snapshots (league_id, espn_id, scoring_period_id);

-- Stat lines per (player, source, kind, season, period); never points. scoring_period_id 0 is the full season.
CREATE TABLE projections (
    sport TEXT NOT NULL CHECK (sport IN ('nfl', 'nba')),
    espn_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'projected' CHECK (kind IN ('projected', 'actual')),
    season INTEGER NOT NULL,
    scoring_period_id INTEGER NOT NULL,
    stats TEXT NOT NULL CHECK (json_valid(stats)),
    as_of TEXT NOT NULL,
    PRIMARY KEY (sport, espn_id, source, kind, season, scoring_period_id)
) STRICT;

CREATE INDEX projections_period ON projections (sport, season, scoring_period_id, kind, source);

-- p_active per player and period, with the designation and the inputs behind it.
CREATE TABLE availability (
    sport TEXT NOT NULL CHECK (sport IN ('nfl', 'nba')),
    espn_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    scoring_period_id INTEGER NOT NULL,
    designation TEXT,
    p_active REAL NOT NULL CHECK (p_active BETWEEN 0 AND 1),
    has_game INTEGER NOT NULL DEFAULT 1 CHECK (has_game IN (0, 1)),
    game_time TEXT,
    inputs TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(inputs)),
    as_of TEXT NOT NULL,
    PRIMARY KEY (sport, espn_id, season, scoring_period_id)
) STRICT;

CREATE INDEX availability_period ON availability (sport, season, scoring_period_id);

-- Deduplicated news (ESPN news API, RotoWire RSS). triaged_at is set once the advisor has processed the item.
CREATE TABLE news_items (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    sport TEXT NOT NULL CHECK (sport IN ('nfl', 'nba')),
    title TEXT NOT NULL,
    body TEXT,
    url TEXT,
    espn_ids TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(espn_ids)),
    published_at TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    triaged_at TEXT,
    UNIQUE (source, external_id)
) STRICT;

CREATE INDEX news_items_published ON news_items (sport, published_at);
CREATE INDEX news_items_untriaged ON news_items (published_at) WHERE triaged_at IS NULL;

-- The advisor's structured reading of a news item for one player (DESIGN section 10, news_triage output).
CREATE TABLE news_signals (
    id INTEGER PRIMARY KEY,
    news_item_id INTEGER NOT NULL REFERENCES news_items (id) ON DELETE CASCADE,
    sport TEXT NOT NULL CHECK (sport IN ('nfl', 'nba')),
    espn_id INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('injury', 'role', 'rest', 'suspension', 'other')),
    severity TEXT NOT NULL,
    games_out INTEGER,
    p_active_delta REAL NOT NULL DEFAULT 0,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    summary TEXT,
    source_url TEXT,
    published_at TEXT NOT NULL,
    created_at TEXT NOT NULL
) STRICT;

CREATE INDEX news_signals_player ON news_signals (sport, espn_id, published_at);
CREATE INDEX news_signals_item ON news_signals (news_item_id);

-- Market values per source (FantasyCalc redraft values, ESPN rank and ownership trends); trade-acceptance input only.
CREATE TABLE market_values (
    sport TEXT NOT NULL CHECK (sport IN ('nfl', 'nba')),
    espn_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    value REAL,
    rank INTEGER,
    position_rank INTEGER,
    trend REAL,
    percent_owned REAL,
    percent_started REAL,
    details TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(details)),
    as_of TEXT NOT NULL,
    PRIMARY KEY (sport, espn_id, source)
) STRICT;

-- Proposals (DESIGN section 11). Status runs proposed -> approved | rejected | expired -> executing -> verified | failed.
-- execution_token is issued on approval and consumed exactly once (token_consumed_at).
CREATE TABLE proposals (
    id INTEGER PRIMARY KEY,
    league_id INTEGER NOT NULL REFERENCES leagues (id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed'
        CHECK (status IN ('proposed', 'approved', 'rejected', 'expired', 'executing', 'verified', 'failed')),
    policy TEXT NOT NULL CHECK (policy IN ('off', 'approve', 'auto')),
    scoring_period_id INTEGER,
    payload TEXT NOT NULL CHECK (json_valid(payload)),
    engine_numbers TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(engine_numbers)),
    rationale TEXT,
    deadline TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided_by TEXT,
    decided_at TEXT,
    execution_token TEXT UNIQUE,
    token_consumed_at TEXT,
    dedupe_key TEXT
) STRICT;

CREATE INDEX proposals_status ON proposals (league_id, status, deadline);
CREATE INDEX proposals_dedupe ON proposals (league_id, dedupe_key) WHERE dedupe_key IS NOT NULL;

-- Execution attempts (DESIGN section 6.3 write safety). `unknown` is a timeout: re-read state before anything else.
CREATE TABLE executions (
    id INTEGER PRIMARY KEY,
    proposal_id INTEGER NOT NULL REFERENCES proposals (id) ON DELETE CASCADE,
    mode TEXT NOT NULL CHECK (mode IN ('api', 'ui')),
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'verified', 'failed', 'unknown', 'dry_run')),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    request TEXT CHECK (request IS NULL OR json_valid(request)),
    response TEXT CHECK (response IS NULL OR json_valid(response)),
    error TEXT,
    verification TEXT CHECK (verification IS NULL OR json_valid(verification)),
    artifacts TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(artifacts)),
    espn_transaction_id TEXT
) STRICT;

CREATE INDEX executions_proposal ON executions (proposal_id, started_at);

-- Replayable decisions (DESIGN principle 5): inputs with as_of stamps, the decision, and later the outcome/metrics.
CREATE TABLE decision_evals (
    id INTEGER PRIMARY KEY,
    league_id INTEGER NOT NULL REFERENCES leagues (id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    season INTEGER NOT NULL,
    scoring_period_id INTEGER,
    proposal_id INTEGER REFERENCES proposals (id) ON DELETE SET NULL,
    decided_at TEXT NOT NULL,
    inputs TEXT NOT NULL CHECK (json_valid(inputs)),
    decision TEXT NOT NULL CHECK (json_valid(decision)),
    outcome TEXT CHECK (outcome IS NULL OR json_valid(outcome)),
    metrics TEXT CHECK (metrics IS NULL OR json_valid(metrics)),
    evaluated_at TEXT
) STRICT;

CREATE INDEX decision_evals_period ON decision_evals (league_id, season, scoring_period_id, kind);

-- Claude API calls, for the daily budget cap and cost reporting (DESIGN section 10).
CREATE TABLE llm_usage (
    id INTEGER PRIMARY KEY,
    called_at TEXT NOT NULL,
    worker TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    batch INTEGER NOT NULL DEFAULT 0 CHECK (batch IN (0, 1)),
    stop_reason TEXT,
    request_id TEXT,
    league_id INTEGER REFERENCES leagues (id) ON DELETE SET NULL
) STRICT;

CREATE INDEX llm_usage_called ON llm_usage (called_at);
