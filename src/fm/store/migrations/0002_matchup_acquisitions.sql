-- ESPN's per-matchup acquisition counter for each team (transactionCounter.matchupAcquisitionTotals, keyed by matchup
-- period id as JSON text). NULL until a sync has read it: the count of acquisitions then falls back to fm's proposals.
ALTER TABLE teams ADD COLUMN matchup_acquisitions TEXT;
