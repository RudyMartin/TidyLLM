-- ===========================================================================
-- QA Validation Review - Complete Database Schema (Stage 2) - CORRECTED
-- ===========================================================================
-- This script creates all required tables, indexes, views, and functions
-- for the Hierarchical LLM QA Validation Review system.
-- Run this on a clean Postgres database or existing one (idempotent).

-- ---------------------------------------------------------------------------
-- Main Tables
-- ---------------------------------------------------------------------------

-- Core review runs table with all Stage 1 and Stage 2 fields
CREATE TABLE IF NOT EXISTS review_runs (
  id BIGSERIAL PRIMARY KEY,
  
  -- Identity & Organization
  reviewer TEXT,
  org TEXT,
  team TEXT,
  process TEXT,
  doc_id TEXT NOT NULL,
  review_date DATE NOT NULL,
  
  -- Audit & Provenance
  ruleset_hash TEXT NOT NULL,
  report_s3_uri TEXT NOT NULL,
  
  -- Stage 1 - Basic Metrics
  runtime_sec INT,
  chunk_count INT,
  finding_count INT,
  tokens_input INT,
  tokens_output INT,
  model_calls JSONB,
  
  -- Stage 2 - Enhanced Analytics
  pipeline_version TEXT,
  doc_pages INT,
  doc_bytes BIGINT,
  input_hash TEXT,
  prompt_hash TEXT,
  model_primary TEXT,
  latency_p50_ms INT,
  latency_p95_ms INT,
  cost_usd NUMERIC(10,4),
  error_count INT,
  timeout_count INT,
  retry_count INT,
  evidence_count INT,
  severity_counts JSONB,
  consensus_status TEXT,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Individual findings for each review run
CREATE TABLE IF NOT EXISTS review_findings (
  run_id BIGINT REFERENCES review_runs(id) ON DELETE CASCADE,
  rule_id TEXT NOT NULL,
  stage TEXT,
  severity TEXT,
  status TEXT,
  evidence JSONB,
  rule_source TEXT,
  PRIMARY KEY (run_id, rule_id)
);

-- ---------------------------------------------------------------------------
-- Event Tracking Tables
-- ---------------------------------------------------------------------------

-- Raw events for detailed tracking
CREATE TABLE IF NOT EXISTS events_raw (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  user_id TEXT,
  org TEXT,
  team TEXT,
  process TEXT,
  event TEXT NOT NULL,    -- e.g., auth_login, session_start, page_view, config_change,
                           -- validation_error, app_error, scoring_update, run_started, run_finished
  payload JSONB           -- free-form data, avoid PII
);

-- Daily aggregated metrics for fast reporting
CREATE TABLE IF NOT EXISTS events_daily (
  day DATE NOT NULL,
  org TEXT,
  team TEXT,
  process TEXT,
  metric TEXT NOT NULL,   -- e.g., active_users, page_views, runs, burgers, errors
  value BIGINT NOT NULL,
  PRIMARY KEY(day, org, team, process, metric)
);

-- ---------------------------------------------------------------------------
-- Indexes for Performance
-- ---------------------------------------------------------------------------

-- Review runs indexes
CREATE INDEX IF NOT EXISTS review_runs_date ON review_runs(review_date);
CREATE INDEX IF NOT EXISTS review_runs_model ON review_runs(model_primary);
CREATE INDEX IF NOT EXISTS review_runs_ruleset ON review_runs(ruleset_hash);
CREATE INDEX IF NOT EXISTS review_runs_org ON review_runs(org);
CREATE INDEX IF NOT EXISTS review_runs_org_team ON review_runs(org, team);
CREATE INDEX IF NOT EXISTS review_runs_org_team_proc_date ON review_runs(org, team, process, review_date);
CREATE INDEX IF NOT EXISTS review_runs_reviewer ON review_runs(reviewer);
CREATE INDEX IF NOT EXISTS review_runs_doc_id ON review_runs(doc_id);

-- Events indexes
CREATE INDEX IF NOT EXISTS events_raw_ts ON events_raw(ts);
CREATE INDEX IF NOT EXISTS events_raw_keys ON events_raw(org, team, process, event);
CREATE INDEX IF NOT EXISTS events_raw_user_event ON events_raw(user_id, event);
CREATE INDEX IF NOT EXISTS events_daily_day ON events_daily(day);
CREATE INDEX IF NOT EXISTS events_daily_org_team ON events_daily(org, team);

-- Findings indexes
CREATE INDEX IF NOT EXISTS review_findings_rule_id ON review_findings(rule_id);
CREATE INDEX IF NOT EXISTS review_findings_run_id ON review_findings(run_id);
CREATE INDEX IF NOT EXISTS review_findings_severity ON review_findings(severity);
CREATE INDEX IF NOT EXISTS review_findings_status ON review_findings(status);

-- Additional performance indexes
CREATE INDEX IF NOT EXISTS events_daily_org_team_proc_day ON events_daily(org, team, process, day);

-- ---------------------------------------------------------------------------
-- Views for Convenience
-- ---------------------------------------------------------------------------

-- Simple view for total burgers served
CREATE OR REPLACE VIEW burgers_served AS
SELECT COALESCE(SUM(finding_count), 0) AS total_burgers 
FROM review_runs;

-- View for latest runs with key metrics
CREATE OR REPLACE VIEW latest_runs AS
SELECT 
  id,
  reviewer,
  org,
  team,
  process,
  doc_id,
  review_date,
  finding_count,
  runtime_sec,
  cost_usd,
  model_primary,
  ruleset_hash,
  created_at,
  ROW_NUMBER() OVER (PARTITION BY org, team, process ORDER BY created_at DESC) as rn
FROM review_runs
ORDER BY created_at DESC;

-- View for daily summary stats
CREATE OR REPLACE VIEW daily_summary AS
SELECT 
  review_date,
  org,
  team,
  process,
  COUNT(*) as runs_count,
  SUM(finding_count) as total_findings,
  AVG(runtime_sec) as avg_runtime_sec,
  SUM(cost_usd) as total_cost,
  SUM(tokens_input + tokens_output) as total_tokens,
  COUNT(DISTINCT reviewer) as unique_reviewers
FROM review_runs
GROUP BY review_date, org, team, process
ORDER BY review_date DESC;

-- ---------------------------------------------------------------------------
-- Functions (CORRECTED - Using $$ delimiters)
-- ---------------------------------------------------------------------------

-- Daily rollup function for events
CREATE OR REPLACE FUNCTION rollup_events_daily(p_start DATE, p_end DATE)
RETURNS VOID AS $$
BEGIN
  -- Active users (distinct user_id with any event)
  INSERT INTO events_daily(day, org, team, process, metric, value)
  SELECT 
    date(ts), 
    org, 
    team, 
    process, 
    'active_users', 
    COUNT(DISTINCT user_id)
  FROM events_raw
  WHERE date(ts) BETWEEN p_start AND p_end
  GROUP BY date(ts), org, team, process
  ON CONFLICT (day, org, team, process, metric) 
  DO UPDATE SET value = EXCLUDED.value;

  -- Page views
  INSERT INTO events_daily(day, org, team, process, metric, value)
  SELECT 
    date(ts), 
    org, 
    team, 
    process, 
    'page_views', 
    COUNT(*)
  FROM events_raw
  WHERE event = 'page_view' AND date(ts) BETWEEN p_start AND p_end
  GROUP BY date(ts), org, team, process
  ON CONFLICT (day, org, team, process, metric) 
  DO UPDATE SET value = EXCLUDED.value;

  -- Runs recorded (from review_runs table)
  INSERT INTO events_daily(day, org, team, process, metric, value)
  SELECT 
    review_date, 
    org, 
    team, 
    process, 
    'runs', 
    COUNT(*)
  FROM review_runs
  WHERE review_date BETWEEN p_start AND p_end
  GROUP BY review_date, org, team, process
  ON CONFLICT (day, org, team, process, metric) 
  DO UPDATE SET value = EXCLUDED.value;

  -- Burgers served per day (sum findings)
  INSERT INTO events_daily(day, org, team, process, metric, value)
  SELECT 
    review_date, 
    org, 
    team, 
    process, 
    'burgers', 
    COALESCE(SUM(finding_count), 0)
  FROM review_runs
  WHERE review_date BETWEEN p_start AND p_end
  GROUP BY review_date, org, team, process
  ON CONFLICT (day, org, team, process, metric) 
  DO UPDATE SET value = EXCLUDED.value;

  -- Error events
  INSERT INTO events_daily(day, org, team, process, metric, value)
  SELECT 
    date(ts), 
    org, 
    team, 
    process, 
    'errors', 
    COUNT(*)
  FROM events_raw
  WHERE event IN ('validation_error', 'app_error') AND date(ts) BETWEEN p_start AND p_end
  GROUP BY date(ts), org, team, process
  ON CONFLICT (day, org, team, process, metric) 
  DO UPDATE SET value = EXCLUDED.value;

  -- Run completion events
  INSERT INTO events_daily(day, org, team, process, metric, value)
  SELECT 
    date(ts), 
    org, 
    team, 
    process, 
    'completed_runs', 
    COUNT(*)
  FROM events_raw
  WHERE event = 'run_finished' AND date(ts) BETWEEN p_start AND p_end
  GROUP BY date(ts), org, team, process
  ON CONFLICT (day, org, team, process, metric) 
  DO UPDATE SET value = EXCLUDED.value;

END; 
$$ LANGUAGE plpgsql;

-- Function to get burger count with filters
CREATE OR REPLACE FUNCTION get_burger_count(
  p_org TEXT DEFAULT NULL,
  p_team TEXT DEFAULT NULL,
  p_process TEXT DEFAULT NULL,
  p_start_date DATE DEFAULT NULL,
  p_end_date DATE DEFAULT NULL
)
RETURNS BIGINT AS $$
DECLARE
  result BIGINT;
BEGIN
  SELECT COALESCE(SUM(finding_count), 0)
  INTO result
  FROM review_runs
  WHERE (p_org IS NULL OR org = p_org)
    AND (p_team IS NULL OR team = p_team)
    AND (p_process IS NULL OR process = p_process)
    AND (p_start_date IS NULL OR review_date >= p_start_date)
    AND (p_end_date IS NULL OR review_date <= p_end_date);
  
  RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old events (for maintenance)
CREATE OR REPLACE FUNCTION cleanup_old_events(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
  deleted_count INTEGER;
BEGIN
  DELETE FROM events_raw 
  WHERE ts < (CURRENT_DATE - INTERVAL '1 day' * days_to_keep);
  
  GET DIAGNOSTICS deleted_count = ROW_COUNT;
  RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------------
-- Triggers (Optional - for automatic maintenance)
-- ---------------------------------------------------------------------------

-- Trigger to automatically update severity counts when findings are inserted
CREATE OR REPLACE FUNCTION update_severity_counts()
RETURNS TRIGGER AS $$
DECLARE
  severity_json JSONB;
BEGIN
  -- Recalculate severity counts for the run
  SELECT jsonb_object_agg(severity, count)
  INTO severity_json
  FROM (
    SELECT 
      COALESCE(severity, 'Unknown') as severity,
      COUNT(*) as count
    FROM review_findings 
    WHERE run_id = COALESCE(NEW.run_id, OLD.run_id)
    GROUP BY severity
  ) counts;
  
  -- Update the review_runs table
  UPDATE review_runs 
  SET severity_counts = COALESCE(severity_json, '{}'::jsonb)
  WHERE id = COALESCE(NEW.run_id, OLD.run_id);
  
  RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create trigger (commented out by default - enable if desired)
-- DROP TRIGGER IF EXISTS trg_update_severity_counts ON review_findings;
-- CREATE TRIGGER trg_update_severity_counts
--   AFTER INSERT OR UPDATE OR DELETE ON review_findings
--   FOR EACH ROW EXECUTE FUNCTION update_severity_counts();

-- ---------------------------------------------------------------------------
-- Data Integrity Constraints
-- ---------------------------------------------------------------------------

-- Optional constraints for data quality
ALTER TABLE review_findings
  ADD CONSTRAINT IF NOT EXISTS chk_severity CHECK (severity IN ('High','Medium','Low'));
ALTER TABLE review_findings
  ADD CONSTRAINT IF NOT EXISTS chk_status   CHECK (status   IN ('PASS','FAIL','WARNING'));

-- Optional JSON GIN indexes for better performance on JSON queries
CREATE INDEX IF NOT EXISTS review_runs_model_calls_gin
  ON review_runs USING GIN (model_calls);
CREATE INDEX IF NOT EXISTS review_runs_severity_counts_gin
  ON review_runs USING GIN (severity_counts);

-- ---------------------------------------------------------------------------
-- Validation & Testing
-- ---------------------------------------------------------------------------

-- Verify tables were created successfully
DO $$
DECLARE
  table_count INTEGER;
BEGIN
  SELECT COUNT(*) INTO table_count
  FROM information_schema.tables 
  WHERE table_schema = 'public' 
    AND table_name IN ('review_runs', 'review_findings', 'events_raw', 'events_daily');
  
  IF table_count = 4 THEN
    RAISE NOTICE 'Schema setup complete! All 4 tables created successfully.';
  ELSE
    RAISE WARNING 'Schema setup incomplete. Expected 4 tables, found %', table_count;
  END IF;
END $$;

-- Check functions compile correctly
SELECT 
  proname,
  CASE WHEN proname IS NOT NULL THEN 'OK' ELSE 'MISSING' END as status
FROM (VALUES 
  ('rollup_events_daily'),
  ('get_burger_count'),
  ('cleanup_old_events'),
  ('update_severity_counts')
) AS expected(fname)
LEFT JOIN pg_proc ON proname = fname;

-- ---------------------------------------------------------------------------
-- Sample Data for Testing (Optional)
-- ---------------------------------------------------------------------------

-- Uncomment to insert sample data:
/*
INSERT INTO review_runs (reviewer, org, team, process, doc_id, review_date, ruleset_hash, report_s3_uri, finding_count, runtime_sec)
VALUES 
  ('alex', 'QA-Americas', 'QA-East', 'QAValidationReview', 'WF-2025-001', '2025-01-15', 'abc123', 's3://bucket/qa_east/alex/report1.md', 5, 120),
  ('maria', 'QA-Americas', 'QA-West', 'QAValidationReview', 'WF-2025-002', '2025-01-16', 'def456', 's3://bucket/qa_west/maria/report2.md', 3, 95),
  ('ravi', 'QA-EMEA', 'QA-UK', 'QAValidationReview', 'WF-2025-003', '2025-01-17', 'ghi789', 's3://bucket/qa_uk/ravi/report3.md', 7, 150)
ON CONFLICT DO NOTHING;
*/

-- ---------------------------------------------------------------------------
-- Maintenance Tasks
-- ---------------------------------------------------------------------------

-- To roll up events for the last 7 days:
-- SELECT rollup_events_daily(CURRENT_DATE - 7, CURRENT_DATE);

-- To clean up events older than 90 days:
-- SELECT cleanup_old_events(90);

-- To get burger count for a specific team:
-- SELECT get_burger_count('QA-Americas', 'QA-East', 'QAValidationReview');

-- ---------------------------------------------------------------------------
-- End of Schema
-- ---------------------------------------------------------------------------