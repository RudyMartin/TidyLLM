-- =============================================================================
-- 04_event_tracking.sql - Event Tracking and Analytics
-- =============================================================================
-- This script creates event tracking tables for analytics
-- Run after 03_embeddings_system.sql

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

-- Indexes for performance
CREATE INDEX IF NOT EXISTS events_raw_ts ON events_raw(ts);
CREATE INDEX IF NOT EXISTS events_raw_event ON events_raw(event);
CREATE INDEX IF NOT EXISTS events_raw_user ON events_raw(user_id);
CREATE INDEX IF NOT EXISTS events_raw_org_team ON events_raw(org, team);

CREATE INDEX IF NOT EXISTS events_daily_day ON events_daily(day);
CREATE INDEX IF NOT EXISTS events_daily_metric ON events_daily(metric);
CREATE INDEX IF NOT EXISTS events_daily_org_team ON events_daily(org, team);

-- Verify setup
SELECT '✅ Event tracking setup complete!' as status;
SELECT COUNT(*) as events_raw_count FROM events_raw;
SELECT COUNT(*) as events_daily_count FROM events_daily;
