-- =============================================================================
-- 02_review_system.sql - Review System Tables
-- =============================================================================
-- This script creates the review system tables for QA validation
-- Run after 01_extensions.sql

-- Main review runs table with all metrics and audit fields
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
  
  -- Basic Metrics
  runtime_sec INT,
  chunk_count INT,
  finding_count INT,
  tokens_input INT,
  tokens_output INT,
  model_calls JSONB,
  
  -- Enhanced Analytics
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

-- Indexes for performance
CREATE INDEX IF NOT EXISTS review_runs_date ON review_runs(review_date);
CREATE INDEX IF NOT EXISTS review_runs_model ON review_runs(model_primary);
CREATE INDEX IF NOT EXISTS review_runs_ruleset ON review_runs(ruleset_hash);
CREATE INDEX IF NOT EXISTS review_runs_org ON review_runs(org);
CREATE INDEX IF NOT EXISTS review_runs_org_team ON review_runs(org, team);
CREATE INDEX IF NOT EXISTS review_runs_org_team_proc_date ON review_runs(org, team, process, review_date);
CREATE INDEX IF NOT EXISTS review_runs_reviewer ON review_runs(reviewer);

-- Verify setup
SELECT '✅ Review system setup complete!' as status;
SELECT COUNT(*) as review_runs_count FROM review_runs;
SELECT COUNT(*) as review_findings_count FROM review_findings;
