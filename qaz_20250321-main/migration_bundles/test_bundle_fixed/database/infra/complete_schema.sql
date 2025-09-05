-- ===========================================================================
-- Complete Database Schema - Review System + Embeddings + Analytics
-- ===========================================================================
-- This script creates the complete database for hierarchical LLM system
-- Combines review runs, embeddings, and event tracking

-- ---------------------------------------------------------------------------
-- Enable Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector for embeddings

-- ---------------------------------------------------------------------------
-- Core Review Tables
-- ---------------------------------------------------------------------------

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

-- ---------------------------------------------------------------------------
-- Embeddings Tables (pgvector + categorical indexing)
-- ---------------------------------------------------------------------------

-- Section embeddings with "Walmart-style" categorical filters
CREATE TABLE IF NOT EXISTS section_gists (
  id BIGSERIAL PRIMARY KEY,
  
  -- Document Identity
  doc_id TEXT NOT NULL,
  section TEXT NOT NULL,
  chunk_ids TEXT[],
  
  -- "Walmart-style" Categorical Filters
  lifecycle TEXT,                      -- "Development", "Validation", "Monitoring"
  stage TEXT,                          -- "Gate1", "Gate2", "Gate3", "Deployment"
  control_owner TEXT,                  -- Named owner/responsible party
  risk_tier TEXT,                      -- "High", "Medium", "Low"
  doc_type TEXT,                       -- "Plan", "Validation", "Monitoring", "Report"
  year INT,                           -- Document year for temporal filtering
  tags TEXT[],                        -- Flexible tagging: ["DQ", "lineage", "thresholds"]
  
  -- Embedding & Content
  embedding_model TEXT NOT NULL,       -- "titan_v2_1024", "cohere_v3", etc.
  embedding vector(1024),              -- pgvector embedding (adjust dimension as needed)
  gist TEXT NOT NULL,                 -- Compressed section summary (~120 tokens)
  
  -- Evidence & Provenance
  evidence_refs TEXT[],               -- S3 URIs to source PDF pages/sections
  s3_uri TEXT,                        -- URI to the .md artifact in S3
  content_hash TEXT,                  -- SHA-1 of the gist content for deduplication
  
  -- Audit Trail
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  
  UNIQUE(doc_id, section)
);

-- Document metadata table
CREATE TABLE IF NOT EXISTS document_metadata (
  doc_id TEXT PRIMARY KEY,
  
  -- Document Properties
  title TEXT,
  doc_type TEXT,
  version TEXT,
  status TEXT,
  
  -- Categorical Organization
  business_unit TEXT,
  process_area TEXT,
  risk_tier TEXT,
  lifecycle_stage TEXT,
  
  -- Document Stats
  total_pages INT,
  total_sections INT,
  total_chunks INT,
  file_size_bytes BIGINT,
  
  -- S3 Storage
  s3_uri TEXT,
  processed_prefix TEXT,
  
  -- Timestamps
  document_date DATE,
  ingested_at TIMESTAMPTZ DEFAULT now(),
  last_processed TIMESTAMPTZ
);

-- Individual chunk embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
  id BIGSERIAL PRIMARY KEY,
  doc_id TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  section TEXT,
  page_num INT,
  chunk_text TEXT NOT NULL,
  embedding_model TEXT NOT NULL,
  embedding vector(1024),
  char_count INT,
  token_estimate INT,
  s3_uri TEXT,
  content_hash TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(doc_id, chunk_id)
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
  event TEXT NOT NULL,
  payload JSONB
);

-- Daily aggregated metrics for fast reporting
CREATE TABLE IF NOT EXISTS events_daily (
  day DATE NOT NULL,
  org TEXT,
  team TEXT,
  process TEXT,
  metric TEXT NOT NULL,
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

-- Review findings indexes
CREATE INDEX IF NOT EXISTS review_findings_rule_id ON review_findings(rule_id);
CREATE INDEX IF NOT EXISTS review_findings_run_id ON review_findings(run_id);
CREATE INDEX IF NOT EXISTS review_findings_severity ON review_findings(severity);
CREATE INDEX IF NOT EXISTS review_findings_status ON review_findings(status);

-- Vector indexes for embeddings (HNSW - good for most use cases)
CREATE INDEX CONCURRENTLY IF NOT EXISTS section_gists_embedding_hnsw 
  ON section_gists USING hnsw (embedding vector_cosine_ops) 
  WITH (m = 16, ef_construction = 200);

CREATE INDEX CONCURRENTLY IF NOT EXISTS document_chunks_embedding_hnsw 
  ON document_chunks USING hnsw (embedding vector_cosine_ops) 
  WITH (m = 16, ef_construction = 200);

-- "Walmart-Style" metadata indexes for fast pre-filtering
CREATE INDEX IF NOT EXISTS section_gists_doc_id ON section_gists (doc_id);
CREATE INDEX IF NOT EXISTS section_gists_lifecycle_stage ON section_gists (lifecycle, stage);
CREATE INDEX IF NOT EXISTS section_gists_risk_tier ON section_gists (risk_tier);
CREATE INDEX IF NOT EXISTS section_gists_control_owner ON section_gists (control_owner);
CREATE INDEX IF NOT EXISTS section_gists_doc_type_year ON section_gists (doc_type, year);
CREATE INDEX IF NOT EXISTS section_gists_embedding_model ON section_gists (embedding_model);
CREATE INDEX IF NOT EXISTS section_gists_tags_gin ON section_gists USING GIN (tags);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS section_gists_tier_lifecycle_stage 
  ON section_gists (risk_tier, lifecycle, stage);
CREATE INDEX IF NOT EXISTS section_gists_doc_type_tier_year 
  ON section_gists (doc_type, risk_tier, year);

-- Document metadata indexes
CREATE INDEX IF NOT EXISTS document_metadata_business_unit ON document_metadata (business_unit);
CREATE INDEX IF NOT EXISTS document_metadata_process_area ON document_metadata (process_area);
CREATE INDEX IF NOT EXISTS document_metadata_risk_tier ON document_metadata (risk_tier);
CREATE INDEX IF NOT EXISTS document_metadata_doc_type ON document_metadata (doc_type);
CREATE INDEX IF NOT EXISTS document_metadata_document_date ON document_metadata (document_date);

-- Chunk indexes
CREATE INDEX IF NOT EXISTS document_chunks_doc_section ON document_chunks (doc_id, section);
CREATE INDEX IF NOT EXISTS document_chunks_embedding_model ON document_chunks (embedding_model);

-- Content hash indexes for deduplication
CREATE INDEX IF NOT EXISTS section_gists_content_hash ON section_gists (content_hash);
CREATE INDEX IF NOT EXISTS document_chunks_content_hash ON document_chunks (content_hash);

-- Events indexes
CREATE INDEX IF NOT EXISTS events_raw_ts ON events_raw(ts);
CREATE INDEX IF NOT EXISTS events_raw_keys ON events_raw(org, team, process, event);
CREATE INDEX IF NOT EXISTS events_raw_user_event ON events_raw(user_id, event);
CREATE INDEX IF NOT EXISTS events_daily_day ON events_daily(day);
CREATE INDEX IF NOT EXISTS events_daily_org_team ON events_daily(org, team);
CREATE INDEX IF NOT EXISTS events_daily_org_team_proc_day ON events_daily(org, team, process, day);

-- JSON GIN indexes for better performance on JSON queries
CREATE INDEX IF NOT EXISTS review_runs_model_calls_gin
  ON review_runs USING GIN (model_calls);
CREATE INDEX IF NOT EXISTS review_runs_severity_counts_gin
  ON review_runs USING GIN (severity_counts);

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
-- Functions
-- ---------------------------------------------------------------------------

-- Hybrid search function for embeddings
CREATE OR REPLACE FUNCTION search_sections(
  query_embedding vector(1024),
  p_lifecycle TEXT DEFAULT NULL,
  p_stage TEXT DEFAULT NULL,
  p_risk_tier TEXT DEFAULT NULL,
  p_doc_type TEXT DEFAULT NULL,
  p_year INT DEFAULT NULL,
  p_tags TEXT[] DEFAULT NULL,
  p_limit INT DEFAULT 10
)
RETURNS TABLE (
  id BIGINT,
  doc_id TEXT,
  section TEXT,
  lifecycle TEXT,
  stage TEXT,
  risk_tier TEXT,
  similarity FLOAT,
  gist TEXT,
  s3_uri TEXT
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    sg.id,
    sg.doc_id,
    sg.section,
    sg.lifecycle,
    sg.stage,
    sg.risk_tier,
    1 - (sg.embedding <=> query_embedding) AS similarity,
    sg.gist,
    sg.s3_uri
  FROM section_gists sg
  WHERE (p_lifecycle IS NULL OR sg.lifecycle = p_lifecycle)
    AND (p_stage IS NULL OR sg.stage = p_stage)
    AND (p_risk_tier IS NULL OR sg.risk_tier = p_risk_tier)
    AND (p_doc_type IS NULL OR sg.doc_type = p_doc_type)
    AND (p_year IS NULL OR sg.year = p_year)
    AND (p_tags IS NULL OR sg.tags && p_tags)  -- Array overlap
    AND sg.embedding IS NOT NULL
  ORDER BY sg.embedding <=> query_embedding
  LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

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

-- Auto-update trigger for section_gists
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_section_gists_updated_at 
  BEFORE UPDATE ON section_gists 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ---------------------------------------------------------------------------
-- Validation & Setup Complete Message
-- ---------------------------------------------------------------------------

-- Verify schema setup
DO $$
DECLARE
  table_count INTEGER;
  extension_exists BOOLEAN;
BEGIN
  -- Check pgvector extension
  SELECT EXISTS(
    SELECT 1 FROM pg_extension WHERE extname = 'vector'
  ) INTO extension_exists;
  
  IF NOT extension_exists THEN
    RAISE WARNING 'pgvector extension not found. Run: CREATE EXTENSION vector;';
  END IF;
  
  -- Check all tables
  SELECT COUNT(*) INTO table_count
  FROM information_schema.tables 
  WHERE table_schema = 'public' 
    AND table_name IN (
      'review_runs', 'review_findings', 'events_raw', 'events_daily',
      'section_gists', 'document_metadata', 'document_chunks'
    );
  
  IF table_count = 7 THEN
    RAISE NOTICE '✅ Complete schema setup successful! All 7 tables created.';
    RAISE NOTICE '📊 Review System: review_runs, review_findings, events_raw, events_daily';
    RAISE NOTICE '🔍 Embeddings: section_gists, document_metadata, document_chunks';
    RAISE NOTICE '⚡ Ready for hierarchical LLM operations!';
  ELSE
    RAISE WARNING '❌ Schema incomplete. Expected 7 tables, found %', table_count;
  END IF;
END $$;

-- ---------------------------------------------------------------------------
-- Usage Examples (commented)
-- ---------------------------------------------------------------------------

-- Example 1: Search high-risk Gate2 validation sections
/*
SELECT * FROM search_sections(
  '[0.1,0.2,0.3,...]'::vector(1024),  -- Your query embedding
  p_lifecycle := 'Validation',
  p_stage := 'Gate2', 
  p_risk_tier := 'High',
  p_limit := 5
);
*/

-- Example 2: Get burger count for a team
/*
SELECT get_burger_count('QA-Americas', 'QA-East', 'QAValidationReview');
*/

-- Example 3: Daily rollup for the last 7 days
/*
SELECT rollup_events_daily(CURRENT_DATE - 7, CURRENT_DATE);
*/

-- ---------------------------------------------------------------------------
-- End of Complete Schema
-- ---------------------------------------------------------------------------