-- ===========================================================================
-- Embeddings Tables with pgvector and Categorical/Tier Indexing
-- ===========================================================================
-- This script creates the embedding storage tables with pgvector support
-- and "Walmart-style" categorical indexing for fast filtered retrieval.
-- Run this AFTER corrected_database_schema.sql

-- ---------------------------------------------------------------------------
-- Enable pgvector Extension
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Section Gists Table (Core Embeddings Storage)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS section_gists (
  id BIGSERIAL PRIMARY KEY,
  
  -- Document Identity
  doc_id TEXT NOT NULL,
  section TEXT NOT NULL,
  chunk_ids TEXT[],                    -- Array of chunk IDs that make up this section
  
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
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Document Metadata Table (for document-level categorization)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_metadata (
  doc_id TEXT PRIMARY KEY,
  
  -- Document Properties
  title TEXT,
  doc_type TEXT,                      -- "ValidationPlan", "MonitoringReport", etc.
  version TEXT,
  status TEXT,                        -- "Draft", "Final", "Archived"
  
  -- Categorical Organization
  business_unit TEXT,                 -- "QA-Americas", "QA-EMEA", etc.
  process_area TEXT,                  -- "DataQuality", "ModelRisk", "Compliance"
  risk_tier TEXT,                     -- "High", "Medium", "Low"
  lifecycle_stage TEXT,               -- Current stage in lifecycle
  
  -- Document Stats
  total_pages INT,
  total_sections INT,
  total_chunks INT,
  file_size_bytes BIGINT,
  
  -- S3 Storage
  s3_uri TEXT,                        -- Original PDF location
  processed_prefix TEXT,              -- S3 prefix for processed artifacts
  
  -- Timestamps
  document_date DATE,                 -- When document was created/effective
  ingested_at TIMESTAMPTZ DEFAULT now(),
  last_processed TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- Chunk Storage Table (for detailed chunk-level embeddings)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
  id BIGSERIAL PRIMARY KEY,
  
  -- Identity
  doc_id TEXT NOT NULL,
  chunk_id TEXT NOT NULL,             -- e.g., "3.2-05"
  section TEXT,                       -- Parent section
  page_num INT,
  
  -- Content
  chunk_text TEXT NOT NULL,
  embedding_model TEXT NOT NULL,
  embedding vector(1024),
  
  -- Metadata
  char_count INT,
  token_estimate INT,
  s3_uri TEXT,                        -- URI to chunk .md file
  content_hash TEXT,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT now(),
  
  UNIQUE(doc_id, chunk_id)
);

-- ---------------------------------------------------------------------------
-- Vector Indexes (Choose Based on Your Use Case)
-- ---------------------------------------------------------------------------

-- Option 1: IVF Flat (good for larger datasets, requires training)
-- Uncomment if you expect >10k vectors and want faster search
/*
CREATE INDEX CONCURRENTLY IF NOT EXISTS section_gists_embedding_ivf 
  ON section_gists USING ivfflat (embedding vector_cosine_ops) 
  WITH (lists = 100);

CREATE INDEX CONCURRENTLY IF NOT EXISTS document_chunks_embedding_ivf 
  ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
  WITH (lists = 100);
*/

-- Option 2: HNSW (great recall, works well for streaming inserts)
-- Recommended for most use cases
CREATE INDEX CONCURRENTLY IF NOT EXISTS section_gists_embedding_hnsw 
  ON section_gists USING hnsw (embedding vector_cosine_ops) 
  WITH (m = 16, ef_construction = 200);

CREATE INDEX CONCURRENTLY IF NOT EXISTS document_chunks_embedding_hnsw 
  ON document_chunks USING hnsw (embedding vector_cosine_ops) 
  WITH (m = 16, ef_construction = 200);

-- ---------------------------------------------------------------------------
-- "Walmart-Style" Metadata Indexes for Fast Pre-filtering
-- ---------------------------------------------------------------------------

-- Section Gists Categorical Indexes
CREATE INDEX IF NOT EXISTS section_gists_doc_id ON section_gists (doc_id);
CREATE INDEX IF NOT EXISTS section_gists_lifecycle_stage ON section_gists (lifecycle, stage);
CREATE INDEX IF NOT EXISTS section_gists_risk_tier ON section_gists (risk_tier);
CREATE INDEX IF NOT EXISTS section_gists_control_owner ON section_gists (control_owner);
CREATE INDEX IF NOT EXISTS section_gists_doc_type_year ON section_gists (doc_type, year);
CREATE INDEX IF NOT EXISTS section_gists_embedding_model ON section_gists (embedding_model);

-- GIN index for flexible tag searching
CREATE INDEX IF NOT EXISTS section_gists_tags_gin ON section_gists USING GIN (tags);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS section_gists_tier_lifecycle_stage 
  ON section_gists (risk_tier, lifecycle, stage);
CREATE INDEX IF NOT EXISTS section_gists_doc_type_tier_year 
  ON section_gists (doc_type, risk_tier, year);

-- Document Metadata Indexes
CREATE INDEX IF NOT EXISTS document_metadata_business_unit ON document_metadata (business_unit);
CREATE INDEX IF NOT EXISTS document_metadata_process_area ON document_metadata (process_area);
CREATE INDEX IF NOT EXISTS document_metadata_risk_tier ON document_metadata (risk_tier);
CREATE INDEX IF NOT EXISTS document_metadata_doc_type ON document_metadata (doc_type);
CREATE INDEX IF NOT EXISTS document_metadata_document_date ON document_metadata (document_date);

-- Chunk Indexes
CREATE INDEX IF NOT EXISTS document_chunks_doc_section ON document_chunks (doc_id, section);
CREATE INDEX IF NOT EXISTS document_chunks_embedding_model ON document_chunks (embedding_model);

-- Content hash indexes for deduplication
CREATE INDEX IF NOT EXISTS section_gists_content_hash ON section_gists (content_hash);
CREATE INDEX IF NOT EXISTS document_chunks_content_hash ON document_chunks (content_hash);

-- ---------------------------------------------------------------------------
-- Hybrid Query Functions (Filter-then-Vector)
-- ---------------------------------------------------------------------------

-- Function: Smart section search with pre-filtering
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

-- Function: Get documents by category with stats
CREATE OR REPLACE FUNCTION get_documents_by_category(
  p_business_unit TEXT DEFAULT NULL,
  p_process_area TEXT DEFAULT NULL,
  p_risk_tier TEXT DEFAULT NULL,
  p_doc_type TEXT DEFAULT NULL
)
RETURNS TABLE (
  doc_id TEXT,
  title TEXT,
  doc_type TEXT,
  risk_tier TEXT,
  total_sections BIGINT,
  latest_processed TIMESTAMPTZ
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    dm.doc_id,
    dm.title,
    dm.doc_type,
    dm.risk_tier,
    COUNT(sg.id) as total_sections,
    MAX(sg.created_at) as latest_processed
  FROM document_metadata dm
  LEFT JOIN section_gists sg ON dm.doc_id = sg.doc_id
  WHERE (p_business_unit IS NULL OR dm.business_unit = p_business_unit)
    AND (p_process_area IS NULL OR dm.process_area = p_process_area)
    AND (p_risk_tier IS NULL OR dm.risk_tier = p_risk_tier)
    AND (p_doc_type IS NULL OR dm.doc_type = p_doc_type)
  GROUP BY dm.doc_id, dm.title, dm.doc_type, dm.risk_tier
  ORDER BY total_sections DESC;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------------
-- Auto-update Triggers
-- ---------------------------------------------------------------------------

-- Trigger to update updated_at timestamp
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
-- Sample Query Patterns for Demo
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

-- Example 2: Find all data quality related sections for 2024
/*
SELECT * FROM section_gists 
WHERE tags @> ARRAY['DQ'] 
  AND year = 2024 
  AND risk_tier = 'High'
ORDER BY created_at DESC;
*/

-- Example 3: Get document stats by business unit
/*
SELECT * FROM get_documents_by_category(
  p_business_unit := 'QA-Americas',
  p_risk_tier := 'High'
);
*/

-- ---------------------------------------------------------------------------
-- Performance Optimization Notes
-- ---------------------------------------------------------------------------

-- For best performance:
-- 1. Always filter by categorical fields BEFORE vector search
-- 2. Use appropriate vector index (HNSW for most cases, IVF for large datasets)
-- 3. Keep embedding dimensions consistent within each model type
-- 4. Use content_hash for deduplication to avoid redundant embeddings
-- 5. Consider partitioning section_gists by year if you have years of data

-- Example optimized query pattern:
-- 1. WHERE risk_tier = 'High' AND lifecycle = 'Validation'  (uses BTREE index)
-- 2. ORDER BY embedding <=> query_vector (uses vector index on filtered set)
-- 3. LIMIT N (stops after finding enough results)

-- ---------------------------------------------------------------------------
-- Validation
-- ---------------------------------------------------------------------------

-- Verify embeddings tables were created
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
  
  -- Check tables
  SELECT COUNT(*) INTO table_count
  FROM information_schema.tables 
  WHERE table_schema = 'public' 
    AND table_name IN ('section_gists', 'document_metadata', 'document_chunks');
  
  IF table_count = 3 THEN
    RAISE NOTICE 'Embeddings schema setup complete! All 3 embedding tables created.';
  ELSE
    RAISE WARNING 'Embeddings schema incomplete. Expected 3 tables, found %', table_count;
  END IF;
END $$;

-- ---------------------------------------------------------------------------
-- End of Embeddings Schema
-- ---------------------------------------------------------------------------