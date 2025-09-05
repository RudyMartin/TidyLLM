-- =============================================================================
-- 03_embeddings_system.sql - Embeddings and Vector Storage
-- =============================================================================
-- This script creates the embeddings system with pgvector support
-- Run after 02_review_system.sql

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

-- Indexes for performance
CREATE INDEX IF NOT EXISTS section_gists_doc_id ON section_gists(doc_id);
CREATE INDEX IF NOT EXISTS section_gists_lifecycle ON section_gists(lifecycle);
CREATE INDEX IF NOT EXISTS section_gists_stage ON section_gists(stage);
CREATE INDEX IF NOT EXISTS section_gists_risk_tier ON section_gists(risk_tier);
CREATE INDEX IF NOT EXISTS section_gists_year ON section_gists(year);
CREATE INDEX IF NOT EXISTS section_gists_embedding_model ON section_gists(embedding_model);

CREATE INDEX IF NOT EXISTS document_chunks_doc_id ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS document_chunks_section ON document_chunks(section);
CREATE INDEX IF NOT EXISTS document_chunks_embedding_model ON document_chunks(embedding_model);

-- Vector similarity search indexes (HNSW for fast approximate search)
CREATE INDEX IF NOT EXISTS section_gists_embedding_idx ON section_gists USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Verify setup
SELECT '✅ Embeddings system setup complete!' as status;
SELECT COUNT(*) as section_gists_count FROM section_gists;
SELECT COUNT(*) as document_metadata_count FROM document_metadata;
SELECT COUNT(*) as document_chunks_count FROM document_chunks;
