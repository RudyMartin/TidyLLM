-- =============================================================================
-- 01_extensions.sql - PostgreSQL Extensions Setup
-- =============================================================================
-- This script enables required PostgreSQL extensions
-- Run this first before any other scripts

-- Enable pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Verify extensions are installed
SELECT '✅ Extensions setup complete!' as status;
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp');
