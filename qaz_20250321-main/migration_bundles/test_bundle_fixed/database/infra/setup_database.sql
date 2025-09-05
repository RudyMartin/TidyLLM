-- ===========================================================================
-- Complete Database Setup Script
-- ===========================================================================
-- This script sets up the complete database schema in the correct order:
-- 1. Core review and event tracking tables
-- 2. pgvector embeddings tables with categorical indexing
--
-- Usage:
--   psql -h your-host -U your-user -d your-database -f setup_database.sql

-- ---------------------------------------------------------------------------
-- Complete Schema Setup (Review System + Embeddings + Analytics)
-- ---------------------------------------------------------------------------
\echo 'Setting up complete schema (review system + embeddings + analytics)...'
\i complete_schema.sql

-- ---------------------------------------------------------------------------
-- Final Validation
-- ---------------------------------------------------------------------------
\echo 'Validating complete setup...'

DO $$
DECLARE
  core_tables INTEGER;
  embedding_tables INTEGER;
  total_functions INTEGER;
  vector_extension BOOLEAN;
BEGIN
  -- Count core tables
  SELECT COUNT(*) INTO core_tables
  FROM information_schema.tables 
  WHERE table_schema = 'public' 
    AND table_name IN ('review_runs', 'review_findings', 'events_raw', 'events_daily');
  
  -- Count embedding tables  
  SELECT COUNT(*) INTO embedding_tables
  FROM information_schema.tables 
  WHERE table_schema = 'public' 
    AND table_name IN ('section_gists', 'document_metadata', 'document_chunks');
  
  -- Count functions
  SELECT COUNT(*) INTO total_functions
  FROM pg_proc 
  WHERE proname IN (
    'rollup_events_daily', 'get_burger_count', 'cleanup_old_events', 
    'update_severity_counts', 'search_sections', 'get_documents_by_category'
  );
  
  -- Check pgvector extension
  SELECT EXISTS(
    SELECT 1 FROM pg_extension WHERE extname = 'vector'
  ) INTO vector_extension;
  
  -- Report status
  RAISE NOTICE '=== Database Setup Summary ===';
  RAISE NOTICE 'Core tables: % / 4', core_tables;
  RAISE NOTICE 'Embedding tables: % / 3', embedding_tables;
  RAISE NOTICE 'Functions: % / 6', total_functions;
  RAISE NOTICE 'pgvector extension: %', CASE WHEN vector_extension THEN 'OK' ELSE 'MISSING' END;
  
  IF core_tables = 4 AND embedding_tables = 3 AND total_functions = 6 AND vector_extension THEN
    RAISE NOTICE '✅ Database setup COMPLETE! All components ready.';
  ELSE
    RAISE WARNING '❌ Database setup INCOMPLETE. Check the logs above.';
  END IF;
END $$;

-- Show table sizes (useful for monitoring)
SELECT 
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public' 
  AND tablename IN (
    'review_runs', 'review_findings', 'events_raw', 'events_daily',
    'section_gists', 'document_metadata', 'document_chunks'
  )
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

\echo '=== Setup complete! ==='