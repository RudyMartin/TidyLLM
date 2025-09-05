-- =============================================================================
-- 00_complete_setup.sql - Complete Production Database Setup
-- =============================================================================
-- This script runs all 4 setup scripts in sequence
-- Use this for complete production deployment

-- Start transaction for atomic setup
BEGIN;

-- Log start
SELECT '🚀 Starting complete production database setup...' as status;

-- 1. Extensions Setup
SELECT '📦 Step 1: Installing PostgreSQL extensions...' as status;
\i 01_extensions.sql

-- 2. Review System Setup  
SELECT '🔍 Step 2: Creating review system tables...' as status;
\i 02_review_system.sql

-- 3. Embeddings System Setup
SELECT '🧠 Step 3: Creating embeddings and vector storage...' as status;
\i 03_embeddings_system.sql

-- 4. Event Tracking Setup
SELECT '📊 Step 4: Creating event tracking tables...' as status;
\i 04_event_tracking.sql

-- 5. MLflow Integration Setup
SELECT '🔗 Step 5: Creating MLflow integration tables...' as status;
\i 05_mlflow_integration.sql

-- Final verification
SELECT '✅ Complete production setup finished successfully!' as status;

-- Show summary
SELECT 
  '📋 Setup Summary:' as summary,
  (SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'public') as total_tables,
  (SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public') as total_indexes,
  (SELECT COUNT(*) FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp')) as extensions_installed,
  (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE '%mlflow%') as mlflow_tables;

COMMIT;

-- Success message
SELECT '🎉 Production database is ready for use!' as final_status;
