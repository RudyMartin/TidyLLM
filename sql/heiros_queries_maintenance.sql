-- ============================================================================
-- HEIROS MAINTENANCE QUERIES
-- Database maintenance, cleanup, and administrative queries
-- ============================================================================

-- DATABASE HEALTH CHECK
-- Comprehensive health check of all HeirOS tables
SELECT 
    'heiros_workflows' as table_name,
    COUNT(*) as row_count,
    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_records,
    MAX(created_date) as latest_record,
    pg_size_pretty(pg_total_relation_size('heiros_workflows')) as table_size
FROM heiros_workflows
UNION ALL
SELECT 
    'heiros_executions' as table_name,
    COUNT(*) as row_count,
    COUNT(CASE WHEN status = 'success' THEN 1 END) as active_records,
    MAX(execution_date) as latest_record,
    pg_size_pretty(pg_total_relation_size('heiros_executions')) as table_size
FROM heiros_executions
UNION ALL
SELECT 
    'heiros_sparse_agreements' as table_name,
    COUNT(*) as row_count,
    COUNT(CASE WHEN status = 'approved' THEN 1 END) as active_records,
    MAX(created_date) as latest_record,
    pg_size_pretty(pg_total_relation_size('heiros_sparse_agreements')) as table_size
FROM heiros_sparse_agreements
UNION ALL
SELECT 
    'heiros_node_templates' as table_name,
    COUNT(*) as row_count,
    COUNT(CASE WHEN is_public THEN 1 END) as active_records,
    MAX(created_date) as latest_record,
    pg_size_pretty(pg_total_relation_size('heiros_node_templates')) as table_size
FROM heiros_node_templates
UNION ALL
SELECT 
    'heiros_audit_trail' as table_name,
    COUNT(*) as row_count,
    COUNT(*) as active_records,
    MAX(timestamp) as latest_record,
    pg_size_pretty(pg_total_relation_size('heiros_audit_trail')) as table_size
FROM heiros_audit_trail
ORDER BY table_name;

-- INDEX USAGE STATISTICS
-- Check index usage and performance
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan,
    CASE WHEN idx_scan = 0 THEN 'UNUSED' ELSE 'USED' END as usage_status
FROM pg_stat_user_indexes 
WHERE schemaname = 'public' 
  AND tablename LIKE 'heiros_%'
ORDER BY tablename, idx_scan DESC;

-- VACUUM AND ANALYZE RECOMMENDATIONS
-- Identify tables needing maintenance
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples,
    CASE 
        WHEN n_dead_tup > n_live_tup * 0.1 THEN 'NEEDS VACUUM'
        ELSE 'OK'
    END as vacuum_recommendation,
    last_vacuum,
    last_analyze
FROM pg_stat_user_tables 
WHERE schemaname = 'public' 
  AND tablename LIKE 'heiros_%'
ORDER BY dead_tuples DESC;

-- CLEANUP OLD COMPLETED EXECUTIONS
-- Archive successful executions older than 6 months
WITH old_executions AS (
    SELECT execution_id 
    FROM heiros_executions 
    WHERE status = 'success' 
      AND execution_date < NOW() - INTERVAL '180 days'
    LIMIT 1000
)
DELETE FROM heiros_executions 
WHERE execution_id IN (SELECT execution_id FROM old_executions);

-- CLEANUP OLD FAILED EXECUTIONS
-- Archive failed executions older than 1 year
WITH old_failed_executions AS (
    SELECT execution_id 
    FROM heiros_executions 
    WHERE status = 'failure' 
      AND execution_date < NOW() - INTERVAL '365 days'
    LIMIT 1000
)
DELETE FROM heiros_executions 
WHERE execution_id IN (SELECT execution_id FROM old_failed_executions);

-- CLEANUP OLD AUDIT TRAIL
-- Remove audit entries based on compliance requirements
-- Keep full transparency for 3 years, others for 1 year
DELETE FROM heiros_audit_trail a
USING heiros_workflows w
WHERE a.workflow_id = w.workflow_id
  AND w.compliance_level != 'full_transparency'
  AND a.timestamp < NOW() - INTERVAL '1 year';

DELETE FROM heiros_audit_trail a
USING heiros_workflows w
WHERE a.workflow_id = w.workflow_id
  AND w.compliance_level = 'full_transparency'
  AND a.timestamp < NOW() - INTERVAL '3 years';

-- CLEANUP ORPHANED AUDIT ENTRIES
-- Remove audit entries for deleted executions
DELETE FROM heiros_audit_trail 
WHERE execution_id NOT IN (
    SELECT execution_id FROM heiros_executions
);

-- EXPIRE OLD SPARSE AGREEMENTS
-- Automatically expire agreements past their expiry date
UPDATE heiros_sparse_agreements 
SET status = 'expired'
WHERE status = 'approved'
  AND expiry_date IS NOT NULL
  AND expiry_date < NOW()
RETURNING agreement_id, title, expiry_date;

-- ARCHIVE INACTIVE WORKFLOWS
-- Mark workflows as archived if not used in 6 months
UPDATE heiros_workflows 
SET 
    status = 'archived',
    updated_date = NOW()
WHERE status = 'inactive'
  AND workflow_id NOT IN (
      SELECT DISTINCT workflow_id 
      FROM heiros_executions 
      WHERE execution_date >= NOW() - INTERVAL '180 days'
  )
  AND updated_date < NOW() - INTERVAL '90 days';

-- REINDEX HEAVILY USED TABLES
-- Reindex tables for performance optimization
REINDEX TABLE heiros_executions;
REINDEX TABLE heiros_audit_trail;

-- UPDATE TABLE STATISTICS
-- Refresh statistics for query optimizer
ANALYZE heiros_workflows;
ANALYZE heiros_executions;
ANALYZE heiros_sparse_agreements;
ANALYZE heiros_node_templates;
ANALYZE heiros_audit_trail;

-- BACKUP VERIFICATION QUERY
-- Verify backup requirements and retention
SELECT 
    'heiros_workflows' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN status = 'active' THEN 1 END) as critical_records,
    MIN(created_date) as oldest_record,
    MAX(updated_date) as newest_record,
    'Critical - Workflow Definitions' as backup_priority
FROM heiros_workflows
UNION ALL
SELECT 
    'heiros_executions' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN execution_date >= NOW() - INTERVAL '30 days' THEN 1 END) as critical_records,
    MIN(execution_date) as oldest_record,
    MAX(execution_date) as newest_record,
    'High - Recent Execution History' as backup_priority
FROM heiros_executions
UNION ALL
SELECT 
    'heiros_audit_trail' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN timestamp >= NOW() - INTERVAL '90 days' THEN 1 END) as critical_records,
    MIN(timestamp) as oldest_record,
    MAX(timestamp) as newest_record,
    'Critical - Compliance Audit Trail' as backup_priority
FROM heiros_audit_trail;

-- DISK SPACE MONITORING
-- Monitor disk usage by HeirOS tables
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(tablename::regclass)) as total_size,
    pg_size_pretty(pg_relation_size(tablename::regclass)) as table_size,
    pg_size_pretty(pg_total_relation_size(tablename::regclass) - pg_relation_size(tablename::regclass)) as index_size,
    ROUND(
        (pg_total_relation_size(tablename::regclass) * 100.0 / 
         (SELECT SUM(pg_total_relation_size(tablename::regclass)) 
          FROM pg_tables 
          WHERE schemaname = 'public' AND tablename LIKE 'heiros_%')), 2
    ) as percentage_of_heiros_data
FROM pg_tables 
WHERE schemaname = 'public' 
  AND tablename LIKE 'heiros_%'
ORDER BY pg_total_relation_size(tablename::regclass) DESC;

-- CONSTRAINT VALIDATION
-- Verify all constraints are functioning properly
SELECT 
    tc.table_name,
    tc.constraint_name,
    tc.constraint_type,
    CASE 
        WHEN tc.constraint_type = 'CHECK' THEN 'Validates data integrity'
        WHEN tc.constraint_type = 'FOREIGN KEY' THEN 'Maintains referential integrity'
        WHEN tc.constraint_type = 'PRIMARY KEY' THEN 'Ensures unique identification'
        WHEN tc.constraint_type = 'UNIQUE' THEN 'Prevents duplicates'
        ELSE 'Other constraint'
    END as purpose
FROM information_schema.table_constraints tc
WHERE tc.table_schema = 'public' 
  AND tc.table_name LIKE 'heiros_%'
ORDER BY tc.table_name, tc.constraint_type;

-- PERFORMANCE MONITORING SETUP
-- Create view for ongoing performance monitoring
CREATE OR REPLACE VIEW heiros_performance_monitor AS
SELECT 
    'Executions per hour (last 24h)' as metric,
    COUNT(*)::text as value,
    'executions' as unit
FROM heiros_executions 
WHERE execution_date >= NOW() - INTERVAL '24 hours'
UNION ALL
SELECT 
    'Average execution time (last 24h)' as metric,
    ROUND(AVG(duration_ms), 2)::text as value,
    'milliseconds' as unit
FROM heiros_executions 
WHERE execution_date >= NOW() - INTERVAL '24 hours'
  AND status = 'success'
UNION ALL
SELECT 
    'Success rate (last 24h)' as metric,
    ROUND(
        (COUNT(CASE WHEN status = 'success' THEN 1 END)::decimal / 
         COUNT(*)) * 100, 2
    )::text as value,
    'percent' as unit
FROM heiros_executions 
WHERE execution_date >= NOW() - INTERVAL '24 hours'
UNION ALL
SELECT 
    'Active workflows' as metric,
    COUNT(*)::text as value,
    'workflows' as unit
FROM heiros_workflows 
WHERE status = 'active'
UNION ALL
SELECT 
    'Approved SPARSE agreements' as metric,
    COUNT(*)::text as value,
    'agreements' as unit
FROM heiros_sparse_agreements 
WHERE status = 'approved';

-- EMERGENCY CLEANUP PROCEDURES
-- Use only in emergency situations
-- Uncommenting these will delete large amounts of data

-- Emergency: Clean all execution history older than 30 days
-- DELETE FROM heiros_executions WHERE execution_date < NOW() - INTERVAL '30 days';

-- Emergency: Clean all audit trail older than 30 days  
-- DELETE FROM heiros_audit_trail WHERE timestamp < NOW() - INTERVAL '30 days';

-- Emergency: Reset all template usage counts
-- UPDATE heiros_node_templates SET usage_count = 0;

-- Emergency: Archive all inactive workflows
-- UPDATE heiros_workflows SET status = 'archived' WHERE status = 'inactive';