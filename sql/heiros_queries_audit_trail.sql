-- ============================================================================
-- HEIROS AUDIT TRAIL TABLE QUERIES
-- Comprehensive audit logging and compliance tracking queries
-- ============================================================================

-- CREATE AUDIT ENTRY
-- Log workflow execution action
INSERT INTO heiros_audit_trail (
    execution_id,
    workflow_id,
    node_id,
    action_type,
    user_id,
    details,
    risk_factors,
    compliance_notes
) VALUES (
    $1,
    $2,
    'document_validation',
    'node_execution',
    'system_user',
    '{"node_type": "action", "duration_ms": 245, "result": "success"}',
    '["file_size_check", "format_validation"]',
    'Standard document validation completed successfully'
);

-- GET AUDIT TRAIL BY EXECUTION
-- Complete audit trail for specific execution
SELECT 
    a.audit_id,
    a.node_id,
    a.action_type,
    a.timestamp,
    a.user_id,
    a.details,
    a.risk_factors,
    a.compliance_notes
FROM heiros_audit_trail a
WHERE a.execution_id = $1
ORDER BY a.timestamp;

-- GET AUDIT TRAIL BY WORKFLOW
-- All audit entries for a specific workflow
SELECT 
    a.audit_id,
    a.execution_id,
    a.node_id,
    a.action_type,
    a.timestamp,
    a.user_id,
    a.details->>'result' as result,
    a.compliance_notes
FROM heiros_audit_trail a
WHERE a.workflow_id = $1
ORDER BY a.timestamp DESC
LIMIT 100;

-- RECENT AUDIT ENTRIES
-- Most recent audit trail entries across all workflows
SELECT 
    a.audit_id,
    w.name as workflow_name,
    a.execution_id,
    a.node_id,
    a.action_type,
    a.timestamp,
    a.user_id,
    a.details->>'result' as result
FROM heiros_audit_trail a
JOIN heiros_workflows w ON a.workflow_id = w.workflow_id
ORDER BY a.timestamp DESC
LIMIT 50;

-- AUDIT ENTRIES BY USER
-- All audit entries for specific user
SELECT 
    a.audit_id,
    w.name as workflow_name,
    a.node_id,
    a.action_type,
    a.timestamp,
    a.details,
    a.compliance_notes
FROM heiros_audit_trail a
JOIN heiros_workflows w ON a.workflow_id = w.workflow_id
WHERE a.user_id = $1
ORDER BY a.timestamp DESC;

-- RISK FACTOR ANALYSIS
-- Audit entries containing specific risk factors
SELECT 
    a.audit_id,
    w.name as workflow_name,
    a.execution_id,
    a.node_id,
    a.action_type,
    a.timestamp,
    a.risk_factors,
    a.compliance_notes
FROM heiros_audit_trail a
JOIN heiros_workflows w ON a.workflow_id = w.workflow_id
WHERE a.risk_factors @> '["' || $1 || '"]'::jsonb
ORDER BY a.timestamp DESC;

-- ACTION TYPE STATISTICS
-- Summary of audit entries by action type
SELECT 
    action_type,
    COUNT(*) as total_entries,
    COUNT(DISTINCT workflow_id) as workflows_affected,
    COUNT(DISTINCT user_id) as users_involved,
    MIN(timestamp) as earliest_entry,
    MAX(timestamp) as latest_entry
FROM heiros_audit_trail 
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY action_type
ORDER BY total_entries DESC;

-- COMPLIANCE AUDIT REPORT
-- Comprehensive compliance report for date range
SELECT 
    w.name as workflow_name,
    w.compliance_level,
    COUNT(a.audit_id) as audit_entries,
    COUNT(DISTINCT a.execution_id) as executions_audited,
    COUNT(DISTINCT a.user_id) as users_involved,
    STRING_AGG(DISTINCT a.action_type, ', ' ORDER BY a.action_type) as action_types,
    MIN(a.timestamp) as audit_period_start,
    MAX(a.timestamp) as audit_period_end
FROM heiros_audit_trail a
JOIN heiros_workflows w ON a.workflow_id = w.workflow_id
WHERE a.timestamp BETWEEN $1 AND $2
GROUP BY w.workflow_id, w.name, w.compliance_level
ORDER BY audit_entries DESC;

-- FAILED OPERATIONS AUDIT
-- Audit trail for failed operations
SELECT 
    a.audit_id,
    w.name as workflow_name,
    a.execution_id,
    a.node_id,
    a.action_type,
    a.timestamp,
    a.user_id,
    a.details->>'error_message' as error_message,
    a.compliance_notes
FROM heiros_audit_trail a
JOIN heiros_workflows w ON a.workflow_id = w.workflow_id
WHERE a.details->>'result' = 'failure'
   OR a.details ? 'error_message'
ORDER BY a.timestamp DESC;

-- NODE EXECUTION AUDIT
-- Audit trail for specific node across all executions
SELECT 
    a.audit_id,
    w.name as workflow_name,
    a.execution_id,
    a.timestamp,
    a.user_id,
    a.details->>'duration_ms' as duration_ms,
    a.details->>'result' as result,
    a.compliance_notes
FROM heiros_audit_trail a
JOIN heiros_workflows w ON a.workflow_id = w.workflow_id
WHERE a.node_id = $1
  AND a.action_type = 'node_execution'
ORDER BY a.timestamp DESC;

-- HOURLY AUDIT ACTIVITY
-- Audit activity distribution by hour
SELECT 
    EXTRACT(HOUR FROM timestamp) as hour_of_day,
    COUNT(*) as audit_entries,
    COUNT(DISTINCT workflow_id) as workflows,
    COUNT(DISTINCT user_id) as users
FROM heiros_audit_trail 
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY EXTRACT(HOUR FROM timestamp)
ORDER BY hour_of_day;

-- WORKFLOW EXECUTION TIMELINE
-- Complete timeline for specific execution with all audit entries
SELECT 
    a.timestamp,
    a.node_id,
    a.action_type,
    a.details->>'duration_ms' as duration_ms,
    a.details->>'result' as result,
    a.user_id,
    a.compliance_notes,
    EXTRACT(EPOCH FROM (a.timestamp - LAG(a.timestamp) OVER (ORDER BY a.timestamp))) as seconds_since_previous
FROM heiros_audit_trail a
WHERE a.execution_id = $1
ORDER BY a.timestamp;

-- SECURITY AUDIT REPORT
-- Security-focused audit report
SELECT 
    DATE(timestamp) as audit_date,
    COUNT(*) as total_activities,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(CASE WHEN action_type IN ('authentication', 'authorization') THEN 1 END) as security_events,
    COUNT(CASE WHEN risk_factors != '[]'::jsonb THEN 1 END) as risk_flagged_activities,
    COUNT(CASE WHEN details->>'result' = 'failure' THEN 1 END) as failed_activities
FROM heiros_audit_trail 
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY audit_date DESC;

-- CLEANUP OLD AUDIT ENTRIES
-- Archive audit entries older than retention period
DELETE FROM heiros_audit_trail 
WHERE timestamp < NOW() - INTERVAL '2 years';

-- AUDIT RETENTION BY COMPLIANCE LEVEL
-- Different retention periods based on compliance requirements
DELETE FROM heiros_audit_trail a
USING heiros_workflows w
WHERE a.workflow_id = w.workflow_id
  AND w.compliance_level = 'minimal'
  AND a.timestamp < NOW() - INTERVAL '90 days';

DELETE FROM heiros_audit_trail a
USING heiros_workflows w
WHERE a.workflow_id = w.workflow_id
  AND w.compliance_level = 'summary_only'
  AND a.timestamp < NOW() - INTERVAL '1 year';

-- USER ACTIVITY SUMMARY
-- Summary of user activity in workflows
SELECT 
    user_id,
    COUNT(DISTINCT workflow_id) as workflows_accessed,
    COUNT(DISTINCT execution_id) as executions_involved,
    COUNT(*) as total_audit_entries,
    MIN(timestamp) as first_activity,
    MAX(timestamp) as last_activity,
    COUNT(CASE WHEN details->>'result' = 'failure' THEN 1 END) as failed_operations
FROM heiros_audit_trail 
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY user_id
ORDER BY total_audit_entries DESC;