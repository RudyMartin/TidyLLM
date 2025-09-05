-- ============================================================================
-- HEIROS EXECUTIONS TABLE QUERIES
-- Workflow execution tracking and monitoring queries
-- ============================================================================

-- CREATE EXECUTION RECORD
-- Start a new workflow execution
INSERT INTO heiros_executions (
    workflow_id,
    status,
    context_data,
    executed_by
) VALUES (
    $1,
    'running',
    '{"user_id": "demo_user", "environment": "production"}',
    'system'
) RETURNING execution_id;

-- UPDATE EXECUTION RESULTS
-- Complete an execution with results
UPDATE heiros_executions 
SET 
    status = $2,
    duration_ms = $3,
    result_data = $4,
    node_results = $5,
    compliance_report = $6,
    nodes_executed = $7
WHERE execution_id = $1;

-- UPDATE EXECUTION ERROR
-- Mark execution as failed with error details
UPDATE heiros_executions 
SET 
    status = 'failure',
    error_message = $2,
    duration_ms = $3
WHERE execution_id = $1;

-- GET EXECUTION BY ID
-- Retrieve specific execution details
SELECT 
    e.execution_id,
    e.workflow_id,
    w.name as workflow_name,
    e.execution_date,
    e.status,
    e.duration_ms,
    e.context_data,
    e.result_data,
    e.node_results,
    e.compliance_report,
    e.executed_by,
    e.error_message,
    e.nodes_executed
FROM heiros_executions e
JOIN heiros_workflows w ON e.workflow_id = w.workflow_id
WHERE e.execution_id = $1;

-- LIST RECENT EXECUTIONS
-- Get recent executions with workflow information
SELECT 
    e.execution_id,
    w.name as workflow_name,
    e.execution_date,
    e.status,
    e.duration_ms,
    e.executed_by,
    e.nodes_executed
FROM heiros_executions e
JOIN heiros_workflows w ON e.workflow_id = w.workflow_id
ORDER BY e.execution_date DESC
LIMIT 50;

-- GET EXECUTIONS BY WORKFLOW
-- All executions for a specific workflow
SELECT 
    execution_id,
    execution_date,
    status,
    duration_ms,
    executed_by,
    nodes_executed,
    CASE 
        WHEN error_message IS NOT NULL THEN 'Has Errors'
        ELSE 'No Errors'
    END as error_status
FROM heiros_executions 
WHERE workflow_id = $1
ORDER BY execution_date DESC;

-- EXECUTION PERFORMANCE METRICS
-- Performance statistics for workflow executions
SELECT 
    w.name as workflow_name,
    COUNT(*) as total_executions,
    COUNT(CASE WHEN e.status = 'success' THEN 1 END) as successful,
    COUNT(CASE WHEN e.status = 'failure' THEN 1 END) as failed,
    AVG(e.duration_ms) as avg_duration_ms,
    MIN(e.duration_ms) as min_duration_ms,
    MAX(e.duration_ms) as max_duration_ms,
    AVG(e.nodes_executed) as avg_nodes_executed
FROM heiros_executions e
JOIN heiros_workflows w ON e.workflow_id = w.workflow_id
WHERE e.execution_date >= NOW() - INTERVAL '30 days'
GROUP BY w.workflow_id, w.name
ORDER BY total_executions DESC;

-- FAILED EXECUTIONS ANALYSIS
-- Detailed analysis of failed executions
SELECT 
    w.name as workflow_name,
    e.execution_date,
    e.duration_ms,
    e.executed_by,
    e.error_message,
    e.context_data->>'user_id' as user_id,
    e.nodes_executed
FROM heiros_executions e
JOIN heiros_workflows w ON e.workflow_id = w.workflow_id
WHERE e.status = 'failure'
  AND e.execution_date >= NOW() - INTERVAL '7 days'
ORDER BY e.execution_date DESC;

-- EXECUTION STATUS SUMMARY
-- Current status of all executions
SELECT 
    status,
    COUNT(*) as count,
    AVG(duration_ms) as avg_duration,
    MIN(execution_date) as earliest,
    MAX(execution_date) as latest
FROM heiros_executions 
GROUP BY status
ORDER BY count DESC;

-- LONG RUNNING EXECUTIONS
-- Find executions that are still running for too long
SELECT 
    e.execution_id,
    w.name as workflow_name,
    e.execution_date,
    e.executed_by,
    EXTRACT(EPOCH FROM (NOW() - e.execution_date)) as seconds_running
FROM heiros_executions e
JOIN heiros_workflows w ON e.workflow_id = w.workflow_id
WHERE e.status = 'running'
  AND e.execution_date < NOW() - INTERVAL '1 hour'
ORDER BY e.execution_date;

-- EXECUTION COMPLIANCE REPORT
-- Compliance data for executions
SELECT 
    e.execution_id,
    w.name as workflow_name,
    w.compliance_level,
    e.execution_date,
    e.compliance_report->>'compliance_score' as compliance_score,
    e.compliance_report->>'risk_factors' as risk_factors,
    e.status
FROM heiros_executions e
JOIN heiros_workflows w ON e.workflow_id = w.workflow_id
WHERE e.compliance_report IS NOT NULL
  AND e.compliance_report != '{}'::jsonb
ORDER BY e.execution_date DESC;

-- CLEANUP OLD EXECUTIONS
-- Archive old successful executions (keep failures longer)
DELETE FROM heiros_executions 
WHERE status = 'success' 
  AND execution_date < NOW() - INTERVAL '180 days';

DELETE FROM heiros_executions 
WHERE status IN ('failure', 'cancelled') 
  AND execution_date < NOW() - INTERVAL '365 days';