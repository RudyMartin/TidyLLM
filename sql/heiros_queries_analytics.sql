-- ============================================================================
-- HEIROS ANALYTICS QUERIES
-- Cross-table analytics and business intelligence queries
-- ============================================================================

-- WORKFLOW PERFORMANCE DASHBOARD
-- Comprehensive workflow performance metrics
SELECT 
    w.workflow_id,
    w.name,
    w.status as workflow_status,
    w.compliance_level,
    COUNT(e.execution_id) as total_executions,
    COUNT(CASE WHEN e.status = 'success' THEN 1 END) as successful_executions,
    COUNT(CASE WHEN e.status = 'failure' THEN 1 END) as failed_executions,
    ROUND(AVG(e.duration_ms), 2) as avg_execution_time_ms,
    ROUND(AVG(e.nodes_executed), 1) as avg_nodes_per_execution,
    MAX(e.execution_date) as last_execution,
    ROUND(
        (COUNT(CASE WHEN e.status = 'success' THEN 1 END)::decimal / 
         NULLIF(COUNT(e.execution_id), 0)) * 100, 2
    ) as success_rate_percent
FROM heiros_workflows w
LEFT JOIN heiros_executions e ON w.workflow_id = e.workflow_id
WHERE w.status = 'active'
GROUP BY w.workflow_id, w.name, w.status, w.compliance_level
ORDER BY total_executions DESC;

-- SYSTEM HEALTH METRICS
-- Overall system health and performance indicators
WITH execution_stats AS (
    SELECT 
        COUNT(*) as total_executions,
        COUNT(CASE WHEN status = 'success' THEN 1 END) as successful,
        COUNT(CASE WHEN status = 'failure' THEN 1 END) as failed,
        COUNT(CASE WHEN status = 'running' THEN 1 END) as currently_running,
        AVG(duration_ms) as avg_duration
    FROM heiros_executions 
    WHERE execution_date >= NOW() - INTERVAL '24 hours'
),
workflow_stats AS (
    SELECT 
        COUNT(*) as total_workflows,
        COUNT(CASE WHEN status = 'active' THEN 1 END) as active_workflows
    FROM heiros_workflows
),
agreement_stats AS (
    SELECT 
        COUNT(*) as total_agreements,
        COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved_agreements
    FROM heiros_sparse_agreements
)
SELECT 
    e.total_executions,
    e.successful,
    e.failed,
    e.currently_running,
    ROUND(e.avg_duration, 2) as avg_duration_ms,
    ROUND((e.successful::decimal / NULLIF(e.total_executions, 0)) * 100, 2) as success_rate_percent,
    w.total_workflows,
    w.active_workflows,
    a.total_agreements,
    a.approved_agreements
FROM execution_stats e
CROSS JOIN workflow_stats w
CROSS JOIN agreement_stats a;

-- USAGE TRENDS BY TIME
-- Workflow execution trends over time
SELECT 
    DATE(e.execution_date) as execution_date,
    COUNT(*) as total_executions,
    COUNT(CASE WHEN e.status = 'success' THEN 1 END) as successful,
    COUNT(CASE WHEN e.status = 'failure' THEN 1 END) as failed,
    COUNT(DISTINCT e.workflow_id) as workflows_used,
    COUNT(DISTINCT e.executed_by) as unique_users,
    ROUND(AVG(e.duration_ms), 2) as avg_duration_ms
FROM heiros_executions e
WHERE e.execution_date >= NOW() - INTERVAL '30 days'
GROUP BY DATE(e.execution_date)
ORDER BY execution_date DESC;

-- TOP PERFORMING WORKFLOWS
-- Best performing workflows by success rate and usage
SELECT 
    w.name,
    COUNT(e.execution_id) as executions,
    COUNT(CASE WHEN e.status = 'success' THEN 1 END) as successful,
    ROUND(
        (COUNT(CASE WHEN e.status = 'success' THEN 1 END)::decimal / 
         COUNT(e.execution_id)) * 100, 2
    ) as success_rate,
    ROUND(AVG(e.duration_ms), 2) as avg_duration_ms,
    COUNT(DISTINCT e.executed_by) as unique_users,
    MAX(e.execution_date) as last_used
FROM heiros_workflows w
JOIN heiros_executions e ON w.workflow_id = e.workflow_id
WHERE w.status = 'active'
  AND e.execution_date >= NOW() - INTERVAL '30 days'
GROUP BY w.workflow_id, w.name
HAVING COUNT(e.execution_id) >= 5
ORDER BY success_rate DESC, executions DESC;

-- PROBLEMATIC WORKFLOWS
-- Workflows with high failure rates or performance issues
SELECT 
    w.name,
    COUNT(e.execution_id) as total_executions,
    COUNT(CASE WHEN e.status = 'failure' THEN 1 END) as failures,
    ROUND(
        (COUNT(CASE WHEN e.status = 'failure' THEN 1 END)::decimal / 
         COUNT(e.execution_id)) * 100, 2
    ) as failure_rate,
    ROUND(AVG(e.duration_ms), 2) as avg_duration_ms,
    STRING_AGG(DISTINCT e.error_message, '; ') as common_errors
FROM heiros_workflows w
JOIN heiros_executions e ON w.workflow_id = e.workflow_id
WHERE e.execution_date >= NOW() - INTERVAL '30 days'
GROUP BY w.workflow_id, w.name
HAVING COUNT(CASE WHEN e.status = 'failure' THEN 1 END) > 0
ORDER BY failure_rate DESC, failures DESC;

-- USER ACTIVITY ANALYSIS
-- User behavior and workflow usage patterns
SELECT 
    e.executed_by,
    COUNT(DISTINCT e.workflow_id) as workflows_used,
    COUNT(e.execution_id) as total_executions,
    COUNT(CASE WHEN e.status = 'success' THEN 1 END) as successful,
    COUNT(CASE WHEN e.status = 'failure' THEN 1 END) as failed,
    ROUND(AVG(e.duration_ms), 2) as avg_execution_time,
    MIN(e.execution_date) as first_execution,
    MAX(e.execution_date) as last_execution
FROM heiros_executions e
WHERE e.execution_date >= NOW() - INTERVAL '30 days'
GROUP BY e.executed_by
ORDER BY total_executions DESC;

-- COMPLIANCE OVERVIEW
-- Compliance levels and audit activity
SELECT 
    w.compliance_level,
    COUNT(DISTINCT w.workflow_id) as workflows,
    COUNT(e.execution_id) as executions,
    COUNT(a.audit_id) as audit_entries,
    ROUND(COUNT(a.audit_id)::decimal / NULLIF(COUNT(e.execution_id), 0), 2) as audit_ratio,
    COUNT(CASE WHEN e.status = 'success' THEN 1 END) as successful_executions
FROM heiros_workflows w
LEFT JOIN heiros_executions e ON w.workflow_id = e.workflow_id
LEFT JOIN heiros_audit_trail a ON e.execution_id = a.execution_id
WHERE w.status = 'active'
GROUP BY w.compliance_level
ORDER BY workflows DESC;

-- SPARSE AGREEMENT UTILIZATION
-- Usage analysis of SPARSE agreements
SELECT 
    sa.title,
    sa.risk_level,
    sa.business_owner,
    sa.execution_count,
    sa.last_execution_date,
    DATE_PART('days', NOW() - sa.last_execution_date) as days_since_last_use,
    CASE 
        WHEN sa.execution_count = 0 THEN 'Unused'
        WHEN sa.last_execution_date < NOW() - INTERVAL '30 days' THEN 'Dormant'
        WHEN sa.execution_count < 10 THEN 'Low Usage'
        ELSE 'Active'
    END as usage_category,
    sa.expiry_date,
    CASE 
        WHEN sa.expiry_date IS NULL THEN 'No Expiry'
        WHEN sa.expiry_date < NOW() THEN 'Expired'
        WHEN sa.expiry_date < NOW() + INTERVAL '30 days' THEN 'Expiring Soon'
        ELSE 'Active'
    END as expiry_status
FROM heiros_sparse_agreements sa
WHERE sa.status = 'approved'
ORDER BY sa.execution_count DESC;

-- NODE TEMPLATE POPULARITY
-- Most and least used node templates
SELECT 
    nt.name,
    nt.node_type,
    nt.category,
    nt.usage_count,
    nt.created_date,
    nt.created_by,
    DATE_PART('days', NOW() - nt.created_date) as days_since_creation,
    CASE 
        WHEN nt.usage_count = 0 THEN 'Unused'
        WHEN nt.usage_count < 5 THEN 'Low Usage'
        WHEN nt.usage_count < 20 THEN 'Medium Usage'
        ELSE 'High Usage'
    END as popularity_level
FROM heiros_node_templates nt
WHERE nt.is_public = true
ORDER BY nt.usage_count DESC;

-- EXECUTION DURATION ANALYSIS
-- Analysis of workflow execution times
WITH duration_stats AS (
    SELECT 
        w.name,
        e.duration_ms,
        NTILE(4) OVER (PARTITION BY w.workflow_id ORDER BY e.duration_ms) as quartile
    FROM heiros_workflows w
    JOIN heiros_executions e ON w.workflow_id = e.workflow_id
    WHERE e.status = 'success'
      AND e.duration_ms IS NOT NULL
      AND e.execution_date >= NOW() - INTERVAL '30 days'
)
SELECT 
    name,
    COUNT(*) as executions,
    ROUND(AVG(duration_ms), 2) as avg_duration_ms,
    ROUND(MIN(duration_ms), 2) as min_duration_ms,
    ROUND(MAX(duration_ms), 2) as max_duration_ms,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms), 2) as median_duration_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms), 2) as p95_duration_ms
FROM duration_stats
GROUP BY name
ORDER BY avg_duration_ms DESC;

-- ERROR PATTERN ANALYSIS
-- Common error patterns and their frequencies
SELECT 
    COALESCE(e.error_message, 'No Error Message') as error_type,
    COUNT(*) as occurrence_count,
    COUNT(DISTINCT e.workflow_id) as workflows_affected,
    COUNT(DISTINCT e.executed_by) as users_affected,
    MIN(e.execution_date) as first_occurrence,
    MAX(e.execution_date) as last_occurrence,
    ARRAY_AGG(DISTINCT w.name) as affected_workflows
FROM heiros_executions e
JOIN heiros_workflows w ON e.workflow_id = w.workflow_id
WHERE e.status = 'failure'
  AND e.execution_date >= NOW() - INTERVAL '30 days'
GROUP BY e.error_message
ORDER BY occurrence_count DESC;

-- WORKFLOW COMPLEXITY METRICS
-- Analysis of workflow complexity and performance correlation
SELECT 
    w.name,
    jsonb_array_length(w.workflow_json->'nodes') as estimated_node_count,
    COUNT(e.execution_id) as executions,
    ROUND(AVG(e.duration_ms), 2) as avg_duration_ms,
    ROUND(AVG(e.nodes_executed), 1) as avg_nodes_executed,
    COUNT(CASE WHEN e.status = 'success' THEN 1 END) as successful,
    ROUND(
        (COUNT(CASE WHEN e.status = 'success' THEN 1 END)::decimal / 
         COUNT(e.execution_id)) * 100, 2
    ) as success_rate
FROM heiros_workflows w
LEFT JOIN heiros_executions e ON w.workflow_id = e.workflow_id
WHERE w.status = 'active'
  AND w.workflow_json ? 'nodes'
GROUP BY w.workflow_id, w.name, w.workflow_json
ORDER BY estimated_node_count DESC;