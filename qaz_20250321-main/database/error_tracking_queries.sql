-- Error Tracking Query Examples
-- These queries demonstrate how to analyze errors that testing misses

-- 1. Get Critical Errors That Need Immediate Attention
SELECT 
    error_id,
    timestamp,
    error_type,
    error_message,
    agent_name,
    task_type,
    resolution_status,
    CASE 
        WHEN resolution_status = 'open' THEN '🚨 URGENT'
        WHEN resolution_status = 'investigating' THEN '🔍 IN PROGRESS'
        ELSE '✅ RESOLVED'
    END as priority
FROM prompt_pipeline_errors
WHERE severity = 'critical' 
    AND resolution_status != 'resolved'
    AND timestamp >= NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC;

-- 2. Error Summary by Hour (Last 24 Hours)
SELECT 
    DATE_TRUNC('hour', timestamp) as hour_bucket,
    severity,
    error_type,
    COUNT(*) as error_count,
    COUNT(DISTINCT agent_name) as affected_agents,
    ROUND(AVG(EXTRACT(EPOCH FROM (NOW() - timestamp)) / 60), 2) as avg_age_minutes
FROM prompt_pipeline_errors
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp), severity, error_type
ORDER BY hour_bucket DESC, error_count DESC;

-- 3. Performance Trends with Error Correlation
SELECT 
    DATE_TRUNC('hour', ph.timestamp) as hour_bucket,
    ph.agent_name,
    ph.task_type,
    COUNT(*) as total_requests,
    COUNT(CASE WHEN ph.success = false THEN 1 END) as failed_requests,
    ROUND((COUNT(CASE WHEN ph.success = false THEN 1 END) * 100.0 / COUNT(*)), 2) as error_rate_percent,
    COUNT(DISTINCT ph.mlflow_run_id) as mlflow_runs_count
FROM prompt_history ph
WHERE ph.timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', ph.timestamp), ph.agent_name, ph.task_type
ORDER BY hour_bucket DESC, total_requests DESC;

-- 4. Error Patterns Analysis
SELECT 
    ep.error_type,
    ep.pattern_description,
    ep.frequency_threshold,
    ep.time_window_minutes,
    ep.severity,
    ep.auto_resolution_action,
    COUNT(ppe.error_id) as actual_occurrences,
    CASE 
        WHEN COUNT(ppe.error_id) >= ep.frequency_threshold THEN '🚨 PATTERN TRIGGERED'
        ELSE '✅ BELOW THRESHOLD'
    END as pattern_status
FROM error_patterns ep
LEFT JOIN prompt_pipeline_errors ppe ON ep.error_type = ppe.error_type
    AND ppe.timestamp >= NOW() - (ep.time_window_minutes || ' minutes')::INTERVAL
WHERE ep.is_active = true
GROUP BY ep.pattern_id, ep.error_type, ep.pattern_description, ep.frequency_threshold, 
         ep.time_window_minutes, ep.severity, ep.auto_resolution_action
ORDER BY actual_occurrences DESC;

-- 5. Agent Health Analysis
SELECT 
    agent_name,
    COUNT(*) as total_errors,
    COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_errors,
    COUNT(CASE WHEN severity = 'warning' THEN 1 END) as warning_errors,
    COUNT(CASE WHEN severity = 'info' THEN 1 END) as info_errors,
    COUNT(CASE WHEN resolution_status = 'resolved' THEN 1 END) as resolved_errors,
    ROUND((COUNT(CASE WHEN resolution_status = 'resolved' THEN 1 END) * 100.0 / COUNT(*)), 2) as resolution_rate_percent,
    MAX(timestamp) as last_error_time
FROM prompt_pipeline_errors
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY agent_name
ORDER BY total_errors DESC;

-- 6. Real-time Context Correlation
SELECT 
    ph.prompt_id,
    ph.agent_name,
    ph.task_type,
    ph.success,
    rtc.context_type,
    rtc.context_data,
    rtc.relevance_score,
    ppe.error_type,
    ppe.severity
FROM prompt_history ph
LEFT JOIN real_time_context rtc ON ph.prompt_id = rtc.prompt_id
LEFT JOIN prompt_pipeline_errors ppe ON ph.prompt_id = ppe.prompt_id
WHERE ph.timestamp >= NOW() - INTERVAL '2 hours'
ORDER BY ph.timestamp DESC;

-- 7. Alert History Analysis
SELECT 
    ah.alert_type,
    ah.recipient,
    ah.status,
    COUNT(*) as alert_count,
    MIN(ah.sent_at) as first_alert,
    MAX(ah.sent_at) as last_alert,
    AVG(EXTRACT(EPOCH FROM (ah.acknowledged_at - ah.sent_at)) / 60) as avg_response_time_minutes
FROM alert_history ah
WHERE ah.sent_at >= NOW() - INTERVAL '24 hours'
GROUP BY ah.alert_type, ah.recipient, ah.status
ORDER BY alert_count DESC;

-- 8. MLflow Integration Analysis
SELECT 
    mi.experiment_name,
    mi.run_name,
    mi.status,
    COUNT(ph.prompt_id) as prompt_count,
    COUNT(CASE WHEN ph.success = false THEN 1 END) as failed_prompts,
    COUNT(ppe.error_id) as error_count,
    COUNT(CASE WHEN ppe.severity = 'critical' THEN 1 END) as critical_errors
FROM mlflow_integration mi
LEFT JOIN prompt_history ph ON mi.mlflow_run_id = ph.mlflow_run_id
LEFT JOIN prompt_pipeline_errors ppe ON mi.mlflow_run_id = ppe.mlflow_run_id
WHERE mi.created_at >= NOW() - INTERVAL '24 hours'
GROUP BY mi.mlflow_run_id, mi.experiment_name, mi.run_name, mi.status
ORDER BY error_count DESC;

-- 9. Batch Processing Status
SELECT 
    batch_id,
    status,
    total_records,
    processed_records,
    failed_records,
    ROUND((processed_records * 100.0 / total_records), 2) as completion_percentage,
    start_time,
    end_time,
    CASE 
        WHEN end_time IS NOT NULL THEN 
            ROUND(EXTRACT(EPOCH FROM (end_time - start_time)) / 60, 2)
        ELSE 
            ROUND(EXTRACT(EPOCH FROM (NOW() - start_time)) / 60, 2)
    END as duration_minutes,
    error_message
FROM batch_processing_status
WHERE start_time >= NOW() - INTERVAL '24 hours'
ORDER BY start_time DESC;

-- 10. Error Resolution Time Analysis
SELECT 
    error_type,
    severity,
    COUNT(*) as total_errors,
    COUNT(CASE WHEN resolution_status = 'resolved' THEN 1 END) as resolved_errors,
    ROUND(AVG(EXTRACT(EPOCH FROM (updated_at - created_at)) / 60), 2) as avg_resolution_time_minutes,
    MIN(EXTRACT(EPOCH FROM (updated_at - created_at)) / 60) as min_resolution_time_minutes,
    MAX(EXTRACT(EPOCH FROM (updated_at - created_at)) / 60) as max_resolution_time_minutes
FROM prompt_pipeline_errors
WHERE resolution_status = 'resolved'
    AND created_at >= NOW() - INTERVAL '24 hours'
GROUP BY error_type, severity
ORDER BY avg_resolution_time_minutes DESC;

-- 11. User Session Error Analysis
SELECT 
    ph.user_id,
    ph.session_id,
    COUNT(ph.prompt_id) as total_prompts,
    COUNT(CASE WHEN ph.success = false THEN 1 END) as failed_prompts,
    COUNT(ppe.error_id) as error_count,
    COUNT(CASE WHEN ppe.severity = 'critical' THEN 1 END) as critical_errors,
    MIN(ph.timestamp) as session_start,
    MAX(ph.timestamp) as session_end,
    ROUND(EXTRACT(EPOCH FROM (MAX(ph.timestamp) - MIN(ph.timestamp)) / 60), 2) as session_duration_minutes
FROM prompt_history ph
LEFT JOIN prompt_pipeline_errors ppe ON ph.prompt_id = ppe.prompt_id
WHERE ph.timestamp >= NOW() - INTERVAL '24 hours'
    AND ph.user_id IS NOT NULL
GROUP BY ph.user_id, ph.session_id
HAVING COUNT(ph.prompt_id) > 1
ORDER BY error_count DESC;

-- 12. Cost Impact Analysis (when MLflow metrics are available)
SELECT 
    ph.agent_name,
    ph.task_type,
    COUNT(ph.prompt_id) as total_prompts,
    COUNT(CASE WHEN ph.success = false THEN 1 END) as failed_prompts,
    COUNT(ppe.error_id) as error_count,
    -- Note: Actual cost data would come from MLflow integration
    -- This is a placeholder for the structure
    'Cost data from MLflow' as cost_source
FROM prompt_history ph
LEFT JOIN prompt_pipeline_errors ppe ON ph.prompt_id = ppe.prompt_id
WHERE ph.timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY ph.agent_name, ph.task_type
ORDER BY error_count DESC;

-- 13. Error Trend Analysis (Last 7 Days)
SELECT 
    DATE_TRUNC('day', timestamp) as day_bucket,
    error_type,
    severity,
    COUNT(*) as error_count,
    COUNT(DISTINCT agent_name) as affected_agents,
    COUNT(DISTINCT DATE_TRUNC('hour', timestamp)) as hours_with_errors
FROM prompt_pipeline_errors
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('day', timestamp), error_type, severity
ORDER BY day_bucket DESC, error_count DESC;

-- 14. System Health Dashboard Query
SELECT 
    'Total Prompts' as metric,
    COUNT(*) as value,
    'count' as unit
FROM prompt_history
WHERE timestamp >= NOW() - INTERVAL '1 hour'

UNION ALL

SELECT 
    'Failed Prompts' as metric,
    COUNT(CASE WHEN success = false THEN 1 END) as value,
    'count' as unit
FROM prompt_history
WHERE timestamp >= NOW() - INTERVAL '1 hour'

UNION ALL

SELECT 
    'Error Rate' as metric,
    ROUND((COUNT(CASE WHEN success = false THEN 1 END) * 100.0 / COUNT(*)), 2) as value,
    'percent' as unit
FROM prompt_history
WHERE timestamp >= NOW() - INTERVAL '1 hour'

UNION ALL

SELECT 
    'Critical Errors' as metric,
    COUNT(*) as value,
    'count' as unit
FROM prompt_pipeline_errors
WHERE severity = 'critical' 
    AND timestamp >= NOW() - INTERVAL '1 hour'

UNION ALL

SELECT 
    'Open Issues' as metric,
    COUNT(*) as value,
    'count' as unit
FROM prompt_pipeline_errors
WHERE resolution_status = 'open'
    AND timestamp >= NOW() - INTERVAL '1 hour';

-- 15. Predictive Error Analysis (Pattern Detection)
SELECT 
    ep.error_type,
    ep.pattern_description,
    COUNT(ppe.error_id) as recent_occurrences,
    ep.frequency_threshold,
    CASE 
        WHEN COUNT(ppe.error_id) >= ep.frequency_threshold * 0.8 THEN '⚠️ APPROACHING THRESHOLD'
        WHEN COUNT(ppe.error_id) >= ep.frequency_threshold THEN '🚨 THRESHOLD EXCEEDED'
        ELSE '✅ NORMAL'
    END as prediction_status
FROM error_patterns ep
LEFT JOIN prompt_pipeline_errors ppe ON ep.error_type = ppe.error_type
    AND ppe.timestamp >= NOW() - (ep.time_window_minutes * 0.5 || ' minutes')::INTERVAL
WHERE ep.is_active = true
GROUP BY ep.pattern_id, ep.error_type, ep.pattern_description, ep.frequency_threshold
ORDER BY recent_occurrences DESC;
