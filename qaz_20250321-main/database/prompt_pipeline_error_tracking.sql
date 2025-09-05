-- Prompt Pipeline Error Tracking Schema
-- This schema captures errors that standard testing misses
-- Integrates with MLflow to avoid duplicating metrics (DRY principle)

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- MLflow Integration Table (links to MLflow runs)
CREATE TABLE IF NOT EXISTS mlflow_integration (
    id SERIAL PRIMARY KEY,
    mlflow_run_id VARCHAR(255) UNIQUE NOT NULL,
    experiment_name VARCHAR(255),
    run_name VARCHAR(255),
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Prompt History Table (Enhanced with MLflow integration)
CREATE TABLE IF NOT EXISTS prompt_history (
    id SERIAL PRIMARY KEY,
    prompt_id VARCHAR(255) UNIQUE NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    prompt_text TEXT NOT NULL,
    response_text TEXT,
    -- Remove duplicated metrics - these come from MLflow
    -- tokens_input INTEGER,
    -- tokens_output INTEGER,
    -- cost_usd DECIMAL(10,4),
    -- response_time_ms INTEGER,
    -- confidence_score DECIMAL(3,2),
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    mlflow_run_id VARCHAR(255) REFERENCES mlflow_integration(mlflow_run_id),
    batch_id VARCHAR(255),
    user_id VARCHAR(100),
    session_id VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Error Tracking Table (focuses on errors, not metrics)
CREATE TABLE IF NOT EXISTS prompt_pipeline_errors (
    id SERIAL PRIMARY KEY,
    error_id VARCHAR(255) UNIQUE NOT NULL DEFAULT uuid_generate_v4()::text,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'warning', 'info')),
    error_type VARCHAR(100) NOT NULL,
    error_message TEXT NOT NULL,
    prompt_id VARCHAR(255) REFERENCES prompt_history(prompt_id),
    mlflow_run_id VARCHAR(255) REFERENCES mlflow_integration(mlflow_run_id),
    agent_name VARCHAR(100),
    task_type VARCHAR(100),
    model_used VARCHAR(100),
    context_data JSONB,
    stack_trace TEXT,
    resolution_status VARCHAR(50) DEFAULT 'open' CHECK (resolution_status IN ('open', 'investigating', 'resolved')),
    resolution_notes TEXT,
    alert_sent BOOLEAN DEFAULT false,
    alert_recipients TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Error Patterns Table
CREATE TABLE IF NOT EXISTS error_patterns (
    id SERIAL PRIMARY KEY,
    pattern_id VARCHAR(255) UNIQUE NOT NULL DEFAULT uuid_generate_v4()::text,
    error_type VARCHAR(100) NOT NULL,
    pattern_description TEXT,
    frequency_threshold INTEGER,
    time_window_minutes INTEGER,
    severity VARCHAR(20) CHECK (severity IN ('critical', 'warning', 'info')),
    auto_resolution_action VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Alert History Table
CREATE TABLE IF NOT EXISTS alert_history (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(255) UNIQUE NOT NULL DEFAULT uuid_generate_v4()::text,
    error_id VARCHAR(255) REFERENCES prompt_pipeline_errors(error_id),
    alert_type VARCHAR(50) NOT NULL CHECK (alert_type IN ('email', 'slack', 'sms', 'dashboard')),
    recipient VARCHAR(255),
    message TEXT,
    sent_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(100),
    status VARCHAR(50) DEFAULT 'sent' CHECK (status IN ('sent', 'delivered', 'acknowledged', 'failed'))
);

-- Real-time Context Table
CREATE TABLE IF NOT EXISTS real_time_context (
    id SERIAL PRIMARY KEY,
    prompt_id VARCHAR(255) REFERENCES prompt_history(prompt_id),
    context_type VARCHAR(50) NOT NULL CHECK (context_type IN ('live_events', 'user_session', 'system_status')),
    context_data JSONB NOT NULL,
    relevance_score DECIMAL(3,2),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Batch Processing Status Table
CREATE TABLE IF NOT EXISTS batch_processing_status (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    total_records INTEGER,
    processed_records INTEGER DEFAULT 0,
    failed_records INTEGER DEFAULT 0,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Remove Performance Metrics Table - this data comes from MLflow
-- CREATE TABLE IF NOT EXISTS performance_metrics (...);

-- Create Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_prompt_history_timestamp ON prompt_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_prompt_history_agent_name ON prompt_history(agent_name);
CREATE INDEX IF NOT EXISTS idx_prompt_history_task_type ON prompt_history(task_type);
CREATE INDEX IF NOT EXISTS idx_prompt_history_batch_id ON prompt_history(batch_id);
CREATE INDEX IF NOT EXISTS idx_prompt_history_success ON prompt_history(success);
CREATE INDEX IF NOT EXISTS idx_prompt_history_mlflow_run_id ON prompt_history(mlflow_run_id);

CREATE INDEX IF NOT EXISTS idx_prompt_pipeline_errors_timestamp ON prompt_pipeline_errors(timestamp);
CREATE INDEX IF NOT EXISTS idx_prompt_pipeline_errors_severity ON prompt_pipeline_errors(severity);
CREATE INDEX IF NOT EXISTS idx_prompt_pipeline_errors_error_type ON prompt_pipeline_errors(error_type);
CREATE INDEX IF NOT EXISTS idx_prompt_pipeline_errors_agent_name ON prompt_pipeline_errors(agent_name);
CREATE INDEX IF NOT EXISTS idx_prompt_pipeline_errors_resolution_status ON prompt_pipeline_errors(resolution_status);
CREATE INDEX IF NOT EXISTS idx_prompt_pipeline_errors_mlflow_run_id ON prompt_pipeline_errors(mlflow_run_id);

CREATE INDEX IF NOT EXISTS idx_real_time_context_prompt_id ON real_time_context(prompt_id);
CREATE INDEX IF NOT EXISTS idx_real_time_context_timestamp ON real_time_context(timestamp);
CREATE INDEX IF NOT EXISTS idx_real_time_context_context_type ON real_time_context(context_type);

CREATE INDEX IF NOT EXISTS idx_alert_history_sent_at ON alert_history(sent_at);
CREATE INDEX IF NOT EXISTS idx_alert_history_alert_type ON alert_history(alert_type);
CREATE INDEX IF NOT EXISTS idx_alert_history_status ON alert_history(status);

CREATE INDEX IF NOT EXISTS idx_mlflow_integration_run_id ON mlflow_integration(mlflow_run_id);
CREATE INDEX IF NOT EXISTS idx_mlflow_integration_experiment_name ON mlflow_integration(experiment_name);

-- Create Views that integrate with MLflow data
CREATE OR REPLACE VIEW error_summary_view AS
SELECT 
    DATE_TRUNC('hour', ppe.timestamp) as hour_bucket,
    ppe.severity,
    ppe.error_type,
    COUNT(*) as error_count,
    COUNT(DISTINCT ppe.agent_name) as affected_agents,
    AVG(EXTRACT(EPOCH FROM (NOW() - ppe.timestamp))) as avg_age_seconds
FROM prompt_pipeline_errors ppe
WHERE ppe.timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', ppe.timestamp), ppe.severity, ppe.error_type
ORDER BY hour_bucket DESC, error_count DESC;

-- Performance trends view that reads from MLflow (via integration)
CREATE OR REPLACE VIEW performance_trends_view AS
SELECT 
    DATE_TRUNC('hour', ph.timestamp) as hour_bucket,
    ph.agent_name,
    ph.task_type,
    -- Note: These metrics would be fetched from MLflow in the application layer
    -- This view provides the structure for MLflow integration
    COUNT(*) as total_requests,
    COUNT(CASE WHEN ph.success = false THEN 1 END) as failed_requests,
    (COUNT(CASE WHEN ph.success = false THEN 1 END) * 100.0 / COUNT(*)) as error_rate_percent,
    COUNT(DISTINCT ph.mlflow_run_id) as mlflow_runs_count
FROM prompt_history ph
WHERE ph.timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', ph.timestamp), ph.agent_name, ph.task_type
ORDER BY hour_bucket DESC, total_requests DESC;

CREATE OR REPLACE VIEW critical_errors_view AS
SELECT 
    ppe.error_id,
    ppe.timestamp,
    ppe.error_type,
    ppe.error_message,
    ppe.agent_name,
    ppe.task_type,
    ppe.resolution_status,
    CASE 
        WHEN ppe.resolution_status = 'open' THEN 'URGENT'
        WHEN ppe.resolution_status = 'investigating' THEN 'IN PROGRESS'
        ELSE 'RESOLVED'
    END as priority,
    ppe.mlflow_run_id
FROM prompt_pipeline_errors ppe
WHERE ppe.severity = 'critical' 
    AND ppe.resolution_status != 'resolved'
    AND ppe.timestamp >= NOW() - INTERVAL '1 hour'
ORDER BY ppe.timestamp DESC;

-- View to get MLflow metrics for a specific run
CREATE OR REPLACE VIEW mlflow_metrics_view AS
SELECT 
    mi.mlflow_run_id,
    mi.experiment_name,
    mi.run_name,
    ph.agent_name,
    ph.task_type,
    ph.timestamp,
    ph.success,
    -- Note: Actual metrics (response_time, cost, etc.) would be fetched from MLflow API
    -- This view provides the structure for integration
    ph.metadata
FROM mlflow_integration mi
JOIN prompt_history ph ON mi.mlflow_run_id = ph.mlflow_run_id
WHERE mi.status = 'active'
ORDER BY ph.timestamp DESC;

-- Create Functions for Common Operations
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create Triggers for updated_at
CREATE TRIGGER update_prompt_history_updated_at 
    BEFORE UPDATE ON prompt_history 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_prompt_pipeline_errors_updated_at 
    BEFORE UPDATE ON prompt_pipeline_errors 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_mlflow_integration_updated_at 
    BEFORE UPDATE ON mlflow_integration 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to get error statistics
CREATE OR REPLACE FUNCTION get_error_statistics(hours_back INTEGER DEFAULT 24)
RETURNS TABLE (
    error_type VARCHAR(100),
    severity VARCHAR(20),
    error_count BIGINT,
    avg_age_minutes NUMERIC,
    resolution_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ppe.error_type,
        ppe.severity,
        COUNT(*) as error_count,
        AVG(EXTRACT(EPOCH FROM (NOW() - ppe.timestamp)) / 60) as avg_age_minutes,
        (COUNT(CASE WHEN ppe.resolution_status = 'resolved' THEN 1 END) * 100.0 / COUNT(*)) as resolution_rate
    FROM prompt_pipeline_errors ppe
    WHERE ppe.timestamp >= NOW() - (hours_back || ' hours')::INTERVAL
    GROUP BY ppe.error_type, ppe.severity
    ORDER BY error_count DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get MLflow run metrics (placeholder for application integration)
CREATE OR REPLACE FUNCTION get_mlflow_metrics_for_run(run_id VARCHAR(255))
RETURNS TABLE (
    metric_name VARCHAR(255),
    metric_value TEXT,
    step INTEGER
) AS $$
BEGIN
    -- This function would integrate with MLflow API to fetch actual metrics
    -- For now, return empty result - implementation in application layer
    RETURN QUERY
    SELECT 
        ''::VARCHAR(255) as metric_name,
        ''::TEXT as metric_value,
        0::INTEGER as step
    WHERE FALSE;
END;
$$ LANGUAGE plpgsql;

-- Function to get performance alerts (based on MLflow data)
CREATE OR REPLACE FUNCTION get_performance_alerts()
RETURNS TABLE (
    alert_type VARCHAR(100),
    agent_name VARCHAR(100),
    alert_message TEXT,
    mlflow_run_id VARCHAR(255)
) AS $$
BEGIN
    -- This function would integrate with MLflow to get actual performance metrics
    -- For now, return alerts based on error patterns
    RETURN QUERY
    SELECT 
        'high_error_rate' as alert_type,
        ph.agent_name,
        'High error rate detected for ' || ph.agent_name as alert_message,
        ph.mlflow_run_id
    FROM prompt_history ph
    WHERE ph.timestamp >= NOW() - INTERVAL '1 hour'
    GROUP BY ph.agent_name, ph.mlflow_run_id
    HAVING (COUNT(CASE WHEN ph.success = false THEN 1 END) * 100.0 / COUNT(*)) > 10;
END;
$$ LANGUAGE plpgsql;

-- Function to register MLflow run
CREATE OR REPLACE FUNCTION register_mlflow_run(
    p_run_id VARCHAR(255),
    p_experiment_name VARCHAR(255),
    p_run_name VARCHAR(255)
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO mlflow_integration (mlflow_run_id, experiment_name, run_name)
    VALUES (p_run_id, p_experiment_name, p_run_name)
    ON CONFLICT (mlflow_run_id) 
    DO UPDATE SET 
        experiment_name = EXCLUDED.experiment_name,
        run_name = EXCLUDED.run_name,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to link prompt to MLflow run
CREATE OR REPLACE FUNCTION link_prompt_to_mlflow(
    p_prompt_id VARCHAR(255),
    p_mlflow_run_id VARCHAR(255)
)
RETURNS VOID AS $$
BEGIN
    UPDATE prompt_history 
    SET mlflow_run_id = p_mlflow_run_id
    WHERE prompt_id = p_prompt_id;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO your_user;
