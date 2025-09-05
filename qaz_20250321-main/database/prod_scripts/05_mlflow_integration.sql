-- =============================================================================
-- MLflow Integration Database Schema
-- =============================================================================
-- This script creates the database schema for MLflow integration
-- with the Unified LLM Gateway system
-- 
-- Run this after setting up PostgreSQL and before starting MLflow server

-- =============================================================================
-- 1. MLflow Core Tables (Required by MLflow)
-- =============================================================================

-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    artifact_location VARCHAR(256),
    lifecycle_stage VARCHAR(32)
);

-- Runs table
CREATE TABLE IF NOT EXISTS runs (
    run_uuid VARCHAR(32) PRIMARY KEY,
    name VARCHAR(250),
    source_type VARCHAR(32),
    source_name VARCHAR(500),
    entry_point_name VARCHAR(200),
    user_id VARCHAR(256),
    status VARCHAR(9),
    start_time BIGINT,
    end_time BIGINT,
    source_version VARCHAR(50),
    lifecycle_stage VARCHAR(32),
    artifact_uri VARCHAR(200),
    experiment_id INTEGER,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- Parameters table
CREATE TABLE IF NOT EXISTS params (
    key VARCHAR(250) NOT NULL,
    value VARCHAR(250) NOT NULL,
    run_uuid VARCHAR(32) NOT NULL,
    PRIMARY KEY (key, run_uuid),
    FOREIGN KEY (run_uuid) REFERENCES runs(run_uuid) ON DELETE CASCADE
);

-- Metrics table
CREATE TABLE IF NOT EXISTS metrics (
    key VARCHAR(250) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    timestamp BIGINT NOT NULL,
    run_uuid VARCHAR(32) NOT NULL,
    step BIGINT DEFAULT 0,
    PRIMARY KEY (key, run_uuid, timestamp, step),
    FOREIGN KEY (run_uuid) REFERENCES runs(run_uuid) ON DELETE CASCADE
);

-- Tags table
CREATE TABLE IF NOT EXISTS tags (
    key VARCHAR(250) NOT NULL,
    value VARCHAR(250) NOT NULL,
    run_uuid VARCHAR(32) NOT NULL,
    PRIMARY KEY (key, run_uuid),
    FOREIGN KEY (run_uuid) REFERENCES runs(run_uuid) ON DELETE CASCADE
);

-- =============================================================================
-- 2. Unified LLM Gateway Integration Tables
-- =============================================================================

-- LLM Gateway Configuration
CREATE TABLE IF NOT EXISTS llm_gateway_config (
    id SERIAL PRIMARY KEY,
    gateway_name VARCHAR(100) NOT NULL UNIQUE,
    gateway_type VARCHAR(50) NOT NULL, -- 'local', 'remote', 'unified'
    base_url VARCHAR(500) NOT NULL,
    api_key VARCHAR(500), -- Encrypted
    model_preferences JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- LLM Call History (Enhanced tracking)
CREATE TABLE IF NOT EXISTS llm_call_history (
    id BIGSERIAL PRIMARY KEY,
    run_uuid VARCHAR(32) REFERENCES runs(run_uuid) ON DELETE CASCADE,
    agent_name VARCHAR(100) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    gateway_type VARCHAR(50) NOT NULL, -- 'local', 'remote'
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    cost_usd NUMERIC(10,6),
    response_time_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    prompt_hash VARCHAR(64), -- SHA-256 hash of prompt
    response_hash VARCHAR(64), -- SHA-256 hash of response
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Batch Processing Tracking
CREATE TABLE IF NOT EXISTS llm_batch_processing (
    batch_id VARCHAR(100) PRIMARY KEY,
    experiment_name VARCHAR(256) NOT NULL,
    total_calls INTEGER DEFAULT 0,
    successful_calls INTEGER DEFAULT 0,
    failed_calls INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost_usd NUMERIC(10,6) DEFAULT 0,
    avg_response_time_ms INTEGER,
    budget_limit_usd NUMERIC(10,6),
    status VARCHAR(50) DEFAULT 'running', -- 'running', 'completed', 'failed'
    started_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    created_by VARCHAR(100)
);

-- =============================================================================
-- 3. Enhanced Document Processing Tables
-- =============================================================================

-- Document Processing Sessions
CREATE TABLE IF NOT EXISTS document_processing_sessions (
    session_id VARCHAR(100) PRIMARY KEY,
    batch_id VARCHAR(100) REFERENCES llm_batch_processing(batch_id),
    experiment_name VARCHAR(256) NOT NULL,
    input_directory VARCHAR(500),
    output_directory VARCHAR(500),
    total_documents INTEGER DEFAULT 0,
    processed_documents INTEGER DEFAULT 0,
    failed_documents INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'running',
    started_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    metadata JSONB
);

-- Document Classification Results
CREATE TABLE IF NOT EXISTS document_classifications (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(100) REFERENCES document_processing_sessions(session_id),
    run_uuid VARCHAR(32) REFERENCES runs(run_uuid),
    document_filename VARCHAR(500) NOT NULL,
    document_hash VARCHAR(64),
    classification VARCHAR(100),
    confidence_score NUMERIC(5,2),
    key_themes TEXT[],
    document_purpose TEXT,
    processing_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Document Metadata Extraction
CREATE TABLE IF NOT EXISTS document_metadata (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(100) REFERENCES document_processing_sessions(session_id),
    run_uuid VARCHAR(32) REFERENCES runs(run_uuid),
    document_filename VARCHAR(500) NOT NULL,
    title VARCHAR(500),
    authors TEXT[],
    document_date DATE,
    version VARCHAR(50),
    organization VARCHAR(200),
    document_type VARCHAR(100),
    key_topics TEXT[],
    summary TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- =============================================================================
-- 4. Performance and Analytics Tables
-- =============================================================================

-- Gateway Performance Metrics
CREATE TABLE IF NOT EXISTS gateway_performance (
    id BIGSERIAL PRIMARY KEY,
    gateway_type VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    date_hour TIMESTAMPTZ NOT NULL,
    total_calls INTEGER DEFAULT 0,
    successful_calls INTEGER DEFAULT 0,
    failed_calls INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost_usd NUMERIC(10,6) DEFAULT 0,
    avg_response_time_ms INTEGER,
    p50_response_time_ms INTEGER,
    p95_response_time_ms INTEGER,
    p99_response_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(gateway_type, model_name, date_hour)
);

-- Cost Tracking by Experiment
CREATE TABLE IF NOT EXISTS experiment_cost_tracking (
    id BIGSERIAL PRIMARY KEY,
    experiment_name VARCHAR(256) NOT NULL,
    date DATE NOT NULL,
    total_calls INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost_usd NUMERIC(10,6) DEFAULT 0,
    avg_cost_per_call NUMERIC(10,6),
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(experiment_name, date)
);

-- =============================================================================
-- 5. Indexes for Performance
-- =============================================================================

-- MLflow indexes
CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_user_id ON runs(user_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_start_time ON runs(start_time);
CREATE INDEX IF NOT EXISTS idx_metrics_run_uuid ON metrics(run_uuid);
CREATE INDEX IF NOT EXISTS idx_metrics_key ON metrics(key);
CREATE INDEX IF NOT EXISTS idx_params_run_uuid ON params(run_uuid);
CREATE INDEX IF NOT EXISTS idx_tags_run_uuid ON tags(run_uuid);

-- LLM Gateway indexes
CREATE INDEX IF NOT EXISTS idx_llm_call_history_run_uuid ON llm_call_history(run_uuid);
CREATE INDEX IF NOT EXISTS idx_llm_call_history_agent_name ON llm_call_history(agent_name);
CREATE INDEX IF NOT EXISTS idx_llm_call_history_task_type ON llm_call_history(task_type);
CREATE INDEX IF NOT EXISTS idx_llm_call_history_gateway_type ON llm_call_history(gateway_type);
CREATE INDEX IF NOT EXISTS idx_llm_call_history_created_at ON llm_call_history(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_call_history_success ON llm_call_history(success);

-- Document processing indexes
CREATE INDEX IF NOT EXISTS idx_document_classifications_session_id ON document_classifications(session_id);
CREATE INDEX IF NOT EXISTS idx_document_classifications_classification ON document_classifications(classification);
CREATE INDEX IF NOT EXISTS idx_document_metadata_session_id ON document_metadata(session_id);
CREATE INDEX IF NOT EXISTS idx_document_metadata_document_type ON document_metadata(document_type);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_gateway_performance_date_hour ON gateway_performance(date_hour);
CREATE INDEX IF NOT EXISTS idx_gateway_performance_gateway_model ON gateway_performance(gateway_type, model_name);
CREATE INDEX IF NOT EXISTS idx_experiment_cost_tracking_date ON experiment_cost_tracking(date);
CREATE INDEX IF NOT EXISTS idx_experiment_cost_tracking_experiment ON experiment_cost_tracking(experiment_name);

-- =============================================================================
-- 6. Views for Analytics
-- =============================================================================

-- LLM Gateway Performance Summary
CREATE OR REPLACE VIEW llm_gateway_performance_summary AS
SELECT 
    gateway_type,
    model_name,
    COUNT(*) as total_calls,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_calls,
    SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed_calls,
    ROUND(SUM(CASE WHEN success THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as success_rate,
    SUM(total_tokens) as total_tokens,
    SUM(cost_usd) as total_cost_usd,
    ROUND(AVG(response_time_ms), 2) as avg_response_time_ms,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time_ms), 2) as p50_response_time_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms), 2) as p95_response_time_ms
FROM llm_call_history
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY gateway_type, model_name
ORDER BY total_calls DESC;

-- Experiment Cost Summary
CREATE OR REPLACE VIEW experiment_cost_summary AS
SELECT 
    e.name as experiment_name,
    COUNT(r.run_uuid) as total_runs,
    SUM(lch.total_tokens) as total_tokens,
    SUM(lch.cost_usd) as total_cost_usd,
    ROUND(AVG(lch.cost_usd), 6) as avg_cost_per_call,
    MIN(r.start_time) as first_run,
    MAX(r.start_time) as last_run
FROM experiments e
LEFT JOIN runs r ON e.experiment_id = r.experiment_id
LEFT JOIN llm_call_history lch ON r.run_uuid = lch.run_uuid
WHERE r.start_time >= EXTRACT(EPOCH FROM (NOW() - INTERVAL '7 days')) * 1000
GROUP BY e.experiment_id, e.name
ORDER BY total_cost_usd DESC;

-- Document Processing Summary
CREATE OR REPLACE VIEW document_processing_summary AS
SELECT 
    dps.session_id,
    dps.experiment_name,
    dps.total_documents,
    dps.processed_documents,
    dps.failed_documents,
    ROUND(dps.processed_documents * 100.0 / dps.total_documents, 2) as success_rate,
    COUNT(DISTINCT dc.classification) as unique_classifications,
    COUNT(DISTINCT dm.document_type) as unique_document_types,
    dps.started_at,
    dps.completed_at,
    EXTRACT(EPOCH FROM (dps.completed_at - dps.started_at)) as processing_time_seconds
FROM document_processing_sessions dps
LEFT JOIN document_classifications dc ON dps.session_id = dc.session_id
LEFT JOIN document_metadata dm ON dps.session_id = dm.session_id
GROUP BY dps.session_id, dps.experiment_name, dps.total_documents, dps.processed_documents, 
         dps.failed_documents, dps.started_at, dps.completed_at
ORDER BY dps.started_at DESC;

-- =============================================================================
-- 7. Functions for Data Management
-- =============================================================================

-- Function to update gateway performance metrics
CREATE OR REPLACE FUNCTION update_gateway_performance()
RETURNS void AS $$
BEGIN
    INSERT INTO gateway_performance (
        gateway_type, model_name, date_hour, total_calls, successful_calls, 
        failed_calls, total_tokens, total_cost_usd, avg_response_time_ms,
        p50_response_time_ms, p95_response_time_ms, p99_response_time_ms
    )
    SELECT 
        gateway_type,
        model_name,
        DATE_TRUNC('hour', created_at) as date_hour,
        COUNT(*) as total_calls,
        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_calls,
        SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed_calls,
        SUM(total_tokens) as total_tokens,
        SUM(cost_usd) as total_cost_usd,
        ROUND(AVG(response_time_ms)) as avg_response_time_ms,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time_ms)) as p50_response_time_ms,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms)) as p95_response_time_ms,
        ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms)) as p99_response_time_ms
    FROM llm_call_history
    WHERE created_at >= NOW() - INTERVAL '1 hour'
    GROUP BY gateway_type, model_name, DATE_TRUNC('hour', created_at)
    ON CONFLICT (gateway_type, model_name, date_hour) 
    DO UPDATE SET
        total_calls = EXCLUDED.total_calls,
        successful_calls = EXCLUDED.successful_calls,
        failed_calls = EXCLUDED.failed_calls,
        total_tokens = EXCLUDED.total_tokens,
        total_cost_usd = EXCLUDED.total_cost_usd,
        avg_response_time_ms = EXCLUDED.avg_response_time_ms,
        p50_response_time_ms = EXCLUDED.p50_response_time_ms,
        p95_response_time_ms = EXCLUDED.p95_response_time_ms,
        p99_response_time_ms = EXCLUDED.p99_response_time_ms;
END;
$$ LANGUAGE plpgsql;

-- Function to update experiment cost tracking
CREATE OR REPLACE FUNCTION update_experiment_cost_tracking()
RETURNS void AS $$
BEGIN
    INSERT INTO experiment_cost_tracking (
        experiment_name, date, total_calls, total_tokens, total_cost_usd, avg_cost_per_call
    )
    SELECT 
        e.name as experiment_name,
        DATE(FROM_UNIXTIME(r.start_time / 1000)) as date,
        COUNT(lch.id) as total_calls,
        SUM(lch.total_tokens) as total_tokens,
        SUM(lch.cost_usd) as total_cost_usd,
        ROUND(AVG(lch.cost_usd), 6) as avg_cost_per_call
    FROM experiments e
    JOIN runs r ON e.experiment_id = r.experiment_id
    JOIN llm_call_history lch ON r.run_uuid = lch.run_uuid
    WHERE DATE(FROM_UNIXTIME(r.start_time / 1000)) = CURRENT_DATE
    GROUP BY e.name, DATE(FROM_UNIXTIME(r.start_time / 1000))
    ON CONFLICT (experiment_name, date) 
    DO UPDATE SET
        total_calls = EXCLUDED.total_calls,
        total_tokens = EXCLUDED.total_tokens,
        total_cost_usd = EXCLUDED.total_cost_usd,
        avg_cost_per_call = EXCLUDED.avg_cost_per_call;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 8. Triggers for Automatic Updates
-- =============================================================================

-- Trigger to update gateway performance when new calls are added
CREATE OR REPLACE FUNCTION trigger_update_gateway_performance()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM update_gateway_performance();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_gateway_performance_trigger
    AFTER INSERT ON llm_call_history
    FOR EACH ROW
    EXECUTE FUNCTION trigger_update_gateway_performance();

-- Trigger to update experiment cost tracking when new calls are added
CREATE OR REPLACE FUNCTION trigger_update_experiment_cost()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM update_experiment_cost_tracking();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_experiment_cost_trigger
    AFTER INSERT ON llm_call_history
    FOR EACH ROW
    EXECUTE FUNCTION trigger_update_experiment_cost();

-- =============================================================================
-- 9. Initial Data
-- =============================================================================

-- Insert default gateway configurations
INSERT INTO llm_gateway_config (gateway_name, gateway_type, base_url, model_preferences) VALUES
('local-zllm', 'local', 'http://localhost:11434', '{"default": "llama2", "classification": "gpt-4", "summarization": "llama2"}'),
('remote-openai', 'remote', 'https://api.openai.com/v1', '{"default": "gpt-3.5-turbo", "classification": "gpt-4", "summarization": "gpt-3.5-turbo"}'),
('unified-gateway', 'unified', 'http://localhost:5000', '{"default": "auto", "classification": "auto", "summarization": "auto"}')
ON CONFLICT (gateway_name) DO NOTHING;

-- =============================================================================
-- 10. Verification Queries
-- =============================================================================

-- Verify MLflow tables
SELECT 'MLflow Core Tables' as category, COUNT(*) as table_count FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name IN ('experiments', 'runs', 'params', 'metrics', 'tags');

-- Verify LLM Gateway tables
SELECT 'LLM Gateway Tables' as category, COUNT(*) as table_count FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name IN ('llm_gateway_config', 'llm_call_history', 'llm_batch_processing');

-- Verify Document Processing tables
SELECT 'Document Processing Tables' as category, COUNT(*) as table_count FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name IN ('document_processing_sessions', 'document_classifications', 'document_metadata');

-- Verify Performance tables
SELECT 'Performance Tables' as category, COUNT(*) as table_count FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name IN ('gateway_performance', 'experiment_cost_tracking');

-- Show all created tables
SELECT 
    table_name,
    pg_size_pretty(pg_total_relation_size(table_name::regclass)) as size
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY pg_total_relation_size(table_name::regclass) DESC;

-- =============================================================================
-- SUCCESS MESSAGE
-- =============================================================================

SELECT '✅ MLflow Integration Database Schema created successfully!' as status;
SELECT '📊 Tables created: ' || COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'public';
SELECT '🔗 Ready to connect Unified LLM Gateway to MLflow!' as next_step;
