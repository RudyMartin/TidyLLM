-- Simple Error Tracking Schema
-- Basic tables for error tracking without complex functions

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- MLflow Integration Table
CREATE TABLE IF NOT EXISTS mlflow_integration (
    id SERIAL PRIMARY KEY,
    mlflow_run_id VARCHAR(255) UNIQUE NOT NULL,
    experiment_name VARCHAR(255),
    run_name VARCHAR(255),
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Prompt History Table
CREATE TABLE IF NOT EXISTS prompt_history (
    id SERIAL PRIMARY KEY,
    prompt_id VARCHAR(255) UNIQUE NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    prompt_text TEXT NOT NULL,
    response_text TEXT,
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

-- Error Tracking Table
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
    context_type VARCHAR(50) NOT NULL,
    context_data JSONB NOT NULL,
    relevance_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Batch Processing Status Table
CREATE TABLE IF NOT EXISTS batch_processing_status (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    total_items INTEGER,
    processed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_prompt_history_timestamp ON prompt_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_prompt_history_agent_name ON prompt_history(agent_name);
CREATE INDEX IF NOT EXISTS idx_prompt_history_success ON prompt_history(success);
CREATE INDEX IF NOT EXISTS idx_prompt_pipeline_errors_timestamp ON prompt_pipeline_errors(timestamp);
CREATE INDEX IF NOT EXISTS idx_prompt_pipeline_errors_severity ON prompt_pipeline_errors(severity);
CREATE INDEX IF NOT EXISTS idx_prompt_pipeline_errors_error_type ON prompt_pipeline_errors(error_type);
CREATE INDEX IF NOT EXISTS idx_alert_history_sent_at ON alert_history(sent_at);
CREATE INDEX IF NOT EXISTS idx_alert_history_status ON alert_history(status);
CREATE INDEX IF NOT EXISTS idx_batch_processing_status_batch_id ON batch_processing_status(batch_id);
CREATE INDEX IF NOT EXISTS idx_batch_processing_status_status ON batch_processing_status(status);

