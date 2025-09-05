-- Mock Data for Prompt Pipeline Error Tracking
-- This creates realistic test data that demonstrates errors testing misses

-- Insert MLflow Integration Data
INSERT INTO mlflow_integration (mlflow_run_id, experiment_name, run_name, status) VALUES
('run_001_abc123', 'llm-gateway-prod', 'qa_document_processing_001', 'active'),
('run_002_def456', 'llm-gateway-prod', 'rag_query_processing_002', 'active'),
('run_003_ghi789', 'llm-gateway-prod', 'sme_context_analysis_003', 'active'),
('run_004_jkl012', 'llm-gateway-prod', 'table_extraction_004', 'active'),
('run_005_mno345', 'llm-gateway-prod', 'live_context_integration_005', 'active'),
('run_006_pqr678', 'llm-gateway-prod', 'batch_processing_006', 'active'),
('run_007_stu901', 'llm-gateway-prod', 'error_tracking_test_007', 'active'),
('run_008_vwx234', 'llm-gateway-prod', 'performance_monitoring_008', 'active'),
('run_009_yz0567', 'llm-gateway-prod', 'cost_optimization_009', 'active'),
('run_010_abc890', 'llm-gateway-prod', 'security_monitoring_010', 'active');

-- Insert Prompt History Data (without duplicating MLflow metrics)
INSERT INTO prompt_history (prompt_id, timestamp, agent_name, task_type, model_used, prompt_text, response_text, success, error_message, mlflow_run_id, batch_id, user_id, session_id, metadata) VALUES
-- Successful prompts
('prompt_001', NOW() - INTERVAL '2 hours', 'qa_orchestrator', 'document_processing', 'gpt-4', 'Analyze this QA document for compliance issues', 'Document analysis completed successfully', true, NULL, 'run_001_abc123', 'batch_001', 'user_001', 'session_001', '{"document_type": "qa_report", "priority": "high"}'),
('prompt_002', NOW() - INTERVAL '1 hour 45 minutes', 'rag_qa_orchestrator', 'retrieval', 'gpt-4', 'Search for information about regulatory compliance', 'Found 5 relevant documents', true, NULL, 'run_002_def456', 'batch_001', 'user_001', 'session_001', '{"search_type": "regulatory", "results_count": 5}'),
('prompt_003', NOW() - INTERVAL '1 hour 30 minutes', 'sme_context_coordinator', 'analysis', 'gpt-4', 'Extract key insights from technical documentation', 'Extracted 12 key insights', true, NULL, 'run_003_ghi789', 'batch_001', 'user_002', 'session_002', '{"insights_count": 12, "confidence": 0.95}'),

-- Prompts with errors (what testing misses)
('prompt_004', NOW() - INTERVAL '1 hour 15 minutes', 'table_extractor_worker', 'table_processing', 'gpt-4', 'Extract table data from complex PDF', NULL, false, 'Table header parsing failed: object of type TableHeader has no len()', 'run_004_jkl012', 'batch_002', 'user_003', 'session_003', '{"pdf_complexity": "high", "table_count": 15}'),
('prompt_005', NOW() - INTERVAL '1 hour', 'live_context_worker', 'context_integration', 'gpt-4', 'Integrate live event data with document analysis', NULL, false, 'Database connection timeout after 30 seconds', 'run_005_mno345', 'batch_002', 'user_003', 'session_003', '{"event_count": 1000, "timeout_threshold": 30}'),
('prompt_006', NOW() - INTERVAL '45 minutes', 'batch_processor', 'batch_processing', 'gpt-4', 'Process 10,000 document records', NULL, false, 'Memory exhaustion: Cannot allocate 2GB for batch processing', 'run_006_pqr678', 'batch_003', 'user_004', 'session_004', '{"batch_size": 10000, "memory_required": "2GB"}'),
('prompt_007', NOW() - INTERVAL '30 minutes', 'error_tracker', 'error_monitoring', 'gpt-4', 'Analyze error patterns across system', NULL, false, 'Rate limit exceeded: 1000 requests per minute', 'run_007_stu901', 'batch_003', 'user_004', 'session_004', '{"request_count": 1000, "rate_limit": 1000}'),
('prompt_008', NOW() - INTERVAL '15 minutes', 'performance_monitor', 'performance_analysis', 'gpt-4', 'Monitor system performance metrics', NULL, false, 'External API timeout: HuggingFace model loading failed', 'run_008_vwx234', 'batch_004', 'user_005', 'session_005', '{"api_provider": "huggingface", "model_size": "large"}'),
('prompt_009', NOW() - INTERVAL '10 minutes', 'cost_optimizer', 'cost_analysis', 'gpt-4', 'Optimize LLM usage costs', NULL, false, 'Cost threshold exceeded: $150 in last hour', 'run_009_yz0567', 'batch_004', 'user_005', 'session_005', '{"cost_threshold": 150, "time_window": "1h"}'),
('prompt_010', NOW() - INTERVAL '5 minutes', 'security_monitor', 'security_analysis', 'gpt-4', 'Detect security violations in prompts', NULL, false, 'Security violation detected: Prompt injection attempt', 'run_010_abc890', 'batch_005', 'user_006', 'session_006', '{"security_level": "high", "violation_type": "injection"}'),

-- More recent prompts with various issues
('prompt_011', NOW() - INTERVAL '4 minutes', 'qa_orchestrator', 'document_processing', 'gpt-4', 'Process extremely long document with 50,000 words', NULL, false, 'Token limit exceeded: Input exceeds 8192 tokens', 'run_001_abc123', 'batch_005', 'user_007', 'session_007', '{"document_length": 50000, "token_limit": 8192}'),
('prompt_012', NOW() - INTERVAL '3 minutes', 'rag_qa_orchestrator', 'retrieval', 'gpt-4', 'Search across 1 million documents', NULL, false, 'Vector database connection pool exhausted', 'run_002_def456', 'batch_005', 'user_007', 'session_007', '{"document_count": 1000000, "connection_pool_size": 10}'),
('prompt_013', NOW() - INTERVAL '2 minutes', 'sme_context_coordinator', 'analysis', 'gpt-4', 'Analyze document with special characters: ñáéíóú', NULL, false, 'Encoding error: UTF-8 conversion failed', 'run_003_ghi789', 'batch_006', 'user_008', 'session_008', '{"encoding": "utf-8", "special_chars": true}'),
('prompt_014', NOW() - INTERVAL '1 minute', 'table_extractor_worker', 'table_processing', 'gpt-4', 'Extract data from corrupted PDF file', NULL, false, 'PDF corruption detected: Cannot read file structure', 'run_004_jkl012', 'batch_006', 'user_008', 'session_008', '{"pdf_status": "corrupted", "file_size": "2MB"}'),
('prompt_015', NOW(), 'live_context_worker', 'context_integration', 'gpt-4', 'Integrate real-time stock market data', NULL, false, 'External API authentication failed: Invalid API key', 'run_005_mno345', 'batch_006', 'user_009', 'session_009', '{"api_provider": "stock_market", "auth_type": "api_key"}');

-- Insert Error Tracking Data (the errors that testing misses)
INSERT INTO prompt_pipeline_errors (error_id, timestamp, severity, error_type, error_message, prompt_id, mlflow_run_id, agent_name, task_type, model_used, context_data, stack_trace, resolution_status, resolution_notes) VALUES
-- Critical Errors (immediate alerts)
('error_001', NOW() - INTERVAL '1 hour 15 minutes', 'critical', 'table_parsing_failure', 'Table header parsing failed: object of type TableHeader has no len()', 'prompt_004', 'run_004_jkl012', 'table_extractor_worker', 'table_processing', 'gpt-4', '{"pdf_complexity": "high", "table_count": 15, "user_facing": true}', 'Traceback (most recent call last):\n  File "table_extractor.py", line 45, in table_parser\n    header_length = len(table_header)\nTypeError: object of type TableHeader has no len()', 'open', NULL),
('error_002', NOW() - INTERVAL '1 hour', 'critical', 'database_timeout', 'Database connection timeout after 30 seconds', 'prompt_005', 'run_005_mno345', 'live_context_worker', 'context_integration', 'gpt-4', '{"event_count": 1000, "timeout_threshold": 30, "user_facing": true}', 'Traceback (most recent call last):\n  File "live_context.py", line 78, in db_connection\n    connection = psycopg2.connect(timeout=30)\npsycopg2.OperationalError: connection timeout', 'investigating', 'Database connection pool exhausted, scaling up connections'),
('error_003', NOW() - INTERVAL '45 minutes', 'critical', 'memory_exhaustion', 'Memory exhaustion: Cannot allocate 2GB for batch processing', 'prompt_006', 'run_006_pqr678', 'batch_processor', 'batch_processing', 'gpt-4', '{"batch_size": 10000, "memory_required": "2GB", "user_facing": true}', 'Traceback (most recent call last):\n  File "batch_processor.py", line 123, in process_batch\n    data = load_large_dataset()\nMemoryError: Cannot allocate 2GB', 'open', NULL),
('error_004', NOW() - INTERVAL '30 minutes', 'critical', 'rate_limit_exceeded', 'Rate limit exceeded: 1000 requests per minute', 'prompt_007', 'run_007_stu901', 'error_tracker', 'error_monitoring', 'gpt-4', '{"request_count": 1000, "rate_limit": 1000, "user_facing": true}', 'Traceback (most recent call last):\n  File "error_tracker.py", line 89, in track_error\n    api_call()\nRateLimitError: 1000 requests per minute exceeded', 'resolved', 'Implemented rate limiting and request queuing'),
('error_005', NOW() - INTERVAL '15 minutes', 'critical', 'external_api_timeout', 'External API timeout: HuggingFace model loading failed', 'prompt_008', 'run_008_vwx234', 'performance_monitor', 'performance_analysis', 'gpt-4', '{"api_provider": "huggingface", "model_size": "large", "user_facing": true}', 'Traceback (most recent call last):\n  File "performance_monitor.py", line 156, in load_model\n    model = AutoModel.from_pretrained(model_name)\nTimeoutError: Model loading timeout', 'investigating', 'Switching to local model cache'),

-- Warning Errors (monitor and alert)
('error_006', NOW() - INTERVAL '1 hour 30 minutes', 'warning', 'high_latency', 'Response time 8.5 seconds exceeds threshold of 5 seconds', 'prompt_001', 'run_001_abc123', 'qa_orchestrator', 'document_processing', 'gpt-4', '{"response_time_ms": 8500, "threshold_ms": 5000, "user_facing": true}', NULL, 'resolved', 'Optimized document processing pipeline'),
('error_007', NOW() - INTERVAL '1 hour 20 minutes', 'warning', 'high_error_rate', 'Error rate 15% exceeds threshold of 10% for table_extractor_worker', 'prompt_004', 'run_004_jkl012', 'table_extractor_worker', 'table_processing', 'gpt-4', '{"error_rate": 15, "threshold": 10, "user_facing": true}', NULL, 'investigating', 'Investigating table parsing algorithm'),
('error_008', NOW() - INTERVAL '1 hour 10 minutes', 'warning', 'model_degradation', 'Confidence score 0.65 below threshold of 0.7', 'prompt_002', 'run_002_def456', 'rag_qa_orchestrator', 'retrieval', 'gpt-4', '{"confidence_score": 0.65, "threshold": 0.7, "user_facing": true}', NULL, 'open', NULL),
('error_009', NOW() - INTERVAL '50 minutes', 'warning', 'resource_exhaustion', 'CPU usage 85% exceeds threshold of 80%', 'prompt_006', 'run_006_pqr678', 'batch_processor', 'batch_processing', 'gpt-4', '{"cpu_usage": 85, "threshold": 80, "user_facing": false}', NULL, 'resolved', 'Scaled up CPU resources'),
('error_010', NOW() - INTERVAL '40 minutes', 'warning', 'cost_threshold', 'Cost $120 exceeds threshold of $100 per hour', 'prompt_009', 'run_009_yz0567', 'cost_optimizer', 'cost_analysis', 'gpt-4', '{"cost_usd": 120, "threshold": 100, "user_facing": false}', NULL, 'resolved', 'Switched to more cost-effective model'),

-- Info Errors (log only)
('error_011', NOW() - INTERVAL '2 hours', 'info', 'retry_success', 'Request succeeded after 2 retries', 'prompt_001', 'run_001_abc123', 'qa_orchestrator', 'document_processing', 'gpt-4', '{"retry_count": 2, "user_facing": false}', NULL, 'resolved', 'Normal retry behavior'),
('error_012', NOW() - INTERVAL '1 hour 50 minutes', 'info', 'fallback_used', 'Used fallback model due to primary model unavailability', 'prompt_003', 'run_003_ghi789', 'sme_context_coordinator', 'analysis', 'gpt-3.5-turbo', '{"fallback_reason": "model_unavailable", "user_facing": false}', NULL, 'resolved', 'Graceful degradation successful'),
('error_013', NOW() - INTERVAL '1 hour 40 minutes', 'info', 'cache_miss', 'Cache miss for document analysis request', 'prompt_002', 'run_002_def456', 'rag_qa_orchestrator', 'retrieval', 'gpt-4', '{"cache_hit_rate": 0.85, "user_facing": false}', NULL, 'open', NULL),
('error_014', NOW() - INTERVAL '1 hour 35 minutes', 'info', 'rate_limit_warning', 'Approaching rate limit: 950/1000 requests per minute', 'prompt_007', 'run_007_stu901', 'error_tracker', 'error_monitoring', 'gpt-4', '{"request_count": 950, "rate_limit": 1000, "user_facing": false}', NULL, 'resolved', 'Rate limiting working as expected'),
('error_015', NOW() - INTERVAL '1 hour 25 minutes', 'info', 'performance_degradation', 'Response time increased by 20% over last hour', 'prompt_008', 'run_008_vwx234', 'performance_monitor', 'performance_analysis', 'gpt-4', '{"performance_decrease": 20, "user_facing": false}', NULL, 'open', NULL);

-- Insert Error Patterns (for intelligent alerting)
INSERT INTO error_patterns (pattern_id, error_type, pattern_description, frequency_threshold, time_window_minutes, severity, auto_resolution_action, is_active) VALUES
('pattern_001', 'table_parsing_failure', 'Table parsing failures occurring frequently with complex PDFs', 3, 60, 'critical', 'switch_to_alternative_parser', true),
('pattern_002', 'database_timeout', 'Database timeouts during high load periods', 5, 30, 'critical', 'scale_database_connections', true),
('pattern_003', 'memory_exhaustion', 'Memory exhaustion with large batch processing', 2, 120, 'critical', 'reduce_batch_size', true),
('pattern_004', 'rate_limit_exceeded', 'Rate limit exceeded due to high request volume', 3, 15, 'critical', 'implement_request_queuing', true),
('pattern_005', 'external_api_timeout', 'External API timeouts affecting model loading', 4, 60, 'warning', 'switch_to_local_cache', true),
('pattern_006', 'high_latency', 'High latency patterns affecting user experience', 10, 30, 'warning', 'optimize_processing_pipeline', true),
('pattern_007', 'high_error_rate', 'High error rates for specific agents', 5, 60, 'warning', 'investigate_agent_health', true),
('pattern_008', 'model_degradation', 'Model confidence degradation over time', 3, 120, 'warning', 'retrain_model', true),
('pattern_009', 'cost_threshold', 'Cost threshold exceeded due to high usage', 2, 60, 'warning', 'switch_to_cost_effective_model', true),
('pattern_010', 'security_violation', 'Security violations detected in prompts', 1, 5, 'critical', 'block_user_session', true);

-- Insert Alert History
INSERT INTO alert_history (alert_id, error_id, alert_type, recipient, message, sent_at, status) VALUES
('alert_001', 'error_001', 'email', 'admin@company.com', '🚨 CRITICAL ALERT: Table parsing failure detected for table_extractor_worker', NOW() - INTERVAL '1 hour 15 minutes', 'acknowledged'),
('alert_002', 'error_002', 'slack', '#alerts', '🚨 CRITICAL ALERT: Database timeout affecting live context integration', NOW() - INTERVAL '1 hour', 'delivered'),
('alert_003', 'error_003', 'sms', '+1234567890', '🚨 CRITICAL ALERT: Memory exhaustion in batch processing', NOW() - INTERVAL '45 minutes', 'sent'),
('alert_004', 'error_004', 'email', 'admin@company.com', '🚨 CRITICAL ALERT: Rate limit exceeded for error tracking system', NOW() - INTERVAL '30 minutes', 'acknowledged'),
('alert_005', 'error_005', 'slack', '#alerts', '🚨 CRITICAL ALERT: External API timeout for HuggingFace model loading', NOW() - INTERVAL '15 minutes', 'delivered'),
('alert_006', 'error_006', 'dashboard', 'performance_dashboard', '⚠️ WARNING: High latency detected for qa_orchestrator', NOW() - INTERVAL '1 hour 30 minutes', 'acknowledged'),
('alert_007', 'error_007', 'email', 'dev@company.com', '⚠️ WARNING: High error rate for table_extractor_worker', NOW() - INTERVAL '1 hour 20 minutes', 'sent'),
('alert_008', 'error_008', 'dashboard', 'model_dashboard', '⚠️ WARNING: Model degradation detected for rag_qa_orchestrator', NOW() - INTERVAL '1 hour 10 minutes', 'delivered'),
('alert_009', 'error_009', 'slack', '#infrastructure', '⚠️ WARNING: High CPU usage detected', NOW() - INTERVAL '50 minutes', 'acknowledged'),
('alert_010', 'error_010', 'email', 'finance@company.com', '⚠️ WARNING: Cost threshold exceeded', NOW() - INTERVAL '40 minutes', 'acknowledged');

-- Insert Real-time Context Data
INSERT INTO real_time_context (prompt_id, context_type, context_data, relevance_score) VALUES
('prompt_001', 'live_events', '{"event_type": "document_upload", "user_activity": "high", "system_load": "medium"}', 0.9),
('prompt_002', 'user_session', '{"session_duration": "45m", "queries_count": 12, "user_preferences": {"language": "en"}}', 0.8),
('prompt_003', 'system_status', '{"cpu_usage": 75, "memory_usage": 60, "active_connections": 25}', 0.7),
('prompt_004', 'live_events', '{"event_type": "pdf_processing", "file_size": "15MB", "complexity": "high"}', 0.95),
('prompt_005', 'system_status', '{"database_connections": 95, "queue_size": 150, "response_time": "slow"}', 0.9),
('prompt_006', 'live_events', '{"event_type": "batch_processing", "batch_size": 10000, "priority": "high"}', 0.85),
('prompt_007', 'user_session', '{"session_duration": "2h", "queries_count": 50, "error_count": 8}', 0.8),
('prompt_008', 'system_status', '{"external_api_status": "degraded", "cache_hit_rate": 0.3, "model_loading_time": "slow"}', 0.9),
('prompt_009', 'live_events', '{"event_type": "cost_monitoring", "hourly_cost": 120, "budget_alert": true}', 0.95),
('prompt_010', 'user_session', '{"session_duration": "10m", "suspicious_activity": true, "security_level": "high"}', 0.9);

-- Insert Batch Processing Status
INSERT INTO batch_processing_status (batch_id, status, total_items, processed_items, failed_items, start_time, end_time, error_message) VALUES
('batch_001', 'completed', 100, 98, 2, NOW() - INTERVAL '3 hours', NOW() - INTERVAL '2 hours 30 minutes', NULL),
('batch_002', 'failed', 50, 25, 25, NOW() - INTERVAL '2 hours', NOW() - INTERVAL '1 hour 45 minutes', 'Memory exhaustion during processing'),
('batch_003', 'processing', 1000, 750, 50, NOW() - INTERVAL '1 hour', NULL, NULL),
('batch_004', 'pending', 500, 0, 0, NULL, NULL, NULL),
('batch_005', 'completed', 200, 195, 5, NOW() - INTERVAL '30 minutes', NOW() - INTERVAL '20 minutes', 'Some records failed due to validation errors'),
('batch_006', 'failed', 100, 0, 100, NOW() - INTERVAL '15 minutes', NOW() - INTERVAL '10 minutes', 'Database connection timeout');

-- Update some errors to show resolution
UPDATE prompt_pipeline_errors 
SET resolution_status = 'resolved', 
    resolution_notes = 'Implemented retry mechanism with exponential backoff',
    updated_at = NOW()
WHERE error_id IN ('error_004', 'error_006', 'error_009', 'error_010');

UPDATE prompt_pipeline_errors 
SET resolution_status = 'investigating', 
    resolution_notes = 'Database team investigating connection pool issues',
    updated_at = NOW()
WHERE error_id IN ('error_002', 'error_005', 'error_007');
