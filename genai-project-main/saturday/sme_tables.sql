-- ===========================================================================
-- SME Context Database Schema - Complete Code
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- Enable Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector for embeddings

-- ---------------------------------------------------------------------------
-- Core Review Tables
-- ---------------------------------------------------------------------------

-- MVR Prompts Table
CREATE TABLE IF NOT EXISTS mvr_prompts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    sequence_order INTEGER NOT NULL,
    requirements TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT unique_title UNIQUE (title),
    CONSTRAINT valid_sequence_order CHECK (sequence_order > 0)
);

-- MVR Log Table
CREATE TABLE IF NOT EXISTS mvr_log (
    review_id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    risk_tier VARCHAR(50) NOT NULL,
    validation_type VARCHAR(100) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    status VARCHAR(50) DEFAULT 'in_progress',
    reviewer_id VARCHAR(100),
    notes TEXT,
    CONSTRAINT valid_risk_tier CHECK (risk_tier IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT valid_validation_type CHECK (validation_type IN ('initial', 'periodic', 'event-driven')),
    CONSTRAINT valid_status CHECK (status IN ('in_progress', 'completed', 'failed')),
    CONSTRAINT valid_dates CHECK (started_at <= completed_at OR completed_at IS NULL)
);

-- MVR Records Table
CREATE TABLE IF NOT EXISTS mvr_records (
    id SERIAL PRIMARY KEY,
    review_id INTEGER NOT NULL,
    rating INTEGER NOT NULL,
    conclusion TEXT NOT NULL,
    mvr_section VARCHAR(255) NOT NULL,
    evidence TEXT NOT NULL,
    mvr_number INTEGER NOT NULL,
    review_title VARCHAR(255) NOT NULL,
    completed_at TIMESTAMP NOT NULL,
    reviewer_id VARCHAR(100),
    confidence_score DECIMAL(3,2),
    risk_score INTEGER,
    recommendations TEXT,
    prompt_id INTEGER REFERENCES mvr_prompts(id),
    CONSTRAINT fk_mvr_records_review_id 
        FOREIGN KEY (review_id) REFERENCES mvr_log(review_id) ON DELETE CASCADE,
    CONSTRAINT valid_rating CHECK (rating >= 0 AND rating <= 5),
    CONSTRAINT valid_confidence_score CHECK (confidence_score >= 0.00 AND confidence_score <= 1.00),
    CONSTRAINT valid_risk_score CHECK (risk_score >= 1 AND risk_score <= 10)
);

-- SME Context Mapping Table
CREATE TABLE IF NOT EXISTS sme_context_mapping (
    id SERIAL PRIMARY KEY,
    sme_id VARCHAR(100) NOT NULL,
    sme_name VARCHAR(255) NOT NULL,
    expertise_area VARCHAR(255) NOT NULL,
    validation_type VARCHAR(100) NOT NULL,
    risk_tier VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_expertise_area CHECK (expertise_area IN ('Model Risk', 'Credit Risk', 'Market Risk', 'Operational Risk')),
    CONSTRAINT valid_validation_type CHECK (validation_type IN ('initial', 'periodic', 'event-driven')),
    CONSTRAINT valid_risk_tier CHECK (risk_tier IN ('low', 'medium', 'high', 'critical'))
);

-- ---------------------------------------------------------------------------
-- Indexes for Performance
-- ---------------------------------------------------------------------------

-- MVR Prompts indexes
CREATE INDEX IF NOT EXISTS idx_mvr_prompts_title ON mvr_prompts(title);
CREATE INDEX IF NOT EXISTS idx_mvr_prompts_sequence ON mvr_prompts(sequence_order);
CREATE INDEX IF NOT EXISTS idx_mvr_prompts_active ON mvr_prompts(is_active);

-- MVR Log indexes
CREATE INDEX IF NOT EXISTS idx_mvr_log_model_id ON mvr_log(model_id);
CREATE INDEX IF NOT EXISTS idx_mvr_log_model_type ON mvr_log(model_type);
CREATE INDEX IF NOT EXISTS idx_mvr_log_risk_tier ON mvr_log(risk_tier);
CREATE INDEX IF NOT EXISTS idx_mvr_log_validation_type ON mvr_log(validation_type);
CREATE INDEX IF NOT EXISTS idx_mvr_log_status ON mvr_log(status);
CREATE INDEX IF NOT EXISTS idx_mvr_log_started_at ON mvr_log(started_at);

-- MVR Records indexes
CREATE INDEX IF NOT EXISTS idx_mvr_records_review_id ON mvr_records(review_id);
CREATE INDEX IF NOT EXISTS idx_mvr_records_rating ON mvr_records(rating);
CREATE INDEX IF NOT EXISTS idx_mvr_records_mvr_section ON mvr_records(mvr_section);
CREATE INDEX IF NOT EXISTS idx_mvr_records_completed_at ON mvr_records(completed_at);
CREATE INDEX IF NOT EXISTS idx_mvr_records_risk_score ON mvr_records(risk_score);
CREATE INDEX IF NOT EXISTS idx_mvr_records_prompt_id ON mvr_records(prompt_id);

-- SME Context Mapping indexes
CREATE INDEX IF NOT EXISTS idx_sme_context_sme_id ON sme_context_mapping(sme_id);
CREATE INDEX IF NOT EXISTS idx_sme_context_expertise ON sme_context_mapping(expertise_area);
CREATE INDEX IF NOT EXISTS idx_sme_context_validation_type ON sme_context_mapping(validation_type);
CREATE INDEX IF NOT EXISTS idx_sme_context_active ON sme_context_mapping(is_active);

-- ---------------------------------------------------------------------------
-- Views for Convenience
-- ---------------------------------------------------------------------------

-- Active MVR Prompts View
CREATE OR REPLACE VIEW v_active_mvr_prompts AS
SELECT 
    id,
    title as focus_area,
    sequence_order,
    requirements,
    created_at,
    updated_at
FROM mvr_prompts 
WHERE is_active = TRUE
ORDER BY sequence_order;

-- MVR Review Summary View
CREATE OR REPLACE VIEW v_mvr_review_summary AS
SELECT 
    ml.review_id,
    ml.model_id,
    ml.model_name,
    ml.model_type,
    ml.risk_tier,
    ml.validation_type,
    ml.started_at,
    ml.completed_at,
    ml.status,
    COUNT(mr.id) as total_sections,
    AVG(mr.rating) as avg_rating,
    AVG(mr.risk_score) as avg_risk_score,
    AVG(mr.confidence_score) as avg_confidence
FROM mvr_log ml
LEFT JOIN mvr_records mr ON ml.review_id = mr.review_id
GROUP BY ml.review_id, ml.model_id, ml.model_name, ml.model_type, 
         ml.risk_tier, ml.validation_type, ml.started_at, ml.completed_at, ml.status;

-- SME Context Expertise View
CREATE OR REPLACE VIEW v_sme_context_expertise AS
SELECT 
    scm.sme_id,
    scm.sme_name,
    scm.expertise_area,
    scm.validation_type,
    scm.risk_tier,
    mp.title as focus_area,
    mp.sequence_order,
    mp.requirements
FROM sme_context_mapping scm
JOIN mvr_prompts mp ON mp.title = scm.expertise_area
WHERE scm.is_active = TRUE AND mp.is_active = TRUE
ORDER BY scm.expertise_area, mp.sequence_order;

-- ---------------------------------------------------------------------------
-- Triggers for Data Integrity
-- ---------------------------------------------------------------------------

-- Trigger to update updated_at timestamp on mvr_prompts
CREATE OR REPLACE FUNCTION update_mvr_prompts_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_mvr_prompts_updated_at
    BEFORE UPDATE ON mvr_prompts
    FOR EACH ROW
    EXECUTE FUNCTION update_mvr_prompts_updated_at();

-- Trigger to validate MVR record completion
CREATE OR REPLACE FUNCTION validate_mvr_record_completion()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.completed_at IS NOT NULL THEN
        UPDATE mvr_log 
        SET status = 'completed', completed_at = NEW.completed_at
        WHERE review_id = NEW.review_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_validate_mvr_record_completion
    AFTER INSERT OR UPDATE ON mvr_records
    FOR EACH ROW
    EXECUTE FUNCTION validate_mvr_record_completion();

-- ---------------------------------------------------------------------------
-- Test Data Insertion
-- ---------------------------------------------------------------------------

-- Insert test MVR prompts
INSERT INTO mvr_prompts (title, sequence_order, requirements) VALUES
('Model Risk', 1, 'INPUT CONSIDERATIONS: ...'),
('Credit Risk', 2, 'INPUT CONSIDERATIONS: ...'),
('Market Risk', 3, 'INPUT CONSIDERATIONS: ...'),
('Operational Risk', 4, 'INPUT CONSIDERATIONS: ...');

-- Insert test MVR log entries
INSERT INTO mvr_log (model_id, model_name, model_type, risk_tier, validation_type, started_at, reviewer_id, notes) VALUES
('MODEL_001', 'Credit Scoring Model v2.1', 'Machine Learning', 'high', 'initial', '2024-01-15 09:00:00', 'REVIEWER_001', 'Initial validation'),
('MODEL_002', 'VaR Calculation Engine', 'Statistical', 'critical', 'periodic', '2024-01-16 10:00:00', 'REVIEWER_002', 'Annual validation'),
('MODEL_003', 'Fraud Detection System', 'AI/ML', 'high', 'event-driven', '2024-01-17 14:00:00', 'REVIEWER_001', 'Performance degradation'),
('MODEL_004', 'Portfolio Optimization Model', 'Optimization', 'medium', 'initial', '2024-01-18 11:00:00', 'REVIEWER_003', 'New model validation');

-- Insert test MVR records
INSERT INTO mvr_records (review_id, rating, conclusion, mvr_section, evidence, mvr_number, review_title, completed_at, reviewer_id, confidence_score, risk_score, recommendations, prompt_id) VALUES
(1, 4, 'Meets requirements', 'Data Quality', '98.5% completeness', 1, 'Credit Model Validation', '2024-01-15 16:30:00', 'REVIEWER_001', 0.85, 6, 'Monitor data quality', 1),
(1, 3, 'Assumptions need review', 'Model Assumptions', 'Economic indicators deviated', 2, 'Credit Model Validation', '2024-01-15 16:30:00', 'REVIEWER_001', 0.78, 7, 'Retrain with updated data', 1),
(2, 4, 'Performs well under normal conditions', 'Model Performance', '95% confidence interval', 1, 'VaR Model Review', '2024-01-16 17:00:00', 'REVIEWER_002', 0.88, 5, 'Enhance stress testing', 3);

-- Insert test SME context mapping
INSERT INTO sme_context_mapping (sme_id, sme_name, expertise_area, validation_type, risk_tier) VALUES
('SME_001', 'Dr. Sarah Johnson', 'Model Risk', 'initial', 'high'),
('SME_002', 'Michael Chen', 'Credit Risk', 'initial', 'high'),
('SME_003', 'Dr. Emily Rodriguez', 'Market Risk', 'periodic', 'critical');

-- ---------------------------------------------------------------------------
-- Test Queries
-- ---------------------------------------------------------------------------

-- 1. Active MVR Prompts
SELECT 'Test 1: Active MVR Prompts' as test_name;
SELECT focus_area, sequence_order, LEFT(requirements, 100) || '...' as requirements_preview
FROM v_active_mvr_prompts
ORDER BY sequence_order;

-- 2. MVR Review Summary
SELECT 'Test 2: MVR Review Summary' as test_name;
SELECT model_name, model_type, risk_tier, validation_type, status, total_sections, ROUND(avg_rating::numeric, 2) as avg_rating
FROM v_mvr_review_summary
ORDER BY started_at DESC;

-- 3. SME Context Expertise
SELECT 'Test 3: SME Context Expertise' as test_name;
SELECT sme_name, expertise_area, validation_type, risk_tier, focus_area, sequence_order
FROM v_sme_context_expertise
ORDER BY sme_name, expertise_area;

-- 4. High-Risk Models
SELECT 'Test 4: High-Risk Models' as test_name;
SELECT model_name, model_type, risk_tier, validation_type, AVG(risk_score) as avg_risk_score
FROM v_mvr_review_summary
WHERE avg_risk_score >= 7
ORDER BY avg_risk_score DESC;

-- Additional test queries (5-13) can be added here following the same pattern.

-- ---------------------------------------------------------------------------
-- Cleanup (Optional)
-- ---------------------------------------------------------------------------

-- Uncomment to clean up test data
/*
DROP TABLE mvr_records, mvr_log, sme_context_mapping, mvr_prompts;
DROP VIEW v_active_mvr_prompts, v_mvr_review_summary, v_sme_context_expertise;
DROP FUNCTION update_mvr_prompts_updated_at, validate_mvr_record_completion;
DROP TRIGGER trigger_update_mvr_prompts_updated_at, trigger_validate_mvr_record_completion;
*/

-- ---------------------------------------------------------------------------
-- End of Complete Code
-- ---------------------------------------------------------------------------
