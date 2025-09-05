-- SME Context Database Schema
-- Subject Matter Expert (SME) Context for Model Validation Review (MVR)

-- ============================================================================
-- MVR PROMPTS TABLE
-- Contains focus areas, sequence for reporting, and detailed requirements
-- ============================================================================

CREATE TABLE mvr_prompts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,  -- focus_area
    sequence_order INTEGER NOT NULL,  -- sequence for reporting
    requirements TEXT NOT NULL,  -- input/output considerations, suggested questions, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Constraints
    CONSTRAINT unique_title UNIQUE (title),
    CONSTRAINT valid_sequence_order CHECK (sequence_order > 0)
);

-- ============================================================================
-- MVR LOG TABLE
-- Tracks review sessions and model information
-- ============================================================================

CREATE TABLE mvr_log (
    review_id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    risk_tier VARCHAR(50) NOT NULL,  -- low, medium, high, critical
    validation_type VARCHAR(100) NOT NULL,  -- initial, periodic, event-driven
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    status VARCHAR(50) DEFAULT 'in_progress',  -- in_progress, completed, failed
    reviewer_id VARCHAR(100),
    notes TEXT,
    
    -- Constraints
    CONSTRAINT valid_risk_tier CHECK (risk_tier IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT valid_validation_type CHECK (validation_type IN ('initial', 'periodic', 'event-driven')),
    CONSTRAINT valid_status CHECK (status IN ('in_progress', 'completed', 'failed')),
    CONSTRAINT valid_dates CHECK (started_at <= completed_at OR completed_at IS NULL)
);

-- ============================================================================
-- MVR RECORDS TABLE
-- Detailed validation results and evidence
-- ============================================================================

CREATE TABLE mvr_records (
    id SERIAL PRIMARY KEY,
    review_id INTEGER NOT NULL,
    rating INTEGER NOT NULL,  -- 0-5 scale
    conclusion TEXT NOT NULL,
    mvr_section VARCHAR(255) NOT NULL,
    evidence TEXT NOT NULL,
    mvr_number INTEGER NOT NULL,
    review_title VARCHAR(255) NOT NULL,
    completed_at TIMESTAMP NOT NULL,
    reviewer_id VARCHAR(100),
    confidence_score DECIMAL(3,2),  -- 0.00-1.00
    risk_score INTEGER,  -- 1-10 scale
    recommendations TEXT,
    
    -- Foreign key constraint
    CONSTRAINT fk_mvr_records_review_id 
        FOREIGN KEY (review_id) REFERENCES mvr_log(review_id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT valid_rating CHECK (rating >= 0 AND rating <= 5),
    CONSTRAINT valid_confidence_score CHECK (confidence_score >= 0.00 AND confidence_score <= 1.00),
    CONSTRAINT valid_risk_score CHECK (risk_score >= 1 AND risk_score <= 10)
);

-- ============================================================================
-- SME CONTEXT MAPPING TABLE
-- Links SME expertise to specific risk categories and validation types
-- ============================================================================

CREATE TABLE sme_context_mapping (
    id SERIAL PRIMARY KEY,
    sme_id VARCHAR(100) NOT NULL,
    sme_name VARCHAR(255) NOT NULL,
    expertise_area VARCHAR(255) NOT NULL,  -- Model Risk, Credit Risk, Market Risk, Operational Risk
    validation_type VARCHAR(100) NOT NULL,
    risk_tier VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_expertise_area CHECK (expertise_area IN ('Model Risk', 'Credit Risk', 'Market Risk', 'Operational Risk')),
    CONSTRAINT valid_validation_type CHECK (validation_type IN ('initial', 'periodic', 'event-driven')),
    CONSTRAINT valid_risk_tier CHECK (risk_tier IN ('low', 'medium', 'high', 'critical'))
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- MVR Prompts indexes
CREATE INDEX idx_mvr_prompts_title ON mvr_prompts(title);
CREATE INDEX idx_mvr_prompts_sequence ON mvr_prompts(sequence_order);
CREATE INDEX idx_mvr_prompts_active ON mvr_prompts(is_active);

-- MVR Log indexes
CREATE INDEX idx_mvr_log_model_id ON mvr_log(model_id);
CREATE INDEX idx_mvr_log_model_type ON mvr_log(model_type);
CREATE INDEX idx_mvr_log_risk_tier ON mvr_log(risk_tier);
CREATE INDEX idx_mvr_log_validation_type ON mvr_log(validation_type);
CREATE INDEX idx_mvr_log_status ON mvr_log(status);
CREATE INDEX idx_mvr_log_started_at ON mvr_log(started_at);

-- MVR Records indexes
CREATE INDEX idx_mvr_records_review_id ON mvr_records(review_id);
CREATE INDEX idx_mvr_records_rating ON mvr_records(rating);
CREATE INDEX idx_mvr_records_mvr_section ON mvr_records(mvr_section);
CREATE INDEX idx_mvr_records_completed_at ON mvr_records(completed_at);
CREATE INDEX idx_mvr_records_risk_score ON mvr_records(risk_score);

-- SME Context Mapping indexes
CREATE INDEX idx_sme_context_sme_id ON sme_context_mapping(sme_id);
CREATE INDEX idx_sme_context_expertise ON sme_context_mapping(expertise_area);
CREATE INDEX idx_sme_context_validation_type ON sme_context_mapping(validation_type);
CREATE INDEX idx_sme_context_active ON sme_context_mapping(is_active);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View for active MVR prompts with sequence
CREATE VIEW v_active_mvr_prompts AS
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

-- View for MVR review summary
CREATE VIEW v_mvr_review_summary AS
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

-- View for SME context with expertise mapping
CREATE VIEW v_sme_context_expertise AS
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

-- ============================================================================
-- TRIGGERS FOR DATA INTEGRITY
-- ============================================================================

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
    -- Ensure review is completed when adding records
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
