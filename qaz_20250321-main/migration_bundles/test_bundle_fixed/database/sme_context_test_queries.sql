-- SME Context Test Queries
-- Comprehensive testing of MVR prompts, logs, and records functionality

-- ============================================================================
-- TEST DATA INSERTION
-- ============================================================================

-- Insert test MVR prompts (focus areas with detailed requirements)
INSERT INTO mvr_prompts (title, sequence_order, requirements) VALUES
('Model Risk', 1, 
'INPUT CONSIDERATIONS:
- Model performance metrics (accuracy, precision, recall, F1-score)
- Data quality assessment (completeness, consistency, timeliness)
- Model documentation completeness
- Training data representativeness
- Model assumptions and limitations

OUTPUT CONSIDERATIONS:
- Risk assessment score (1-10 scale)
- Validation status (pass/fail/conditional)
- Specific findings and recommendations
- Required remediation actions
- Timeline for implementation

SUGGESTED QUESTIONS:
1. Does the model meet performance thresholds?
2. Is the training data representative of production data?
3. Are model assumptions still valid?
4. What are the key risk factors?
5. What monitoring is required post-deployment?'),

('Credit Risk', 2, 
'INPUT CONSIDERATIONS:
- Credit scoring model performance
- Default rate analysis
- Portfolio concentration metrics
- Economic scenario impact
- Regulatory compliance status

OUTPUT CONSIDERATIONS:
- Credit risk assessment
- Portfolio risk score
- Concentration risk analysis
- Stress testing results
- Regulatory compliance findings

SUGGESTED QUESTIONS:
1. How does the model perform across different credit segments?
2. What is the impact of economic downturns?
3. Are concentration limits being exceeded?
4. Is the model compliant with fair lending regulations?
5. What stress testing scenarios should be considered?'),

('Market Risk', 3, 
'INPUT CONSIDERATIONS:
- VaR model accuracy
- Market volatility analysis
- Correlation breakdown detection
- Liquidity risk assessment
- Stress testing results

OUTPUT CONSIDERATIONS:
- Market risk score
- VaR model validation
- Liquidity risk assessment
- Stress testing outcomes
- Risk limit recommendations

SUGGESTED QUESTIONS:
1. Is the VaR model capturing tail risk adequately?
2. How do correlations change during stress periods?
3. What is the liquidity impact of large positions?
4. Are stress testing scenarios comprehensive?
5. Are risk limits appropriate for current market conditions?'),

('Operational Risk', 4, 
'INPUT CONSIDERATIONS:
- System reliability metrics
- Process efficiency analysis
- Human error rates
- Third-party risk assessment
- Business continuity planning

OUTPUT CONSIDERATIONS:
- Operational risk score
- System reliability assessment
- Process improvement recommendations
- Business continuity readiness
- Risk mitigation strategies

SUGGESTED QUESTIONS:
1. What are the critical system dependencies?
2. How robust are the backup and recovery procedures?
3. What is the impact of third-party failures?
4. Are there adequate controls for human error?
5. Is the business continuity plan tested and current?');

-- Insert test MVR log entries
INSERT INTO mvr_log (model_id, model_name, model_type, risk_tier, validation_type, started_at, reviewer_id, notes) VALUES
('MODEL_001', 'Credit Scoring Model v2.1', 'Machine Learning', 'high', 'initial', '2024-01-15 09:00:00', 'REVIEWER_001', 'Initial validation of new credit scoring model'),
('MODEL_002', 'VaR Calculation Engine', 'Statistical', 'critical', 'periodic', '2024-01-16 10:00:00', 'REVIEWER_002', 'Annual validation of VaR model'),
('MODEL_003', 'Fraud Detection System', 'AI/ML', 'high', 'event-driven', '2024-01-17 14:00:00', 'REVIEWER_001', 'Validation triggered by performance degradation'),
('MODEL_004', 'Portfolio Optimization Model', 'Optimization', 'medium', 'initial', '2024-01-18 11:00:00', 'REVIEWER_003', 'New portfolio optimization model validation');

-- Insert test MVR records
INSERT INTO mvr_records (review_id, rating, conclusion, mvr_section, evidence, mvr_number, review_title, completed_at, reviewer_id, confidence_score, risk_score, recommendations) VALUES
(1, 4, 'Model meets performance requirements with minor issues identified', 'Data Quality Assessment', 'Training data completeness: 98.5%, Data consistency: 95.2%, Timeliness: 99.1%', 1, 'Credit Scoring Model Validation', '2024-01-15 16:30:00', 'REVIEWER_001', 0.85, 6, 'Implement additional data quality monitoring'),
(1, 3, 'Model assumptions need review due to changing economic conditions', 'Model Assumptions', 'Economic indicators show deviation from training period assumptions', 2, 'Credit Scoring Model Validation', '2024-01-15 16:30:00', 'REVIEWER_001', 0.78, 7, 'Retrain model with updated economic data'),
(1, 5, 'Documentation is comprehensive and well-maintained', 'Documentation Review', 'All required documentation present and up-to-date', 3, 'Credit Scoring Model Validation', '2024-01-15 16:30:00', 'REVIEWER_001', 0.92, 3, 'Continue current documentation practices'),
(2, 4, 'VaR model performs well under normal conditions', 'Model Performance', 'Backtesting results: 95% confidence interval maintained', 1, 'VaR Model Annual Review', '2024-01-16 17:00:00', 'REVIEWER_002', 0.88, 5, 'Enhance stress testing scenarios'),
(2, 2, 'Stress testing reveals vulnerabilities in extreme scenarios', 'Stress Testing', 'Model fails under 2008-like crisis scenarios', 2, 'VaR Model Annual Review', '2024-01-16 17:00:00', 'REVIEWER_002', 0.75, 8, 'Implement additional stress testing and model enhancements'),
(3, 1, 'Significant performance degradation detected', 'Performance Monitoring', 'Accuracy dropped from 95% to 78% over last 30 days', 1, 'Fraud Detection System Review', '2024-01-17 18:00:00', 'REVIEWER_001', 0.95, 9, 'Immediate model retraining required'),
(4, 4, 'Model performs adequately for intended use case', 'Model Validation', 'Portfolio optimization results within acceptable parameters', 1, 'Portfolio Optimization Validation', '2024-01-18 15:00:00', 'REVIEWER_003', 0.82, 5, 'Monitor performance and implement improvements');

-- Insert test SME context mapping
INSERT INTO sme_context_mapping (sme_id, sme_name, expertise_area, validation_type, risk_tier) VALUES
('SME_001', 'Dr. Sarah Johnson', 'Model Risk', 'initial', 'high'),
('SME_001', 'Dr. Sarah Johnson', 'Model Risk', 'periodic', 'high'),
('SME_002', 'Michael Chen', 'Credit Risk', 'initial', 'high'),
('SME_002', 'Michael Chen', 'Credit Risk', 'event-driven', 'critical'),
('SME_003', 'Dr. Emily Rodriguez', 'Market Risk', 'periodic', 'critical'),
('SME_003', 'Dr. Emily Rodriguez', 'Market Risk', 'event-driven', 'high'),
('SME_004', 'James Wilson', 'Operational Risk', 'initial', 'medium'),
('SME_004', 'James Wilson', 'Operational Risk', 'periodic', 'medium');

-- ============================================================================
-- TEST QUERIES
-- ============================================================================

-- 1. Get all active MVR prompts ordered by sequence
SELECT 'Test 1: Active MVR Prompts' as test_name;
SELECT 
    title as focus_area,
    sequence_order,
    LEFT(requirements, 100) || '...' as requirements_preview
FROM v_active_mvr_prompts
ORDER BY sequence_order;

-- 2. Get MVR review summary with statistics
SELECT 'Test 2: MVR Review Summary' as test_name;
SELECT 
    model_name,
    model_type,
    risk_tier,
    validation_type,
    status,
    total_sections,
    ROUND(avg_rating::numeric, 2) as avg_rating,
    ROUND(avg_risk_score::numeric, 2) as avg_risk_score,
    ROUND(avg_confidence::numeric, 2) as avg_confidence
FROM v_mvr_review_summary
ORDER BY started_at DESC;

-- 3. Get SME context with expertise mapping
SELECT 'Test 3: SME Context Expertise' as test_name;
SELECT 
    sme_name,
    expertise_area,
    validation_type,
    risk_tier,
    focus_area,
    sequence_order
FROM v_sme_context_expertise
ORDER BY sme_name, expertise_area, sequence_order;

-- 4. Find high-risk models requiring attention
SELECT 'Test 4: High-Risk Models' as test_name;
SELECT 
    ml.model_name,
    ml.model_type,
    ml.risk_tier,
    ml.validation_type,
    COUNT(mr.id) as sections_reviewed,
    AVG(mr.risk_score) as avg_risk_score,
    MAX(mr.risk_score) as max_risk_score
FROM mvr_log ml
JOIN mvr_records mr ON ml.review_id = mr.review_id
WHERE mr.risk_score >= 7
GROUP BY ml.review_id, ml.model_name, ml.model_type, ml.risk_tier, ml.validation_type
ORDER BY avg_risk_score DESC;

-- 5. Get validation performance by reviewer
SELECT 'Test 5: Reviewer Performance' as test_name;
SELECT 
    mr.reviewer_id,
    COUNT(DISTINCT mr.review_id) as reviews_conducted,
    COUNT(mr.id) as total_sections,
    AVG(mr.rating) as avg_rating,
    AVG(mr.confidence_score) as avg_confidence,
    AVG(mr.risk_score) as avg_risk_score
FROM mvr_records mr
WHERE mr.reviewer_id IS NOT NULL
GROUP BY mr.reviewer_id
ORDER BY avg_rating DESC;

-- 6. Find models with low ratings requiring immediate attention
SELECT 'Test 6: Low-Rated Models' as test_name;
SELECT 
    ml.model_name,
    ml.model_type,
    ml.risk_tier,
    mr.mvr_section,
    mr.rating,
    mr.risk_score,
    mr.conclusion,
    mr.recommendations
FROM mvr_log ml
JOIN mvr_records mr ON ml.review_id = mr.review_id
WHERE mr.rating <= 2
ORDER BY mr.rating, mr.risk_score DESC;

-- 7. Get validation trends over time
SELECT 'Test 7: Validation Trends' as test_name;
SELECT 
    DATE(mr.completed_at) as review_date,
    COUNT(DISTINCT mr.review_id) as reviews_completed,
    COUNT(mr.id) as sections_reviewed,
    AVG(mr.rating) as avg_rating,
    AVG(mr.risk_score) as avg_risk_score
FROM mvr_records mr
GROUP BY DATE(mr.completed_at)
ORDER BY review_date DESC;

-- 8. Find SME expertise gaps
SELECT 'Test 8: SME Expertise Gaps' as test_name;
SELECT 
    mp.title as focus_area,
    mp.sequence_order,
    COUNT(scm.id) as sme_count
FROM mvr_prompts mp
LEFT JOIN sme_context_mapping scm ON mp.title = scm.expertise_area AND scm.is_active = TRUE
WHERE mp.is_active = TRUE
GROUP BY mp.id, mp.title, mp.sequence_order
HAVING COUNT(scm.id) = 0
ORDER BY mp.sequence_order;

-- 9. Get comprehensive risk assessment
SELECT 'Test 9: Comprehensive Risk Assessment' as test_name;
SELECT 
    ml.model_name,
    ml.risk_tier,
    ml.validation_type,
    ml.status,
    COUNT(mr.id) as sections_reviewed,
    AVG(mr.rating) as avg_rating,
    AVG(mr.risk_score) as avg_risk_score,
    CASE 
        WHEN AVG(mr.risk_score) >= 8 THEN 'Critical'
        WHEN AVG(mr.risk_score) >= 6 THEN 'High'
        WHEN AVG(mr.risk_score) >= 4 THEN 'Medium'
        ELSE 'Low'
    END as calculated_risk_level
FROM mvr_log ml
LEFT JOIN mvr_records mr ON ml.review_id = mr.review_id
GROUP BY ml.review_id, ml.model_name, ml.risk_tier, ml.validation_type, ml.status
ORDER BY avg_risk_score DESC;

-- 10. Get validation efficiency metrics
SELECT 'Test 10: Validation Efficiency' as test_name;
SELECT 
    ml.validation_type,
    COUNT(DISTINCT ml.review_id) as total_reviews,
    AVG(EXTRACT(EPOCH FROM (ml.completed_at - ml.started_at))/3600) as avg_hours_to_complete,
    COUNT(mr.id) as total_sections,
    AVG(mr.rating) as avg_rating,
    AVG(mr.confidence_score) as avg_confidence
FROM mvr_log ml
LEFT JOIN mvr_records mr ON ml.review_id = mr.review_id
WHERE ml.completed_at IS NOT NULL
GROUP BY ml.validation_type
ORDER BY avg_hours_to_complete;

-- ============================================================================
-- COMPLEX ANALYTICAL QUERIES
-- ============================================================================

-- 11. Risk correlation analysis
SELECT 'Test 11: Risk Correlation Analysis' as test_name;
WITH risk_analysis AS (
    SELECT 
        ml.model_type,
        ml.risk_tier,
        AVG(mr.risk_score) as avg_risk_score,
        AVG(mr.rating) as avg_rating,
        COUNT(mr.id) as section_count
    FROM mvr_log ml
    JOIN mvr_records mr ON ml.review_id = mr.review_id
    GROUP BY ml.model_type, ml.risk_tier
)
SELECT 
    model_type,
    risk_tier,
    ROUND(avg_risk_score::numeric, 2) as avg_risk_score,
    ROUND(avg_rating::numeric, 2) as avg_rating,
    section_count,
    CASE 
        WHEN avg_risk_score > 7 AND avg_rating < 3 THEN 'High Risk, Low Performance'
        WHEN avg_risk_score < 4 AND avg_rating > 4 THEN 'Low Risk, High Performance'
        WHEN avg_risk_score > 7 AND avg_rating > 4 THEN 'High Risk, High Performance'
        ELSE 'Standard Performance'
    END as risk_performance_category
FROM risk_analysis
ORDER BY avg_risk_score DESC, avg_rating;

-- 12. SME workload distribution
SELECT 'Test 12: SME Workload Distribution' as test_name;
SELECT 
    scm.sme_name,
    scm.expertise_area,
    COUNT(DISTINCT ml.review_id) as reviews_assigned,
    COUNT(mr.id) as sections_reviewed,
    AVG(mr.rating) as avg_rating,
    AVG(mr.confidence_score) as avg_confidence
FROM sme_context_mapping scm
LEFT JOIN mvr_log ml ON ml.reviewer_id = scm.sme_id
LEFT JOIN mvr_records mr ON ml.review_id = mr.review_id
WHERE scm.is_active = TRUE
GROUP BY scm.sme_id, scm.sme_name, scm.expertise_area
ORDER BY reviews_assigned DESC, avg_rating DESC;

-- 13. Validation quality assessment
SELECT 'Test 13: Validation Quality Assessment' as test_name;
SELECT 
    ml.model_name,
    ml.validation_type,
    ml.status,
    COUNT(mr.id) as sections_reviewed,
    AVG(mr.rating) as avg_rating,
    AVG(mr.confidence_score) as avg_confidence,
    AVG(mr.risk_score) as avg_risk_score,
    CASE 
        WHEN AVG(mr.rating) >= 4 AND AVG(mr.confidence_score) >= 0.8 THEN 'High Quality'
        WHEN AVG(mr.rating) >= 3 AND AVG(mr.confidence_score) >= 0.6 THEN 'Medium Quality'
        ELSE 'Low Quality'
    END as quality_assessment
FROM mvr_log ml
JOIN mvr_records mr ON ml.review_id = mr.review_id
GROUP BY ml.review_id, ml.model_name, ml.validation_type, ml.status
ORDER BY quality_assessment, avg_rating DESC;

-- ============================================================================
-- CLEANUP (Optional - for testing)
-- ============================================================================

-- Uncomment to clean up test data
/*
DELETE FROM mvr_records;
DELETE FROM mvr_log;
DELETE FROM sme_context_mapping;
DELETE FROM mvr_prompts;
*/
