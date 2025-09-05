-- ============================================================================
-- HEIROS SPARSE AGREEMENTS TABLE QUERIES
-- SPARSE (Structured Pre-Approved Reasoning for Systematic Execution) management
-- ============================================================================

-- CREATE SPARSE AGREEMENT
-- Insert new SPARSE agreement for approval
INSERT INTO heiros_sparse_agreements (
    title,
    description,
    business_purpose,
    business_owner,
    technical_owner,
    risk_level,
    status,
    agreement_json,
    expiry_date
) VALUES (
    'Automated Document Classification',
    'ML-based document type classification for uploaded files',
    'Streamline document intake process while maintaining audit trail',
    'Risk Management Team',
    'AI Systems Team',
    'low',
    'pending',
    '{"conditions": [...], "approved_actions": [...], "compliance_frameworks": [...]}',
    NOW() + INTERVAL '365 days'
);

-- LIST ALL AGREEMENTS
-- Get all SPARSE agreements with status
SELECT 
    agreement_id,
    title,
    description,
    business_owner,
    technical_owner,
    risk_level,
    status,
    created_date,
    approved_date,
    expiry_date,
    execution_count
FROM heiros_sparse_agreements 
ORDER BY created_date DESC;

-- GET AGREEMENT BY ID
-- Retrieve specific agreement with full JSON
SELECT 
    agreement_id,
    title,
    description,
    business_purpose,
    business_owner,
    technical_owner,
    risk_level,
    status,
    agreement_json,
    created_date,
    approved_date,
    expiry_date,
    execution_count,
    last_execution_date
FROM heiros_sparse_agreements 
WHERE agreement_id = $1;

-- APPROVE AGREEMENT
-- Update agreement status to approved
UPDATE heiros_sparse_agreements 
SET 
    status = 'approved',
    approved_date = NOW()
WHERE agreement_id = $1
  AND status = 'pending';

-- REJECT AGREEMENT
-- Update agreement status to rejected
UPDATE heiros_sparse_agreements 
SET 
    status = 'rejected',
    approved_date = NOW()
WHERE agreement_id = $1
  AND status = 'pending';

-- INCREMENT EXECUTION COUNT
-- Track usage of approved agreements
UPDATE heiros_sparse_agreements 
SET 
    execution_count = execution_count + 1,
    last_execution_date = NOW()
WHERE agreement_id = $1
  AND status = 'approved';

-- GET PENDING APPROVALS
-- List agreements awaiting approval
SELECT 
    agreement_id,
    title,
    business_purpose,
    business_owner,
    technical_owner,
    risk_level,
    created_date,
    EXTRACT(DAYS FROM (NOW() - created_date)) as days_pending
FROM heiros_sparse_agreements 
WHERE status = 'pending'
ORDER BY created_date;

-- GET APPROVED AGREEMENTS
-- List all currently approved agreements
SELECT 
    agreement_id,
    title,
    business_owner,
    risk_level,
    approved_date,
    expiry_date,
    execution_count,
    last_execution_date,
    CASE 
        WHEN expiry_date IS NULL THEN 'No Expiry'
        WHEN expiry_date < NOW() THEN 'EXPIRED'
        WHEN expiry_date < NOW() + INTERVAL '30 days' THEN 'EXPIRING SOON'
        ELSE 'ACTIVE'
    END as expiry_status
FROM heiros_sparse_agreements 
WHERE status = 'approved'
ORDER BY expiry_date NULLS LAST;

-- AGREEMENTS BY RISK LEVEL
-- Summary of agreements by risk category
SELECT 
    risk_level,
    COUNT(*) as total,
    COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved,
    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
    COUNT(CASE WHEN status = 'rejected' THEN 1 END) as rejected,
    SUM(execution_count) as total_executions
FROM heiros_sparse_agreements 
GROUP BY risk_level
ORDER BY 
    CASE risk_level 
        WHEN 'critical' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        WHEN 'low' THEN 4
        WHEN 'minimal' THEN 5
    END;

-- EXPIRING AGREEMENTS
-- Find agreements expiring within 30 days
SELECT 
    agreement_id,
    title,
    business_owner,
    technical_owner,
    expiry_date,
    EXTRACT(DAYS FROM (expiry_date - NOW())) as days_until_expiry,
    execution_count
FROM heiros_sparse_agreements 
WHERE status = 'approved'
  AND expiry_date IS NOT NULL
  AND expiry_date <= NOW() + INTERVAL '30 days'
  AND expiry_date > NOW()
ORDER BY expiry_date;

-- EXPIRE AGREEMENTS
-- Automatically expire agreements past their expiry date
UPDATE heiros_sparse_agreements 
SET status = 'expired'
WHERE status = 'approved'
  AND expiry_date IS NOT NULL
  AND expiry_date < NOW();

-- AGREEMENT USAGE STATISTICS
-- Usage statistics for approved agreements
SELECT 
    title,
    business_owner,
    risk_level,
    approved_date,
    execution_count,
    last_execution_date,
    CASE 
        WHEN last_execution_date IS NULL THEN 'Never Used'
        WHEN last_execution_date < NOW() - INTERVAL '30 days' THEN 'Unused (30+ days)'
        WHEN last_execution_date < NOW() - INTERVAL '7 days' THEN 'Low Usage (7+ days)'
        ELSE 'Active'
    END as usage_status
FROM heiros_sparse_agreements 
WHERE status = 'approved'
ORDER BY execution_count DESC, last_execution_date DESC;

-- SEARCH AGREEMENTS BY PURPOSE
-- Find agreements by business purpose keywords
SELECT 
    agreement_id,
    title,
    business_purpose,
    business_owner,
    status,
    risk_level
FROM heiros_sparse_agreements 
WHERE business_purpose ILIKE '%' || $1 || '%'
   OR title ILIKE '%' || $1 || '%'
ORDER BY 
    CASE status 
        WHEN 'approved' THEN 1
        WHEN 'pending' THEN 2
        ELSE 3
    END,
    title;

-- AGREEMENTS BY OWNER
-- List agreements by business owner
SELECT 
    business_owner,
    COUNT(*) as total_agreements,
    COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved,
    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
    SUM(execution_count) as total_executions,
    MAX(last_execution_date) as latest_execution
FROM heiros_sparse_agreements 
GROUP BY business_owner
ORDER BY total_agreements DESC;

-- RENEW EXPIRING AGREEMENT
-- Extend expiry date for active agreement
UPDATE heiros_sparse_agreements 
SET expiry_date = NOW() + INTERVAL '365 days'
WHERE agreement_id = $1
  AND status = 'approved';