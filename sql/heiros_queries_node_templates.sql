-- ============================================================================
-- HEIROS NODE TEMPLATES TABLE QUERIES
-- Reusable workflow node template management queries
-- ============================================================================

-- CREATE NODE TEMPLATE
-- Insert new reusable node template
INSERT INTO heiros_node_templates (
    name,
    description,
    node_type,
    template_json,
    category,
    version,
    is_public,
    created_by
) VALUES (
    'Document Validation Node',
    'Standard document format and size validation',
    'action',
    '{"validation_rules": {"max_size": 52428800, "allowed_formats": [".pdf", ".txt"]}}',
    'document_processing',
    '1.0',
    true,
    'system_admin'
);

-- LIST ALL TEMPLATES
-- Get all node templates with usage statistics
SELECT 
    template_id,
    name,
    description,
    node_type,
    category,
    version,
    created_date,
    usage_count,
    is_public,
    created_by
FROM heiros_node_templates 
ORDER BY usage_count DESC, created_date DESC;

-- GET TEMPLATE BY ID
-- Retrieve specific template with full JSON definition
SELECT 
    template_id,
    name,
    description,
    node_type,
    template_json,
    category,
    version,
    created_date,
    usage_count,
    is_public,
    created_by
FROM heiros_node_templates 
WHERE template_id = $1;

-- GET TEMPLATES BY TYPE
-- Find templates by node type
SELECT 
    template_id,
    name,
    description,
    category,
    version,
    usage_count,
    created_by
FROM heiros_node_templates 
WHERE node_type = $1
  AND is_public = true
ORDER BY usage_count DESC, name;

-- GET TEMPLATES BY CATEGORY
-- Find templates in specific category
SELECT 
    template_id,
    name,
    description,
    node_type,
    version,
    usage_count
FROM heiros_node_templates 
WHERE category = $1
  AND is_public = true
ORDER BY usage_count DESC, name;

-- SEARCH TEMPLATES
-- Search templates by name or description
SELECT 
    template_id,
    name,
    description,
    node_type,
    category,
    usage_count
FROM heiros_node_templates 
WHERE (name ILIKE '%' || $1 || '%' OR description ILIKE '%' || $1 || '%')
  AND is_public = true
ORDER BY usage_count DESC, name;

-- INCREMENT TEMPLATE USAGE
-- Track when template is used in workflow
UPDATE heiros_node_templates 
SET usage_count = usage_count + 1
WHERE template_id = $1;

-- UPDATE TEMPLATE
-- Update template definition and increment version
UPDATE heiros_node_templates 
SET 
    description = $2,
    template_json = $3,
    version = $4,
    category = $5
WHERE template_id = $1;

-- POPULAR TEMPLATES
-- Most frequently used templates
SELECT 
    name,
    node_type,
    category,
    usage_count,
    created_date,
    created_by
FROM heiros_node_templates 
WHERE is_public = true
  AND usage_count > 0
ORDER BY usage_count DESC
LIMIT 20;

-- TEMPLATES BY NODE TYPE SUMMARY
-- Summary statistics by node type
SELECT 
    node_type,
    COUNT(*) as total_templates,
    COUNT(CASE WHEN is_public THEN 1 END) as public_templates,
    SUM(usage_count) as total_usage,
    AVG(usage_count) as avg_usage,
    MAX(usage_count) as max_usage
FROM heiros_node_templates 
GROUP BY node_type
ORDER BY total_usage DESC;

-- UNUSED TEMPLATES
-- Templates that have never been used
SELECT 
    template_id,
    name,
    description,
    node_type,
    category,
    created_date,
    created_by
FROM heiros_node_templates 
WHERE usage_count = 0
  AND created_date < NOW() - INTERVAL '30 days'
ORDER BY created_date;

-- TEMPLATES BY CREATOR
-- Templates grouped by creator
SELECT 
    created_by,
    COUNT(*) as total_templates,
    COUNT(CASE WHEN is_public THEN 1 END) as public_templates,
    SUM(usage_count) as total_usage,
    MAX(created_date) as latest_template
FROM heiros_node_templates 
GROUP BY created_by
ORDER BY total_usage DESC;

-- CATEGORY STATISTICS
-- Usage statistics by category
SELECT 
    category,
    COUNT(*) as template_count,
    SUM(usage_count) as total_usage,
    AVG(usage_count) as avg_usage,
    MIN(created_date) as earliest_template,
    MAX(created_date) as latest_template
FROM heiros_node_templates 
WHERE is_public = true
GROUP BY category
ORDER BY total_usage DESC;

-- RECENT TEMPLATES
-- Recently created templates
SELECT 
    template_id,
    name,
    description,
    node_type,
    category,
    created_date,
    created_by,
    is_public
FROM heiros_node_templates 
WHERE created_date >= NOW() - INTERVAL '30 days'
ORDER BY created_date DESC;

-- MAKE TEMPLATE PRIVATE
-- Change template visibility to private
UPDATE heiros_node_templates 
SET is_public = false
WHERE template_id = $1;

-- MAKE TEMPLATE PUBLIC
-- Change template visibility to public
UPDATE heiros_node_templates 
SET is_public = true
WHERE template_id = $1;

-- DELETE UNUSED TEMPLATE
-- Remove template that has never been used
DELETE FROM heiros_node_templates 
WHERE template_id = $1
  AND usage_count = 0;

-- TEMPLATE USAGE TREND
-- Templates with increasing usage over time
SELECT 
    name,
    node_type,
    category,
    usage_count,
    CASE 
        WHEN usage_count = 0 THEN 'Unused'
        WHEN usage_count < 5 THEN 'Low Usage'
        WHEN usage_count < 20 THEN 'Medium Usage'
        ELSE 'High Usage'
    END as usage_category
FROM heiros_node_templates 
WHERE is_public = true
ORDER BY usage_count DESC;

-- DUPLICATE TEMPLATE CHECK
-- Find potentially duplicate templates by name similarity
SELECT 
    t1.template_id as template1_id,
    t1.name as template1_name,
    t2.template_id as template2_id,
    t2.name as template2_name,
    t1.node_type
FROM heiros_node_templates t1
JOIN heiros_node_templates t2 ON t1.template_id < t2.template_id
WHERE t1.node_type = t2.node_type
  AND similarity(t1.name, t2.name) > 0.8
ORDER BY similarity(t1.name, t2.name) DESC;