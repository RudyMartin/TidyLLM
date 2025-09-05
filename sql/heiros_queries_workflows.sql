-- ============================================================================
-- HEIROS WORKFLOWS TABLE QUERIES
-- Core workflow management queries for TidyLLM-HeirOS system
-- ============================================================================

-- CREATE WORKFLOW
-- Insert a new workflow definition
INSERT INTO heiros_workflows (
    name, 
    description, 
    compliance_level, 
    workflow_json, 
    version, 
    status,
    created_by,
    tags
) VALUES (
    'Document Processing Workflow',
    'Automated document validation and classification workflow',
    'full_transparency',
    '{"root": {"node_id": "doc_process", "type": "sequence", "children": [...]}}',
    '1.0',
    'active',
    'system_admin',
    '["document", "automation", "compliance"]'
);

-- LIST ALL WORKFLOWS
-- Get all workflows with basic information
SELECT 
    workflow_id,
    name,
    description,
    compliance_level,
    version,
    status,
    created_date,
    created_by,
    tags
FROM heiros_workflows 
ORDER BY created_date DESC;

-- GET WORKFLOW BY ID
-- Retrieve specific workflow with full JSON definition
SELECT 
    workflow_id,
    name,
    description,
    compliance_level,
    workflow_json,
    version,
    status,
    created_date,
    updated_date,
    created_by,
    tags
FROM heiros_workflows 
WHERE workflow_id = $1;

-- SEARCH WORKFLOWS BY NAME
-- Find workflows by partial name match
SELECT 
    workflow_id,
    name,
    description,
    status,
    created_date
FROM heiros_workflows 
WHERE name ILIKE '%' || $1 || '%'
  AND status = 'active'
ORDER BY name;

-- GET WORKFLOWS BY TAG
-- Find workflows containing specific tags
SELECT 
    workflow_id,
    name,
    description,
    tags,
    created_date
FROM heiros_workflows 
WHERE tags @> '["' || $1 || '"]'::jsonb
ORDER BY created_date DESC;

-- UPDATE WORKFLOW STATUS
-- Change workflow status (active/inactive/archived)
UPDATE heiros_workflows 
SET 
    status = $2,
    updated_date = NOW()
WHERE workflow_id = $1;

-- UPDATE WORKFLOW DEFINITION
-- Update workflow JSON and increment version
UPDATE heiros_workflows 
SET 
    workflow_json = $2,
    version = $3,
    updated_date = NOW()
WHERE workflow_id = $1;

-- GET WORKFLOW STATISTICS
-- Summary statistics for all workflows
SELECT 
    status,
    COUNT(*) as count,
    COUNT(CASE WHEN compliance_level = 'full_transparency' THEN 1 END) as full_transparency,
    COUNT(CASE WHEN compliance_level = 'summary_only' THEN 1 END) as summary_only,
    COUNT(CASE WHEN compliance_level = 'minimal' THEN 1 END) as minimal
FROM heiros_workflows 
GROUP BY status
ORDER BY status;

-- ARCHIVE OLD WORKFLOWS
-- Set old inactive workflows to archived status
UPDATE heiros_workflows 
SET 
    status = 'archived',
    updated_date = NOW()
WHERE status = 'inactive' 
  AND updated_date < NOW() - INTERVAL '90 days';

-- DELETE WORKFLOW
-- Remove workflow and cascade to related records
DELETE FROM heiros_workflows 
WHERE workflow_id = $1;