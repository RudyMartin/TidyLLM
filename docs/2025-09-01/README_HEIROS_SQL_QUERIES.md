# HeirOS SQL Queries Documentation

## Overview
This directory contains a comprehensive set of SQL query files for managing TidyLLM-HeirOS workflow database operations. Each file is organized by functional area and contains production-ready queries with parameters and comments.

## Query Files

### 1. `heiros_queries_workflows.sql`
**Core workflow management queries**
- Create, read, update, delete workflow definitions
- Search workflows by name, tags, compliance level
- Workflow status management and statistics
- Archive and cleanup operations

**Key queries:**
- `CREATE WORKFLOW` - Insert new workflow definition
- `LIST ALL WORKFLOWS` - Get all workflows with basic info
- `GET WORKFLOW BY ID` - Retrieve specific workflow with full JSON
- `SEARCH WORKFLOWS BY NAME` - Find workflows by partial match
- `GET WORKFLOWS BY TAG` - Find workflows by JSONB tag search

### 2. `heiros_queries_executions.sql`
**Workflow execution tracking and monitoring**
- Track workflow execution lifecycle
- Performance metrics and analysis
- Error tracking and diagnostics
- Compliance reporting

**Key queries:**
- `CREATE EXECUTION RECORD` - Start new workflow execution
- `UPDATE EXECUTION RESULTS` - Complete execution with results
- `EXECUTION PERFORMANCE METRICS` - Performance statistics
- `FAILED EXECUTIONS ANALYSIS` - Detailed failure analysis
- `LONG RUNNING EXECUTIONS` - Find stuck executions

### 3. `heiros_queries_sparse_agreements.sql`
**SPARSE agreement management**
- Create and manage SPARSE agreements
- Approval workflow tracking
- Risk assessment and compliance
- Usage statistics and renewal

**Key queries:**
- `CREATE SPARSE AGREEMENT` - Insert new agreement
- `APPROVE AGREEMENT` / `REJECT AGREEMENT` - Approval workflow
- `GET PENDING APPROVALS` - List agreements awaiting approval
- `EXPIRING AGREEMENTS` - Find agreements needing renewal
- `AGREEMENT USAGE STATISTICS` - Track agreement utilization

### 4. `heiros_queries_node_templates.sql`
**Reusable workflow node template management**
- Create and manage node templates
- Usage tracking and statistics
- Template discovery and search
- Version management

**Key queries:**
- `CREATE NODE TEMPLATE` - Insert new template
- `GET TEMPLATES BY TYPE` - Find templates by node type
- `SEARCH TEMPLATES` - Search by name/description
- `POPULAR TEMPLATES` - Most frequently used templates
- `INCREMENT TEMPLATE USAGE` - Track template usage

### 5. `heiros_queries_audit_trail.sql`
**Comprehensive audit logging and compliance**
- Complete execution audit trails
- Risk factor tracking
- Compliance reporting
- Security audit analysis

**Key queries:**
- `CREATE AUDIT ENTRY` - Log workflow action
- `GET AUDIT TRAIL BY EXECUTION` - Complete execution audit
- `RISK FACTOR ANALYSIS` - Find entries with specific risks
- `COMPLIANCE AUDIT REPORT` - Comprehensive compliance report
- `SECURITY AUDIT REPORT` - Security-focused analysis

### 6. `heiros_queries_analytics.sql`
**Cross-table analytics and business intelligence**
- Performance dashboards
- Usage trend analysis
- Error pattern analysis
- System health metrics

**Key queries:**
- `WORKFLOW PERFORMANCE DASHBOARD` - Comprehensive metrics
- `SYSTEM HEALTH METRICS` - Overall system indicators
- `TOP PERFORMING WORKFLOWS` - Best performing workflows
- `PROBLEMATIC WORKFLOWS` - High failure rate analysis
- `USER ACTIVITY ANALYSIS` - User behavior patterns

### 7. `heiros_queries_maintenance.sql`
**Database maintenance and administrative tasks**
- Health checks and monitoring
- Cleanup and archiving
- Performance optimization
- Backup verification

**Key queries:**
- `DATABASE HEALTH CHECK` - Table size and record counts
- `CLEANUP OLD EXECUTIONS` - Archive old execution data
- `EXPIRE OLD SPARSE AGREEMENTS` - Automatic expiry management
- `PERFORMANCE MONITORING SETUP` - Create monitoring view
- `DISK SPACE MONITORING` - Track storage usage

## Usage Guidelines

### Parameter Notation
Queries use `$1`, `$2`, etc. for parameterized values:
```sql
WHERE workflow_id = $1  -- Replace with actual workflow UUID
WHERE name ILIKE '%' || $1 || '%'  -- Replace $1 with search term
```

### Common Parameters
- **Workflow ID**: UUID format (e.g., `123e4567-e89b-12d3-a456-426614174000`)
- **Date ranges**: Use PostgreSQL interval notation (e.g., `NOW() - INTERVAL '30 days'`)
- **Status values**: `active`, `inactive`, `archived` for workflows
- **Execution status**: `pending`, `running`, `success`, `failure`, `cancelled`

### Best Practices

1. **Always use transactions** for multi-statement operations
2. **Test queries** in development environment first
3. **Use LIMIT clauses** for large result sets
4. **Monitor execution plans** for performance
5. **Regular maintenance** using cleanup queries

### Security Considerations

- All queries assume proper user permissions
- Sensitive data is logged in audit trails
- Cleanup operations are irreversible
- Emergency cleanup procedures require admin privileges

## Integration with TidyLLM

These queries integrate with the TidyLLM-HeirOS system:

```python
import psycopg2
from psycopg2.extras import RealDictCursor

# Connection setup (use your settings.yaml credentials)
conn = psycopg2.connect(
    host='your-postgres-host',
    database='vectorqa',
    user='vectorqa_user',
    password='your-password'
)

# Example: Get workflow performance
cursor = conn.cursor(cursor_factory=RealDictCursor)
cursor.execute(open('heiros_queries_analytics.sql').read().split(';')[0])
results = cursor.fetchall()
```

## Monitoring and Alerts

Use these queries for monitoring:
- **System health**: Run analytics queries daily
- **Performance alerts**: Monitor execution times and failure rates
- **Capacity planning**: Track disk usage and growth trends
- **Compliance audits**: Regular audit trail analysis

## Support and Maintenance

- **Backup strategy**: Critical data identified in maintenance queries
- **Retention policies**: Automated cleanup based on compliance requirements
- **Index optimization**: Regular reindexing for performance
- **Statistics updates**: Keep query optimizer informed

---

*Generated for TidyLLM-HeirOS v1.0 - PostgreSQL Database Integration*