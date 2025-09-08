#!/usr/bin/env python3
"""
Create TidyLLM-HeirOS workflow tables in PostgreSQL database
"""
import psycopg2
from psycopg2.extras import RealDictCursor

# Import UnifiedSessionManager for secure database connections
try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from start_unified_sessions import UnifiedSessionManager
    UNIFIED_SESSION_AVAILABLE = True
except ImportError:
    UNIFIED_SESSION_AVAILABLE = False

def create_workflow_tables():
    """Create all TidyLLM-HeirOS workflow tables"""
    
    # SECURITY: Use UnifiedSessionManager instead of hardcoded credentials
    if UNIFIED_SESSION_AVAILABLE:
        print("Using UnifiedSessionManager for secure database access")
        session_manager = UnifiedSessionManager()
        conn = session_manager.get_postgres_connection()
    else:
        print("WARNING: Falling back to direct psycopg2 connection")
        conn = psycopg2.connect(
            host='vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com',
            port=5432,
            database='vectorqa',
            user='vectorqa_user', 
            password='REMOVED_PASSWORD',
            sslmode='require'
        )
    
    cursor = conn.cursor()
    
    print("="*80)
    print("CREATING TIDYLLM-HEIROS WORKFLOW TABLES")
    print("="*80)
    
    # SQL for creating workflow tables
    table_definitions = {
        "heiros_workflows": """
            CREATE TABLE IF NOT EXISTS heiros_workflows (
                workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                description TEXT,
                compliance_level VARCHAR(50) DEFAULT 'full_transparency',
                workflow_json JSONB NOT NULL,
                version VARCHAR(20) DEFAULT '1.0',
                status VARCHAR(20) DEFAULT 'active',
                created_date TIMESTAMP DEFAULT NOW(),
                updated_date TIMESTAMP DEFAULT NOW(),
                created_by VARCHAR(100),
                tags JSONB DEFAULT '[]'::jsonb,
                
                CONSTRAINT valid_status CHECK (status IN ('active', 'inactive', 'archived')),
                CONSTRAINT valid_compliance CHECK (compliance_level IN ('full_transparency', 'summary_only', 'minimal', 'regulatory'))
            );
            
            CREATE INDEX IF NOT EXISTS idx_heiros_workflows_name ON heiros_workflows(name);
            CREATE INDEX IF NOT EXISTS idx_heiros_workflows_status ON heiros_workflows(status);
            CREATE INDEX IF NOT EXISTS idx_heiros_workflows_created ON heiros_workflows(created_date);
        """,
        
        "heiros_executions": """
            CREATE TABLE IF NOT EXISTS heiros_executions (
                execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                workflow_id UUID NOT NULL,
                execution_date TIMESTAMP DEFAULT NOW(),
                status VARCHAR(20) DEFAULT 'pending',
                duration_ms INTEGER,
                context_data JSONB DEFAULT '{}'::jsonb,
                result_data JSONB DEFAULT '{}'::jsonb,
                node_results JSONB DEFAULT '[]'::jsonb,
                compliance_report JSONB DEFAULT '{}'::jsonb,
                executed_by VARCHAR(100),
                error_message TEXT,
                execution_path TEXT,
                nodes_executed INTEGER DEFAULT 0,
                
                CONSTRAINT valid_execution_status CHECK (status IN ('pending', 'running', 'success', 'failure', 'cancelled')),
                FOREIGN KEY (workflow_id) REFERENCES heiros_workflows(workflow_id) ON DELETE CASCADE
            );
            
            CREATE INDEX IF NOT EXISTS idx_heiros_executions_workflow ON heiros_executions(workflow_id);
            CREATE INDEX IF NOT EXISTS idx_heiros_executions_date ON heiros_executions(execution_date);
            CREATE INDEX IF NOT EXISTS idx_heiros_executions_status ON heiros_executions(status);
            CREATE INDEX IF NOT EXISTS idx_heiros_executions_user ON heiros_executions(executed_by);
        """,
        
        "heiros_sparse_agreements": """
            CREATE TABLE IF NOT EXISTS heiros_sparse_agreements (
                agreement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                title VARCHAR(255) NOT NULL,
                description TEXT,
                business_purpose TEXT,
                business_owner VARCHAR(100),
                technical_owner VARCHAR(100),
                risk_level VARCHAR(20) DEFAULT 'medium',
                status VARCHAR(20) DEFAULT 'pending',
                agreement_json JSONB NOT NULL,
                created_date TIMESTAMP DEFAULT NOW(),
                approved_date TIMESTAMP,
                expiry_date TIMESTAMP,
                execution_count INTEGER DEFAULT 0,
                last_execution_date TIMESTAMP,
                
                CONSTRAINT valid_risk_level CHECK (risk_level IN ('minimal', 'low', 'medium', 'high', 'critical')),
                CONSTRAINT valid_agreement_status CHECK (status IN ('pending', 'approved', 'rejected', 'expired', 'suspended'))
            );
            
            CREATE INDEX IF NOT EXISTS idx_heiros_agreements_status ON heiros_sparse_agreements(status);
            CREATE INDEX IF NOT EXISTS idx_heiros_agreements_risk ON heiros_sparse_agreements(risk_level);
            CREATE INDEX IF NOT EXISTS idx_heiros_agreements_owner ON heiros_sparse_agreements(business_owner);
            CREATE INDEX IF NOT EXISTS idx_heiros_agreements_expiry ON heiros_sparse_agreements(expiry_date);
        """,
        
        "heiros_node_templates": """
            CREATE TABLE IF NOT EXISTS heiros_node_templates (
                template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                description TEXT,
                node_type VARCHAR(50) NOT NULL,
                template_json JSONB NOT NULL,
                category VARCHAR(100) DEFAULT 'general',
                version VARCHAR(20) DEFAULT '1.0',
                created_date TIMESTAMP DEFAULT NOW(),
                usage_count INTEGER DEFAULT 0,
                is_public BOOLEAN DEFAULT true,
                created_by VARCHAR(100),
                
                CONSTRAINT valid_node_type CHECK (node_type IN ('sequence', 'selector', 'parallel', 'condition', 'action', 'sparse_decision', 'dynamic_flow'))
            );
            
            CREATE INDEX IF NOT EXISTS idx_heiros_templates_type ON heiros_node_templates(node_type);
            CREATE INDEX IF NOT EXISTS idx_heiros_templates_category ON heiros_node_templates(category);
            CREATE INDEX IF NOT EXISTS idx_heiros_templates_usage ON heiros_node_templates(usage_count);
        """,
        
        "heiros_audit_trail": """
            CREATE TABLE IF NOT EXISTS heiros_audit_trail (
                audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                execution_id UUID,
                workflow_id UUID,
                node_id VARCHAR(255),
                action_type VARCHAR(100) NOT NULL,
                timestamp TIMESTAMP DEFAULT NOW(),
                user_id VARCHAR(100),
                details JSONB DEFAULT '{}'::jsonb,
                risk_factors JSONB DEFAULT '[]'::jsonb,
                compliance_notes TEXT,
                
                FOREIGN KEY (execution_id) REFERENCES heiros_executions(execution_id) ON DELETE CASCADE,
                FOREIGN KEY (workflow_id) REFERENCES heiros_workflows(workflow_id) ON DELETE CASCADE
            );
            
            CREATE INDEX IF NOT EXISTS idx_heiros_audit_execution ON heiros_audit_trail(execution_id);
            CREATE INDEX IF NOT EXISTS idx_heiros_audit_workflow ON heiros_audit_trail(workflow_id);
            CREATE INDEX IF NOT EXISTS idx_heiros_audit_timestamp ON heiros_audit_trail(timestamp);
            CREATE INDEX IF NOT EXISTS idx_heiros_audit_user ON heiros_audit_trail(user_id);
        """
    }
    
    # Create each table
    for table_name, sql in table_definitions.items():
        try:
            print(f"\nCreating table: {table_name}")
            cursor.execute(sql)
            print(f"SUCCESS: {table_name} created")
        except Exception as e:
            print(f"ERROR creating {table_name}: {e}")
    
    # Commit all changes
    conn.commit()
    
    # Verify tables were created
    print(f"\n" + "-"*50)
    print("VERIFICATION - Created Tables:")
    print("-"*50)
    
    cursor.execute("""
        SELECT table_name, 
               (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as columns
        FROM information_schema.tables t
        WHERE table_schema = 'public' 
          AND table_name LIKE 'heiros_%'
        ORDER BY table_name;
    """)
    
    heiros_tables = cursor.fetchall()
    
    for table in heiros_tables:
        print(f"  {table[0]:<30} {table[1]:>3} columns")
    
    if not heiros_tables:
        print("  No HeirOS tables found - check for errors above")
    
    print(f"\n" + "="*80)
    print(f"WORKFLOW TABLES CREATION COMPLETE")
    print(f"Total HeirOS tables: {len(heiros_tables)}")
    print("="*80)
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    create_workflow_tables()