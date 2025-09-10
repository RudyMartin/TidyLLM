# TidyLLM-HeirOS Workflow Creation & Management Guide

**Complete guide for creating, saving, and retrieving hierarchical workflows in TidyLLM-HeirOS**

---

## 📋 **Table of Contents**

1. [Quick Start - Create Your First Workflow](#quick-start)
2. [Workflow Node Types](#node-types) 
3. [Building Hierarchical Workflows](#building-workflows)
4. [Saving Workflows](#saving-workflows)
5. [Loading and Retrieving Workflows](#loading-workflows)
6. [SPARSE Agreements Integration](#sparse-agreements)
7. [Execution and Monitoring](#execution)
8. [Advanced Patterns](#advanced-patterns)

---

## 🚀 **Quick Start - Create Your First Workflow** {#quick-start}

### **1. Basic Setup**
```python
#!/usr/bin/env python3
import sys
sys.path.append('tidyllm-heiros/src/dag-manager')
sys.path.append('tidyllm-heiros/src/sparse-agreement')

from hierarchical_dag_manager import *

# Create DAG manager
dag_manager = HierarchicalDAGManager(
    name="My First Workflow",
    compliance_level=ComplianceLevel.FULL_TRANSPARENCY
)
```

### **2. Create Simple Workflow**
```python
# Root sequence - steps executed in order
root = SequenceNode(
    node_id="my_workflow",
    name="Document Processing",
    description="Process and analyze documents"
)

# Step 1: Validation
validation = ActionNode(
    node_id="validate",
    name="Document Validation", 
    description="Check document format and security",
    action=lambda context: {
        "status": "success",
        "message": "Document validated",
        "file_type": context.get("file_extension", "pdf")
    }
)

# Step 2: Processing choice
processor = SelectorNode(
    node_id="process_choice",
    name="Processing Path",
    description="Choose processing method"
)

# Add children to build hierarchy
root.add_child(validation)
root.add_child(processor)

# Add to DAG manager
dag_manager.add_root_node(root)
```

### **3. Execute Workflow**
```python
# Execute with context
context = {
    "document_id": "doc_001",
    "file_extension": "pdf",
    "user_id": "analyst_1"
}

result = dag_manager.execute_dag(context)
print(f"Status: {result.get('status')}")
```

---

## 🏗️ **Workflow Node Types** {#node-types}

### **Core Node Types Available:**

#### **1. SequenceNode** - Execute Children in Order
```python
sequence = SequenceNode(
    node_id="intake_sequence",
    name="Document Intake Process",
    description="Sequential document processing steps"
)

# Children execute: Step 1 → Step 2 → Step 3
sequence.add_child(step1).add_child(step2).add_child(step3)
```

#### **2. SelectorNode** - Execute First Successful Child
```python
selector = SelectorNode(
    node_id="analysis_choice", 
    name="Analysis Method Selection",
    description="Try different analysis methods until one succeeds"
)

# Children execute: Try Method A → If fails, try Method B → etc.
selector.add_child(method_a).add_child(method_b).add_child(manual_review)
```

#### **3. ActionNode** - Execute Specific Task
```python
action = ActionNode(
    node_id="classify_doc",
    name="Document Classification",
    description="Classify document type using ML model",
    action=lambda context: {
        "status": "success",
        "document_type": "mvr_report", 
        "confidence": 0.92,
        "classification_time": 1.2
    }
)
```

#### **4. ConditionNode** - Boolean Decision Point
```python
condition = ConditionNode(
    node_id="size_check",
    name="Document Size Check", 
    description="Check if document is under size limit",
    condition=lambda context: context.get("file_size", 0) < 50_000_000
)
```

#### **5. ParallelNode** - Execute Children Simultaneously
```python
parallel = ParallelNode(
    node_id="parallel_analysis",
    name="Parallel Document Analysis",
    description="Run multiple analyses simultaneously"
)

# All children execute at the same time
parallel.add_child(extract_text).add_child(extract_images).add_child(extract_metadata)
```

#### **6. SparseDecisionNode** - Pre-Approved Decision
```python
sparse_node = SparseDecisionNode(
    node_id="approved_classification",
    name="Pre-Approved Document Classification",
    sparse_agreement_id="039e177a-979a-4c12-9fa5-268a0ee014ba"
)
```

#### **7. DynamicFlowNode** - AI-Generated Workflow
```python
dynamic = DynamicFlowNode(
    node_id="ai_workflow",
    name="AI-Generated Analysis Flow",
    description="Let AI determine the best workflow for complex cases"
)
```

---

## 🔨 **Building Hierarchical Workflows** {#building-workflows}

### **Pattern 1: Sequential Processing Pipeline**
```python
def create_document_pipeline():
    """Standard document processing pipeline"""
    
    # Main pipeline
    pipeline = SequenceNode("doc_pipeline", "Document Processing Pipeline")
    
    # Phase 1: Intake
    intake = SequenceNode("intake", "Document Intake")
    intake.add_child(ActionNode("upload", "Upload Document", action=upload_handler))
    intake.add_child(ActionNode("validate", "Validate Document", action=validate_handler))
    
    # Phase 2: Analysis Selection
    analysis = SelectorNode("analysis", "Analysis Selection")
    analysis.add_child(ActionNode("simple", "Simple Analysis", action=simple_analysis))
    analysis.add_child(ActionNode("complex", "Complex Analysis", action=complex_analysis))
    analysis.add_child(ActionNode("manual", "Manual Review", action=manual_review))
    
    # Phase 3: Report Generation
    reporting = ParallelNode("reporting", "Generate Reports")
    reporting.add_child(ActionNode("summary", "Summary Report", action=generate_summary))
    reporting.add_child(ActionNode("compliance", "Compliance Report", action=compliance_report))
    
    # Build complete pipeline
    pipeline.add_child(intake).add_child(analysis).add_child(reporting)
    
    return pipeline
```

### **Pattern 2: Risk-Based Decision Tree**
```python
def create_risk_workflow():
    """Risk-based document processing with compliance checks"""
    
    root = SequenceNode("risk_workflow", "Risk-Based Processing")
    
    # Risk assessment
    risk_check = ConditionNode(
        "risk_check", 
        "Risk Level Check",
        condition=lambda ctx: ctx.get("risk_score", 0) > 0.7
    )
    
    # High risk path
    high_risk = SequenceNode("high_risk", "High Risk Processing")
    high_risk.add_child(ActionNode("detailed_scan", "Detailed Security Scan", action=security_scan))
    high_risk.add_child(ActionNode("supervisor_review", "Supervisor Review", action=supervisor_review))
    
    # Standard path
    standard = ActionNode("standard_process", "Standard Processing", action=standard_process)
    
    # Risk-based selector
    risk_selector = SelectorNode("risk_selector", "Risk-Based Routing")
    risk_selector.add_child(high_risk).add_child(standard)
    
    root.add_child(risk_check).add_child(risk_selector)
    return root
```

### **Pattern 3: SPARSE Agreement Integration**
```python
def create_compliance_workflow():
    """Workflow with pre-approved SPARSE agreements"""
    
    root = SequenceNode("compliance_workflow", "Compliance-First Workflow")
    
    # Pre-approved document classification
    classification = SparseDecisionNode(
        "doc_classify",
        "Document Classification (Pre-Approved)", 
        sparse_agreement_id="039e177a-979a-4c12-9fa5-268a0ee014ba"
    )
    
    # Pre-approved standard compliance check
    compliance_check = SparseDecisionNode(
        "compliance_check",
        "Standard Compliance Validation",
        sparse_agreement_id="a1dcc693-5b2f-4c8e-9f1e-8d7c6a5b4321"
    )
    
    # Dynamic flow for uncertain cases
    dynamic_path = DynamicFlowNode(
        "dynamic_analysis",
        "AI-Generated Analysis Path"
    )
    
    # Compliance selector
    compliance_selector = SelectorNode("compliance_selector", "Compliance Path Selection")
    compliance_selector.add_child(compliance_check).add_child(dynamic_path)
    
    root.add_child(classification).add_child(compliance_selector)
    return root
```

---

## 💾 **Saving Workflows** {#saving-workflows}

### **Method 1: Save to JSON File**
```python
import json
from datetime import datetime

def save_workflow_to_file(dag_manager, filename):
    """Save complete workflow to JSON file"""
    
    workflow_data = {
        "workflow_metadata": {
            "name": dag_manager.name,
            "created_date": datetime.now().isoformat(),
            "compliance_level": dag_manager.compliance_level.value,
            "version": "1.0"
        },
        "root_nodes": [],
        "global_context": dag_manager.global_context
    }
    
    # Serialize each root node
    for root in dag_manager.root_nodes:
        workflow_data["root_nodes"].append(serialize_node(root))
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(workflow_data, f, indent=2)
    
    print(f"Workflow saved to {filename}")

def serialize_node(node):
    """Convert node to serializable dictionary"""
    node_data = {
        "node_id": node.node_id,
        "name": node.name,
        "description": node.description,
        "node_type": node.node_type.value,
        "children": [serialize_node(child) for child in node.children]
    }
    
    # Add type-specific data
    if hasattr(node, 'sparse_agreement_id'):
        node_data["sparse_agreement_id"] = node.sparse_agreement_id
    
    if hasattr(node, 'condition') and node.condition:
        node_data["condition_description"] = "Custom condition function"
    
    return node_data

# Usage
dag_manager = create_document_pipeline_dag()
save_workflow_to_file(dag_manager, "workflows/document_processing_v1.json")
```

### **Method 2: Save to Database**
```python
def save_workflow_to_database(dag_manager, db_connection):
    """Save workflow to PostgreSQL database"""
    
    workflow_record = {
        "workflow_id": str(uuid.uuid4()),
        "name": dag_manager.name,
        "compliance_level": dag_manager.compliance_level.value,
        "created_date": datetime.now(),
        "workflow_json": json.dumps(serialize_dag(dag_manager)),
        "status": "active",
        "version": "1.0"
    }
    
    # Insert into workflows table
    cursor = db_connection.cursor()
    cursor.execute("""
        INSERT INTO workflows (workflow_id, name, compliance_level, created_date, workflow_json, status, version)
        VALUES (%(workflow_id)s, %(name)s, %(compliance_level)s, %(created_date)s, %(workflow_json)s, %(status)s, %(version)s)
    """, workflow_record)
    
    db_connection.commit()
    print(f"Workflow saved to database with ID: {workflow_record['workflow_id']}")
    
    return workflow_record['workflow_id']
```

### **Method 3: Export with SPARSE Agreements**
```python
def export_complete_workflow(dag_manager, export_dir):
    """Export workflow with all dependencies"""
    
    os.makedirs(export_dir, exist_ok=True)
    
    # 1. Save main workflow
    workflow_file = os.path.join(export_dir, "workflow.json")
    save_workflow_to_file(dag_manager, workflow_file)
    
    # 2. Export referenced SPARSE agreements
    sparse_dir = os.path.join(export_dir, "sparse_agreements")
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Find all SPARSE agreement references
    agreement_ids = find_sparse_agreement_references(dag_manager)
    
    for agreement_id in agreement_ids:
        agreement_file = f"tidyllm-heiros/examples/demo_sparse_agreements/{agreement_id}.json"
        if os.path.exists(agreement_file):
            shutil.copy(agreement_file, os.path.join(sparse_dir, f"{agreement_id}.json"))
    
    # 3. Create metadata file
    metadata = {
        "export_date": datetime.now().isoformat(),
        "workflow_name": dag_manager.name,
        "sparse_agreements": len(agreement_ids),
        "compliance_level": dag_manager.compliance_level.value
    }
    
    with open(os.path.join(export_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Complete workflow exported to {export_dir}")
```

---

## 📂 **Loading and Retrieving Workflows** {#loading-workflows}

### **Method 1: Load from JSON File**
```python
def load_workflow_from_file(filename):
    """Load workflow from JSON file"""
    
    with open(filename, 'r') as f:
        workflow_data = json.load(f)
    
    # Create DAG manager
    dag_manager = HierarchicalDAGManager(
        name=workflow_data["workflow_metadata"]["name"],
        compliance_level=ComplianceLevel(workflow_data["workflow_metadata"]["compliance_level"])
    )
    
    # Restore global context
    dag_manager.global_context.update(workflow_data.get("global_context", {}))
    
    # Rebuild root nodes
    for root_data in workflow_data["root_nodes"]:
        root_node = deserialize_node(root_data)
        dag_manager.add_root_node(root_node)
    
    print(f"Workflow loaded: {dag_manager.name}")
    return dag_manager

def deserialize_node(node_data):
    """Convert dictionary back to node object"""
    
    node_type = NodeType(node_data["node_type"])
    
    # Create appropriate node type
    if node_type == NodeType.SEQUENCE:
        node = SequenceNode(
            node_data["node_id"],
            node_data["name"], 
            node_data["description"]
        )
    elif node_type == NodeType.SELECTOR:
        node = SelectorNode(
            node_data["node_id"],
            node_data["name"],
            node_data["description"]
        )
    elif node_type == NodeType.ACTION:
        # Note: Action functions need to be reconnected
        node = ActionNode(
            node_data["node_id"],
            node_data["name"],
            node_data["description"],
            action=lambda ctx: {"status": "success", "message": "Loaded action"}
        )
    elif node_type == NodeType.SPARSE_DECISION:
        node = SparseDecisionNode(
            node_data["node_id"],
            node_data["name"],
            node_data.get("sparse_agreement_id")
        )
    # ... handle other node types
    
    # Rebuild children
    for child_data in node_data.get("children", []):
        child_node = deserialize_node(child_data)
        node.add_child(child_node)
    
    return node

# Usage
dag_manager = load_workflow_from_file("workflows/document_processing_v1.json")
```

### **Method 2: Load from Database**
```python
def load_workflow_from_database(workflow_id, db_connection):
    """Load workflow from database by ID"""
    
    cursor = db_connection.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT * FROM workflows WHERE workflow_id = %s AND status = 'active'
    """, (workflow_id,))
    
    workflow_record = cursor.fetchone()
    if not workflow_record:
        raise ValueError(f"Workflow {workflow_id} not found")
    
    # Parse workflow JSON
    workflow_data = json.loads(workflow_record["workflow_json"])
    
    # Create DAG manager
    dag_manager = HierarchicalDAGManager(
        name=workflow_record["name"],
        compliance_level=ComplianceLevel(workflow_record["compliance_level"])
    )
    
    # Rebuild from stored data
    rebuild_dag_from_data(dag_manager, workflow_data)
    
    print(f"Workflow loaded from database: {dag_manager.name}")
    return dag_manager

def list_saved_workflows(db_connection):
    """List all saved workflows"""
    
    cursor = db_connection.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT workflow_id, name, compliance_level, created_date, version, status
        FROM workflows 
        ORDER BY created_date DESC
    """)
    
    workflows = cursor.fetchall()
    
    print("Available Workflows:")
    for wf in workflows:
        print(f"  ID: {wf['workflow_id']}")
        print(f"  Name: {wf['name']}")
        print(f"  Compliance: {wf['compliance_level']}")
        print(f"  Created: {wf['created_date']}")
        print(f"  Status: {wf['status']}")
        print()
    
    return workflows
```

### **Method 3: Load Workflow Templates**
```python
def load_workflow_template(template_name):
    """Load predefined workflow templates"""
    
    templates = {
        "document_processing": create_document_pipeline,
        "risk_assessment": create_risk_workflow,
        "compliance_review": create_compliance_workflow,
        "mvr_analysis": create_mvr_workflow
    }
    
    if template_name not in templates:
        raise ValueError(f"Template '{template_name}' not found. Available: {list(templates.keys())}")
    
    # Create DAG manager with template
    dag_manager = HierarchicalDAGManager(
        name=f"Template: {template_name}",
        compliance_level=ComplianceLevel.FULL_TRANSPARENCY
    )
    
    # Add template workflow
    template_workflow = templates[template_name]()
    dag_manager.add_root_node(template_workflow)
    
    print(f"Template loaded: {template_name}")
    return dag_manager

# Usage
dag_manager = load_workflow_template("document_processing")
```

---

## 📋 **SPARSE Agreements Integration** {#sparse-agreements}

### **Creating SPARSE Agreements**
```python
def create_sparse_agreement():
    """Create new SPARSE agreement for workflow"""
    
    from sparse_system import SparseAgreementManager, RiskLevel
    
    sparse_manager = SparseAgreementManager()
    
    # Create new agreement
    agreement = sparse_manager.create_agreement(
        title="Automated Document Classification",
        business_purpose="Streamline document intake with audit trail",
        business_owner="Risk Management Team",
        technical_owner="AI Systems Team",
        risk_level=RiskLevel.LOW
    )
    
    # Add execution conditions
    agreement.add_condition(
        description="Document size under 50MB",
        condition_type="context_check",
        parameters={"max_size": 50_000_000},
        mandatory=True
    )
    
    # Add approved actions
    agreement.add_action(
        name="ML Classification",
        action_type="ml_classification",
        parameters={
            "model": "document_classifier_v2",
            "confidence_threshold": 0.85
        }
    )
    
    # Add stakeholder approvals
    agreement.add_stakeholder_approval(
        name="Jane Smith",
        role="VP Risk Management", 
        comments="Approved for standard workflow"
    )
    
    # Finalize agreement
    agreement_id = sparse_manager.approve_agreement(agreement)
    
    print(f"SPARSE agreement created: {agreement_id}")
    return agreement_id
```

### **Using SPARSE Agreements in Workflows**
```python
def create_workflow_with_sparse():
    """Create workflow using SPARSE agreements"""
    
    # Create or load existing agreement
    agreement_id = "039e177a-979a-4c12-9fa5-268a0ee014ba"
    
    # Use in workflow
    workflow = SequenceNode("sparse_workflow", "SPARSE-Enabled Workflow")
    
    # Pre-approved classification step
    classification = SparseDecisionNode(
        "classify_sparse",
        "Document Classification (Pre-Approved)",
        sparse_agreement_id=agreement_id
    )
    
    # Standard processing after classification
    processing = ActionNode(
        "process_doc",
        "Document Processing",
        action=lambda ctx: process_document(ctx["classification_result"])
    )
    
    workflow.add_child(classification).add_child(processing)
    
    return workflow
```

---

## ⚡ **Execution and Monitoring** {#execution}

### **Execute Workflow with Full Monitoring**
```python
def execute_with_monitoring(dag_manager, context):
    """Execute workflow with comprehensive monitoring"""
    
    print(f"Executing workflow: {dag_manager.name}")
    print("="*50)
    
    # 1. Pre-execution validation
    validation_report = dag_manager.validate_dag()
    if validation_report.get("has_errors"):
        print("Validation errors found:")
        for error in validation_report.get("errors", []):
            print(f"  - {error}")
        return None
    
    # 2. Execute workflow
    start_time = datetime.now()
    result = dag_manager.execute_dag(context)
    execution_time = datetime.now() - start_time
    
    # 3. Generate reports
    compliance_report = dag_manager.generate_compliance_report()
    
    # 4. Display results
    print(f"Execution completed in {execution_time.total_seconds():.2f} seconds")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Nodes executed: {len(result.get('node_results', []))}")
    print(f"Audit completeness: {compliance_report.get('audit_completeness', 0):.1%}")
    
    # 5. Show execution path
    print("\nExecution Path:")
    for node_summary in compliance_report.get("nodes_summary", []):
        status_icon = "✓" if node_summary["status"] == "success" else "✗"
        print(f"  {status_icon} {node_summary['hierarchy_path']}")
    
    return result

# Usage
context = {
    "document_id": "doc_12345",
    "file_size": 2_000_000,
    "file_extension": "pdf",
    "user_id": "analyst_1",
    "risk_score": 0.3
}

result = execute_with_monitoring(dag_manager, context)
```

### **Workflow Execution History**
```python
def track_execution_history(dag_manager, result, context):
    """Track and store execution history"""
    
    execution_record = {
        "execution_id": str(uuid.uuid4()),
        "workflow_name": dag_manager.name,
        "execution_date": datetime.now(),
        "status": result.get("status"),
        "duration": result.get("duration", 0),
        "context": context,
        "node_results": result.get("node_results", []),
        "compliance_report": dag_manager.generate_compliance_report()
    }
    
    # Save to execution history file
    history_file = "workflow_executions.jsonl"
    with open(history_file, 'a') as f:
        f.write(json.dumps(execution_record) + "\n")
    
    return execution_record["execution_id"]
```

---

## 🔧 **Advanced Patterns** {#advanced-patterns}

### **Pattern 1: Conditional Workflows**
```python
def create_conditional_workflow():
    """Workflow that adapts based on document properties"""
    
    root = SequenceNode("conditional_workflow", "Adaptive Document Processing")
    
    # Document analysis
    analysis = ActionNode("analyze", "Document Analysis", action=analyze_document)
    
    # Conditional routing based on analysis
    routing = ConditionNode(
        "route_decision",
        "Processing Route Decision",
        condition=lambda ctx: ctx.get("document_complexity") > 0.8
    )
    
    # Complex document path
    complex_path = SequenceNode("complex_path", "Complex Document Processing")
    complex_path.add_child(ActionNode("deep_analysis", "Deep Analysis", action=deep_analysis))
    complex_path.add_child(ActionNode("human_review", "Human Review", action=human_review))
    
    # Simple document path  
    simple_path = ActionNode("simple_process", "Simple Processing", action=simple_process)
    
    # Route selector
    router = SelectorNode("router", "Processing Router")
    router.add_child(complex_path).add_child(simple_path)
    
    root.add_child(analysis).add_child(routing).add_child(router)
    return root
```

### **Pattern 2: Error Handling and Retry Logic**
```python
def create_resilient_workflow():
    """Workflow with built-in error handling and retry logic"""
    
    root = SequenceNode("resilient_workflow", "Resilient Processing Workflow")
    
    # Retry wrapper for critical operations
    def retry_action(func, max_retries=3):
        def wrapper(context):
            for attempt in range(max_retries):
                try:
                    result = func(context)
                    if result.get("status") == "success":
                        return result
                except Exception as e:
                    if attempt == max_retries - 1:
                        return {"status": "failure", "error": str(e)}
                    context["retry_attempt"] = attempt + 1
            return {"status": "failure", "error": "Max retries exceeded"}
        return wrapper
    
    # Critical processing with retry
    critical_step = ActionNode(
        "critical_process",
        "Critical Processing (with retry)",
        action=retry_action(critical_processing_function)
    )
    
    # Error recovery path
    recovery = SelectorNode("recovery", "Error Recovery")
    recovery.add_child(ActionNode("auto_fix", "Automatic Fix", action=auto_fix_errors))
    recovery.add_child(ActionNode("manual_intervention", "Manual Intervention", action=manual_fix))
    
    root.add_child(critical_step).add_child(recovery)
    return root
```

### **Pattern 3: Multi-Stage Approval Workflow**
```python
def create_approval_workflow():
    """Multi-stage approval workflow for sensitive documents"""
    
    root = SequenceNode("approval_workflow", "Multi-Stage Approval Process")
    
    # Stage 1: Initial screening
    screening = SequenceNode("screening", "Initial Screening")
    screening.add_child(ActionNode("security_scan", "Security Scan", action=security_scan))
    screening.add_child(ActionNode("content_filter", "Content Filter", action=content_filter))
    
    # Stage 2: Risk assessment
    risk_assessment = ActionNode("risk_assess", "Risk Assessment", action=assess_risk)
    
    # Stage 3: Approval routing based on risk
    approval_router = SelectorNode("approval_router", "Approval Routing")
    
    # Low risk - automatic approval
    auto_approve = ConditionNode(
        "auto_approve_check",
        "Auto-Approval Check", 
        condition=lambda ctx: ctx.get("risk_score", 1.0) < 0.3
    )
    auto_approve.add_child(ActionNode("auto_approval", "Automatic Approval", action=auto_approve_doc))
    
    # Medium risk - supervisor approval
    supervisor_approve = ConditionNode(
        "supervisor_check",
        "Supervisor Approval Check",
        condition=lambda ctx: ctx.get("risk_score", 1.0) < 0.7
    )
    supervisor_approve.add_child(ActionNode("supervisor_approval", "Supervisor Approval", action=supervisor_approve_doc))
    
    # High risk - committee approval
    committee_approve = ActionNode("committee_approval", "Committee Approval", action=committee_approve_doc)
    
    approval_router.add_child(auto_approve).add_child(supervisor_approve).add_child(committee_approve)
    
    root.add_child(screening).add_child(risk_assessment).add_child(approval_router)
    return root
```

---

## 📝 **Complete Example: MVR Document Workflow**

```python
#!/usr/bin/env python3
"""
Complete example: MVR Document Processing Workflow
Demonstrates all key concepts in a real-world scenario
"""

import sys
sys.path.append('tidyllm-heiros/src/dag-manager')
sys.path.append('tidyllm-heiros/src/sparse-agreement')

from hierarchical_dag_manager import *
import json
from datetime import datetime

def create_mvr_workflow():
    """Complete MVR document processing workflow"""
    
    # Create DAG manager
    dag_manager = HierarchicalDAGManager(
        name="MVR Document Processing System",
        compliance_level=ComplianceLevel.FULL_TRANSPARENCY
    )
    
    # Root workflow
    mvr_root = SequenceNode("mvr_root", "MVR Processing Workflow")
    
    # Phase 1: Document Intake
    intake_phase = SequenceNode("intake", "Document Intake Phase")
    
    # SPARSE-approved classification
    classification = SparseDecisionNode(
        "doc_classify_sparse",
        "Document Type Classification (Pre-Approved)",
        sparse_agreement_id="039e177a-979a-4c12-9fa5-268a0ee014ba"
    )
    
    validation = ActionNode(
        "doc_validation", 
        "Document Validation",
        action=lambda ctx: {
            "status": "success",
            "validation_score": 0.95,
            "issues_found": []
        }
    )
    
    intake_phase.add_child(classification).add_child(validation)
    
    # Phase 2: Analysis Path Selection
    analysis_selector = SelectorNode("analysis_selector", "Analysis Path Selection")
    
    # Standard compliance path (SPARSE-approved)
    standard_path = SequenceNode("standard_compliance", "Standard Compliance Path")
    standard_compliance = SparseDecisionNode(
        "standard_compliance_sparse",
        "Standard MVR Compliance Check",
        sparse_agreement_id="a1dcc693-5b2f-4c8e-9f1e-8d7c6a5b4321"
    )
    standard_path.add_child(standard_compliance)
    
    # Complex analysis path (AI-generated)
    complex_path = DynamicFlowNode(
        "complex_analysis_dynamic",
        "Complex Analysis Dynamic Flow"
    )
    
    # Manual review fallback
    manual_review = ActionNode(
        "manual_review_required",
        "Manual Review Required", 
        action=lambda ctx: {"status": "success", "assigned_to": "senior_analyst"}
    )
    
    analysis_selector.add_child(standard_path).add_child(complex_path).add_child(manual_review)
    
    # Phase 3: Report Generation
    report_phase = ParallelNode("report_generation", "Report Generation Phase")
    
    findings_report = ActionNode(
        "findings_summary",
        "Generate Findings Summary",
        action=lambda ctx: {"status": "success", "findings_count": 12}
    )
    
    compliance_report = ActionNode(
        "compliance_report_generation", 
        "Generate Compliance Report",
        action=lambda ctx: {"status": "success", "compliance_score": 0.87}
    )
    
    report_phase.add_child(findings_report).add_child(compliance_report)
    
    # Phase 4: Final Review
    final_review = SelectorNode("final_review", "Final Review Process")
    
    auto_approve = ConditionNode(
        "auto_approve",
        "Automatic Approval Check",
        condition=lambda ctx: ctx.get("compliance_score", 0) > 0.85
    )
    auto_approve.add_child(ActionNode("auto_approval", "Automatic Approval", action=lambda ctx: {"status": "approved"}))
    
    human_review = ActionNode(
        "human_review_required",
        "Human Review Required",
        action=lambda ctx: {"status": "pending_review", "reviewer": "compliance_team"}
    )
    
    final_review.add_child(auto_approve).add_child(human_review)
    
    # Build complete workflow
    mvr_root.add_child(intake_phase).add_child(analysis_selector).add_child(report_phase).add_child(final_review)
    
    # Add to DAG manager
    dag_manager.add_root_node(mvr_root)
    
    return dag_manager

def save_and_load_example():
    """Demonstrate saving and loading workflows"""
    
    # Create workflow
    dag_manager = create_mvr_workflow()
    
    # Save to file
    save_workflow_to_file(dag_manager, "mvr_workflow.json")
    
    # Load from file
    loaded_dag = load_workflow_from_file("mvr_workflow.json")
    
    # Execute loaded workflow
    context = {
        "document_id": "mvr_001",
        "document_type": "mvr_report", 
        "file_size": 1_500_000,
        "risk_score": 0.4,
        "user_id": "analyst_smith"
    }
    
    result = execute_with_monitoring(loaded_dag, context)
    
    return result

if __name__ == "__main__":
    # Run complete example
    result = save_and_load_example()
    print(f"Complete workflow example finished with status: {result.get('status')}")
```

---

## 🎯 **Summary**

### **Key Takeaways:**

1. **Hierarchical Structure** - Build workflows as trees, not complex graphs
2. **Node Types** - Mix sequences, selectors, actions, and conditions as needed
3. **SPARSE Integration** - Use pre-approved agreements for compliance
4. **Save/Load Flexibility** - JSON files, databases, or complete exports
5. **Monitoring & Audit** - Full transparency with compliance reporting
6. **Error Resilience** - Built-in retry logic and recovery paths

### **Best Practices:**

- ✅ **Start Simple** - Begin with basic sequences and selectors
- ✅ **Use SPARSE** - Pre-approve common decisions for faster execution  
- ✅ **Save Frequently** - Version control your workflows
- ✅ **Monitor Everything** - Full audit trails for compliance
- ✅ **Plan for Errors** - Always have fallback paths
- ✅ **Test Thoroughly** - Validate workflows before production

### **Next Steps:**

1. **Create your first workflow** using the quick start guide
2. **Set up SPARSE agreements** for repeated decisions
3. **Implement saving/loading** for your use case
4. **Add monitoring and reporting** for production use
5. **Build advanced patterns** as your needs grow

**TidyLLM-HeirOS provides enterprise-grade workflow management with complete corporate compliance and audit trails!** 🚀

---

*Generated: 2025-09-01 | TidyLLM-HeirOS Workflow Management Guide v1.0*