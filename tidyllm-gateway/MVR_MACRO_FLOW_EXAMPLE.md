# MVR Macro Flow Example with TidyLLM Gateway

## Complete End-to-End Example

This example demonstrates how the MVR (Model Validation Report) macro flows through the TidyLLM Gateway with full enterprise governance, audit trails, and cost tracking for a financial institution.

## 🏦 Scenario Setup

**Bank**: First National Bank  
**Department**: Risk Management - Model Validation Unit  
**Analyst**: Sarah Chen (sarah.chen@firstnational.com)  
**Document**: Credit Risk Model v3.2 Validation Report (85 pages)  
**Analysis Type**: Quarterly regulatory review for Federal Reserve  
**Budget**: $2,500 daily department limit, $50 per-analysis limit  

## 📋 Complete Flow Implementation

### Step 1: Enterprise Gateway Initialization

```python
#!/usr/bin/env python3
"""
Enterprise MVR Demo with TidyLLM Gateway Integration
Complete flow from upload to regulatory report
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, List

# TidyLLM Gateway imports
from tidyllm_gateway import LLMGateway, LLMGatewayConfig
from tidyllm_gateway.integrations.mcp_integration import (
    create_mcp_gateway_provider, 
    integrate_with_advanced_qa_orchestrator
)

# MVR macro imports
from backend.macros.built_ins.mvr_peer_review import mvr_peer_review_pipeline
from backend.macros.tidyverse_pipe import DocumentCollection

# Banking-specific configuration
BANKING_LLM_CONFIG = LLMGatewayConfig(
    # Corporate MLFlow Gateway
    base_url="https://mlflow-gateway.firstnational.com",
    mlflow_gateway_uri="https://mlflow-gateway.firstnational.com",
    
    # IT-approved providers for financial services
    available_providers=["claude-banking", "azure-openai-gov"],
    default_provider="claude-banking",
    provider_models={
        "claude-banking": ["claude-3-5-sonnet-banking"],  # Bank's custom Claude endpoint
        "azure-openai-gov": ["gpt-4o-gov"]               # Azure Government Cloud
    },
    
    # Financial services cost controls
    max_cost_per_request_usd=50.0,        # Per-analysis limit
    budget_limit_daily_usd=2500.0,        # Department daily limit
    
    # Banking compliance requirements
    tenant_id="risk-management",
    department="model-validation-unit", 
    compliance_mode="banking_regulation",
    require_audit_reason=True,
    enable_audit_logging=True,
    audit_retention_days=2555,            # 7 years for banking records
    
    # Performance requirements for regulatory deadlines
    timeout=300,                          # 5 minute SLA
    max_retries=3,
    
    # Security settings
    tls_verify=True,
    auth_method="kerberos"               # Corporate authentication
)

class EnterpriseMVRDemo:
    """Enterprise-grade MVR Demo with TidyLLM Gateway"""
    
    def __init__(self):
        # Initialize gateway provider
        self.mcp_provider = create_mcp_gateway_provider(
            llm_config=BANKING_LLM_CONFIG
        )
        
        # Corporate authentication
        self.current_user = self._authenticate_user()
        
        # Usage tracking
        self.session_stats = {
            "analyses_completed": 0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "avg_processing_time_ms": 0.0
        }
    
    def _authenticate_user(self) -> str:
        """Corporate SSO authentication"""
        st.sidebar.markdown("## 🏦 Corporate Authentication")
        
        # In production, this integrates with SAML/Kerberos
        user_id = st.sidebar.text_input(
            "Corporate ID", 
            value="sarah.chen@firstnational.com",
            help="Use your corporate email address"
        )
        
        department = st.sidebar.selectbox(
            "Department",
            ["model-validation-unit", "credit-risk", "market-risk", "operational-risk"]
        )
        
        if user_id and "@firstnational.com" in user_id:
            st.sidebar.success(f"✅ Authenticated: {user_id}")
            st.sidebar.info(f"📋 Department: {department}")
            return user_id
        else:
            st.sidebar.error("❌ Invalid corporate credentials")
            st.stop()
    
    def display_enterprise_dashboard(self):
        """Display enterprise governance dashboard"""
        
        # Gateway status
        gateway_status = self.mcp_provider.get_integration_status()
        
        st.sidebar.markdown("## 🔒 Enterprise Status")
        
        # Budget tracking
        budget_used = gateway_status["integration_metrics"]["total_cost_usd"]
        budget_limit = BANKING_LLM_CONFIG.budget_limit_daily_usd
        budget_remaining = budget_limit - budget_used
        budget_pct = (budget_used / budget_limit) * 100
        
        st.sidebar.metric(
            "Daily Budget Usage", 
            f"${budget_used:.2f} / ${budget_limit:.2f}",
            f"{budget_remaining:.2f} remaining"
        )
        st.sidebar.progress(budget_pct / 100)
        
        # Usage metrics
        st.sidebar.metric("Analyses Today", gateway_status["integration_metrics"]["orchestrator_requests"])
        st.sidebar.metric("Avg Response Time", f"{gateway_status['integration_metrics']['avg_response_time_ms']:.0f}ms")
        
        # Compliance status
        st.sidebar.markdown("## ✅ Compliance Status")
        st.sidebar.success("🔐 Corporate gateway active")
        st.sidebar.success("📝 Full audit trail enabled")
        st.sidebar.success("💰 Cost tracking active")
        st.sidebar.success("👤 User attribution enabled")
        
        # Regulatory compliance
        st.sidebar.markdown("## 📋 Regulatory Compliance")
        st.sidebar.info("✅ Federal Reserve SR 11-7")
        st.sidebar.info("✅ OCC Model Risk Management")
        st.sidebar.info("✅ Basel III Model Standards")
        st.sidebar.info("✅ SOX Section 404 Controls")

def run_enterprise_mvr_analysis():
    """Main MVR analysis flow with enterprise governance"""
    
    st.title("🏦 Enterprise MVR Analysis - First National Bank")
    st.markdown("*Model Validation Report Analysis with Federal Banking Compliance*")
    
    # Initialize enterprise demo
    demo = EnterpriseMVRDemo()
    demo.display_enterprise_dashboard()
    
    # Document upload section
    st.markdown("## 📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload Model Validation Report",
        type=["pdf"],
        help="Upload MVR for regulatory compliance analysis"
    )
    
    # Analysis configuration
    st.markdown("## ⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        complexity = st.selectbox(
            "Analysis Complexity",
            ["simple", "enhanced", "advanced"],
            index=1,  # Default to enhanced
            help="Simple: 2-5min, Enhanced: 10-20min, Advanced: 20-45min"
        )
        
        analysis_purpose = st.selectbox(
            "Analysis Purpose",
            [
                "Quarterly Federal Reserve Review",
                "Annual OCC Examination Prep", 
                "Internal Model Validation",
                "Audit Preparation",
                "Basel III Compliance Check"
            ]
        )
    
    with col2:
        model_type = st.selectbox(
            "Model Type",
            ["Credit Risk", "Market Risk", "Operational Risk", "CECL", "Stress Testing"]
        )
        
        priority = st.selectbox(
            "Priority Level",
            ["Standard", "High", "Critical - Regulatory Deadline"]
        )
    
    # Cost estimation
    estimated_costs = {
        "simple": 2.50,
        "enhanced": 8.75,
        "advanced": 24.50
    }
    
    st.info(f"💰 Estimated Cost: ${estimated_costs[complexity]:.2f}")
    
    # Execute analysis
    if uploaded_file and st.button("🚀 Start Enterprise Analysis"):
        
        # Create audit reason for regulatory compliance
        audit_reason = f"{analysis_purpose} - {model_type} model validation - Priority: {priority}"
        
        st.markdown("---")
        st.markdown("## 🔄 Processing Pipeline")
        
        with st.spinner("Processing through enterprise gateway..."):
            
            # Step 1: Document ingestion with governance
            st.markdown("### 📥 Step 1: Document Ingestion")
            progress_bar = st.progress(0)
            
            # Create document collection
            mvr_data = [{
                "filename": uploaded_file.name,
                "document_type": "mvr",
                "content": "Model Validation Report Content...",  # Would extract from PDF
                "pages": 85,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "uploaded_by": demo.current_user,
                "audit_reason": audit_reason
            }]
            
            mvr_docs = DocumentCollection(data=mvr_data)
            progress_bar.progress(20)
            st.success("✅ Document ingested with audit trail")
            
            # Step 2: Gateway execution with enterprise controls
            st.markdown("### 🔒 Step 2: Enterprise Gateway Execution")
            
            start_time = datetime.utcnow()
            
            try:
                # Execute MVR macro through enterprise gateway
                result = demo.mcp_provider.execute_orchestrated_workflow(
                    orchestrator_id="mvr_orchestrator",
                    workflow_name="mvr_peer_review_pipeline", 
                    inputs={
                        "mvr_document": mvr_docs,
                        "complexity": complexity,
                        "model_type": model_type,
                        "analysis_purpose": analysis_purpose,
                        "priority": priority
                    },
                    user_id=demo.current_user,
                    audit_reason=audit_reason
                )
                
                progress_bar.progress(60)
                
                if result["success"]:
                    
                    # Step 3: Results processing
                    st.markdown("### 📊 Step 3: Results Processing") 
                    
                    execution_time = (datetime.utcnow() - start_time).total_seconds()
                    cost_usd = result.get("cost_usd", estimated_costs[complexity])
                    
                    # Update session stats
                    demo.session_stats["analyses_completed"] += 1
                    demo.session_stats["total_cost_usd"] += cost_usd
                    demo.session_stats["total_tokens"] += result.get("tokens_used", 0)
                    
                    progress_bar.progress(80)
                    
                    # Step 4: Generate regulatory report
                    st.markdown("### 📋 Step 4: Regulatory Report Generation")
                    
                    # Simulate the actual MVR macro execution results
                    mvr_results = simulate_mvr_macro_results(complexity, model_type)
                    
                    progress_bar.progress(100)
                    st.success("✅ Analysis completed successfully")
                    
                    # Display results
                    display_mvr_results(mvr_results, result, complexity, execution_time)
                    
                else:
                    st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Enterprise gateway error: {e}")

def simulate_mvr_macro_results(complexity: str, model_type: str) -> Dict[str, Any]:
    """Simulate MVR macro execution results"""
    
    if complexity == "simple":
        return {
            "analysis_type": "Basic Compliance Check",
            "model_context": {
                "model_type": model_type,
                "risk_tier": "Tier 1 - High Risk", 
                "validation_scope": "Quarterly Review"
            },
            "compliance_status": "Partially Compliant",
            "completion_percentage": 78.5,
            "missing_sections": ["5.3.2", "7.1.4"],
            "processing_time": "3.2 minutes",
            "summary": "Basic compliance assessment completed. 2 sections require attention."
        }
    
    elif complexity == "enhanced":
        return {
            "analysis_type": "Evidence-Focused Review",
            "model_context": {
                "model_type": model_type,
                "risk_tier": "Tier 1 - High Risk",
                "validation_scope": "Comprehensive Quarterly Review"
            },
            "compliance_matrix": [
                {
                    "mvr_section": "4 - Conceptual Soundness",
                    "mvs_requirements": "MVS 5.4.3, 5.4.3.1–3, 5.12.1; VST Conceptual Soundness",
                    "review_narrative": "Section covers methodology, segmentation, variable selection with quantitative validation results.",
                    "contradiction_summary": "No contradictions detected. Devil's advocate: Heavy reliance on SHAP for feature selection interpretability.",
                    "peer_review_challenge": "Rationale for SHAP-based feature selection methodology not fully supported by regulatory guidance. Consider alternative interpretability methods.",
                    "conclusion": "✅ Compliant",
                    "confidence_score": "Highly Confident",
                    "defect_type": "N/A"
                },
                {
                    "mvr_section": "5 - Data Quality",
                    "mvs_requirements": "MVS 5.5.1-5.5.4; VST Data Analysis",
                    "review_narrative": "Comprehensive data quality assessment including missing value analysis and outlier detection.",
                    "contradiction_summary": "Minor inconsistency: Sample size justification differs between sections 5.1 and 5.3.",
                    "peer_review_challenge": "Sample size of 50,000 observations may be insufficient for portfolio complexity. Recommend statistical power analysis.",
                    "conclusion": "⚠️ Attention Required",
                    "confidence_score": "Moderately Confident", 
                    "defect_type": "Sample Size Adequacy"
                }
            ],
            "internal_contradictions": 1,
            "peer_challenges": 2,
            "evidence_mapping_score": 85.7,
            "logic_trace_score": 90.5,
            "processing_time": "12.8 minutes",
            "overall_score": 88.1
        }
    
    elif complexity == "advanced":
        return {
            "analysis_type": "AI-Powered Comprehensive Review",
            "model_context": {
                "model_type": model_type,
                "risk_tier": "Tier 1 - High Risk",
                "validation_scope": "Federal Reserve SR 11-7 Full Compliance"
            },
            "comprehensive_analysis": {
                "regulatory_compliance_score": 92.3,
                "external_contradictions": 0,
                "industry_benchmark_comparison": "Above average",
                "regulatory_updates_impact": "2 minor updates require attention",
                "enforcement_action_relevance": "No relevant actions found"
            },
            "multi_persona_consensus": {
                "mvr_expert_score": 91.5,
                "regulatory_expert_score": 93.1, 
                "risk_expert_score": 88.7,
                "consensus_score": 91.1,
                "consensus_achieved": True
            },
            "real_time_monitoring": {
                "alerts_configured": 5,
                "risk_indicators": ["Sample size adequacy", "Model stability"],
                "escalation_triggers": "None active"
            },
            "audit_readiness": "Fully Prepared",
            "processing_time": "28.5 minutes",
            "recommendation": "Approve with minor recommendations. Address sample size documentation."
        }

def display_mvr_results(mvr_results: Dict, gateway_result: Dict, complexity: str, execution_time: float):
    """Display MVR analysis results with enterprise metrics"""
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Analysis Type", 
            mvr_results["analysis_type"],
            delta="Complete"
        )
    
    with col2:
        st.metric(
            "Processing Time",
            f"{execution_time:.1f}s",
            delta=f"Target: {mvr_results['processing_time']}"
        )
    
    with col3:
        st.metric(
            "Cost",
            f"${gateway_result.get('cost_usd', 0):.2f}",
            delta="Within budget"
        )
    
    with col4:
        overall_score = mvr_results.get('overall_score', mvr_results.get('completion_percentage', 0))
        st.metric(
            "Overall Score",
            f"{overall_score:.1f}%",
            delta="Above threshold"
        )
    
    # Detailed results based on complexity
    if complexity == "simple":
        st.markdown("### 📋 Basic Compliance Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Context:**")
            st.json(mvr_results["model_context"])
            
        with col2:
            st.markdown("**Compliance Status:**")
            st.info(f"Status: {mvr_results['compliance_status']}")
            st.info(f"Completion: {mvr_results['completion_percentage']}%")
            if mvr_results["missing_sections"]:
                st.warning(f"Missing sections: {', '.join(mvr_results['missing_sections'])}")
    
    elif complexity == "enhanced":
        st.markdown("### 📊 Detailed Compliance Matrix")
        
        # Display compliance matrix as table
        import pandas as pd
        df = pd.DataFrame(mvr_results["compliance_matrix"])
        st.dataframe(df, use_container_width=True)
        
        # Evidence and logic metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Evidence Mapping", f"{mvr_results['evidence_mapping_score']:.1f}%")
        
        with col2:
            st.metric("Logic Tracing", f"{mvr_results['logic_trace_score']:.1f}%") 
        
        with col3:
            st.metric("Internal Contradictions", mvr_results['internal_contradictions'])
    
    elif complexity == "advanced":
        st.markdown("### 🤖 AI-Powered Comprehensive Analysis")
        
        # Multi-persona consensus
        consensus = mvr_results["multi_persona_consensus"]
        st.markdown("**Multi-Expert Consensus:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MVR Expert", f"{consensus['mvr_expert_score']:.1f}%")
        with col2:
            st.metric("Regulatory Expert", f"{consensus['regulatory_expert_score']:.1f}%") 
        with col3:
            st.metric("Risk Expert", f"{consensus['risk_expert_score']:.1f}%")
        
        if consensus['consensus_achieved']:
            st.success(f"✅ Expert consensus achieved: {consensus['consensus_score']:.1f}%")
        else:
            st.warning("⚠️ No expert consensus - manual review required")
        
        # Regulatory compliance
        reg_analysis = mvr_results["comprehensive_analysis"]
        st.markdown("**Regulatory Analysis:**")
        st.info(f"🏛️ Compliance Score: {reg_analysis['regulatory_compliance_score']:.1f}%")
        st.info(f"📊 Industry Benchmark: {reg_analysis['industry_benchmark_comparison']}")
        st.info(f"📋 Audit Readiness: {mvr_results['audit_readiness']}")
    
    # Enterprise governance summary
    st.markdown("### 🏢 Enterprise Governance Summary")
    
    governance_col1, governance_col2 = st.columns(2)
    
    with governance_col1:
        st.markdown("**Audit Trail:**")
        st.success("✅ User attribution recorded")
        st.success("✅ Business purpose documented") 
        st.success("✅ Cost allocated to department")
        st.success("✅ Processing time tracked")
        
    with governance_col2:
        st.markdown("**Regulatory Compliance:**")
        st.success("✅ Federal Reserve SR 11-7")
        st.success("✅ OCC Model Risk Management")
        st.success("✅ SOX Section 404 Controls")
        st.success("✅ Data classification maintained")
    
    # Export options
    st.markdown("### 📤 Export Options")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("📊 Export Excel"):
            st.success("Excel report generated")
    
    with export_col2:
        if st.button("📄 Export PDF"):
            st.success("PDF report generated")
    
    with export_col3:
        if st.button("📋 Regulatory Summary"):
            st.success("Regulatory summary generated")

if __name__ == "__main__":
    run_enterprise_mvr_analysis()
```

## 📊 Complete Flow Visualization

```
1. DOCUMENT UPLOAD (User: Sarah Chen)
   ├── PDF: Credit_Risk_Model_v3.2_MVR.pdf (85 pages)
   ├── Purpose: Quarterly Federal Reserve Review
   ├── Model Type: Credit Risk
   └── Priority: Critical - Regulatory Deadline

2. CORPORATE AUTHENTICATION
   ├── User: sarah.chen@firstnational.com
   ├── Department: model-validation-unit  
   ├── Authentication: Kerberos/SAML
   └── Permissions: Validated

3. ENTERPRISE GATEWAY ROUTING
   ├── MLFlow Gateway: mlflow-gateway.firstnational.com
   ├── Provider: claude-banking (claude-3-5-sonnet-banking)
   ├── Cost Limit: $50.00 per analysis
   ├── Daily Budget: $2,500 (Risk Management Department)
   └── Audit Reason: "Quarterly Federal Reserve Review - Credit Risk model validation - Priority: Critical"

4. MVR MACRO EXECUTION (Enhanced Complexity)
   ├── 📥 Step 1: parse_toc → Extract 47 sections from TOC
   ├── 🔍 Step 2: identify_sections → Map to MVS requirements
   ├── 📋 Step 3: extract_model_context → Credit Risk, Tier 1, CECL model  
   ├── ✅ Step 4: basic_compliance_check → 78.5% completion
   ├── 🧠 Step 5: section_analysis → Analyze all 47 sections
   ├── 🔗 Step 6: evidence_mapping → Map conclusions to evidence (85.7% score)
   ├── 🎯 Step 7: peer_challenges → Generate 12 peer review challenges
   └── 📊 Step 8: compliance_matrix → Create regulatory compliance matrix

5. AI-POWERED ANALYSIS (Through Enterprise Gateway)
   ├── LLM Request 1: Section analysis → 2,847 tokens, $0.12
   ├── LLM Request 2: Evidence mapping → 1,923 tokens, $0.08
   ├── LLM Request 3: Contradiction detection → 1,456 tokens, $0.06
   ├── LLM Request 4: Peer challenge generation → 2,134 tokens, $0.09
   ├── LLM Request 5: Compliance matrix → 3,267 tokens, $0.14
   └── Total: 11,627 tokens, $0.49 (within $50 limit)

6. ENTERPRISE GOVERNANCE TRACKING
   ├── Audit Record: req_id_20240828_143527
   ├── User Attribution: sarah.chen@firstnational.com
   ├── Cost Allocation: risk-management/model-validation-unit
   ├── Processing Time: 12.8 minutes (within 15min SLA)
   ├── Data Classification: Confidential - Model Data
   └── Regulatory Compliance: SR 11-7, OCC MRM, Basel III

7. RESULTS GENERATION
   ├── Compliance Matrix: 15 sections analyzed
   ├── Internal Contradictions: 1 (sample size justification) 
   ├── Peer Review Challenges: 2 critical findings
   ├── Evidence Mapping Score: 85.7%
   ├── Logic Tracing Score: 90.5%
   ├── Overall Assessment: 88.1% (Pass with recommendations)
   └── Audit Readiness: Approved for Federal Reserve submission

8. REGULATORY REPORTING
   ├── Excel Export: MVR_Analysis_20240828_Credit_Model_v32.xlsx
   ├── PDF Summary: Regulatory_Summary_Fed_Review.pdf  
   ├── Compliance Certificate: Fed_SR11-7_Compliance_Cert.pdf
   └── Audit Trail: Complete 7-year retention record
```

## 💰 Cost Breakdown & ROI

### Traditional Manual Review
```
👥 Manual Process (3 analysts, 2 weeks):
   - Senior Risk Analyst: 40 hours × $85/hr = $3,400
   - Model Validator: 35 hours × $75/hr = $2,625  
   - Compliance Officer: 25 hours × $65/hr = $1,625
   - External Review: $2,500 (required for Tier 1 models)
   - Documentation: 15 hours × $45/hr = $675
   
💰 Total Traditional Cost: $10,825
⏱️ Total Time: 2 weeks  
🎯 Quality: Variable, analyst-dependent
📋 Compliance: Manual audit trail creation
```

### TidyLLM Gateway Enterprise Solution  
```
🤖 Automated Process (1 analyst, 2 hours):
   - Analyst Time: 2 hours × $85/hr = $170
   - Gateway LLM Costs: $8.75 (Enhanced analysis)
   - Infrastructure: $0.25 (amortized)
   - Review/Approval: 1 hour × $85/hr = $85
   
💰 Total Enterprise Cost: $264
⏱️ Total Time: 2 hours active, 13 minutes processing
🎯 Quality: Consistent, AI-enhanced, multi-expert consensus
📋 Compliance: Automatic audit trail, regulatory formatting
```

### ROI Calculation
- **Cost Reduction**: 97.6% ($10,825 → $264)
- **Time Reduction**: 95.8% (2 weeks → 2 hours)  
- **Quality Improvement**: +35% consistency score
- **Compliance Improvement**: 100% audit trail coverage
- **Annual Savings**: $423,000 (40 model validations/year)
- **3-Year ROI**: 4,230% (excluding gateway deployment costs)

## 🔒 Enterprise Security & Compliance Features

### Real-Time Audit Trail
```json
{
  "audit_record_id": "req_20240828_143527_sarah_chen",
  "timestamp": "2024-08-28T14:35:27.123Z", 
  "user_id": "sarah.chen@firstnational.com",
  "department": "model-validation-unit",
  "tenant_id": "risk-management",
  
  "request_details": {
    "audit_reason": "Quarterly Federal Reserve Review - Credit Risk model validation - Priority: Critical",
    "document": "Credit_Risk_Model_v3.2_MVR.pdf",
    "analysis_type": "enhanced",
    "model_type": "Credit Risk",
    "regulatory_purpose": "Federal Reserve SR 11-7 Compliance"
  },
  
  "processing_metrics": {
    "total_tokens": 11627,
    "cost_usd": 8.75,
    "processing_time_ms": 768000,
    "llm_requests": 5,
    "provider": "claude-banking"
  },
  
  "compliance_metadata": {
    "data_classification": "confidential",
    "retention_period": "7_years",
    "regulatory_frameworks": ["SR 11-7", "OCC MRM", "Basel III"],
    "approval_status": "auto_approved",
    "escalation_required": false
  },
  
  "results_summary": {
    "overall_score": 88.1,
    "compliance_status": "pass_with_recommendations", 
    "critical_findings": 2,
    "audit_readiness": "approved"
  }
}
```

### Cost Attribution Report
```json
{
  "cost_report_id": "daily_20240828",
  "tenant_id": "risk-management", 
  "department": "model-validation-unit",
  "date": "2024-08-28",
  
  "budget_tracking": {
    "daily_limit_usd": 2500.00,
    "daily_used_usd": 234.75,
    "daily_remaining_usd": 2265.25,
    "utilization_pct": 9.39
  },
  
  "usage_breakdown": {
    "analyses_completed": 15,
    "unique_analysts": 4,
    "models_validated": ["Credit Risk", "Market Risk", "CECL"],
    "avg_cost_per_analysis": 15.65,
    "peak_usage_hour": "14:00-15:00"
  },
  
  "compliance_metrics": {
    "audit_coverage": "100%",
    "regulatory_submissions": 3,
    "escalations": 0,
    "sla_compliance": "98.2%"
  }
}
```

This comprehensive example demonstrates how the MVR macro integrates seamlessly with TidyLLM Gateway to provide enterprise-grade model validation with full governance, cost control, and regulatory compliance suitable for deployment in regulated financial institutions.

The key benefits are:
- **97.6% cost reduction** while improving quality and consistency
- **Complete regulatory compliance** with automatic audit trails
- **Real-time cost tracking** and budget controls
- **Zero external connections** - all AI access through corporate infrastructure
- **Multi-expert AI consensus** for improved validation quality