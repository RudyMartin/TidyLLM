"""
Corporate Flow Agreements - Enterprise Workflow Contracts
=========================================================

Pre-configured agreements for corporate users who need:
- Controlled LLM access via MLFlow Gateway
- Audit trails and compliance
- Cost controls and budgets
- IT-approved workflows
"""

from typing import Dict, Any
from datetime import datetime, timedelta

from .base import BaseFlowAgreement, FlowAgreementConfig


class CorporateAgreement(BaseFlowAgreement):
    """
    Corporate Flow Agreement for enterprise users.
    
    Provides pre-configured access to LLM Gateway with:
    - Corporate IT controls
    - Full audit trails
    - Cost management
    - Compliance requirements
    """
    
    @classmethod
    def document_processing(cls, company: str = "Enterprise Corp"):
        """Standard corporate document processing agreement."""
        config = FlowAgreementConfig(
            agreement_id=f"corporate_docs_{company.lower().replace(' ', '_')}",
            agreement_type="Corporate Document Processing",
            created_by=f"IT Admin - {company}",
            valid_until=datetime.now() + timedelta(days=365),
            max_files_per_day=1000,
            max_cost_per_month=500.0,
            approved_gateways=["llm"],
            audit_requirements=[
                "log_all_requests",
                "retain_audit_trail_90_days",
                "compliance_scan_enabled",
                "cost_tracking_enabled"
            ],
            auto_optimizations=[
                "batch_similar_documents",
                "cache_repeated_queries",
                "optimize_token_usage"
            ]
        )
        return cls(config)
    
    @classmethod
    def data_analysis(cls, company: str = "Enterprise Corp"):
        """Corporate data analysis with privacy controls."""
        config = FlowAgreementConfig(
            agreement_id=f"corporate_data_{company.lower().replace(' ', '_')}",
            agreement_type="Corporate Data Analysis",
            created_by=f"Data Team - {company}",
            max_files_per_day=500,
            max_cost_per_month=1000.0,
            approved_gateways=["llm"],
            audit_requirements=[
                "log_all_requests",
                "pii_detection_enabled",
                "data_classification_required",
                "retain_audit_trail_30_days"
            ],
            auto_optimizations=[
                "privacy_preserving_processing",
                "data_minimization",
                "secure_caching"
            ]
        )
        return cls(config)
    
    @classmethod
    def executive_briefings(cls, company: str = "Enterprise Corp"):
        """High-priority executive document processing."""
        config = FlowAgreementConfig(
            agreement_id=f"executive_{company.lower().replace(' ', '_')}",
            agreement_type="Executive Document Processing",
            created_by=f"Executive Assistant - {company}",
            max_files_per_day=100,
            max_cost_per_month=200.0,
            approved_gateways=["llm"],
            audit_requirements=[
                "log_all_requests",
                "executive_priority_processing",
                "confidentiality_controls",
                "retain_audit_trail_365_days"
            ],
            auto_optimizations=[
                "priority_queue_processing",
                "executive_summary_generation",
                "key_insights_extraction"
            ]
        )
        return cls(config)
    
    def validate(self) -> bool:
        """Validate corporate agreement requirements."""
        if not self.is_valid():
            return False
        
        # Corporate agreements require LLM gateway
        if "llm" not in self.config.approved_gateways:
            return False
        
        # Must have audit requirements
        if not self.config.audit_requirements:
            return False
        
        return True
    
    def get_gateway_config(self) -> Dict[str, Any]:
        """Get LLM Gateway configuration for corporate use."""
        return {
            'mlflow_gateway_uri': 'http://corporate-mlflow-gateway:5000',
            'require_audit_reason': True,
            'max_cost_per_request_usd': 2.0,
            'budget_limit_daily_usd': self.config.max_cost_per_month / 30 if self.config.max_cost_per_month else None,
            'temperature_limits': (0.0, 0.7),  # Conservative for corporate use
            'audit_trail': True,
            'compliance_mode': True
        }
    
    def get_drop_zone_config(self) -> Dict[str, Any]:
        """Get drop zone configuration for corporate processing."""
        return {
            'name': f'corporate_zone_{self.config.agreement_id}',
            'agent': 'llm',
            'zone_dirs': [f'./corporate_drop_zones/{self.config.agreement_type.lower().replace(" ", "_")}'],
            'file_patterns': ['*.pdf', '*.docx', '*.xlsx', '*.pptx'],
            'events': ['created'],
            'model': 'claude-3-sonnet',
            'workflow_prompt': self._get_corporate_workflow_prompt(),
            'compliance_tasks': self.config.audit_requirements,
            'max_file_size': 50 * 1024 * 1024,  # 50MB limit
            'processing_timeout': 600,  # 10 minutes
            'create_zone_dir_if_not_exists': True
        }
    
    def _get_corporate_workflow_prompt(self) -> str:
        """Get the workflow prompt for corporate processing."""
        return f"""
        Corporate Document Processing Workflow
        =====================================
        
        Agreement: {self.config.agreement_type}
        Company: {self.config.created_by}
        
        Processing Guidelines:
        1. Maintain strict confidentiality
        2. Generate executive summaries
        3. Extract key business insights
        4. Flag any compliance issues
        5. Provide actionable recommendations
        
        Audit Requirements: {', '.join(self.config.audit_requirements)}
        
        Please process this document following corporate standards.
        """
    
    def get_welcome_message(self) -> str:
        """Corporate welcome message."""
        return f"""
        🏢 Welcome to TidyLLM Corporate Service
        
        Your {self.config.agreement_type} agreement is now active.
        
        ✅ LLM Gateway configured with corporate controls
        ✅ Audit trail enabled
        ✅ Cost limits: ${self.config.max_cost_per_month}/month
        ✅ Processing limit: {self.config.max_files_per_day} files/day
        
        Drop your documents in the corporate drop zone to begin processing.
        All activities are logged for compliance and audit purposes.
        """
    
    def get_quick_start_guide(self) -> list:
        """Corporate quick start guide."""
        return [
            f"📁 Drop zone ready at: ./corporate_drop_zones/{self.config.agreement_type.lower().replace(' ', '_')}",
            "📄 Supported files: PDF, DOCX, XLSX, PPTX",
            "🔒 All processing logged for audit compliance",
            "💰 Cost tracking enabled with monthly limits",
            f"📊 Processing capacity: {self.config.max_files_per_day} files/day",
            "🚨 Any compliance issues will be flagged automatically",
            "📞 Contact IT Admin for support or limit increases"
        ]