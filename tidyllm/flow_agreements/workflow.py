"""
Workflow Flow Agreements - Process Automation Workflow Contracts
===============================================================

Pre-configured agreements for workflow automation that need:
- Document processing pipelines
- Automated workflow execution
- Process orchestration
- Integration with existing systems
"""

from typing import Dict, Any
from datetime import datetime, timedelta

from .base import BaseFlowAgreement, FlowAgreementConfig


class WorkflowAgreement(BaseFlowAgreement):
    """
    Workflow Flow Agreement for process automation and orchestration.
    
    Provides pre-configured access to workflow systems with:
    - Automated pipeline execution
    - Process orchestration
    - Integration capabilities
    - Workflow monitoring
    """
    
    @classmethod
    def document_pipeline(cls, workflow_name: str = "Document Processing"):
        """Standard document processing pipeline agreement."""
        config = FlowAgreementConfig(
            agreement_id=f"workflow_docs_{workflow_name.lower().replace(' ', '_')}",
            agreement_type="Workflow Document Pipeline",
            created_by=f"Workflow System - {workflow_name}",
            max_files_per_day=2000,  # High throughput for automated processing
            max_cost_per_month=300.0,
            approved_gateways=["llm", "dspy"],
            audit_requirements=[
                "log_all_pipeline_steps",
                "track_processing_metrics",
                "retain_workflow_evidence",
                "monitor_pipeline_health"
            ],
            auto_optimizations=[
                "batch_processing_enabled",
                "pipeline_optimization",
                "automatic_retry_logic",
                "throughput_monitoring"
            ]
        )
        return cls(config)
    
    @classmethod
    def automated_analysis(cls, workflow_name: str = "Analysis Pipeline"):
        """Automated analysis workflow agreement."""
        config = FlowAgreementConfig(
            agreement_id=f"workflow_analysis_{workflow_name.lower().replace(' ', '_')}",
            agreement_type="Workflow Automated Analysis",
            created_by=f"Analysis System - {workflow_name}",
            max_files_per_day=1000,
            max_cost_per_month=200.0,
            approved_gateways=["llm", "dspy"],
            audit_requirements=[
                "log_analysis_results",
                "track_quality_metrics",
                "retain_analysis_artifacts",
                "monitor_accuracy_trends"
            ],
            auto_optimizations=[
                "intelligent_batching",
                "quality_optimization",
                "adaptive_processing",
                "result_validation"
            ]
        )
        return cls(config)
    
    @classmethod
    def integration_workflow(cls, workflow_name: str = "Integration Flow"):
        """System integration workflow agreement."""
        config = FlowAgreementConfig(
            agreement_id=f"workflow_integration_{workflow_name.lower().replace(' ', '_')}",
            agreement_type="Workflow System Integration",
            created_by=f"Integration System - {workflow_name}",
            max_files_per_day=5000,  # Very high throughput for integrations
            max_cost_per_month=500.0,
            approved_gateways=["llm", "dspy"],
            audit_requirements=[
                "log_integration_events",
                "track_data_flow",
                "monitor_system_health",
                "retain_integration_logs"
            ],
            auto_optimizations=[
                "high_throughput_mode",
                "integration_optimization",
                "error_recovery",
                "load_balancing"
            ]
        )
        return cls(config)
    
    def validate(self) -> bool:
        """Validate workflow agreement requirements."""
        if not self.is_valid():
            return False
        
        # Workflow agreements require LLM or DSPy gateway
        if not any(gw in self.config.approved_gateways for gw in ["llm", "dspy"]):
            return False
        
        # Must have processing optimization
        if not self.config.auto_optimizations:
            return False
        
        return True
    
    def get_gateway_config(self) -> Dict[str, Any]:
        """Get gateway configuration for workflow processing."""
        return {
            'gateway_type': 'workflow_optimized',
            'max_cost_per_request_usd': 2.0,
            'budget_limit_daily_usd': self.config.max_cost_per_month / 30 if self.config.max_cost_per_month else None,
            'temperature': 0.3,  # Consistent results for workflows
            'audit_trail': True,
            'workflow_mode': True,
            'batch_processing': True,
            'retry_logic': True,
            'monitoring_enabled': True
        }
    
    def get_drop_zone_config(self) -> Dict[str, Any]:
        """Get drop zone configuration for workflow processing."""
        return {
            'name': f'workflow_zone_{self.config.agreement_id}',
            'agent': 'workflow_processor',
            'zone_dirs': [f'./workflow_zones/{self.config.agreement_type.lower().replace(" ", "_")}'],
            'file_patterns': ['*.pdf', '*.docx', '*.txt', '*.md', '*.json', '*.csv'],
            'events': ['created', 'modified'],
            'model': 'claude-3-sonnet',
            'workflow_prompt': self._get_workflow_processing_prompt(),
            'batch_processing': True,
            'max_file_size': 100 * 1024 * 1024,  # 100MB limit for workflows
            'processing_timeout': 1800,  # 30 minutes for complex workflows
            'retry_attempts': 3,
            'create_zone_dir_if_not_exists': True
        }
    
    def _get_workflow_processing_prompt(self) -> str:
        """Get the workflow prompt for automated processing."""
        return f"""
        Workflow Processing System - {self.config.agreement_type}
        ========================================================
        
        Workflow: {self.config.created_by}
        Mode: Automated Processing Pipeline
        
        Processing Guidelines:
        1. Execute workflow steps in sequence
        2. Validate data quality at each step
        3. Generate detailed processing logs
        4. Handle errors gracefully with retries
        5. Optimize for throughput and accuracy
        6. Maintain audit trail for compliance
        
        Optimization Focus: High throughput, quality assurance, error recovery
        
        Process this content through the automated workflow pipeline.
        """
    
    def get_welcome_message(self) -> str:
        """Workflow welcome message."""
        return f"""
        ⚙️ Welcome to TidyLLM Workflow Service
        
        Your {self.config.agreement_type} is now active.
        
        [OK] Workflow gateway configured
        [OK] Batch processing enabled
        [OK] Cost limits: ${self.config.max_cost_per_month}/month
        [OK] Processing capacity: {self.config.max_files_per_day} files/day
        [OK] Automated retry logic enabled
        [OK] Pipeline monitoring active
        
        Workflow zones are ready for automated processing.
        Perfect for high-throughput document pipelines and process automation.
        """
    
    def get_quick_start_guide(self) -> list:
        """Workflow quick start guide."""
        return [
            f"📁 Workflow zone: ./workflow_zones/{self.config.agreement_type.lower().replace(' ', '_')}",
            "📄 Supported files: PDF, DOCX, TXT, MD, JSON, CSV",
            "⚙️ Batch processing: Automatic batching enabled",
            "🔄 Retry logic: 3 automatic retries on failures",
            f"💰 Budget: ${self.config.max_cost_per_month}/month for workflows",
            f"⚡ Throughput: {self.config.max_files_per_day} files/day",
            "📊 Monitoring: Pipeline health tracking enabled",
            "🔧 Perfect for: Document pipelines, process automation, system integration"
        ]