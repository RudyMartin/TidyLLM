"""
Developer Flow Agreements - AI Experimentation Workflow Contracts
================================================================

Pre-configured agreements for developers who need:
- Quick DSPy experimentation
- Model testing workflows  
- Development/testing access
- Flexible configuration options
"""

from typing import Dict, Any
from datetime import datetime, timedelta

from .base import BaseFlowAgreement, FlowAgreementConfig


class DeveloperAgreement(BaseFlowAgreement):
    """
    Developer Flow Agreement for AI experimentation and testing.
    
    Provides pre-configured access to DSPy and testing workflows with:
    - Development-friendly settings
    - Flexible model access
    - Testing and experimentation focus
    """
    
    @classmethod
    def ai_experimentation(cls, developer: str = "Developer"):
        """Standard AI experimentation agreement for developers."""
        config = FlowAgreementConfig(
            agreement_id=f"dev_ai_experiment_{developer.lower().replace(' ', '_')}",
            agreement_type="Developer AI Experimentation",
            created_by=f"Developer - {developer}",
            valid_until=datetime.now() + timedelta(days=90),  # 90-day dev cycles
            max_files_per_day=100,
            max_cost_per_month=50.0,  # Lower cost limits for experimentation
            approved_gateways=["dspy", "direct"],
            audit_requirements=[
                "log_experiment_results",
                "track_model_performance",
                "retain_dev_artifacts_30_days"
            ],
            auto_optimizations=[
                "fast_iteration_mode",
                "experiment_tracking",
                "model_comparison_enabled"
            ]
        )
        return cls(config)
    
    @classmethod
    def model_testing(cls, developer: str = "Developer"):
        """Model testing and validation agreement."""
        config = FlowAgreementConfig(
            agreement_id=f"dev_model_test_{developer.lower().replace(' ', '_')}",
            agreement_type="Developer Model Testing",
            created_by=f"Developer - {developer}",
            max_files_per_day=50,
            max_cost_per_month=25.0,
            approved_gateways=["dspy"],
            audit_requirements=[
                "log_test_results",
                "model_performance_metrics",
                "test_case_tracking"
            ],
            auto_optimizations=[
                "automated_testing",
                "performance_benchmarking",
                "result_comparison"
            ]
        )
        return cls(config)
    
    @classmethod
    def quick_prototype(cls, developer: str = "Developer"):
        """Quick prototyping agreement for rapid development."""
        config = FlowAgreementConfig(
            agreement_id=f"dev_prototype_{developer.lower().replace(' ', '_')}",
            agreement_type="Developer Quick Prototype",
            created_by=f"Developer - {developer}",
            max_files_per_day=200,
            max_cost_per_month=10.0,  # Very low cost for prototyping
            approved_gateways=["dspy"],
            audit_requirements=[
                "basic_logging"
            ],
            auto_optimizations=[
                "rapid_iteration",
                "minimal_overhead"
            ]
        )
        return cls(config)
    
    def validate(self) -> bool:
        """Validate developer agreement requirements."""
        if not self.is_valid():
            return False
        
        # Developer agreements require DSPy or direct access
        if not any(gw in self.config.approved_gateways for gw in ["dspy", "direct"]):
            return False
        
        return True
    
    def get_gateway_config(self) -> Dict[str, Any]:
        """Get DSPy gateway configuration for development use."""
        return {
            'gateway_type': 'dspy',
            'development_mode': True,
            'max_cost_per_request_usd': 0.10,  # Low cost per request for dev
            'budget_limit_daily_usd': self.config.max_cost_per_month / 30 if self.config.max_cost_per_month else None,
            'temperature_limits': (0.0, 1.0),  # Full temperature range for experimentation
            'audit_trail': False,  # Simplified logging for dev
            'experiment_tracking': True,
            'rapid_iteration_mode': True
        }
    
    def get_drop_zone_config(self) -> Dict[str, Any]:
        """Get drop zone configuration for development processing."""
        return {
            'name': f'dev_zone_{self.config.agreement_id}',
            'agent': 'dspy',
            'zone_dirs': [f'./dev_drop_zones/{self.config.agreement_type.lower().replace(" ", "_")}'],
            'file_patterns': ['*.txt', '*.md', '*.json', '*.csv', '*.py'],
            'events': ['created'],
            'model': 'claude-3-haiku',  # Faster, cheaper model for dev
            'workflow_prompt': self._get_developer_workflow_prompt(),
            'development_mode': True,
            'max_file_size': 10 * 1024 * 1024,  # 10MB limit
            'processing_timeout': 120,  # 2 minutes
            'create_zone_dir_if_not_exists': True
        }
    
    def _get_developer_workflow_prompt(self) -> str:
        """Get the workflow prompt for development processing."""
        return f"""
        Developer Workflow - {self.config.agreement_type}
        ===============================================
        
        Developer: {self.config.created_by}
        Mode: Experimentation and Testing
        
        Processing Guidelines:
        1. Focus on rapid iteration and feedback
        2. Generate detailed experiment logs
        3. Compare model performance metrics
        4. Provide actionable development insights
        5. Flag potential issues early
        
        Development Focus: Fast feedback, detailed metrics, experimentation support
        
        Process this content for development insights and experimentation.
        """
    
    def get_welcome_message(self) -> str:
        """Developer welcome message."""
        return f"""
        🚀 Welcome to TidyLLM Developer Service
        
        Your {self.config.agreement_type} agreement is now active.
        
        [OK] DSPy gateway configured for experimentation
        [OK] Development mode enabled  
        [OK] Cost limits: ${self.config.max_cost_per_month}/month
        [OK] Processing limit: {self.config.max_files_per_day} files/day
        [OK] Rapid iteration mode enabled
        
        Drop your files in the development drop zone to begin experimentation.
        Perfect for AI model testing, prototyping, and development workflows.
        """
    
    def get_quick_start_guide(self) -> list:
        """Developer quick start guide."""
        return [
            f"📁 Drop zone ready at: ./dev_drop_zones/{self.config.agreement_type.lower().replace(' ', '_')}",
            "📄 Supported files: TXT, MD, JSON, CSV, PY",
            "🚀 Development mode: Fast iteration enabled",
            "📊 Experiment tracking: Performance metrics logged",
            f"💰 Budget: ${self.config.max_cost_per_month}/month for experimentation",
            f"⚡ Processing: {self.config.max_files_per_day} files/day",
            "🔧 Perfect for: Model testing, prototyping, AI experimentation"
        ]