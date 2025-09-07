"""
MVR Analysis Flow Agreement - Automated Model Validation Report Analysis
"""

from .base import BaseFlowAgreement, FlowAgreementConfig
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from pathlib import Path


@dataclass
class MVRAnalysisConfig(FlowAgreementConfig):
    """Configuration specific to MVR Analysis."""
    report_types: List[str] = None
    prompt_templates: Dict[str, str] = None
    output_directory: str = None
    complexity_level: str = "enhanced"  # basic, enhanced, or advanced
    
    def __post_init__(self):
        super().__post_init__()
        if self.report_types is None:
            self.report_types = ["compliance", "intelligence", "knowledge"]
        if self.prompt_templates is None:
            self.prompt_templates = {
                "compliance": "JB_Overview_Prompt.md",
                "intelligence": "comprehensive_whitepaper_analysis.md",
                "knowledge": "toc_extraction_prompt.md",
                "enhanced": "comprehensive_whitepaper_analysis_enhanced.md"
            }
        if self.output_directory is None:
            self.output_directory = "./mvr_reports"


class MVRAnalysisFlowAgreement(BaseFlowAgreement):
    """
    Flow Agreement for MVR (Model Validation Report) Analysis.
    
    This agreement enables automated analysis of Model Validation Reports
    to produce three types of PDF reports:
    1. Compliance Validation Report
    2. Document Intelligence Report  
    3. Knowledge Base Expansion Report
    """
    
    def __init__(self, config: Optional[MVRAnalysisConfig] = None):
        if config is None:
            config = MVRAnalysisConfig(
                agreement_id="mvr-analysis-001",
                agreement_type="MVR Analysis",
                created_by="system",
                approved_gateways=["ai_processing", "corporate_llm", "workflow_optimizer"]
            )
        super().__init__(config)
        self.config: MVRAnalysisConfig = config
        self.report_generators = {}
        self._initialize_report_generators()
    
    def _initialize_report_generators(self):
        """Initialize the report generator instances."""
        from ..knowledge_systems.core.domain_rag import DomainRAG
        
        # Initialize generators for each report type
        self.report_generators = {
            "compliance": ComplianceReportGenerator(self.config),
            "intelligence": IntelligenceReportGenerator(self.config),
            "knowledge": KnowledgeReportGenerator(self.config)
        }
    
    def validate(self) -> bool:
        """Validate that this agreement can be used."""
        # Check if prompt templates exist
        prompt_dir = Path("qaz_20250321-main/src/assets/prompts/favorites")
        
        for template in self.config.prompt_templates.values():
            template_path = prompt_dir / template
            if not template_path.exists():
                print(f"Warning: Template {template} not found at {template_path}")
                # Don't fail validation, just warn
        
        # Check output directory is writable
        output_path = Path(self.config.output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        return self.is_valid()
    
    def get_gateway_config(self) -> Dict[str, Any]:
        """Get the gateway configuration for MVR analysis."""
        return {
            "ai_processing": {
                "model": "gpt-4",
                "temperature": 0.1,  # Low temperature for consistency
                "max_tokens": 4000,
                "system_prompt": "You are an expert Model Validation analyst."
            },
            "corporate_llm": {
                "endpoint": "corporate-llm-endpoint",
                "api_key": os.environ.get("CORPORATE_LLM_KEY", ""),
                "timeout": 300
            },
            "workflow_optimizer": {
                "batch_size": 5,
                "parallel_processing": True,
                "cache_enabled": True
            }
        }
    
    def get_drop_zone_config(self) -> Dict[str, Any]:
        """Get the drop zone configuration for MVR documents."""
        return {
            "input_path": "./mvr_dropzone",
            "accepted_formats": [".pdf", ".docx", ".doc"],
            "auto_process": True,
            "scan_interval": 30,  # seconds
            "preprocessing": {
                "extract_toc": True,
                "extract_figures": True,
                "extract_tables": True,
                "extract_references": True
            }
        }
    
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process an MVR document through all report generators.
        
        Args:
            document_path: Path to the MVR document
            
        Returns:
            Dict containing paths to generated reports
        """
        from ..gateways import get_gateway
        
        results = {
            "document": document_path,
            "timestamp": datetime.now().isoformat(),
            "reports": {}
        }
        
        # Process through each report generator
        for report_type in self.config.report_types:
            if report_type in self.report_generators:
                generator = self.report_generators[report_type]
                
                # Select appropriate gateway
                if report_type == "compliance":
                    gateway = get_gateway("corporate_llm", **self.get_gateway_config()["corporate_llm"])
                elif report_type == "intelligence":
                    gateway = get_gateway("ai_processing", **self.get_gateway_config()["ai_processing"])
                else:  # knowledge
                    gateway = get_gateway("workflow_optimizer", **self.get_gateway_config()["workflow_optimizer"])
                
                # Generate report
                report_path = generator.generate(document_path, gateway)
                results["reports"][report_type] = report_path
        
        return results
    
    def get_welcome_message(self) -> str:
        """Get welcome message for MVR Analysis agreement."""
        return f"""
        🔬 MVR Analysis Flow Agreement Activated!
        
        Ready to analyze Model Validation Reports with:
        ✅ Compliance Validation (using {self.config.prompt_templates['compliance']})
        📊 Document Intelligence (using {self.config.prompt_templates['intelligence']})
        🔍 Knowledge Extraction (using {self.config.prompt_templates['knowledge']})
        
        Complexity Level: {self.config.complexity_level.upper()}
        Output Directory: {self.config.output_directory}
        """
    
    def get_quick_start_guide(self) -> List[str]:
        """Get quick start guide for MVR analysis."""
        return [
            "1. Place your MVR document (PDF/DOCX) in the drop zone",
            "2. The system will automatically detect and process it",
            "3. Three PDF reports will be generated:",
            "   - Compliance Validation Report",
            "   - Document Intelligence Report",
            "   - Knowledge Base Expansion Report",
            f"4. Find your reports in: {self.config.output_directory}",
            "5. Review reports and iterate as needed"
        ]


class ComplianceReportGenerator:
    """Generator for Compliance Validation Reports."""
    
    def __init__(self, config: MVRAnalysisConfig):
        self.config = config
        self.template = config.prompt_templates.get("compliance")
    
    def generate(self, document_path: str, gateway) -> str:
        """Generate compliance validation report."""
        output_path = Path(self.config.output_directory) / f"compliance_{Path(document_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Load prompt template
        prompt_path = Path("qaz_20250321-main/src/assets/prompts/favorites") / self.template
        
        # Process document through gateway
        # This would integrate with your existing gateway system
        
        print(f"Generating Compliance Report: {output_path}")
        return str(output_path)


class IntelligenceReportGenerator:
    """Generator for Document Intelligence Reports."""
    
    def __init__(self, config: MVRAnalysisConfig):
        self.config = config
        self.template = config.prompt_templates.get("intelligence")
        self.enhanced_template = config.prompt_templates.get("enhanced")
    
    def generate(self, document_path: str, gateway) -> str:
        """Generate document intelligence report."""
        output_path = Path(self.config.output_directory) / f"intelligence_{Path(document_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Use enhanced template if complexity level is advanced
        if self.config.complexity_level == "advanced":
            template = self.enhanced_template
        else:
            template = self.template
        
        print(f"Generating Intelligence Report: {output_path}")
        return str(output_path)


class KnowledgeReportGenerator:
    """Generator for Knowledge Base Expansion Reports."""
    
    def __init__(self, config: MVRAnalysisConfig):
        self.config = config
        self.template = config.prompt_templates.get("knowledge")
    
    def generate(self, document_path: str, gateway) -> str:
        """Generate knowledge base expansion report."""
        output_path = Path(self.config.output_directory) / f"knowledge_{Path(document_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        print(f"Generating Knowledge Report: {output_path}")
        return str(output_path)