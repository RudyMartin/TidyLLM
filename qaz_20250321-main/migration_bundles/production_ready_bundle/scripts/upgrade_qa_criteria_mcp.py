#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA Criteria Upgrade with MCP Framework

This script demonstrates the complete QA criteria upgrade workflow using the MCP framework.
It integrates real document processing, field extraction, and QA report generation.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('qa_upgrade_mcp.log')
    ]
)

logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import MCP components
from backend.mcp.orchestrators.qa_orchestrator import QAOrchestrator
from backend.mcp.coordinators.dspy_coordinator import DSPyCoordinator

# Import utility components
from upgrade_latex import LaTeXProcessor, create_enhanced_latex_report


class QACriteriaUpgraderMCP:
    """QA Criteria Upgrader using MCP Framework"""
    
    def __init__(self, config_path: str = "dev_configs/qa_criteria_full.yaml"):
        self.config_path = Path(config_path)
        self.output_dir = Path("data/output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MCP components
        self.qa_orchestrator = None
        self.dspy_coordinator = None
        self.latex_processor = None
        
        logger.info("🔧 Initializing MCP-based QA Criteria Upgrader")

    def initialize_components(self):
        """Initialize MCP components"""
        try:
            logger.info("🔧 Initializing QA Orchestrator...")
            self.qa_orchestrator = QAOrchestrator(str(self.config_path))
            logger.info("✅ QA Orchestrator initialized")
            
            logger.info("🔧 Initializing DSPy Coordinator...")
            self.dspy_coordinator = DSPyCoordinator()
            logger.info("✅ DSPy Coordinator initialized")
            
            logger.info("🔧 Initializing LaTeX Processor...")
            self.latex_processor = LaTeXProcessor(str(self.output_dir))
            logger.info("✅ LaTeX Processor initialized")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error initializing MCP components: {e}")
            return False

    def run_mcp_workflow(self) -> Dict[str, Any]:
        """Run the complete MCP-based QA workflow"""
        
        try:
            logger.info("🚀 Starting MCP-based QA Workflow")
            logger.info("=" * 60)
            
            # Step 1: Process documents using QA Orchestrator
            logger.info("📋 Step 1: Processing documents with QA Orchestrator")
            workflow_result = self.qa_orchestrator.process_qa_documents()
            
            if workflow_result["status"] != "completed":
                logger.error(f"❌ MCP workflow failed: {workflow_result.get('error', 'Unknown error')}")
                return workflow_result
            
            # Step 2: Generate enhanced reports
            logger.info("📄 Step 2: Generating enhanced reports")
            enhanced_reports = self._generate_enhanced_reports(workflow_result)
            
            # Step 3: Compile LaTeX to PDF
            logger.info("📄 Step 3: Compiling LaTeX to PDF")
            pdf_results = self._compile_latex_reports(enhanced_reports)
            
            # Create comprehensive result
            result = {
                "workflow_id": workflow_result["workflow_id"],
                "status": "completed",
                "mcp_workflow": workflow_result,
                "enhanced_reports": enhanced_reports,
                "pdf_results": pdf_results,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("✅ MCP-based QA Workflow completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ MCP workflow failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _generate_enhanced_reports(self, workflow_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate enhanced reports from MCP workflow results"""
        
        try:
            enhanced_reports = {}
            
            # Get report data from workflow
            report_result = workflow_result.get("report_result", {})
            if report_result.get("status") == "success":
                qa_report = report_result.get("qa_report")
                
                if qa_report:
                    # Generate enhanced LaTeX report
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    enhanced_latex_path = self.output_dir / f"qa_enhanced_report_{timestamp}.tex"
                    
                    # Convert QAReport to dict if needed
                    if hasattr(qa_report, '__dict__'):
                        qa_report_dict = qa_report.__dict__
                    else:
                        qa_report_dict = qa_report
                    
                    # Create enhanced LaTeX report
                    enhanced_content = create_enhanced_latex_report(qa_report_dict)
                    
                    with open(enhanced_latex_path, 'w', encoding='utf-8') as f:
                        f.write(enhanced_content)
                    
                    enhanced_reports["enhanced_latex"] = str(enhanced_latex_path)
                    logger.info(f"✅ Enhanced LaTeX report: {enhanced_latex_path}")
            
            return enhanced_reports
            
        except Exception as e:
            logger.error(f"❌ Enhanced report generation failed: {e}")
            return {}

    def _compile_latex_reports(self, enhanced_reports: Dict[str, str]) -> Dict[str, str]:
        """Compile LaTeX reports to PDF"""
        
        try:
            pdf_results = {}
            
            for report_type, latex_path in enhanced_reports.items():
                if latex_path.endswith('.tex'):
                    pdf_path = self.latex_processor.compile_latex_to_pdf(latex_path)
                    if pdf_path:
                        pdf_results[report_type] = pdf_path
                        logger.info(f"✅ Compiled PDF: {pdf_path}")
            
            return pdf_results
            
        except Exception as e:
            logger.error(f"❌ LaTeX compilation failed: {e}")
            return {}

    def display_results(self, result: Dict[str, Any]):
        """Display workflow results"""
        
        print("\n🎉 MCP-based QA Criteria Upgrade Complete!")
        print("=" * 60)
        
        if result["status"] == "completed":
            workflow_result = result["mcp_workflow"]
            
            # Display workflow summary
            print(f"📊 Workflow ID: {workflow_result['workflow_id']}")
            print(f"📋 Documents Processed: {workflow_result.get('document_result', {}).get('document_count', 0)}")
            print(f"🔍 Fields Extracted: {workflow_result.get('extraction_result', {}).get('field_count', 0)}")
            
            # Display report paths
            report_result = workflow_result.get("report_result", {})
            if report_result.get("status") == "success":
                print(f"📄 JSON Report: {report_result.get('json_report_path', 'N/A')}")
                print(f"📄 LaTeX Report: {report_result.get('latex_report_path', 'N/A')}")
            
            # Display enhanced reports
            enhanced_reports = result.get("enhanced_reports", {})
            for report_type, path in enhanced_reports.items():
                print(f"📄 {report_type.title()}: {path}")
            
            # Display PDF results
            pdf_results = result.get("pdf_results", {})
            for report_type, path in pdf_results.items():
                print(f"📄 {report_type.title()} PDF: {path}")
            
        else:
            print(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
        
        print(f"\n📁 Check the 'data/output' directory for generated files")
        print(f"📄 LaTeX files can be compiled to PDF using pdflatex")


def main():
    """Main function to run the MCP-based QA criteria upgrade"""
    
    print("🚀 QA Criteria Upgrade with MCP Framework")
    print("=" * 60)
    
    # Create MCP-based upgrader instance
    upgrader = QACriteriaUpgraderMCP()
    
    # Initialize components
    if not upgrader.initialize_components():
        print("❌ Failed to initialize MCP components")
        return
    
    # Run MCP workflow
    result = upgrader.run_mcp_workflow()
    
    # Display results
    upgrader.display_results(result)


if __name__ == "__main__":
    main()
