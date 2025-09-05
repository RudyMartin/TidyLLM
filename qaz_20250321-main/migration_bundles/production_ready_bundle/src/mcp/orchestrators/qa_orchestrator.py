"""
QA Orchestrator

Orchestrates the QA document processing workflow using MCP framework.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from pathlib import Path

from ..protocol.message_schemas import MessageType, MessagePriority
from ..protocol.communication import MCPProtocol
from ..context.context_manager import MCPContextManager

# Import our real components
from ...core.document_processor import DocumentProcessor
from ...core.qa_report_generator import QAReportGenerator


class QAOrchestrator:
    """Orchestrates QA document processing workflow"""
    
    def __init__(self, config_path: str = "dev_configs/qa_criteria_full.yaml"):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.protocol = MCPProtocol()
        self.context_manager = MCPContextManager()
        self.workflow_id = str(uuid.uuid4())
        
        # Initialize real components
        self.document_processor = DocumentProcessor()
        self.qa_generator = QAReportGenerator(config_path)
        self.output_dir = Path("data/output")
        self.output_dir.mkdir(exist_ok=True)

    def process_qa_documents(self, 
                           files: List[Dict[str, Any]] = None,
                           team_num: str = "TEAM001",
                           process_name: str = "QA Validation Review",
                           reviewer_name: str = "System",
                           review_id: str = None,
                           model_type: str = "Research Document",
                           risk_tier: str = "Medium",
                           custom_prompt: str = None) -> Dict[str, Any]:
        """Process QA documents through the MCP workflow"""
        
        try:
            self.logger.info(f"🚀 Starting QA workflow {self.workflow_id}")
            
            # Create workflow context
            workflow_context = self._create_workflow_context(
                files, team_num, process_name, reviewer_name, 
                review_id, model_type, risk_tier, custom_prompt
            )
            
            # Step 1: Process documents (real implementation)
            self.logger.info("Step 1: Processing documents from input directory")
            document_result = self._process_documents(workflow_context)
            
            # Step 2: Extract key fields using real document processor
            self.logger.info("Step 2: Extracting key fields from documents")
            extraction_result = self._extract_key_fields(document_result, workflow_context)
            
            # Step 3: Generate QA HealthCheck Report using real QA generator
            self.logger.info("Step 3: Generating QA HealthCheck Report")
            report_result = self._generate_qa_report(document_result, extraction_result, workflow_context)
            
            return {
                "workflow_id": self.workflow_id,
                "status": "completed",
                "document_result": document_result,
                "extraction_result": extraction_result,
                "report_result": report_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"QA workflow failed: {e}")
            return {
                "workflow_id": self.workflow_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _create_workflow_context(self, 
                               files: List[Dict[str, Any]],
                               team_num: str,
                               process_name: str,
                               reviewer_name: str,
                               review_id: str,
                               model_type: str,
                               risk_tier: str,
                               custom_prompt: str = None) -> str:
        """Create workflow context"""
        
        context_data = {
            "workflow_type": "qa_document_processing",
            "files": files,
            "team_num": team_num,
            "process_name": process_name,
            "reviewer_name": reviewer_name,
            "review_id": review_id,
            "model_type": model_type,
            "risk_tier": risk_tier,
            "custom_prompt": custom_prompt,
            "s3_path": f"usecase-qa/teams/{team_num}/{process_name}/{reviewer_name}/{review_id}",
            "created_at": datetime.now().isoformat()
        }
        
        context = self.context_manager.create_context(
            context_data=context_data,
            source_layer="qa_orchestrator",
            expiry_hours=24
        )
        
        return context.context_id

    def _process_documents(self, context_id: str) -> Dict[str, Any]:
        """Process documents using real document processor"""
        
        try:
            self.logger.info("📋 Processing documents from input directory")
            
            # Process all documents in input directory
            processed_documents = self.document_processor.process_all_documents()
            
            if not processed_documents:
                self.logger.warning("⚠️ No documents found in input directory")
                return {
                    "status": "warning",
                    "message": "No documents found in input directory",
                    "processed_documents": []
                }
            
            self.logger.info(f"✅ Processed {len(processed_documents)} documents")
            
            return {
                "status": "success",
                "processed_documents": processed_documents,
                "document_count": len(processed_documents),
                "processing_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Document processing failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _extract_key_fields(self, document_result: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Extract key fields using real document processor"""
        
        try:
            if document_result["status"] != "success":
                raise ValueError("Document processing failed")
            
            processed_documents = document_result["processed_documents"]
            
            # Extract metadata fields from documents
            extracted_fields = self.document_processor.extract_metadata_fields(processed_documents)
            
            self.logger.info(f"✅ Extracted {len(extracted_fields)} metadata fields")
            
            return {
                "status": "success",
                "extracted_fields": extracted_fields,
                "field_count": len(extracted_fields),
                "extraction_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Field extraction failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _generate_qa_report(self, document_result: Dict[str, Any], 
                          extraction_result: Dict[str, Any], 
                          context_id: str) -> Dict[str, Any]:
        """Generate QA HealthCheck Report using real QA generator"""
        
        try:
            if document_result["status"] != "success":
                raise ValueError("Document processing failed")
            
            if extraction_result["status"] != "success":
                raise ValueError("Field extraction failed")
            
            processed_documents = document_result["processed_documents"]
            extracted_fields = extraction_result["extracted_fields"]
            
            # Generate QA report using real QA generator
            qa_report = self.qa_generator.generate_report(processed_documents, extracted_fields)
            
            # Generate reports in multiple formats
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # JSON report
            json_path = self.output_dir / f"qa_healthcheck_report_{timestamp}.json"
            json_result = self.qa_generator.generate_json_report(str(json_path))
            
            # LaTeX report
            latex_path = self.output_dir / f"qa_healthcheck_report_{timestamp}.tex"
            latex_result = self.qa_generator.generate_latex_report(str(latex_path))
            
            self.logger.info(f"✅ Generated QA reports: {json_path}, {latex_path}")
            
            return {
                "status": "success",
                "qa_report": qa_report,
                "json_report_path": str(json_path),
                "latex_report_path": str(latex_path),
                "report_generation_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
