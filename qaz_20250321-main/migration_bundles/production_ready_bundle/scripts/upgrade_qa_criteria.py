#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA Criteria Upgrade and Report Generation Script

This script provides a comprehensive upgrade path for QA criteria processing,
integrating YAML configuration, document analysis, and enhanced LaTeX report generation.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import the QA report generator
from src.backend.core.qa_report_generator import QAReportGenerator

# Import the document processor
from src.backend.core.document_processor import DocumentProcessor

# Import the LaTeX utility
from upgrade_latex import LaTeXProcessor, create_enhanced_latex_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QACriteriaUpgrader:
    """Main class for upgrading QA criteria processing and report generation"""
    
    def __init__(self, config_path: str = "dev_configs/qa_criteria_full.yaml"):
        self.config_path = Path(config_path)
        self.qa_generator = None
        self.latex_processor = None
        self.document_processor = None
        self.output_dir = Path("data/output")
        self.output_dir.mkdir(exist_ok=True)
        
    def initialize_components(self):
        """Initialize QA generator, document processor, and LaTeX processor"""
        try:
            logger.info("🔧 Initializing QA Report Generator...")
            self.qa_generator = QAReportGenerator(str(self.config_path))
            logger.info("✅ QA Report Generator initialized")
            
            logger.info("🔧 Initializing Document Processor...")
            self.document_processor = DocumentProcessor()
            logger.info("✅ Document Processor initialized")
            
            logger.info("🔧 Initializing LaTeX Processor...")
            self.latex_processor = LaTeXProcessor(str(self.output_dir))
            logger.info("✅ LaTeX Processor initialized")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            return False
    
    def process_documents(self, documents: List[Dict[str, Any]], 
                         extracted_fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process documents and generate QA report"""
        
        if not self.qa_generator:
            logger.error("❌ QA Generator not initialized")
            return None
        
        try:
            logger.info("📊 Generating QA HealthCheck Report...")
            report = self.qa_generator.generate_report(documents, extracted_fields)
            logger.info("✅ QA Report generated successfully")
            return report
        except Exception as e:
            logger.error(f"❌ Error generating QA report: {e}")
            return None
    
    def load_real_documents(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Load and process real documents from input directory"""
        
        if not self.document_processor:
            logger.error("❌ Document Processor not initialized")
            return [], {}
        
        try:
            logger.info("📋 Loading real documents from input directory...")
            
            # Process all documents in input directory
            processed_documents = self.document_processor.process_all_documents()
            
            if not processed_documents:
                logger.warning("⚠️ No documents found in input directory")
                return [], {}
            
            # Extract metadata fields from documents
            extracted_fields = self.document_processor.extract_metadata_fields(processed_documents)
            
            logger.info(f"✅ Loaded {len(processed_documents)} documents")
            logger.info(f"✅ Extracted {len(extracted_fields)} metadata fields")
            
            return processed_documents, extracted_fields
            
        except Exception as e:
            logger.error(f"❌ Error loading real documents: {e}")
            return [], {}
    
    def generate_enhanced_reports(self, qa_report_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate enhanced reports in multiple formats"""
        
        results = {}
        
        try:
            # Generate JSON report
            logger.info("📄 Generating JSON report...")
            json_path = self.qa_generator.generate_json_report()
            results['json'] = json_path
            logger.info(f"✅ JSON report: {json_path}")
            
            # Generate basic LaTeX report
            logger.info("📄 Generating basic LaTeX report...")
            latex_path = self.qa_generator.generate_latex_report()
            results['latex'] = latex_path
            logger.info(f"✅ LaTeX report: {latex_path}")
            
            # Generate enhanced LaTeX report
            logger.info("📄 Generating enhanced LaTeX report...")
            enhanced_latex_path = create_enhanced_latex_report(qa_report_data)
            results['enhanced_latex'] = enhanced_latex_path
            logger.info(f"✅ Enhanced LaTeX report: {enhanced_latex_path}")
            
            # Compile LaTeX to PDF (if LaTeX is available)
            if self.latex_processor and self.latex_processor.check_latex_installation():
                logger.info("📄 Compiling LaTeX to PDF...")
                
                # Compile basic LaTeX
                basic_pdf = self.latex_processor.compile_latex_to_pdf(
                    latex_path, 
                    f"qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                )
                if basic_pdf:
                    results['pdf'] = basic_pdf
                    logger.info(f"✅ PDF report: {basic_pdf}")
                
                # Compile enhanced LaTeX
                enhanced_pdf = self.latex_processor.compile_latex_to_pdf(
                    enhanced_latex_path,
                    f"enhanced_qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                )
                if enhanced_pdf:
                    results['enhanced_pdf'] = enhanced_pdf
                    logger.info(f"✅ Enhanced PDF report: {enhanced_pdf}")
            else:
                logger.warning("⚠️ LaTeX not available - skipping PDF compilation")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error generating reports: {e}")
            return results
    
    def validate_configuration(self) -> bool:
        """Validate QA criteria configuration"""
        
        try:
            logger.info("🔍 Validating QA criteria configuration...")
            
            if not self.config_path.exists():
                logger.error(f"❌ Configuration file not found: {self.config_path}")
                return False
            
            # Load and validate YAML
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['checklist_categories', 'scoring_rules', 'workflow_steps']
            for section in required_sections:
                if section not in config:
                    logger.error(f"❌ Missing required section: {section}")
                    return False
            
            # Check categories
            categories = config.get('checklist_categories', [])
            if not categories:
                logger.error("❌ No checklist categories found")
                return False
            
            # Check criteria in each category
            total_criteria = 0
            for category in categories:
                criteria = category.get('criteria', [])
                if not criteria:
                    logger.warning(f"⚠️ No criteria found in category: {category.get('name', 'Unknown')}")
                total_criteria += len(criteria)
            
            logger.info(f"✅ Configuration validated: {len(categories)} categories, {total_criteria} criteria")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating configuration: {e}")
            return False
    
    def create_sample_documents(self) -> List[Dict[str, Any]]:
        """Create sample documents for testing"""
        
        sample_documents = [
            {
                'filename': 'sample_validation_report.pdf',
                'content': '''
                Review ID: REV00001
                Model Type: Machine Learning
                Risk Tier: High
                Model ID: ML-2024-001
                Model Name: Credit Risk Assessment Model v2.1
                Version: 2.1.0
                Authors: Dr. Sarah Johnson, Dr. Michael Chen
                Date: 08-22-2024
                Validation Type: Comprehensive Review
                
                This document presents the comprehensive validation methodology for the Credit Risk Assessment Model.
                The model has been developed using advanced machine learning techniques and validated against
                historical data spanning five years.
                
                Data Quality and Content Control:
                - Source data validation has been performed on all input variables
                - Data lineage is fully documented from source systems through model processing
                - Missing data handling procedures are implemented and tested
                - Data quality metrics are established and monitored continuously
                
                Governance and Compliance:
                - Model development documentation is complete and comprehensive
                - Business use case is clearly defined for credit risk assessment
                - Model limitations are documented with impact assessments
                - Governance approval has been obtained from the Risk Committee
                
                Validation Processes:
                - Validation methodology is appropriate for machine learning models
                - Validation scope covers all critical model components
                - Validation testing includes backtesting and stress testing
                - Validation results are thoroughly documented
                
                Testing and Performance:
                - Independent review has been conducted by qualified personnel
                - Review findings have been addressed and resolved
                - Testing procedures are validated and appropriate
                - Testing results are verified and independently validated
                
                The model demonstrates strong performance across all validation criteria
                and meets regulatory requirements for credit risk assessment.
                '''
            }
        ]
        
        return sample_documents
    
    def create_sample_extracted_fields(self) -> Dict[str, Any]:
        """Create sample extracted fields for testing"""
        
        return {
            'review_id': 'REV00001',
            'model_type': 'Machine Learning',
            'risk_tier': 'High',
            'model_id': 'ML-2024-001',
            'model_name': 'Credit Risk Assessment Model v2.1',
            'version': '2.1.0',
            'authors': ['Dr. Sarah Johnson', 'Dr. Michael Chen'],
            'date': '08-22-2024',
            'validation_type': 'Comprehensive Review',
            'reviewer_name': 'Alex',
            'team_num': 'QA Team 1',
            'process_name': 'QA Validation Review'
        }
    
    def run_complete_workflow(self, use_sample_data: bool = True) -> Dict[str, Any]:
        """Run the complete QA criteria upgrade workflow"""
        
        logger.info("🚀 Starting QA Criteria Upgrade Workflow")
        logger.info("=" * 60)
        
        # Step 1: Initialize components
        if not self.initialize_components():
            return {'status': 'failed', 'error': 'Component initialization failed'}
        
        # Step 2: Validate configuration
        if not self.validate_configuration():
            return {'status': 'failed', 'error': 'Configuration validation failed'}
        
        # Step 3: Prepare data
        if use_sample_data:
            logger.info("📋 Using sample data for demonstration")
            documents = self.create_sample_documents()
            extracted_fields = self.create_sample_extracted_fields()
        else:
            logger.info("📋 Loading real documents from input folder")
            documents, extracted_fields = self.load_real_documents()
            
            if not documents:
                logger.warning("⚠️ No real documents found, falling back to sample data")
                documents = self.create_sample_documents()
                extracted_fields = self.create_sample_extracted_fields()
        
        # Step 4: Process documents
        qa_report = self.process_documents(documents, extracted_fields)
        if not qa_report:
            return {'status': 'failed', 'error': 'Document processing failed'}
        
        # Step 5: Generate reports
        report_paths = self.generate_enhanced_reports(qa_report)
        
        # Step 6: Create summary
        summary = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'qa_report': qa_report,
            'report_paths': report_paths,
            'summary': {
                'overall_score': qa_report.overall_score,
                'overall_status': qa_report.overall_status,
                'total_criteria': qa_report.total_criteria,
                'passed_criteria': qa_report.passed_criteria,
                'failed_criteria': qa_report.failed_criteria,
                'conditional_criteria': qa_report.conditional_criteria
            }
        }
        
        logger.info("🎉 QA Criteria Upgrade Workflow Complete!")
        return summary
    
    def display_results(self, summary: Dict[str, Any]):
        """Display workflow results"""
        
        if summary['status'] == 'success':
            logger.info("\n📊 Workflow Results Summary:")
            logger.info("=" * 40)
            
            # Display QA report summary
            qa_summary = summary['summary']
            logger.info(f"Overall Score: {qa_summary['overall_score']:.1f}%")
            logger.info(f"Overall Status: {qa_summary['overall_status'].title()}")
            logger.info(f"Total Criteria: {qa_summary['total_criteria']}")
            logger.info(f"Passed: {qa_summary['passed_criteria']}")
            logger.info(f"Failed: {qa_summary['failed_criteria']}")
            logger.info(f"Conditional: {qa_summary['conditional_criteria']}")
            
            # Display generated files
            logger.info("\n📄 Generated Files:")
            logger.info("=" * 20)
            for format_type, path in summary['report_paths'].items():
                if path:
                    logger.info(f"{format_type.upper()}: {path}")
            
            # Display recommendations
            qa_report = summary['qa_report']
            if qa_report.recommendations:
                logger.info("\n💡 Top Recommendations:")
                logger.info("=" * 25)
                for i, rec in enumerate(qa_report.recommendations[:5], 1):
                    logger.info(f"{i}. {rec}")
            
            if qa_report.next_steps:
                logger.info("\n🎯 Next Steps:")
                logger.info("=" * 15)
                for i, step in enumerate(qa_report.next_steps, 1):
                    logger.info(f"{i}. {step}")
        
        else:
            logger.error(f"❌ Workflow failed: {summary.get('error', 'Unknown error')}")


def main():
    """Main function to run the QA criteria upgrade"""
    
    print("🚀 QA Criteria Upgrade and Report Generation")
    print("=" * 60)
    
    # Create upgrader instance
    upgrader = QACriteriaUpgrader()
    
    # Run complete workflow with real documents
    summary = upgrader.run_complete_workflow(use_sample_data=False)
    
    # Display results
    upgrader.display_results(summary)
    
    print(f"\n📁 Check the 'data/output' directory for generated files")
    print(f"📄 LaTeX files can be compiled to PDF using pdflatex")


if __name__ == "__main__":
    main()
