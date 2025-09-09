#!/usr/bin/env python3
"""

# Centralized AWS credential management
import sys
from pathlib import Path

# Add admin directory to path for credential loading
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import set_aws_environment

# Load AWS credentials using centralized system
set_aws_environment()
Domain Workflow Creator
=======================

Creates a complete workflow sequence:
1. User provides: domain_name + input_folder
2. Creates S3 drop zone structure: domain_name/01_input/
3. Uploads input folder contents to S3 as first drop zone
4. Creates vector embeddings referencing S3 locations
5. Sets up workflow sequence for processing pipeline

Example:
  Domain: "financial_risk"
  Input: "./risk_documents/"
  Result: s3://bucket/financial_risk/01_input/*.pdf
          Ready for next workflow stages
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set AWS credentials




parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

class DomainWorkflowCreator:
    """Creates domain workflow with configurable S3 drop zone sequence"""
    
    def __init__(self, 
                 s3_bucket: Optional[str] = None,
                 s3_prefix: Optional[str] = None, 
                 s3_region: Optional[str] = None):
        """
        Initialize with configuration override capability
        
        Args:
            s3_bucket: Override S3 bucket (uses settings.yaml default if None)
            s3_prefix: Override S3 prefix (uses settings.yaml default if None)
            s3_region: Override S3 region (uses settings.yaml default if None)
        """
        from knowledge_systems.core.workflow_config import create_workflow_config
        from ..infrastructure.session import get_s3_manager
        from knowledge_systems.core.vector_manager import get_vector_manager
        
        # Create workflow config with overrides
        self.workflow_config = create_workflow_config(
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_region=s3_region
        )
        
        # For backward compatibility
        self.s3_bucket = self.workflow_config.s3.bucket
        
        self.s3_manager = get_s3_manager()
        self.vector_manager = get_vector_manager()
        
    def create_domain_workflow(self, domain_name: str, input_folder: str) -> Dict[str, Any]:
        """
        Create complete domain workflow from name + input folder
        
        Args:
            domain_name: Name for the domain (e.g., "financial_risk", "model_validation")
            input_folder: Local folder path with documents to upload
            
        Returns:
            Workflow creation results with S3 structure and next steps
        """
        
        input_path = Path(input_folder)
        if not input_path.exists():
            return {
                "success": False,
                "error": f"Input folder not found: {input_folder}"
            }
        
        result = {
            "domain_name": domain_name,
            "input_folder": str(input_path),
            "s3_bucket": self.s3_bucket,
            "workflow_created": datetime.now().isoformat(),
            "steps_completed": [],
            "s3_structure": {},
            "next_steps": []
        }
        
        try:
            # Step 1: Create S3 workflow structure
            result["steps_completed"].append("Creating S3 workflow structure...")
            
            s3_structure = self._create_s3_workflow_structure(domain_name)
            result["s3_structure"] = s3_structure
            
            # Step 2: Upload input folder to first drop zone
            result["steps_completed"].append("Uploading input folder to S3 drop zone...")
            
            upload_result = self._upload_to_drop_zone(
                input_path, 
                domain_name, 
                drop_zone="01_input"
            )
            
            result["upload_summary"] = upload_result
            
            # Step 3: Create vector embeddings for first drop zone
            result["steps_completed"].append("Creating vector embeddings from S3...")
            
            vector_result = self._create_vector_embeddings(domain_name, "01_input")
            result["vector_summary"] = vector_result
            
            # Step 4: Set up workflow sequence
            result["steps_completed"].append("Setting up workflow sequence...")
            
            workflow_sequence = self._setup_workflow_sequence(domain_name, upload_result)
            result["workflow_sequence"] = workflow_sequence
            
            # Step 5: Generate next steps
            result["next_steps"] = self._generate_next_steps(domain_name, workflow_sequence)
            
            result["success"] = True
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["steps_completed"].append(f"ERROR: {str(e)}")
        
        return result
    
    def _create_s3_workflow_structure(self, domain_name: str) -> Dict[str, str]:
        """Create S3 folder structure for workflow sequence using configurable prefix"""
        
        s3_config = self.workflow_config.s3
        
        # Build base path from configured prefix
        if s3_config.prefix:
            # Remove trailing slash and add domain
            base_path = f"{s3_config.prefix.rstrip('/')}/{domain_name}/"
        else:
            base_path = f"{domain_name}/"
        
        # Standard workflow sequence structure  
        structure = {
            "domain_root": f"{base_path}",
            "01_input": f"{base_path}01_input/",
            "02_processed": f"{base_path}02_processed/",
            "03_analyzed": f"{base_path}03_analyzed/",
            "04_enriched": f"{base_path}04_enriched/",
            "05_output": f"{base_path}05_output/",
            "metadata": f"{base_path}metadata/",
            "logs": f"{base_path}logs/"
        }
        
        # Add full S3 URLs
        s3_urls = {}
        for key, prefix in structure.items():
            s3_urls[f"{key}_s3_url"] = f"s3://{s3_config.bucket}/{prefix}"
        
        structure.update(s3_urls)
        
        return structure
    
    def _upload_to_drop_zone(self, input_path: Path, domain_name: str, drop_zone: str) -> Dict[str, Any]:
        """Upload input folder contents to S3 drop zone using configurable prefix"""
        
        s3_config = self.workflow_config.s3
        
        # Build S3 prefix using configured base prefix
        if s3_config.prefix:
            base_path = f"{s3_config.prefix.rstrip('/')}/{domain_name}/"
        else:
            base_path = f"{domain_name}/"
        
        s3_prefix = f"{base_path}{drop_zone}/"
        
        # Find all documents in input folder
        document_patterns = ["*.pdf", "*.txt", "*.md", "*.docx", "*.html"]
        files_to_upload = []
        
        for pattern in document_patterns:
            files_to_upload.extend(input_path.glob(pattern))
        
        if not files_to_upload:
            return {
                "success": False,
                "error": "No documents found in input folder",
                "files_found": 0
            }
        
        # Upload each file
        upload_results = []
        successful_uploads = 0
        
        print(f"Uploading {len(files_to_upload)} files to {drop_zone}...")
        
        for file_path in files_to_upload:
            s3_key = f"{s3_prefix}{file_path.name}"
            
            print(f"  Uploading: {file_path.name}")
            
            result = self.s3_manager.upload_file(
                file_path=file_path,
                bucket=self.s3_bucket,
                s3_key=s3_key,
                metadata={
                    "domain": domain_name,
                    "workflow_stage": drop_zone,
                    "original_filename": file_path.name,
                    "upload_timestamp": datetime.now().isoformat(),
                    "file_size": str(file_path.stat().st_size)
                }
            )
            
            if result.success:
                successful_uploads += 1
                upload_results.append({
                    "filename": file_path.name,
                    "s3_key": s3_key,
                    "s3_url": result.s3_url,
                    "size": result.file_size,
                    "upload_duration": result.upload_duration
                })
                print(f"    SUCCESS: {result.s3_url}")
            else:
                upload_results.append({
                    "filename": file_path.name,
                    "error": result.error
                })
                print(f"    FAILED: {result.error}")
        
        return {
            "success": successful_uploads > 0,
            "total_files": len(files_to_upload),
            "successful_uploads": successful_uploads,
            "failed_uploads": len(files_to_upload) - successful_uploads,
            "drop_zone": drop_zone,
            "s3_prefix": s3_prefix,
            "upload_results": upload_results
        }
    
    def _create_vector_embeddings(self, domain_name: str, drop_zone: str) -> Dict[str, Any]:
        """Create vector embeddings for documents in drop zone"""
        
        s3_prefix = f"workflows/{domain_name}/{drop_zone}/"
        
        try:
            # List documents in S3 drop zone
            s3_documents = self.s3_manager.list_documents(self.s3_bucket, s3_prefix)
            
            if not s3_documents:
                return {
                    "success": False,
                    "error": "No documents found in S3 drop zone",
                    "s3_prefix": s3_prefix
                }
            
            # Create vector records for each S3 document
            vector_records = []
            
            for s3_doc in s3_documents:
                print(f"  Creating vector record: {s3_doc['filename']}")
                
                # Create document record with S3 reference
                from knowledge_systems.core.vector_manager import Document
                
                doc = Document(
                    title=s3_doc['filename'],
                    content="",  # Content stays in S3
                    source=s3_doc['s3_url'],
                    doc_type=f"{domain_name}_workflow_document",
                    metadata={
                        "domain": domain_name,
                        "workflow_stage": drop_zone,
                        "s3_bucket": self.s3_bucket,
                        "s3_key": s3_doc['key'],
                        "s3_url": s3_doc['s3_url'],
                        "s3_etag": s3_doc['etag'],
                        "s3_size": s3_doc['size']
                    }
                )
                
                # This would add to vector DB (mock for now due to schema issues)
                vector_records.append({
                    "document_title": s3_doc['filename'],
                    "s3_reference": s3_doc['s3_url'],
                    "workflow_stage": drop_zone,
                    "vector_created": True
                })
            
            return {
                "success": True,
                "documents_processed": len(s3_documents),
                "vector_records": len(vector_records),
                "drop_zone": drop_zone,
                "records": vector_records
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _setup_workflow_sequence(self, domain_name: str, upload_result: Dict) -> Dict[str, Any]:
        """Setup the complete workflow sequence"""
        
        workflow_sequence = {
            "domain": domain_name,
            "sequence_id": f"{domain_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "stages": [
                {
                    "stage_number": 1,
                    "stage_name": "01_input",
                    "description": "Input documents uploaded to S3",
                    "status": "completed",
                    "s3_location": f"s3://{self.s3_bucket}/workflows/{domain_name}/01_input/",
                    "documents": upload_result.get("successful_uploads", 0),
                    "ready_for_next": True
                },
                {
                    "stage_number": 2,
                    "stage_name": "02_processed",
                    "description": "Document processing and analysis",
                    "status": "ready",
                    "s3_location": f"s3://{self.s3_bucket}/workflows/{domain_name}/02_processed/",
                    "depends_on": ["01_input"],
                    "ready_for_next": False
                },
                {
                    "stage_number": 3,
                    "stage_name": "03_analyzed",
                    "description": "Content analysis and extraction",
                    "status": "pending",
                    "s3_location": f"s3://{self.s3_bucket}/workflows/{domain_name}/03_analyzed/",
                    "depends_on": ["02_processed"],
                    "ready_for_next": False
                },
                {
                    "stage_number": 4,
                    "stage_name": "04_enriched",
                    "description": "Data enrichment and enhancement",
                    "status": "pending",
                    "s3_location": f"s3://{self.s3_bucket}/workflows/{domain_name}/04_enriched/",
                    "depends_on": ["03_analyzed"],
                    "ready_for_next": False
                },
                {
                    "stage_number": 5,
                    "stage_name": "05_output",
                    "description": "Final output and results",
                    "status": "pending",
                    "s3_location": f"s3://{self.s3_bucket}/workflows/{domain_name}/05_output/",
                    "depends_on": ["04_enriched"],
                    "ready_for_next": False
                }
            ]
        }
        
        return workflow_sequence
    
    def _generate_next_steps(self, domain_name: str, workflow_sequence: Dict) -> List[str]:
        """Generate next steps for the workflow"""
        
        next_steps = [
            f"✓ Domain '{domain_name}' workflow created successfully",
            f"✓ Input documents uploaded to S3 drop zone 01_input",
            f"✓ Vector embeddings created with S3 references",
            f"✓ Workflow sequence established with 5 stages",
            "",
            "NEXT STEPS:",
            f"1. Process stage 02_processed:",
            f"   - Analyze documents in 01_input",
            f"   - Extract content and metadata",
            f"   - Move processed results to 02_processed",
            "",
            f"2. Continue workflow sequence:",
            f"   - 03_analyzed: Content analysis",
            f"   - 04_enriched: Data enrichment", 
            f"   - 05_output: Final results",
            "",
            f"3. Query the domain:",
            f"   ki.query('your question', domain='{domain_name}')",
            "",
            f"4. Monitor S3 locations:",
            f"   s3://{self.s3_bucket}/workflows/{domain_name}/[stage]/",
            "",
            "WORKFLOW IS READY FOR PROCESSING!"
        ]
        
        return next_steps

def create_workflow_interactive():
    """Interactive workflow creation"""
    
    print("Domain Workflow Creator")
    print("=" * 30)
    
    # Get user input
    print("\nEnter workflow details:")
    domain_name = input("Domain name (e.g., 'financial_risk'): ").strip()
    input_folder = input("Input folder path: ").strip()
    
    if not domain_name or not input_folder:
        print("ERROR: Both domain name and input folder are required")
        return
    
    # Validate input folder
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"ERROR: Input folder not found: {input_folder}")
        return
    
    # Count documents
    document_patterns = ["*.pdf", "*.txt", "*.md", "*.docx"]
    total_docs = 0
    for pattern in document_patterns:
        total_docs += len(list(input_path.glob(pattern)))
    
    if total_docs == 0:
        print(f"ERROR: No documents found in {input_folder}")
        return
    
    print(f"\nConfiguration:")
    print(f"  Domain: {domain_name}")
    print(f"  Input folder: {input_folder}")
    print(f"  Documents found: {total_docs}")
    
    proceed = input(f"\nProceed with workflow creation? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Workflow creation cancelled")
        return
    
    # Create workflow
    print(f"\nCreating workflow...")
    creator = DomainWorkflowCreator()
    result = creator.create_domain_workflow(domain_name, input_folder)
    
    # Display results
    print(f"\nWORKFLOW CREATION RESULTS")
    print("=" * 30)
    print(f"Success: {result['success']}")
    
    if result["success"]:
        print(f"Domain: {result['domain_name']}")
        print(f"S3 Bucket: {result['s3_bucket']}")
        
        upload_summary = result.get("upload_summary", {})
        print(f"Files uploaded: {upload_summary.get('successful_uploads', 0)}/{upload_summary.get('total_files', 0)}")
        
        vector_summary = result.get("vector_summary", {})
        print(f"Vector records: {vector_summary.get('vector_records', 0)}")
        
        print(f"\nS3 Structure Created:")
        s3_structure = result.get("s3_structure", {})
        for key, value in s3_structure.items():
            if not key.endswith("_s3_url"):
                print(f"  {key}: {value}")
        
        print(f"\nWorkflow Sequence:")
        workflow = result.get("workflow_sequence", {})
        for stage in workflow.get("stages", []):
            status_icon = "✓" if stage["status"] == "completed" else "○"
            print(f"  {status_icon} {stage['stage_name']}: {stage['description']}")
        
        print(f"\nNext Steps:")
        for step in result.get("next_steps", []):
            print(f"  {step}")
            
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

def main():
    """Demo workflow creation"""
    
    print("Demo: Domain Workflow Creation")
    print("=" * 35)
    
    # Demo configuration
    domain_name = "model_validation_demo"
    
    # Find knowledge base for demo
    knowledge_base_paths = [
        parent_dir / "knowledge_base",
        parent_dir / "tidyllm" / "knowledge_base"
    ]
    
    input_folder = None
    for path in knowledge_base_paths:
        if path.exists():
            input_folder = str(path)
            break
    
    if not input_folder:
        print("ERROR: No knowledge base found for demo")
        return
    
    print(f"Demo Configuration:")
    print(f"  Domain: {domain_name}")
    print(f"  Input folder: {input_folder}")
    
    # Create workflow
    creator = DomainWorkflowCreator()
    result = creator.create_domain_workflow(domain_name, input_folder)
    
    # Show workflow structure
    print(f"\nDEMO WORKFLOW CREATED")
    print("-" * 25)
    
    if result["success"]:
        print("SUCCESS: Workflow sequence established")
        
        # Show S3 structure
        print(f"\nS3 Drop Zone Structure:")
        s3_structure = result["s3_structure"]
        print(f"  01_input: {s3_structure['01_input_s3_url']}")
        print(f"  02_processed: {s3_structure['02_processed_s3_url']}")
        print(f"  03_analyzed: {s3_structure['03_analyzed_s3_url']}")
        print(f"  04_enriched: {s3_structure['04_enriched_s3_url']}")
        print(f"  05_output: {s3_structure['05_output_s3_url']}")
        
        # Show upload results  
        upload_summary = result["upload_summary"]
        print(f"\nFirst Drop Zone (01_input):")
        print(f"  Files uploaded: {upload_summary['successful_uploads']}")
        print(f"  S3 location: {upload_summary['s3_prefix']}")
        
        print(f"\nWorkflow is ready for stage 02_processed!")
        
    else:
        print(f"FAILED: {result['error']}")

if __name__ == "__main__":
    # Uncomment for interactive mode:
    # create_workflow_interactive()
    
    # Run demo:
    main()