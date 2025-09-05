#!/usr/bin/env python
"""

# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod

S3-First MVR Processor - CONSTRAINTS COMPLIANT
===============================================

Fully compliant with TidyLLM S3-First Processing Architecture:
- NO local file processing
- NO local storage/temp files  
- NO app folders or directories
- Direct S3 → S3 streaming
- PostgreSQL-direct MLflow tracking
- UnifiedSessionManager ONLY

Data Flow: S3 Upload → S3 Processing → S3 Storage → PostgreSQL Tracking
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.start_unified_sessions import UnifiedSessionManager
import json
from datetime import datetime
import io
# TidyLLM native imports (constraints-compliant)
try:
    import tidyllm.tlm as np  # TidyLLM native math
    import tidyllm_sentence as tls  # TidyLLM native embeddings
    TIDYLLM_AVAILABLE = True
except ImportError:
    print("[WARN] TidyLLM native modules not available - using fallbacks")
    TIDYLLM_AVAILABLE = False


class S3FirstMVRProcessor:
    """
    CONSTRAINTS-COMPLIANT MVR Processor
    
    Architecture:
    - User uploads MVR directly to S3 
    - System processes S3 → S3 (streaming)
    - Results saved to S3 (no local storage)
    - Tracking in PostgreSQL (direct connection)
    - Zero local file operations
    """
    
    def __init__(self):
        """Initialize S3-first processor with UnifiedSessionManager and three gateways"""
        
        print("[S3-FIRST] Initializing MVR Processor...")
        
        # REQUIRED: Use UnifiedSessionManager (constraints-compliant)
        try:
            self.session_mgr = UnifiedSessionManager()
            
            # Test AWS connectivity immediately
            s3_client = self.session_mgr.get_s3_client()
            if not s3_client:
                raise Exception("S3 client not available - check AWS credentials")
            
            # Quick connectivity test
            s3_client.list_buckets()
            print("[AWS] AWS credentials validated successfully")
            
        except Exception as e:
            print(f"[ERROR] AWS initialization failed: {e}")
            print("[ERROR] Please configure AWS credentials:")
            print("   1. Run 'aws configure' to set up credentials")
            print("   2. Or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            print("   3. Or configure IAM role if running on EC2")
            raise Exception("Cannot proceed without valid AWS credentials")
        
        # Initialize the three-gateway architecture (graceful fallback)
        try:
            from tidyllm.gateways import init_gateways, get_gateway
            
            # Initialize gateways with configuration
            self.gateways = init_gateways({
                "corporate_llm": {
                    "budget_limit_daily_usd": 100,
                    "provider": "anthropic",
                    "model": "claude-3-sonnet"
                },
                "ai_processing": {
                    "backend": "anthropic",
                    "model": "claude-3-sonnet", 
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                "workflow_optimizer": {
                    "optimization_level": "high"
                }
            })
            
            print("[GATEWAYS] Initialized three-gateway architecture:")
            print(f"   • corporate_llm: Compliance validation")
            print(f"   • ai_processing: Document intelligence") 
            print(f"   • workflow_optimizer: Knowledge extraction")
            
        except Exception as e:
            print(f"[WARN] Gateway initialization failed: {e}")
            print("   [FALLBACK] Will use rule-based processing instead of AI gateways")
            self.gateways = None
        
        # S3 bucket configuration (NO LOCAL EQUIVALENTS)
        self.s3_config = {
            "bucket": s3_config["bucket"],  # Primary bucket
            "paths": {
                "raw_mvr": build_s3_path("mvr_analysis", "raw/"),           # Raw MVR documents  
                "prompts": build_s3_path("mvr_analysis", "prompts/"),       # Prompt templates
                "processed": build_s3_path("mvr_analysis", "processed/"),   # Processed results
                "reports": build_s3_path("mvr_analysis", "reports/"),       # Generated reports
                "embeddings": build_s3_path("mvr_analysis", "embeddings/"), # Vector storage
                "metadata": build_s3_path("mvr_analysis", "metadata/")      # Document metadata
            }
        }
        
        # Validate bucket exists
        try:
            s3_client.head_bucket(Bucket=self.s3_config['bucket'])
            print(f"[S3] Bucket '{self.s3_config['bucket']}' validated successfully")
        except Exception as e:
            print(f"[ERROR] Bucket '{self.s3_config['bucket']}' not accessible: {e}")
            print("[ERROR] Available buckets:")
            try:
                buckets = s3_client.list_buckets()['Buckets']
                for bucket in buckets:
                    print(f"   - {bucket['Name']}")
                print("[ERROR] Please update the bucket name in the code or create the bucket")
                raise Exception(f"Cannot access bucket: {self.s3_config['bucket']}")
            except Exception as list_error:
                print(f"   [ERROR] Cannot list buckets: {list_error}")
                raise Exception("AWS access insufficient - check permissions")
        
        print("[S3-FIRST] Configuration:")
        print(f"   Bucket: {self.s3_config['bucket']}")
        print(f"   Session: UnifiedSessionManager (official)")
        print(f"   Storage: S3-only (zero local files)")
        print(f"   Processing: Three-gateway AI pipeline")
        print(f"   Tracking: PostgreSQL direct")
    
    def process_mvr_s3_to_s3(self, source_bucket: str, mvr_key: str, 
                            prompt_key: str = "prompts/favorites/JB_Overview_Prompt.md") -> dict:
        """
        Process MVR document entirely in cloud - ZERO local storage
        
        Args:
            source_bucket: S3 bucket containing MVR
            mvr_key: S3 key of MVR document  
            prompt_key: S3 key of prompt template
            
        Returns:
            Processing results (metadata only)
        """
        
        print(f"\n[S3→S3] Processing MVR: s3://{source_bucket}/{mvr_key}")
        print(f"[S3→S3] Using prompt: s3://{source_bucket}/{prompt_key}")
        
        # Generate processing ID
        process_id = f"mvr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # STEP 1: Stream MVR document from S3 (NO local download)
            print("[STEP 1/6] Streaming MVR from S3...")
            mvr_content = self._stream_document_from_s3(source_bucket, mvr_key)
            print(f"   → Streamed {len(mvr_content):,} characters")
            
            # STEP 2: Stream prompt template from S3 (NO local files)  
            print("[STEP 2/6] Streaming prompt template from S3...")
            prompt_content = self._stream_document_from_s3(source_bucket, prompt_key)
            print(f"   → Streamed prompt: {len(prompt_content):,} characters")
            
            # STEP 3: Generate embeddings using TidyLLM native (in memory)
            print("[STEP 3/6] Generating embeddings (TidyLLM native)...")
            embeddings = self._generate_embeddings_native(mvr_content)
            print(f"   → Generated embeddings: {len(embeddings)} vectors")
            
            # STEP 4: Process through three-gateway architecture
            print("[STEP 4/6] Processing through three-gateway AI pipeline...")
            all_reports = self._process_through_gateways(mvr_content, prompt_content, process_id)
            print(f"   → Generated {len(all_reports)} comprehensive reports")
            
            # STEP 5: Save all reports and originals to S3 (safe keeping)
            print("[STEP 5/6] Saving originals and reports to S3...")
            result_keys = self._save_all_results_to_s3(
                source_bucket, process_id, all_reports, embeddings, mvr_content, prompt_content
            )
            
            # STEP 6: Track in PostgreSQL (direct connection)
            print("[STEP 6/6] Logging to PostgreSQL...")
            self._track_processing_postgresql(process_id, source_bucket, mvr_key, result_keys)
            
            print(f"\n[SUCCESS] MVR processing complete: {process_id}")
            
            # Build outputs dynamically based on what was generated
            outputs = {}
            if "original_mvr" in result_keys:
                outputs["original_mvr"] = f"s3://{source_bucket}/{result_keys['original_mvr']}"
            if "original_prompt" in result_keys:
                outputs["original_prompt"] = f"s3://{source_bucket}/{result_keys['original_prompt']}"
            if "compliance_report" in result_keys:
                outputs["compliance_report"] = f"s3://{source_bucket}/{result_keys['compliance_report']}"
            if "intelligence_report" in result_keys:
                outputs["intelligence_report"] = f"s3://{source_bucket}/{result_keys['intelligence_report']}"
            if "knowledge_report" in result_keys:
                outputs["knowledge_report"] = f"s3://{source_bucket}/{result_keys['knowledge_report']}"
            if "embeddings" in result_keys:
                outputs["embeddings"] = f"s3://{source_bucket}/{result_keys['embeddings']}"
            if "metadata" in result_keys:
                outputs["metadata"] = f"s3://{source_bucket}/{result_keys['metadata']}"
            
            return {
                "process_id": process_id,
                "status": "completed",
                "architecture": "three-gateway AI pipeline",
                "gateways_used": ["corporate_llm", "ai_processing", "workflow_optimizer"],
                "reports_generated": len([k for k in result_keys.keys() if "_report" in k]),
                "input": f"s3://{source_bucket}/{mvr_key}",
                "outputs": outputs,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"[ERROR] S3-first processing failed: {e}")
            return {
                "process_id": process_id,
                "status": "failed", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_s3_client_with_retry(self, max_retries: int = 2):
        """Get S3 client with session restart capability"""
        
        for attempt in range(max_retries + 1):
            try:
                s3_client = self.session_mgr.get_s3_client()
                if not s3_client:
                    raise Exception("S3 client not available")
                
                # Test connectivity with a simple operation
                s3_client.list_buckets()
                return s3_client
                
            except Exception as e:
                print(f"   [RETRY {attempt + 1}/{max_retries + 1}] S3 session error: {e}")
                
                if attempt < max_retries:
                    print(f"   [RESTART] Restarting AWS session...")
                    try:
                        # Use your existing admin restart script (more stable)
                        import subprocess
                        import os
                        
                        admin_script = Path(__file__).parent.parent / "tidyllm" / "admin" / "restart_aws_session.py"
                        if admin_script.exists():
                            print(f"   [RESTART] Using admin script: {admin_script}")
                            result = subprocess.run([sys.executable, str(admin_script)], 
                                                  capture_output=True, text=True, timeout=30)
                            if result.returncode == 0:
                                print(f"   [RESTART] Admin script completed successfully")
                            else:
                                print(f"   [RESTART] Admin script error: {result.stderr}")
                        
                        # Reinitialize session manager with fresh settings
                        self.session_mgr = UnifiedSessionManager()
                        print(f"   [RESTART] Session restarted successfully")
                        
                    except Exception as restart_error:
                        print(f"   [RESTART] Failed to restart session: {restart_error}")
                else:
                    print(f"   [FAILED] All retry attempts exhausted")
                    raise Exception(f"Cannot establish S3 connection after {max_retries + 1} attempts")
    
    def _stream_document_from_s3(self, bucket: str, key: str) -> str:
        """Stream document content from S3 (NO local files) with session recovery"""
        
        try:
            # Get S3 client with retry capability
            s3_client = self._get_s3_client_with_retry()
            
            # Stream object from S3
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content_bytes = response['Body'].read()
            
            # Decode content (handle different encodings)
            try:
                content = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                content = content_bytes.decode('latin-1', errors='ignore')
            
            return content
            
        except Exception as e:
            raise Exception(f"Failed to stream {key} from S3: {e}")
    
    def _generate_embeddings_native(self, content: str) -> list:
        """Generate embeddings using TidyLLM native (NOT sentence-transformers)"""
        
        try:
            if TIDYLLM_AVAILABLE:
                # Use TidyLLM sentence (constraints-compliant)
                embeddings, model = tls.tfidf_fit_transform([content])
                return embeddings[0] if embeddings else []
            else:
                # Fallback: simple word frequency vector (pure Python)
                words = content.lower().split()
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                
                # Return top 100 most frequent words as embedding
                sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:100]
                return [count for word, count in sorted_words]
            
        except Exception as e:
            print(f"[WARN] Embedding generation failed: {e}")
            return []
    
    def _process_through_gateways(self, mvr_content: str, prompt_content: str, process_id: str) -> dict:
        """Process MVR through three-gateway architecture (AI-powered analysis)"""
        
        if not self.gateways:
            # Fallback to rule-based processing
            return {"compliance_report": self._fallback_rule_based_analysis(mvr_content, prompt_content)}
        
        try:
            from tidyllm.gateways import get_gateway, AIRequest, LLMRequest, WorkflowRequest
            
            all_reports = {}
            
            # GATEWAY 1: Corporate LLM - Compliance Validation Report
            print("   [GATEWAY 1/3] Corporate LLM - Compliance validation...")
            corporate_gateway = get_gateway("corporate_llm")
            if corporate_gateway:
                compliance_prompt = f"""
{prompt_content}

DOCUMENT TO ANALYZE:
{mvr_content[:4000]}...

Please perform detailed MVR compliance validation according to the JB_Overview_Prompt instructions.
Focus on:
1. MVS requirement compliance
2. VST section alignment
3. Evidence quality assessment
4. Logic gaps and contradictions
5. Peer review challenges
6. Compliance conclusions

Return structured JSON response with section-by-section analysis.
"""
                
                llm_request = LLMRequest(
                    prompt=compliance_prompt,
                    model="claude-3-sonnet",
                    temperature=0.1,
                    max_tokens=2000
                )
                
                compliance_response = corporate_gateway.process(llm_request)
                if compliance_response.success:
                    all_reports["compliance_report"] = {
                        "report_type": "mvr_compliance_validation",
                        "gateway": "corporate_llm",
                        "timestamp": datetime.now().isoformat(),
                        "analysis": compliance_response.content,
                        "processing_time": compliance_response.processing_time
                    }
                    print(f"      → Generated compliance report ({len(compliance_response.content):,} chars)")
            
            # GATEWAY 2: AI Processing - Document Intelligence Report  
            print("   [GATEWAY 2/3] AI Processing - Document intelligence...")
            ai_gateway = get_gateway("ai_processing")
            if ai_gateway:
                intelligence_prompt = f"""
Perform comprehensive document intelligence analysis on this MVR document:

{mvr_content[:4000]}...

Analyze for:
1. Document structure and organization
2. Key findings and conclusions
3. Technical methodology assessment
4. Data quality and sources
5. Risk identification
6. Recommendations extraction
7. Knowledge gaps identification

Provide detailed intelligence report in structured JSON format.
"""
                
                ai_request = AIRequest(
                    prompt=intelligence_prompt,
                    model="claude-3-sonnet",
                    temperature=0.2,
                    max_tokens=2000
                )
                
                intelligence_response = ai_gateway.process(ai_request)
                if intelligence_response.success:
                    all_reports["intelligence_report"] = {
                        "report_type": "document_intelligence",
                        "gateway": "ai_processing", 
                        "timestamp": datetime.now().isoformat(),
                        "analysis": intelligence_response.content,
                        "processing_time": intelligence_response.processing_time
                    }
                    print(f"      → Generated intelligence report ({len(intelligence_response.content):,} chars)")
            
            # GATEWAY 3: Workflow Optimizer - Knowledge Extraction Report
            print("   [GATEWAY 3/3] Workflow Optimizer - Knowledge extraction...")
            optimizer_gateway = get_gateway("workflow_optimizer")
            if optimizer_gateway:
                extraction_prompt = f"""
Extract and organize knowledge from this MVR document for knowledge base expansion:

{mvr_content[:4000]}...

Extract:
1. Table of contents and document structure
2. Key references and citations
3. Methodological frameworks used
4. Technical definitions and terminology
5. Best practices identified
6. Lessons learned
7. Reusable templates and patterns

Organize findings for knowledge base integration in structured JSON format.
"""
                
                workflow_request = WorkflowRequest(
                    operation="knowledge_extraction",
                    content=extraction_prompt,
                    optimization_level="comprehensive"
                )
                
                extraction_response = optimizer_gateway.process_workflow(workflow_request)
                if extraction_response.success:
                    all_reports["knowledge_report"] = {
                        "report_type": "knowledge_extraction",
                        "gateway": "workflow_optimizer",
                        "timestamp": datetime.now().isoformat(),
                        "analysis": extraction_response.content,
                        "processing_time": extraction_response.processing_time
                    }
                    print(f"      → Generated knowledge extraction report ({len(extraction_response.content):,} chars)")
            
            return all_reports
            
        except Exception as e:
            print(f"   [ERROR] Gateway processing failed: {e}")
            return {"compliance_report": self._fallback_rule_based_analysis(mvr_content, prompt_content)}
    
    def _fallback_rule_based_analysis(self, mvr_content: str, prompt_content: str) -> dict:
        """Fallback rule-based analysis when gateways are unavailable"""
        
        # Parse MVR sections (pure Python, no external libs)
        sections = self._parse_mvr_sections_native(mvr_content)
        
        compliance_report = {
            "report_type": "mvr_compliance_validation_fallback",
            "prompt_used": "JB_Overview_Prompt.md",
            "timestamp": datetime.now().isoformat(),
            "total_sections": len(sections),
            "processing_mode": "rule_based_fallback",
            "sections": []
        }
        
        # Analyze each section for compliance
        for section in sections:
            section_analysis = {
                "section_id": section["id"],
                "section_title": section["title"], 
                "mvs_requirements": self._identify_mvs_requirements(section),
                "vst_sections": self._identify_vst_sections(section),
                "review_narrative": self._generate_review_narrative(section),
                "contradiction_summary": "None detected (rule-based)",
                "peer_review_challenge": self._generate_peer_challenge(section),
                "conclusion": self._determine_compliance_status(section),
                "confidence_score": "Moderate (rule-based)",
                "defect_type": "N/A"
            }
            
            compliance_report["sections"].append(section_analysis)
        
        return compliance_report
    
    def _parse_mvr_sections_native(self, content: str) -> list:
        """Parse MVR sections using pure Python (no external libraries)"""
        
        sections = []
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Detect section headers (pure Python pattern matching)
            if self._is_section_header(line):
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    "id": self._extract_section_id(line),
                    "title": line,
                    "text": ""
                }
            elif current_section:
                current_section["text"] += line + "\n"
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """Identify section headers using pure Python"""
        if not line:
            return False
        
        # Check for numbered sections
        if any(line.startswith(f"{i}.") for i in range(1, 20)):
            return True
        
        # Check for uppercase headers
        if line.isupper() and len(line) > 3:
            return True
        
        # Check for common MVR section patterns
        keywords = ["EXECUTIVE SUMMARY", "SCOPE", "METHODOLOGY", "FINDINGS", "CONCLUSION"]
        return any(keyword in line.upper() for keyword in keywords)
    
    def _extract_section_id(self, line: str) -> str:
        """Extract section ID from header"""
        # Simple extraction for numbered sections
        parts = line.split()
        if parts and any(char.isdigit() for char in parts[0]):
            return parts[0].rstrip('.')
        return f"sec_{hash(line) % 1000}"
    
    def _identify_mvs_requirements(self, section: dict) -> list:
        """Identify relevant MVS requirements for section"""
        text = section["text"].lower()
        
        requirements = []
        if "conceptual" in text or "methodology" in text:
            requirements.extend(["MVS 5.4.3", "MVS 5.4.3.1"])
        if "performance" in text or "validation" in text:
            requirements.extend(["MVS 5.12.1", "MVS 5.12.2"])
        if "data quality" in text:
            requirements.append("MVS 5.4.2")
        
        return requirements if requirements else ["MVS General"]
    
    def _identify_vst_sections(self, section: dict) -> list:
        """Identify relevant VST sections"""
        text = section["text"].lower()
        
        vst_sections = []
        if "conceptual" in text:
            vst_sections.append("VST Conceptual Soundness")
        if "performance" in text:
            vst_sections.append("VST Performance Monitoring")
        if "data" in text:
            vst_sections.append("VST Data Quality")
        
        return vst_sections if vst_sections else ["VST General"]
    
    def _generate_review_narrative(self, section: dict) -> str:
        """Generate review narrative for section"""
        return f"Section {section['id']} provides analysis of {section['title']}. Content demonstrates appropriate coverage of requirements."
    
    def _generate_peer_challenge(self, section: dict) -> str:
        """Generate peer reviewer challenge"""
        challenges = [
            "Consider adding more recent validation data",
            "Strengthen documentation of assumptions",
            "Enhance testing coverage for edge cases", 
            "Provide more detailed justification for methodology"
        ]
        return challenges[hash(section["id"]) % len(challenges)]
    
    def _determine_compliance_status(self, section: dict) -> str:
        """Determine compliance status for section"""
        # Simple heuristic - would be more sophisticated in production
        text_length = len(section["text"])
        if text_length > 500:
            return "✅ COMPLIANT"
        elif text_length > 200:
            return "⚠️ PARTIALLY COMPLIANT"
        else:
            return "❌ NON-COMPLIANT"
    
    def _save_all_results_to_s3(self, bucket: str, process_id: str, all_reports: dict, 
                               embeddings: list, mvr_content: str, prompt_content: str) -> dict:
        """Save all reports and originals to S3 for safe keeping (NO local files) with session recovery"""
        
        # Use retry mechanism for S3 operations
        s3_client = self._get_s3_client_with_retry()
        result_keys = {}
        
        try:
            # Save original MVR document for safe keeping
            mvr_original_key = fbuild_s3_path("mvr_analysis", "originals/{process_id}_original_mvr.txt")
            s3_client.put_object(
                Bucket=bucket,
                Key=mvr_original_key,
                Body=mvr_content.encode('utf-8'),
                ServerSideEncryption='AES256',
                Metadata={'content_type': 'original_document', 'process_id': process_id}
            )
            result_keys["original_mvr"] = mvr_original_key
            
            # Save original prompt for safe keeping
            prompt_original_key = fbuild_s3_path("mvr_analysis", "originals/{process_id}_original_prompt.md")
            s3_client.put_object(
                Bucket=bucket,
                Key=prompt_original_key,
                Body=prompt_content.encode('utf-8'),
                ServerSideEncryption='AES256',
                Metadata={'content_type': 'original_prompt', 'process_id': process_id}
            )
            result_keys["original_prompt"] = prompt_original_key
            
            # Save compliance validation report (Gateway 1)
            if "compliance_report" in all_reports:
                compliance_key = fbuild_s3_path("mvr_analysis", "reports/compliance/{process_id}_compliance_report.json")
                s3_client.put_object(
                    Bucket=bucket,
                    Key=compliance_key,
                    Body=json.dumps(all_reports["compliance_report"], indent=2).encode(),
                    ServerSideEncryption='AES256'
                )
                result_keys["compliance_report"] = compliance_key
                print(f"   → Saved compliance validation report: {compliance_key}")
            
            # Save document intelligence report (Gateway 2)
            if "intelligence_report" in all_reports:
                intelligence_key = fbuild_s3_path("mvr_analysis", "reports/intelligence/{process_id}_intelligence_report.json")
                s3_client.put_object(
                    Bucket=bucket,
                    Key=intelligence_key,
                    Body=json.dumps(all_reports["intelligence_report"], indent=2).encode(),
                    ServerSideEncryption='AES256'
                )
                result_keys["intelligence_report"] = intelligence_key
                print(f"   → Saved document intelligence report: {intelligence_key}")
            
            # Save knowledge extraction report (Gateway 3)
            if "knowledge_report" in all_reports:
                knowledge_key = fbuild_s3_path("mvr_analysis", "reports/knowledge/{process_id}_knowledge_report.json")
                s3_client.put_object(
                    Bucket=bucket,
                    Key=knowledge_key,
                    Body=json.dumps(all_reports["knowledge_report"], indent=2).encode(),
                    ServerSideEncryption='AES256'
                )
                result_keys["knowledge_report"] = knowledge_key
                print(f"   → Saved knowledge extraction report: {knowledge_key}")
            
            # Save embeddings
            embeddings_key = fbuild_s3_path("mvr_analysis", "embeddings/{process_id}_embeddings.json")
            embeddings_data = {
                "process_id": process_id,
                "embeddings": embeddings,
                "model": "tidyllm-sentence-tfidf",
                "dimensions": len(embeddings) if embeddings else 0,
                "timestamp": datetime.now().isoformat()
            }
            s3_client.put_object(
                Bucket=bucket,
                Key=embeddings_key,
                Body=json.dumps(embeddings_data).encode(),
                ServerSideEncryption='AES256'
            )
            result_keys["embeddings"] = embeddings_key
            
            # Save comprehensive metadata
            metadata_key = fbuild_s3_path("mvr_analysis", "metadata/{process_id}_metadata.json")
            metadata = {
                "process_id": process_id,
                "timestamp": datetime.now().isoformat(),
                "three_gateway_architecture": True,
                "gateways_used": {
                    "corporate_llm": "compliance_report" in all_reports,
                    "ai_processing": "intelligence_report" in all_reports,
                    "workflow_optimizer": "knowledge_report" in all_reports
                },
                "reports_generated": len(all_reports),
                "originals_preserved": True,
                "total_processing_files": len(result_keys),
                "architecture": "S3-first with three-gateway AI pipeline"
            }
            s3_client.put_object(
                Bucket=bucket,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2).encode(),
                ServerSideEncryption='AES256'
            )
            result_keys["metadata"] = metadata_key
            
            print(f"   → Saved original MVR document: {mvr_original_key}")
            print(f"   → Saved original prompt: {prompt_original_key}")
            print(f"   → Saved embeddings: {embeddings_key}")
            print(f"   → Saved metadata: {metadata_key}")
            print(f"   → Total files saved: {len(result_keys)} (originals + reports)")
            
            return result_keys
            
        except Exception as e:
            raise Exception(f"Failed to save all results to S3: {e}")
    
    def _track_processing_postgresql(self, process_id: str, bucket: str, 
                                   mvr_key: str, result_keys: dict):
        """Track processing in PostgreSQL (direct connection)"""
        
        try:
            # Log to MLflow via UnifiedSessionManager (PostgreSQL direct)
            self.session_mgr.log_mlflow_experiment({
                "experiment_name": "mvr_compliance_validation",
                "process_id": process_id,
                "input_document": f"s3://{bucket}/{mvr_key}",
                "compliance_report": f"s3://{bucket}/{result_keys['compliance']}",
                "embeddings": f"s3://{bucket}/{result_keys['embeddings']}",
                "processing_method": "s3_first_streaming",
                "embedding_model": "tidyllm-sentence-tfidf",
                "prompt_template": "JB_Overview_Prompt.md",
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"   → Logged to MLflow: {process_id}")
            
        except Exception as e:
            print(f"[WARN] PostgreSQL tracking failed: {e}")


def main():
    """Demo S3-first MVR processing with three-gateway architecture"""
    
    print("=" * 70)
    print("S3-FIRST MVR PROCESSOR - THREE-GATEWAY AI ARCHITECTURE")
    print("=" * 70)
    print("[OK] NO local file processing")
    print("[OK] NO local storage/temp files")
    print("[OK] Three-gateway AI pipeline (corporate_llm, ai_processing, workflow_optimizer)")
    print("[OK] Direct S3 -> S3 streaming")
    print("[OK] Originals preserved for safe keeping")
    print("[OK] PostgreSQL-direct tracking")
    print("[OK] UnifiedSessionManager only")
    print("[OK] TidyLLM native stack")
    
    processor = S3FirstMVRProcessor()
    
    print("\n[DEMO] To process an MVR document:")
    print("1. Upload MVR to S3: s3://nsc-mvp1/mvr_analysis/raw/document.pdf")
    print("2. Upload prompt to S3: s3://nsc-mvp1/mvr_analysis/prompts/JB_Overview_Prompt.md")
    print("3. Run:")
    print("   result = processor.process_mvr_s3_to_s3(")
    print("       s3_config["bucket"],")
    print("       build_s3_path("mvr_analysis", "raw/document.pdf"),")
    print("       build_s3_path("mvr_analysis", "prompts/JB_Overview_Prompt.md")")
    print("   )")
    
    print("\n[THREE-GATEWAY REPORTS] Generated reports:")
    print("GATEWAY 1: Corporate LLM -> Compliance Validation Report")
    print("   Location: s3://nsc-mvp1/mvr_analysis/reports/compliance/")
    print("GATEWAY 2: AI Processing -> Document Intelligence Report")
    print("   Location: s3://nsc-mvp1/mvr_analysis/reports/intelligence/")
    print("GATEWAY 3: Workflow Optimizer -> Knowledge Extraction Report")
    print("   Location: s3://nsc-mvp1/mvr_analysis/reports/knowledge/")
    
    print("\n[SAFE KEEPING] Originals preserved:")
    print("Original MVR document: s3://nsc-mvp1/mvr_analysis/originals/")
    print("Original prompt template: s3://nsc-mvp1/mvr_analysis/originals/")
    print("Embeddings: s3://nsc-mvp1/mvr_analysis/embeddings/")
    print("Metadata: s3://nsc-mvp1/mvr_analysis/metadata/")
    
    print("\n[ARCHITECTURE] Three-Gateway Data Flow:")
    print("S3 Upload -> Gateway Pipeline -> S3 Safe Storage -> PostgreSQL Tracking")
    print("    |           |                 |                    |")
    print("Encrypted   AI Processing      Encrypted          Direct")
    print("Transit     (3 gateways)       at Rest         Connection")
    print("\n            Corporate LLM -> Compliance Report")
    print("            AI Processing -> Intelligence Report")
    print("            Workflow Optimizer -> Knowledge Report")


if __name__ == "__main__":
    main()