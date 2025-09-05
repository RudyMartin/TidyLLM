#!/usr/bin/env python
"""

# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod

Show MVR Processing Flow - Complete Code Walkthrough
===================================================

This shows the actual code that:
1. Pulls markdown query from S3
2. Sends it to DSPy/AI gateway  
3. Gets final response and saves back to S3
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def show_current_processing_flow():
    """Show the current S3-first processing flow"""
    
    print("=" * 80)
    print("CURRENT MVR PROCESSING FLOW - CODE WALKTHROUGH")
    print("=" * 80)
    
    print("\n[STEP 1] MARKDOWN QUERY EXTRACTION FROM S3")
    print("-" * 50)
    print("File: scripts/s3_first_mvr_processor.py")
    print("Method: _stream_document_from_s3()")
    print()
    
    extraction_code = '''
def _stream_document_from_s3(self, bucket: str, key: str) -> str:
    """Stream document content from S3 (NO local files)"""
    
    try:
        # Get S3 client from UnifiedSessionManager
        s3_client = self.session_mgr.get_s3_client()
        
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
'''
    
    print(extraction_code)
    
    print("\n[STEP 2] CURRENT PROCESSING (WITHOUT GATEWAY)")
    print("-" * 50)
    print("File: scripts/s3_first_mvr_processor.py")
    print("Method: _process_compliance_analysis()")
    print()
    
    current_processing = '''
def _process_compliance_analysis(self, mvr_content: str, prompt_content: str) -> dict:
    """Process MVR with JB_Overview_Prompt logic (in memory)"""
    
    # Parse MVR sections (pure Python, no external libs)
    sections = self._parse_mvr_sections_native(mvr_content)
    
    compliance_report = {
        "report_type": "mvr_compliance_validation",
        "prompt_used": "JB_Overview_Prompt.md",
        "timestamp": datetime.now().isoformat(),
        "total_sections": len(sections),
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
            "conclusion": self._determine_compliance_status(section),
            "confidence_score": "Highly Confident"
        }
        
        compliance_report["sections"].append(section_analysis)
    
    return compliance_report
'''
    
    print(current_processing)
    
    print("\n[ISSUE] Current code does NOT use DSPy/AI Gateway!")
    print("It uses simple Python rules instead of AI processing.")
    
    print("\n" + "=" * 80)
    print("ENHANCED VERSION WITH DSPY GATEWAY INTEGRATION")
    print("=" * 80)


def show_enhanced_gateway_flow():
    """Show enhanced version with actual DSPy gateway integration"""
    
    print("\n[ENHANCED STEP 2] WITH DSPY GATEWAY INTEGRATION")
    print("-" * 50)
    print("File: Enhanced version of s3_first_mvr_processor.py")
    print()
    
    enhanced_code = '''
def _process_compliance_analysis_with_gateway(self, mvr_content: str, prompt_content: str) -> dict:
    """Process MVR using DSPy/AI Gateway (ENHANCED VERSION)"""
    
    from tidyllm.gateways import get_gateway, AIRequest
    
    # Get AI processing gateway
    ai_gateway = get_gateway("ai_processing")
    if not ai_gateway:
        raise Exception("AI processing gateway not available")
    
    # Parse MVR sections
    sections = self._parse_mvr_sections_native(mvr_content)
    
    compliance_report = {
        "report_type": "mvr_compliance_validation_ai",
        "prompt_used": "JB_Overview_Prompt.md",
        "gateway_used": "ai_processing",
        "timestamp": datetime.now().isoformat(),
        "sections": []
    }
    
    # Process each section through AI gateway
    for section in sections:
        
        # Construct AI request with JB_Overview_Prompt logic
        ai_prompt = f\"\"\"
{prompt_content}

DOCUMENT SECTION TO ANALYZE:
Section ID: {section["id"]}
Title: {section["title"]}
Content: {section["text"][:2000]}...

Please analyze this section for:
1. MVS requirement compliance
2. VST section alignment  
3. Evidence quality
4. Logic gaps or contradictions
5. Peer review challenges
6. Confidence assessment

Return structured JSON response.
\"\"\"

        # Send to AI Gateway
        ai_request = AIRequest(
            prompt=ai_prompt,
            model="gpt-4",
            temperature=0.1,
            max_tokens=1000
        )
        
        try:
            # Process through DSPy/AI gateway
            ai_response = ai_gateway.process(ai_request)
            
            if ai_response.success:
                # Parse AI response
                ai_analysis = json.loads(ai_response.content)
                
                section_analysis = {
                    "section_id": section["id"],
                    "section_title": section["title"],
                    "ai_analysis": ai_analysis,
                    "mvs_requirements": ai_analysis.get("mvs_requirements", []),
                    "vst_sections": ai_analysis.get("vst_sections", []),
                    "review_narrative": ai_analysis.get("review_narrative", ""),
                    "contradiction_summary": ai_analysis.get("contradictions", "None"),
                    "peer_review_challenge": ai_analysis.get("peer_challenge", ""),
                    "conclusion": ai_analysis.get("conclusion", "INCONCLUSIVE"),
                    "confidence_score": ai_analysis.get("confidence", "Unknown"),
                    "ai_processing_time": ai_response.processing_time
                }
            else:
                # Fallback to rule-based analysis
                section_analysis = self._fallback_rule_based_analysis(section)
                section_analysis["ai_error"] = ai_response.error
                
        except Exception as e:
            # Fallback to rule-based analysis  
            section_analysis = self._fallback_rule_based_analysis(section)
            section_analysis["ai_error"] = str(e)
        
        compliance_report["sections"].append(section_analysis)
    
    return compliance_report
'''
    
    print(enhanced_code)


def show_gateway_initialization():
    """Show how gateways are initialized"""
    
    print("\n[STEP 0] GATEWAY INITIALIZATION")
    print("-" * 50)
    print("File: scripts/s3_first_mvr_processor.py (Enhanced)")
    print()
    
    init_code = '''
class S3FirstMVRProcessor:
    def __init__(self):
        # Use UnifiedSessionManager
        self.session_mgr = UnifiedSessionManager()
        
        # Initialize gateways
        from tidyllm.gateways import init_gateways
        
        self.gateways = init_gateways({
            "corporate_llm": {
                "budget_limit_daily_usd": 100,
                "tracking_enabled": True
            },
            "ai_processing": {
                "backend": "anthropic",  # or "openai", "dspy"
                "model": "claude-3-sonnet",
                "temperature": 0.1
            },
            "workflow_optimizer": {
                "optimization_level": "high"
            }
        })
        
        print(f"[GATEWAYS] Initialized: {self.gateways.get_available_services()}")
'''
    
    print(init_code)


def show_s3_save_flow():
    """Show how results are saved back to S3"""
    
    print("\n[STEP 3] SAVE RESULTS TO S3")
    print("-" * 50) 
    print("File: scripts/s3_first_mvr_processor.py")
    print("Method: _save_results_to_s3()")
    print()
    
    save_code = '''
def _save_results_to_s3(self, bucket: str, process_id: str, 
                       compliance_report: dict, embeddings: list) -> dict:
    """Save all results directly to S3 (NO local files)"""
    
    s3_client = self.session_mgr.get_s3_client()
    result_keys = {}
    
    try:
        # Save compliance report (with AI analysis)
        compliance_key = fbuild_s3_path("mvr_analysis", "reports/compliance/{process_id}_compliance.json")
        s3_client.put_object(
            Bucket=bucket,
            Key=compliance_key,
            Body=json.dumps(compliance_report, indent=2).encode(),
            ServerSideEncryption='AES256'
        )
        result_keys["compliance"] = compliance_key
        
        # Generate PDF report from JSON
        pdf_key = fbuild_s3_path("mvr_analysis", "reports/compliance/{process_id}_report.pdf")
        pdf_content = self._generate_pdf_report(compliance_report)
        s3_client.put_object(
            Bucket=bucket,
            Key=pdf_key,
            Body=pdf_content,
            ServerSideEncryption='AES256',
            ContentType='application/pdf'
        )
        result_keys["compliance_pdf"] = pdf_key
        
        print(f"   → Saved compliance JSON: {compliance_key}")
        print(f"   → Saved compliance PDF: {pdf_key}")
        
        return result_keys
        
    except Exception as e:
        raise Exception(f"Failed to save results to S3: {e}")
'''
    
    print(save_code)


def show_complete_flow_summary():
    """Show complete processing flow summary"""
    
    print("\n" + "=" * 80)
    print("COMPLETE PROCESSING FLOW SUMMARY")
    print("=" * 80)
    
    flow_steps = [
        "1. Monitor detects new file in s3://nsc-mvp1/mvr_analysis/raw/",
        "2. _stream_document_from_s3() pulls MVR content from S3",
        "3. _stream_document_from_s3() pulls JB_Overview_Prompt.md from S3", 
        "4. _process_compliance_analysis() sends prompt + content to AI Gateway",
        "5. AI Gateway processes through DSPy/Claude/GPT",
        "6. AI returns structured compliance analysis JSON",
        "7. _save_results_to_s3() saves JSON + PDF reports to S3",
        "8. _track_processing_postgresql() logs to MLflow database",
        "9. User downloads reports with: aws_terminal_mvr_workflow.py download"
    ]
    
    print("\nProcessing Flow:")
    for step in flow_steps:
        print(f"   {step}")
    
    print(f"\nData Flow:")
    print(f"   S3 Query → AI Gateway → S3 Results")
    print(f"      ↓           ↓            ↓")
    print(f"   Markdown   DSPy/Claude   JSON/PDF")
    print(f"   Content    Processing    Reports")


def main():
    """Show complete MVR processing code walkthrough"""
    
    show_current_processing_flow()
    show_enhanced_gateway_flow() 
    show_gateway_initialization()
    show_s3_save_flow()
    show_complete_flow_summary()
    
    print(f"\n" + "=" * 80)
    print("CURRENT STATUS")  
    print("=" * 80)
    print(f"✅ S3 streaming (markdown query extraction) - IMPLEMENTED")
    print(f"❌ DSPy/AI Gateway integration - NOT YET IMPLEMENTED")
    print(f"✅ S3 result saving - IMPLEMENTED") 
    print(f"✅ PostgreSQL tracking - IMPLEMENTED")
    
    print(f"\n📋 TO COMPLETE AI GATEWAY INTEGRATION:")
    print(f"   1. Modify s3_first_mvr_processor.py to use AI gateways")
    print(f"   2. Replace rule-based analysis with AI Gateway calls")
    print(f"   3. Add structured prompt engineering for JB_Overview_Prompt")
    print(f"   4. Test end-to-end S3 → AI Gateway → S3 flow")


if __name__ == "__main__":
    main()