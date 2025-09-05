#!/usr/bin/env python3
"""
Simple PDF RAG Test

This script tests the PDF RAG functionality with a simple text prompt
and tracks activity in MLflow for monitoring.
"""

import sys
import os
from pathlib import Path
import logging
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplePDFRAGTest:
    """Simple PDF RAG test with MLflow tracking"""
    
    def __init__(self):
        self.test_document = Path("input/tests/test_document.pdf")
        self.mlflow_experiment_name = "pdf_rag_test"
        
    def test_pdf_rag_with_prompt(self):
        """Test PDF RAG with a simple text prompt"""
        print("\n" + "="*80)
        print("🎯 SIMPLE PDF RAG TEST")
        print("="*80)
        
        if not self.test_document.exists():
            print("❌ Test document not found")
            return False
        
        try:
            # Initialize RAG QA Orchestrator
            from backend.mcp.orchestrators.rag_qa_orchestrator import RAGQAOrchestrator
            
            print("🔄 Initializing RAG QA Orchestrator...")
            rag_orchestrator = RAGQAOrchestrator()
            
            # Prepare test document
            files = [{
                'filename': self.test_document.name,
                'content': 'Sample document content for RAG processing',
                'size': 1024
            }]
            
            # Test prompt
            test_prompt = "Provide summary of the main topics and key findings in this document"
            
            print(f"📄 Processing document: {self.test_document.name}")
            print(f"🔍 Test prompt: {test_prompt}")
            
            # Process with RAG
            start_time = time.time()
            rag_result = rag_orchestrator.process_whitepaper_rag(files)
            processing_time = time.time() - start_time
            
            print(f"✅ RAG processing completed in {processing_time:.2f}s")
            print(f"📊 Status: {rag_result.get('status', 'unknown')}")
            print(f"📄 Documents processed: {rag_result.get('documents_processed', 0)}")
            print(f"🔍 Chunks created: {rag_result.get('chunks_created', 0)}")
            
            # Track in MLflow
            self.track_mlflow_activity(rag_result, processing_time, test_prompt)
            
            return True
            
        except Exception as e:
            print(f"❌ PDF RAG test failed: {e}")
            return False
    
    def track_mlflow_activity(self, rag_result, processing_time, prompt):
        """Track activity in MLflow"""
        try:
            import mlflow
            from backend.llm.unified_llm_gateway import UnifiedLLMGateway
            
            # Set up MLflow experiment
            mlflow.set_experiment(self.mlflow_experiment_name)
            
            with mlflow.start_run(run_name=f"pdf_rag_test_{int(time.time())}"):
                # Log parameters
                mlflow.log_param("test_type", "pdf_rag")
                mlflow.log_param("document", self.test_document.name)
                mlflow.log_param("prompt", prompt)
                mlflow.log_param("processing_time", processing_time)
                
                # Log metrics
                mlflow.log_metric("processing_time_seconds", processing_time)
                mlflow.log_metric("documents_processed", rag_result.get('documents_processed', 0))
                mlflow.log_metric("chunks_created", rag_result.get('chunks_created', 0))
                mlflow.log_metric("success", 1 if rag_result.get('status') == 'completed' else 0)
                
                # Log results
                mlflow.log_dict(rag_result, "rag_results.json")
                
                # Log model info
                mlflow.log_param("embedding_model", "all-mpnet-base-v2")
                mlflow.log_param("database_used", rag_result.get('database_used', False))
                
                print("✅ MLflow tracking completed")
                
                # Get MLflow tracking URI
                tracking_uri = mlflow.get_tracking_uri()
                print(f"📊 MLflow Tracking URI: {tracking_uri}")
                
                # Get experiment info
                experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
                if experiment:
                    print(f"📈 MLflow Experiment ID: {experiment.experiment_id}")
                    print(f"📈 MLflow Experiment Name: {experiment.name}")
                
        except Exception as e:
            print(f"⚠️ MLflow tracking failed: {e}")
    
    def test_unified_llm_gateway(self):
        """Test Unified LLM Gateway with simple prompt"""
        print("\n" + "="*80)
        print("🎯 UNIFIED LLM GATEWAY TEST")
        print("="*80)
        
        try:
            from backend.llm.unified_llm_gateway import UnifiedLLMGateway
            
            # Initialize gateway
            print("🔄 Initializing Unified LLM Gateway...")
            gateway = UnifiedLLMGateway(
                experiment_name="pdf_rag_gateway_test",
                enable_tracking=True
            )
            
            # Test prompt
            test_prompt = "Provide a brief summary of machine learning concepts"
            
            print(f"🔍 Test prompt: {test_prompt}")
            
            # Call LLM
            start_time = time.time()
            response = gateway.call_llm(
                agent_name="document_qa",
                task_type="question_answering",
                prompt=test_prompt
            )
            processing_time = time.time() - start_time
            
            print(f"✅ LLM Gateway response received in {processing_time:.2f}s")
            print(f"🤖 Response: {response.content[:200]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ LLM Gateway test failed: {e}")
            return False
    
    def run_complete_test(self):
        """Run the complete PDF RAG test"""
        print("🚀 SIMPLE PDF RAG TEST")
        print("="*80)
        print("Testing PDF RAG functionality with MLflow tracking")
        print("="*80)
        
        # Test 1: PDF RAG
        pdf_rag_success = self.test_pdf_rag_with_prompt()
        
        # Test 2: LLM Gateway
        llm_gateway_success = self.test_unified_llm_gateway()
        
        # Summary
        print("\n" + "="*80)
        print("🎯 TEST SUMMARY")
        print("="*80)
        print(f"✅ PDF RAG Test: {'PASSED' if pdf_rag_success else 'FAILED'}")
        print(f"✅ LLM Gateway Test: {'PASSED' if llm_gateway_success else 'FAILED'}")
        
        if pdf_rag_success and llm_gateway_success:
            print("🎉 All tests passed!")
            print("📊 Check MLflow for detailed activity tracking")
        else:
            print("⚠️ Some tests failed")
        
        return pdf_rag_success and llm_gateway_success

def main():
    """Main function"""
    test = SimplePDFRAGTest()
    success = test.run_complete_test()
    
    if success:
        print("\n🎉 PDF RAG test completed successfully!")
        print("📊 MLflow activity tracking enabled")
    else:
        print("\n⚠️ Test completed with issues")

if __name__ == "__main__":
    main()
