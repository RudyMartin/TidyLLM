#!/usr/bin/env python3
"""
Simple Research Paper Peer Review with Embeddings
Working version for tomorrow - no bells and whistles

Usage: python simple_research_peer_review.py paper.pdf
"""

import sys
import os
from pathlib import Path

# Add tidyllm to path
sys.path.append('tidyllm')

from tidyllm.compliance import ModelRiskMonitor
from tidyllm.vectorqa.whitepapers.embeddings_enhanced_qa import EmbeddingsEnhancedQA
from tidyllm.core import tidyllm, LLMConnector

def research_peer_review_flow(pdf_path: str):
    """Simple research paper peer review with embeddings"""
    
    print(f"🔄 RESEARCH PEER REVIEW FLOW")
    print(f"📄 Processing: {pdf_path}")
    print("=" * 60)
    
    # Step 1: Extract text from PDF (simple version)
    print("1. Extracting text from PDF...")
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        print(f"   SUCCESS: Extracted {len(text)} characters")
    except ImportError:
        # Fallback - just read if it's a text file
        with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        print(f"   FALLBACK: Read {len(text)} characters as text")
    
    # Step 2: Generate embeddings
    print("\n2. Generating embeddings...")
    try:
        embeddings_qa = EmbeddingsEnhancedQA()
        embeddings = embeddings_qa.generate_embeddings([text])
        print(f"   SUCCESS: Generated embeddings vector of size {len(embeddings[0])}")
    except Exception as e:
        print(f"   ERROR: Embeddings failed: {e}")
        embeddings = None
    
    # Step 3: Compliance peer review analysis
    print("\n3. Running compliance peer review...")
    monitor = ModelRiskMonitor()
    
    # Simple peer review check
    peer_review_result = monitor.validate_documentation(text)
    print(f"   SUCCESS: Compliance score: {peer_review_result.get('overall_score', 'N/A')}")
    
    # Step 4: Generate peer review using LLM
    print("\n4. Generating AI peer review...")
    try:
        llm = tidyllm()
        
        prompt = f"""
        PEER REVIEW ANALYSIS
        
        Please provide a peer review of this research methodology:
        
        {text[:2000]}...
        
        Focus on:
        1. Methodology soundness
        2. Regulatory compliance (SR 11-7 if applicable)
        3. Data quality and assumptions
        4. Model validation approach
        5. Limitations and risks
        
        Provide specific feedback and recommendations.
        """
        
        review = llm(prompt)
        print(f"   SUCCESS: Generated {len(review)} character peer review")
        
    except Exception as e:
        print(f"   ERROR: AI peer review failed: {e}")
        review = "AI peer review unavailable"
    
    # Step 5: Save results
    print("\n5. Saving results...")
    output_file = f"peer_review_results_{Path(pdf_path).stem}.txt"
    
    with open(output_file, 'w') as f:
        f.write("RESEARCH PAPER PEER REVIEW RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Paper: {pdf_path}\n")
        f.write(f"Embeddings: {'Generated' if embeddings else 'Failed'}\n")
        f.write(f"Compliance Score: {peer_review_result.get('overall_score', 'N/A')}\n\n")
        f.write("COMPLIANCE ANALYSIS:\n")
        f.write(str(peer_review_result) + "\n\n")
        f.write("AI PEER REVIEW:\n")
        f.write(review + "\n")
    
    print(f"   SUCCESS: Results saved to {output_file}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 PEER REVIEW COMPLETE!")
    print(f"📊 Results: {output_file}")
    print(f"🔍 Embeddings: {'✅' if embeddings else '❌'}")
    print(f"📋 Compliance: {peer_review_result.get('overall_score', 'N/A')}")
    print("=" * 60)
    
    return {
        'pdf_path': pdf_path,
        'embeddings_generated': embeddings is not None,
        'compliance_score': peer_review_result.get('overall_score'),
        'output_file': output_file,
        'status': 'completed'
    }

def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python simple_research_peer_review.py paper.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    try:
        result = research_peer_review_flow(pdf_path)
        sys.exit(0 if result['status'] == 'completed' else 1)
    except Exception as e:
        print(f"ERROR: Flow failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()