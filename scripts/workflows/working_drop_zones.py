#!/usr/bin/env python3
"""
Working Drop Zones for Research Peer Review
Uses existing TidyLLM systems, bypasses TidyDSPy import issues
"""

import os
import sys
import time
import yaml
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add tidyllm to path
sys.path.append('tidyllm')

from tidyllm.compliance import ModelRiskMonitor
from tidyllm.vectorqa.whitepapers.embeddings_enhanced_qa import EmbeddingsEnhancedQA
from tidyllm.core import tidyllm

class ResearchDropZoneHandler(FileSystemEventHandler):
    """Handle files dropped in research peer review zone."""
    
    def __init__(self):
        self.llm = None
        self.embeddings_qa = None
        self.compliance_monitor = None
        
    def initialize(self):
        """Initialize components."""
        try:
            print("🔄 Initializing drop zone components...")
            self.llm = tidyllm()
            self.embeddings_qa = EmbeddingsEnhancedQA()
            self.compliance_monitor = ModelRiskMonitor()
            print("✅ Drop zone components initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def on_created(self, event):
        """Process files when they are dropped in the zone."""
        if event.is_directory:
            return
            
        file_path = event.src_path
        print(f"\n🔔 NEW FILE DETECTED: {file_path}")
        
        # Only process PDFs and text files
        if not file_path.lower().endswith(('.pdf', '.txt', '.docx')):
            print(f"⏭️  Skipping unsupported file type")
            return
            
        # Wait a moment for file to be fully written
        time.sleep(2)
        
        try:
            self.process_research_paper(file_path)
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    def process_research_paper(self, file_path: str):
        """Process a research paper through peer review workflow."""
        print(f"🔄 Starting peer review workflow for: {Path(file_path).name}")
        
        # Initialize if needed
        if not self.llm and not self.initialize():
            print("❌ Failed to initialize - aborting")
            return
        
        # Step 1: Extract text
        print("📄 Step 1: Extracting text...")
        text = self.extract_text(file_path)
        if not text:
            print("❌ Text extraction failed")
            return
        print(f"✅ Extracted {len(text)} characters")
        
        # Step 2: Generate embeddings
        print("🧠 Step 2: Generating embeddings...")
        embeddings_success = self.generate_embeddings(text)
        print(f"{'✅' if embeddings_success else '⚠️'} Embeddings: {'Generated' if embeddings_success else 'Failed'}")
        
        # Step 3: Compliance analysis
        print("📋 Step 3: Running compliance analysis...")
        compliance_result = self.run_compliance_check(text)
        print(f"✅ Compliance score: {compliance_result.get('overall_score', 'N/A')}")
        
        # Step 4: AI peer review
        print("🤖 Step 4: Generating AI peer review...")
        peer_review = self.generate_peer_review(text)
        print(f"✅ Generated peer review ({len(peer_review)} chars)")
        
        # Step 5: Save results
        print("💾 Step 5: Saving results...")
        output_file = self.save_results(file_path, text, embeddings_success, compliance_result, peer_review)
        
        print(f"\n🎉 PEER REVIEW COMPLETE!")
        print(f"📊 Results saved to: {output_file}")
        print("=" * 60)
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from file."""
        try:
            if file_path.lower().endswith('.pdf'):
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                    return text
                except ImportError:
                    print("⚠️  PyPDF2 not available, reading as text")
                    
            # Fallback to text reading
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
                
        except Exception as e:
            print(f"❌ Text extraction error: {e}")
            return ""
    
    def generate_embeddings(self, text: str) -> bool:
        """Generate embeddings for the text."""
        try:
            embeddings = self.embeddings_qa.generate_embeddings([text])
            return embeddings is not None and len(embeddings) > 0
        except Exception as e:
            print(f"⚠️  Embeddings error: {e}")
            return False
    
    def run_compliance_check(self, text: str) -> dict:
        """Run compliance analysis on the text."""
        try:
            return self.compliance_monitor.validate_documentation(text)
        except Exception as e:
            print(f"⚠️  Compliance check error: {e}")
            return {"overall_score": "Error", "error": str(e)}
    
    def generate_peer_review(self, text: str) -> str:
        """Generate AI peer review."""
        try:
            prompt = f"""
RESEARCH PEER REVIEW ANALYSIS

Please provide a comprehensive peer review of this research methodology:

{text[:3000]}...

Focus on:
1. Methodology soundness and rigor
2. Regulatory compliance (SR 11-7 if applicable)  
3. Data quality and underlying assumptions
4. Model validation approach and completeness
5. Identified limitations and risk factors
6. Recommendations for improvement

Provide specific, actionable feedback in a professional peer review format.
"""
            return self.llm(prompt)
        except Exception as e:
            return f"AI peer review error: {e}"
    
    def save_results(self, file_path: str, text: str, embeddings_success: bool, 
                    compliance_result: dict, peer_review: str) -> str:
        """Save all results to output file in results drop zone."""
        # Create results drop zone
        results_dir = Path('./drop_zones/results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        stem = Path(file_path).stem
        output_file = results_dir / f"peer_review_{stem}_{int(time.time())}.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("RESEARCH PAPER PEER REVIEW RESULTS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Original File: {file_path}\n")
                f.write(f"Processed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Text Length: {len(text)} characters\n")
                f.write(f"Embeddings Generated: {'Yes' if embeddings_success else 'No'}\n")
                f.write(f"Compliance Score: {compliance_result.get('overall_score', 'N/A')}\n")
                f.write("\n" + "=" * 50 + "\n\n")
                
                f.write("COMPLIANCE ANALYSIS RESULTS:\n")
                f.write("-" * 30 + "\n")
                for key, value in compliance_result.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("AI PEER REVIEW:\n")
                f.write("-" * 30 + "\n")
                f.write(peer_review)
                f.write("\n\n")
                
                f.write("EXTRACTED TEXT (First 2000 chars):\n")
                f.write("-" * 30 + "\n")
                f.write(text[:2000])
                if len(text) > 2000:
                    f.write("\n[... truncated ...]\n")
                    
            return str(output_file)
        except Exception as e:
            print(f"❌ Save error: {e}")
            return f"./drop_zones/results/save_error_{int(time.time())}.txt"

def create_drop_zone_config():
    """Create drop zone configuration."""
    config = {
        'research_peer_review': {
            'name': 'Research Peer Review Drop Zone',
            'directory': './drop_zones/research_peer_review',
            'patterns': ['*.pdf', '*.txt', '*.docx'],
            'description': 'Drop research papers here for automated peer review'
        },
        'results': {
            'name': 'Results Drop Zone',
            'directory': './drop_zones/results', 
            'patterns': ['*.txt'],
            'description': 'Peer review results appear here automatically'
        }
    }
    
    with open('drop_zone_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config

def setup_drop_zone_directories():
    """Setup the drop zone directories."""
    # Input drop zone
    input_dir = Path('./drop_zones/research_peer_review')
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Results drop zone  
    results_dir = Path('./drop_zones/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create README for input zone
    input_readme = input_dir / 'README.txt'
    with open(input_readme, 'w') as f:
        f.write("""
RESEARCH PEER REVIEW DROP ZONE - INPUT

Drop your research papers here (.pdf, .txt, .docx) and they will be
automatically processed through:

1. Text extraction
2. Embeddings generation  
3. Compliance analysis (SR 11-7)
4. AI peer review
5. Results saved to ../results/ folder

The system monitors this folder for new files and processes them immediately.
        """.strip())
    
    # Create README for results zone
    results_readme = results_dir / 'README.txt'
    with open(results_readme, 'w') as f:
        f.write("""
RESEARCH PEER REVIEW DROP ZONE - RESULTS

Peer review results appear here automatically when you drop files in
the research_peer_review folder.

Each result file contains:
- Compliance analysis (SR 11-7 standards)
- AI-generated peer review
- Embeddings status
- Complete audit trail
- Original text excerpt

Files named: peer_review_[original_filename]_[timestamp].txt
        """.strip())
    
    return input_dir, results_dir

def main():
    """Main drop zone monitoring."""
    print("🚀 TIDYLLM RESEARCH PEER REVIEW DROP ZONES")
    print("=" * 60)
    
    # Setup
    config = create_drop_zone_config()
    input_dir, results_dir = setup_drop_zone_directories()
    
    print(f"📁 Input drop zone: {input_dir}")
    print(f"📊 Results drop zone: {results_dir}")
    print(f"📋 Configuration saved to: drop_zone_config.yaml")
    
    # Initialize handler
    event_handler = ResearchDropZoneHandler()
    if not event_handler.initialize():
        print("❌ Failed to initialize - exiting")
        return
    
    # Setup file monitoring
    observer = Observer()
    observer.schedule(event_handler, str(input_dir), recursive=False)
    observer.start()
    
    print(f"\n👀 Monitoring {input_dir} for new files...")
    print("📝 Supported formats: PDF, TXT, DOCX")
    print(f"📊 Results will appear in: {results_dir}")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping drop zone monitoring...")
        observer.stop()
    
    observer.join()
    print("✅ Drop zone monitoring stopped")

if __name__ == "__main__":
    main()