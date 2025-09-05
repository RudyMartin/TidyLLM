"""
RAG2DAG Drop Zone Interface
===========================

Automatic workflow triggering when files and queries are dropped into folders.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from .converter import RAG2DAGConverter
from .config import RAG2DAGConfig


class RAG2DAGDropZoneHandler(FileSystemEventHandler):
    """File system event handler for RAG2DAG drop zones."""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = Path(workspace_path)
        self.documents_path = self.workspace_path / "documents"
        self.queries_path = self.workspace_path / "queries"
        self.results_path = self.workspace_path / "results"
        
        # Ensure directories exist
        for path in [self.documents_path, self.queries_path, self.results_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Create README files for guidance
        self._create_readme_files()
        
        # Configuration
        self.config = RAG2DAGConfig.create_default_config()
        self.converter = RAG2DAGConverter(self.config)
        
        # Workflow tracking
        self.active_workflows: Dict[str, Dict] = {}
        
        print(f"RAG2DAG Drop Zone initialized at: {workspace_path}")
        print(f"  📁 Drop documents in: {self.documents_path}")
        print(f"  ❓ Drop queries in: {self.queries_path}")
        print(f"  📊 Results appear in: {self.results_path}")
    
    def _create_readme_files(self):
        """Create helpful README files in drop zone directories."""
        
        # Documents README
        doc_readme = self.documents_path / "README.md"
        if not doc_readme.exists():
            doc_readme.write_text("""# Documents Folder

Drop your files here for RAG2DAG analysis:

## Supported File Types
- PDF documents (*.pdf)
- Word documents (*.docx, *.doc) 
- Text files (*.txt)
- Markdown files (*.md)

## Tips
- Drop 1-10 files for optimal processing
- Related documents work best together
- Clear file names help with analysis

## What Happens Next
1. Drop your files here
2. Drop a query in the queries/ folder
3. RAG2DAG automatically creates optimized workflow
4. Results appear in results/ folder

---
*RAG2DAG - Intelligent Document Analysis*
""")

        # Queries README
        query_readme = self.queries_path / "README.md"
        if not query_readme.exists():
            query_readme.write_text("""# Queries Folder

Drop text files with your questions here:

## How to Create Queries
Create a `.txt` file with your question, like:

**research_question.txt:**
```
What are the main findings across these research papers?
```

**compliance_check.txt:**
```
Extract all compliance requirements and deadlines
```

**methodology_comparison.txt:**
```
Compare the methodologies used in these studies
```

## Query Tips
- Be specific about what you want
- Use action words: "extract", "compare", "summarize", "analyze"
- Mention output format if needed: "create a table", "list the findings"

## Automatic Processing
- When documents + query are both present → workflow starts
- Progress shown in results/workflow_status.json
- Final results in results/final_report.md

---
*RAG2DAG - Intelligent Document Analysis*
""")

        # Results README (initial)
        results_readme = self.results_path / "README.md"
        if not results_readme.exists():
            results_readme.write_text("""# Results Folder

Your RAG2DAG analysis results will appear here:

## File Structure
When a workflow runs, you'll see:

```
results/
├── workflow_YYYYMMDD_HHMMSS/
│   ├── workflow.json          # Complete workflow configuration
│   ├── workflow_status.json   # Real-time progress updates
│   ├── extracted_content/     # Parallel extraction results
│   │   ├── facts.json
│   │   ├── quotes.json
│   │   └── methodology.json
│   ├── synthesis.json         # Combined analysis
│   └── final_report.md        # Generated summary report
└── README.md                  # This file
```

## Monitoring Progress
- Check `workflow_status.json` for real-time updates
- Status values: "starting", "processing", "completed", "failed"
- Progress shows which nodes are complete/in-progress

## Getting Results
- **Quick Answer**: Read `final_report.md`
- **Detailed Data**: Check JSON files in `extracted_content/`
- **Full Workflow**: Review `workflow.json`

---
*RAG2DAG - Intelligent Document Analysis*
""")
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Ignore hidden files, temp files, and README files
        if file_path.name.startswith('.') or file_path.name.startswith('~'):
            return
        if file_path.name == "README.md":
            return
            
        print(f"📁 File detected: {file_path.name}")
        
        # Give the file a moment to be fully written
        time.sleep(1)
        
        # Check if we can trigger a workflow
        self._check_and_trigger_workflow()
    
    def _check_and_trigger_workflow(self):
        """Check if we have both documents and queries to trigger workflow."""
        
        # Get document files
        doc_files = self._get_files_in_directory(self.documents_path)
        if not doc_files:
            return
        
        # Get query files
        query_files = self._get_files_in_directory(self.queries_path)
        if not query_files:
            return
        
        print(f"🔍 Conditions met: {len(doc_files)} documents, {len(query_files)} queries")
        
        # Use the first query file (or combine multiple queries)
        query = self._read_query_file(query_files[0])
        if not query:
            print("❌ Could not read query file")
            return
        
        # Trigger workflow
        try:
            workflow_id = self._create_and_run_workflow(query, doc_files)
            print(f"🚀 Workflow started: {workflow_id}")
            
            # Move processed files to avoid re-triggering
            self._archive_processed_files(doc_files, query_files, workflow_id)
            
        except Exception as e:
            print(f"❌ Error creating workflow: {e}")
            self._write_error_status(str(e))
    
    def _get_files_in_directory(self, directory: Path) -> List[Path]:
        """Get processable files from directory."""
        if not directory.exists():
            return []
        
        supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md'}
        files = []
        
        for file_path in directory.iterdir():
            if (file_path.is_file() and 
                file_path.suffix.lower() in supported_extensions and
                file_path.name != "README.md"):
                files.append(file_path)
        
        return files
    
    def _read_query_file(self, query_file: Path) -> Optional[str]:
        """Read query from text file."""
        try:
            with open(query_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Handle different query formats
            if query_file.suffix.lower() == '.json':
                # JSON format: {"query": "...", "context": {...}}
                data = json.loads(content)
                return data.get('query', content)
            else:
                # Plain text
                return content
            
        except Exception as e:
            print(f"Error reading query file {query_file}: {e}")
            return None
    
    def _create_and_run_workflow(self, query: str, doc_files: List[Path]) -> str:
        """Create and start RAG2DAG workflow."""
        
        # Convert paths to strings
        file_paths = [str(f) for f in doc_files]
        
        print(f"📋 Creating workflow...")
        print(f"   Query: \"{query[:50]}{'...' if len(query) > 50 else ''}\"")
        print(f"   Files: {len(file_paths)} documents")
        
        # Generate workflow
        workflow = self.converter.create_workflow_from_query(query, file_paths)
        workflow_id = workflow['workflow_id']
        
        # Create results directory
        workflow_dir = self.results_path / workflow_id
        workflow_dir.mkdir(exist_ok=True)
        
        # Save workflow configuration
        workflow_file = workflow_dir / "workflow.json"
        with open(workflow_file, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        # Create initial status
        status = {
            "workflow_id": workflow_id,
            "status": "starting",
            "pattern": workflow['pattern_name'],
            "complexity_score": workflow['complexity_score'],
            "estimated_cost": workflow['estimated_cost_factor'] * 1.50,
            "created_at": workflow['workflow_id'].split('_')[1:],  # Extract timestamp
            "progress": {node['node_id']: "pending" for node in workflow['dag_nodes']},
            "estimated_completion": None,
            "error": None
        }
        
        # Save status
        status_file = workflow_dir / "workflow_status.json"
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        # Also save to main results directory for easy access
        main_status_file = self.results_path / "workflow_status.json"
        with open(main_status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        print(f"✅ Workflow created: {workflow['pattern_name']}")
        print(f"   Complexity: {workflow['complexity_score']}/10")
        print(f"   Nodes: {len(workflow['dag_nodes'])}")
        print(f"   Estimated cost: ${status['estimated_cost']:.2f}")
        print(f"   Results in: {workflow_dir}")
        
        # Start workflow execution (placeholder)
        self._simulate_workflow_execution(workflow, workflow_dir)
        
        return workflow_id
    
    def _simulate_workflow_execution(self, workflow: Dict[str, Any], workflow_dir: Path):
        """Simulate workflow execution for demo (replace with actual execution)."""
        import threading
        import random
        
        def execute_workflow():
            workflow_id = workflow['workflow_id']
            status_file = workflow_dir / "workflow_status.json"
            
            # Update status to processing
            status = json.loads(status_file.read_text())
            status['status'] = 'processing'
            status_file.write_text(json.dumps(status, indent=2))
            
            # Simulate node execution
            total_nodes = len(workflow['dag_nodes'])
            for i, node in enumerate(workflow['dag_nodes']):
                # Simulate processing time
                time.sleep(random.uniform(2, 8))
                
                # Update progress
                status['progress'][node['node_id']] = 'completed'
                
                # Create mock output
                extracted_dir = workflow_dir / "extracted_content"
                extracted_dir.mkdir(exist_ok=True)
                
                if node['operation'] == 'extract':
                    output_file = extracted_dir / f"{node['node_id']}.json"
                    mock_data = {
                        "node_id": node['node_id'],
                        "operation": node['operation'],
                        "instruction": node['instruction'],
                        "extracted_items": [
                            f"Mock extracted item {j+1} from {node['operation']}"
                            for j in range(random.randint(3, 12))
                        ],
                        "confidence_score": random.uniform(0.7, 0.95),
                        "processing_time_seconds": random.uniform(2, 8)
                    }
                    output_file.write_text(json.dumps(mock_data, indent=2))
                
                # Update status
                progress_pct = int((i + 1) / total_nodes * 100)
                status['progress_percentage'] = progress_pct
                status_file.write_text(json.dumps(status, indent=2))
                
                print(f"   ✅ {node['operation']}: {node['node_id']} completed ({progress_pct}%)")
            
            # Generate final report
            final_report = workflow_dir / "final_report.md"
            report_content = f"""# RAG2DAG Analysis Report

**Workflow ID:** {workflow_id}
**Pattern:** {workflow['pattern_name']}
**Query:** {workflow['query']}
**Files Processed:** {len(workflow['files'])}

## Summary

This analysis used the {workflow['pattern_name']} pattern to process {len(workflow['files'])} documents.

## Key Findings

1. **Document Analysis**: Successfully extracted content from all {len(workflow['files'])} input documents
2. **Pattern Recognition**: {workflow['pattern_name']} pattern was automatically selected based on query analysis
3. **Processing Efficiency**: Workflow completed with {len(workflow['dag_nodes'])} processing nodes

## Extracted Content

The following content was extracted and analyzed:

{chr(10).join([f"- **{node['operation'].title()}**: {node['instruction']}" for node in workflow['dag_nodes']])}

## Technical Details

- **Complexity Score**: {workflow['complexity_score']}/10
- **Cost Factor**: {workflow['estimated_cost_factor']}x
- **Processing Time**: {workflow['execution_plan']['total_estimated_time_seconds']}s estimated
- **Parallel Efficiency**: {len([n for n in workflow['dag_nodes'] if n.get('parallel_group')])} parallel operations

## Files Analyzed

{chr(10).join([f"- {f}" for f in workflow['files']])}

---

*Generated by RAG2DAG - Intelligent Document Analysis*
*Report created: {workflow_id}*
"""
            
            final_report.write_text(report_content)
            
            # Final status update
            status['status'] = 'completed'
            status['progress_percentage'] = 100
            status['completed_at'] = workflow_id  # Use workflow ID timestamp
            status_file.write_text(json.dumps(status, indent=2))
            
            print(f"🎉 Workflow completed: {workflow_id}")
            print(f"   📄 Report: {final_report}")
            print(f"   📊 Extracted content: {extracted_dir}")
        
        # Start execution in background thread
        thread = threading.Thread(target=execute_workflow, daemon=True)
        thread.start()
    
    def _archive_processed_files(self, doc_files: List[Path], query_files: List[Path], workflow_id: str):
        """Move processed files to archive to avoid re-processing."""
        
        # Create archive directory
        archive_dir = self.workspace_path / "archive" / workflow_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Move documents
        doc_archive = archive_dir / "documents"
        doc_archive.mkdir(exist_ok=True)
        for doc_file in doc_files:
            doc_file.rename(doc_archive / doc_file.name)
        
        # Move queries
        query_archive = archive_dir / "queries"  
        query_archive.mkdir(exist_ok=True)
        for query_file in query_files:
            query_file.rename(query_archive / query_file.name)
        
        print(f"📦 Files archived to: {archive_dir}")
    
    def _write_error_status(self, error_message: str):
        """Write error status to results."""
        error_status = {
            "status": "error",
            "error": error_message,
            "timestamp": time.time()
        }
        
        error_file = self.results_path / "error_status.json"
        with open(error_file, 'w') as f:
            json.dump(error_status, f, indent=2)


class RAG2DAGDropZone:
    """Main RAG2DAG Drop Zone interface."""
    
    def __init__(self, workspace_path: str = "rag2dag_workspace"):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(exist_ok=True)
        
        self.handler = RAG2DAGDropZoneHandler(self.workspace_path)
        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.workspace_path), recursive=True)
    
    def start(self):
        """Start monitoring the drop zone."""
        print(f"Starting RAG2DAG Drop Zone at: {self.workspace_path}")
        print("=" * 50)
        print("📁 Drop your documents in: documents/")
        print("❓ Drop your questions in: queries/")
        print("📊 Results will appear in: results/")
        print("=" * 50)
        print("Press Ctrl+C to stop")
        
        self.observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping RAG2DAG Drop Zone...")
            self.observer.stop()
        
        self.observer.join()
    
    def stop(self):
        """Stop monitoring the drop zone."""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()


def main():
    """Run RAG2DAG Drop Zone interface."""
    import sys
    
    workspace = sys.argv[1] if len(sys.argv) > 1 else "rag2dag_workspace"
    
    drop_zone = RAG2DAGDropZone(workspace)
    drop_zone.start()


if __name__ == "__main__":
    main()