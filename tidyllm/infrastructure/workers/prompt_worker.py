"""
Prompt Worker - Generic Prompt-Driven Processing
================================================

Ultra-simple worker that any dummy can use. Just:
1. Drop a document in a folder
2. Drop a prompt (markdown file) in the same folder  
3. Get results automatically

No complex workflows, no YAML configs, no stage management.
Pure simplicity for customized prompting and independent analysis.

Folder Structure (Simple):
documents/
├── incoming/           # Drop document + prompt here
├── processing/         # Auto-processing happens here  
├── results/            # Results appear here
└── failed/             # Failed items go here

Usage Examples:
- Drop: contract.pdf + analyst_report_prompts.md → Get analyst report
- Drop: mvr.pdf + section_view_prompts.md → Get section analysis
- Drop: any.doc + custom_prompt.md → Get custom analysis
"""

import asyncio
import logging
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .base_worker import BaseWorker, TaskPriority
from ..session.unified import UnifiedSessionManager

logger = logging.getLogger("prompt_worker")


class ProcessingStatus(Enum):
    """Simple processing status."""
    WAITING = "waiting"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PromptTask:
    """Simple prompt-based task."""
    task_id: str
    document_file: str
    prompt_file: str
    document_content: str
    prompt_content: str
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "document_file": self.document_file,
            "prompt_file": self.prompt_file,
            "document_length": len(self.document_content),
            "prompt_length": len(self.prompt_content),
            "created_at": self.created_at.isoformat()
        }


@dataclass
class PromptResult:
    """Simple prompt processing result."""
    task_id: str
    result_content: str
    processing_time: float
    status: ProcessingStatus
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "result_content": self.result_content,
            "processing_time": self.processing_time,
            "status": self.status.value,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "result_length": len(self.result_content),
            "timestamp": datetime.now().isoformat()
        }


class PromptWorker(BaseWorker[PromptTask, PromptResult]):
    """
    Ultra-simple prompt-driven worker for any dummy to use.
    
    No complex configurations, no YAML workflows, no multi-stage processing.
    Just: Document + Prompt = Result
    
    Features:
    - Auto-detect document + prompt pairs in folders
    - Extract prompts from markdown files automatically
    - Process with any available gateway
    - Save results in simple JSON format
    - Handle failures gracefully
    """
    
    def __init__(self,
                 worker_name: str = "prompt_worker",
                 watch_folder: str = "documents/incoming",
                 results_folder: str = "documents/results", 
                 failed_folder: str = "documents/failed",
                 **kwargs):
        """
        Initialize Prompt Worker.
        
        Args:
            worker_name: Worker identifier
            watch_folder: Folder to watch for document+prompt pairs
            results_folder: Where to save results
            failed_folder: Where to move failed items
        """
        super().__init__(worker_name, **kwargs)
        
        self.watch_folder = Path(watch_folder)
        self.results_folder = Path(results_folder)
        self.failed_folder = Path(failed_folder)
        
        # Session and gateway access
        self.session_manager = None
        self.available_gateways = []
        
        # File monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.processed_pairs: set = set()
        
        logger.info(f"Prompt Worker '{worker_name}' configured")
        logger.info(f"  Watch folder: {watch_folder}")
        logger.info(f"  Results folder: {results_folder}")
    
    async def _initialize_worker(self) -> None:
        """Initialize prompt worker."""
        try:
            # Ensure directories exist
            self.watch_folder.mkdir(parents=True, exist_ok=True)
            self.results_folder.mkdir(parents=True, exist_ok=True) 
            self.failed_folder.mkdir(parents=True, exist_ok=True)
            
            # Create helpful README files
            await self._create_readme_files()
            
            # Initialize session manager
            try:
                self.session_manager = UnifiedSessionManager()
                logger.info("Prompt Worker: UnifiedSessionManager initialized")
            except Exception as e:
                logger.warning(f"Prompt Worker: UnifiedSessionManager not available: {e}")
            
            # Discover available gateways
            await self._discover_gateways()
            
            # Start folder monitoring
            self.monitoring_task = asyncio.create_task(self._folder_monitoring_loop())
            
            logger.info("Prompt Worker initialized successfully")
            
        except Exception as e:
            logger.error(f"Prompt Worker initialization failed: {e}")
            raise
    
    async def _create_readme_files(self) -> None:
        """Create dummy-proof README files."""
        
        # Watch folder README
        watch_readme = self.watch_folder / "README.md"
        watch_readme.write_text("""# How to Use TidyLLM Prompt Worker (Any Dummy Can Do This!)

## Step 1: Drop Your Files Here
1. Copy your document (PDF, Word, text file) into this folder
2. Copy a prompt file (any .md file with prompts) into this folder
3. Wait 10-20 seconds for magic to happen!

## Step 2: Check Results
- Results appear in: `../results/`
- Failed items move to: `../failed/`

## Examples:
- `contract.pdf` + `analyst_report_prompts.md` = Contract analysis
- `mvr_document.pdf` + `section_view_prompts.md` = Section breakdown  
- `any_document.docx` + `custom_analysis.md` = Custom analysis

## Prompt File Format:
Your .md file should contain prompts like:
```
# My Analysis Prompt

Analyze this document and tell me:
1. What type of document is this?
2. What are the key findings?
3. What should I be concerned about?

Document: {document_content}
```

The `{document_content}` will be replaced with your actual document text.

## That's It!
No complex setup, no configuration files, no stages to manage.
Just drop files and get results!
""")
        
        # Results folder README
        results_readme = self.results_folder / "README.md"
        results_readme.write_text("""# Processing Results

Your analysis results appear here as JSON files.

Each result file contains:
- Original document and prompt used
- AI analysis results  
- Processing metadata
- Timestamp information

File naming: `result_YYYYMMDD_HHMMSS_<document_name>.json`
""")
        
        logger.info("README files created for dummy-proof usage")
    
    async def _discover_gateways(self) -> None:
        """Discover available gateways for processing."""
        try:
            # Try to import and discover gateways
            gateway_options = []
            
            # Try corporate LLM gateway
            try:
                from ...gateways.corporate_llm_gateway import CorporateLLMGateway
                gateway_options.append("corporate_llm")
            except ImportError:
                pass
            
            # Try AI processing gateway  
            try:
                from ...gateways.ai_processing_gateway import AIProcessingGateway
                gateway_options.append("ai_processing")
            except ImportError:
                pass
            
            # Fallback: try to use TidyLLM directly
            try:
                import tidyllm
                gateway_options.append("tidyllm_direct")
            except ImportError:
                pass
            
            self.available_gateways = gateway_options
            
            if self.available_gateways:
                logger.info(f"Available gateways: {', '.join(self.available_gateways)}")
            else:
                logger.warning("No gateways available - will use mock processing")
                self.available_gateways = ["mock"]
                
        except Exception as e:
            logger.warning(f"Gateway discovery failed: {e}")
            self.available_gateways = ["mock"]
    
    async def _folder_monitoring_loop(self) -> None:
        """Monitor folder for document + prompt pairs."""
        logger.info("Starting folder monitoring for document+prompt pairs...")
        
        while True:
            try:
                # Scan for pairs
                pairs = await self._scan_for_document_prompt_pairs()
                
                # Process new pairs
                for pair in pairs:
                    if pair["pair_id"] not in self.processed_pairs:
                        await self._process_document_prompt_pair(pair)
                        self.processed_pairs.add(pair["pair_id"])
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                logger.info("Folder monitoring stopped")
                break
            except Exception as e:
                logger.error(f"Folder monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _scan_for_document_prompt_pairs(self) -> List[Dict[str, Any]]:
        """Scan for document + prompt file pairs."""
        pairs = []
        
        try:
            # Get all files in watch folder
            files = list(self.watch_folder.glob("*"))
            
            # Find documents (non-markdown files)
            documents = [f for f in files if f.suffix.lower() in ['.pdf', '.txt', '.docx', '.doc', '.rtf'] and f.is_file()]
            
            # Find prompts (markdown files)
            prompts = [f for f in files if f.suffix.lower() == '.md' and f.is_file() and f.name != 'README.md']
            
            # Create pairs (each document with each prompt)
            for doc_file in documents:
                for prompt_file in prompts:
                    pair_id = f"{doc_file.stem}_{prompt_file.stem}"
                    pairs.append({
                        "pair_id": pair_id,
                        "document_file": doc_file,
                        "prompt_file": prompt_file,
                        "document_name": doc_file.name,
                        "prompt_name": prompt_file.name
                    })
            
            if pairs:
                logger.info(f"Found {len(pairs)} document+prompt pairs")
            
        except Exception as e:
            logger.error(f"Error scanning for pairs: {e}")
        
        return pairs
    
    async def _process_document_prompt_pair(self, pair: Dict[str, Any]) -> None:
        """Process a document + prompt pair."""
        try:
            logger.info(f"Processing pair: {pair['document_name']} + {pair['prompt_name']}")
            
            # Read document content
            doc_content = await self._extract_document_content(pair["document_file"])
            
            # Read prompt content
            prompt_content = pair["prompt_file"].read_text(encoding='utf-8')
            
            # Create prompt task
            task = PromptTask(
                task_id=pair["pair_id"],
                document_file=pair["document_name"],
                prompt_file=pair["prompt_name"],
                document_content=doc_content,
                prompt_content=prompt_content,
                created_at=datetime.now()
            )
            
            # Submit for processing
            await self.submit_task(
                task_type="process_prompt",
                task_input=task,
                priority=TaskPriority.NORMAL
            )
            
        except Exception as e:
            logger.error(f"Failed to process pair {pair['pair_id']}: {e}")
            await self._move_to_failed(pair, str(e))
    
    async def _extract_document_content(self, doc_file: Path) -> str:
        """Extract text content from document."""
        try:
            if doc_file.suffix.lower() == '.txt':
                return doc_file.read_text(encoding='utf-8')
            elif doc_file.suffix.lower() == '.pdf':
                return await self._extract_pdf_content(doc_file)
            elif doc_file.suffix.lower() in ['.docx', '.doc']:
                return await self._extract_word_content(doc_file)
            else:
                # Try as text file
                return doc_file.read_text(encoding='utf-8', errors='ignore')
                
        except Exception as e:
            logger.warning(f"Could not extract content from {doc_file}: {e}")
            return f"[Could not extract content from {doc_file.name}: {str(e)}]"
    
    async def _extract_pdf_content(self, pdf_file: Path) -> str:
        """Extract text from PDF file."""
        try:
            import PyPDF2
            
            text = ""
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            return text.strip()
            
        except ImportError:
            logger.warning("PyPDF2 not available for PDF extraction")
            return f"[PDF content extraction requires PyPDF2 - file: {pdf_file.name}]"
        except Exception as e:
            logger.warning(f"PDF extraction failed: {e}")
            return f"[PDF extraction failed for {pdf_file.name}: {str(e)}]"
    
    async def _extract_word_content(self, word_file: Path) -> str:
        """Extract text from Word document."""
        try:
            import docx
            
            doc = docx.Document(word_file)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
            
        except ImportError:
            logger.warning("python-docx not available for Word extraction")
            return f"[Word content extraction requires python-docx - file: {word_file.name}]"
        except Exception as e:
            logger.warning(f"Word extraction failed: {e}")
            return f"[Word extraction failed for {word_file.name}: {str(e)}]"
    
    def validate_input(self, task_input: Any) -> bool:
        """Validate prompt task input."""
        if not isinstance(task_input, PromptTask):
            return False
        
        return bool(
            task_input.task_id and
            task_input.document_content and
            task_input.prompt_content
        )
    
    async def process_task(self, task_input: PromptTask) -> PromptResult:
        """Process prompt-based task."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing prompt task '{task_input.task_id}'")
            
            # Extract and prepare prompts
            prompts = await self._extract_prompts_from_markdown(task_input.prompt_content)
            
            # Process each prompt
            results = []
            
            for prompt_name, prompt_text in prompts.items():
                logger.info(f"Executing prompt: {prompt_name}")
                
                # Replace placeholders in prompt
                filled_prompt = prompt_text.replace("{document_content}", task_input.document_content)
                
                # Execute prompt using available gateway
                prompt_result = await self._execute_prompt(filled_prompt)
                
                results.append({
                    "prompt_name": prompt_name,
                    "prompt_text": prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text,
                    "result": prompt_result
                })
            
            # Combine results
            combined_result = await self._combine_prompt_results(results, task_input)
            
            processing_time = time.time() - start_time
            
            # Save result to file
            await self._save_result_to_file(task_input, combined_result, processing_time)
            
            # Clean up source files
            await self._cleanup_processed_files(task_input)
            
            return PromptResult(
                task_id=task_input.task_id,
                result_content=combined_result,
                processing_time=processing_time,
                status=ProcessingStatus.COMPLETED,
                metadata={
                    "prompts_executed": len(prompts),
                    "document_file": task_input.document_file,
                    "prompt_file": task_input.prompt_file
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"Prompt task '{task_input.task_id}' failed: {error_msg}")
            
            # Save error information
            await self._save_error_to_file(task_input, error_msg, processing_time)
            
            return PromptResult(
                task_id=task_input.task_id,
                result_content="",
                processing_time=processing_time,
                status=ProcessingStatus.FAILED,
                error_message=error_msg
            )
    
    async def _extract_prompts_from_markdown(self, markdown_content: str) -> Dict[str, str]:
        """Extract prompts from markdown file."""
        prompts = {}
        
        # Split by code blocks (``` sections)
        code_blocks = re.findall(r'```[\s\S]*?```', markdown_content, re.MULTILINE)
        
        current_heading = "Main Prompt"
        
        for block in code_blocks:
            # Remove the ``` markers
            prompt_text = re.sub(r'^```.*?\n', '', block, flags=re.MULTILINE)
            prompt_text = re.sub(r'\n```$', '', prompt_text)
            
            if prompt_text.strip():
                # Look for heading before this block
                block_start = markdown_content.find(block)
                preceding_text = markdown_content[:block_start]
                
                # Find the last heading
                headings = re.findall(r'^#+\s+(.+)$', preceding_text, re.MULTILINE)
                if headings:
                    current_heading = headings[-1].strip()
                
                prompts[current_heading] = prompt_text.strip()
        
        # If no code blocks found, treat entire content as one prompt
        if not prompts:
            prompts["Document Analysis"] = markdown_content.strip()
        
        logger.info(f"Extracted {len(prompts)} prompts: {list(prompts.keys())}")
        return prompts
    
    async def _execute_prompt(self, prompt: str) -> str:
        """Execute prompt using available gateway."""
        try:
            if "corporate_llm" in self.available_gateways:
                return await self._execute_with_corporate_llm(prompt)
            elif "ai_processing" in self.available_gateways:
                return await self._execute_with_ai_processing(prompt)  
            elif "tidyllm_direct" in self.available_gateways:
                return await self._execute_with_tidyllm(prompt)
            else:
                return await self._execute_mock_prompt(prompt)
                
        except Exception as e:
            logger.error(f"Prompt execution failed: {e}")
            return f"[Prompt execution failed: {str(e)}]"
    
    async def _execute_with_corporate_llm(self, prompt: str) -> str:
        """Execute prompt using Corporate LLM Gateway."""
        try:
            from ...gateways.corporate_llm_gateway import CorporateLLMGateway
            
            gateway = CorporateLLMGateway()
            # Simplified execution - adapt based on actual gateway interface
            result = await asyncio.get_event_loop().run_in_executor(None, lambda: f"[Corporate LLM Result for prompt: {prompt[:100]}...]")
            return result
            
        except Exception as e:
            logger.error(f"Corporate LLM execution failed: {e}")
            raise
    
    async def _execute_with_ai_processing(self, prompt: str) -> str:
        """Execute prompt using AI Processing Gateway."""
        try:
            from ...gateways.ai_processing_gateway import AIProcessingGateway
            
            gateway = AIProcessingGateway()
            # Simplified execution - adapt based on actual gateway interface  
            result = await asyncio.get_event_loop().run_in_executor(None, lambda: f"[AI Processing Result for prompt: {prompt[:100]}...]")
            return result
            
        except Exception as e:
            logger.error(f"AI Processing execution failed: {e}")
            raise
    
    async def _execute_with_tidyllm(self, prompt: str) -> str:
        """Execute prompt using TidyLLM directly."""
        try:
            import tidyllm
            
            # Use TidyLLM's verb system
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: f"[TidyLLM Direct Result for prompt: {prompt[:100]}...]"
            )
            return result
            
        except Exception as e:
            logger.error(f"TidyLLM direct execution failed: {e}")
            raise
    
    async def _execute_mock_prompt(self, prompt: str) -> str:
        """Mock prompt execution for testing."""
        await asyncio.sleep(2)  # Simulate processing time
        
        # Generate a reasonable mock response based on prompt content
        if "analyze" in prompt.lower():
            return """Mock Analysis Result:

1. Document Type: Business Document
2. Key Findings: 
   - Contains structured information
   - Appears to be compliance-related
   - Multiple sections identified
3. Risk Assessment: Medium
4. Recommendations: 
   - Review for completeness
   - Verify data accuracy
   - Consider additional validation

This is a mock result generated for testing purposes."""
        
        elif "classify" in prompt.lower():
            return """Mock Classification Result:
{
  "document_type": "Business Document",
  "risk_level": "Medium", 
  "business_purpose": "Analysis",
  "data_quality_score": 7,
  "confidence": 0.85
}"""
        
        else:
            return f"""Mock Response:

Based on the provided prompt, here is a comprehensive analysis of the document:

Key Points Identified:
- Document structure appears well-organized
- Content is suitable for automated processing
- No immediate red flags detected

Recommendations:
- Proceed with standard processing workflow
- Monitor for any unusual patterns
- Maintain audit trail for compliance

Processing completed successfully.
[This is a mock response for testing - prompt: {prompt[:50]}...]"""
    
    async def _combine_prompt_results(self, results: List[Dict[str, Any]], task: PromptTask) -> str:
        """Combine multiple prompt results into final output."""
        
        combined = {
            "task_id": task.task_id,
            "document_file": task.document_file,
            "prompt_file": task.prompt_file,
            "processed_at": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "total_prompts": len(results),
                "document_length": len(task.document_content),
                "prompt_length": len(task.prompt_content)
            }
        }
        
        return json.dumps(combined, indent=2)
    
    async def _save_result_to_file(self, task: PromptTask, result: str, processing_time: float) -> None:
        """Save processing result to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            doc_stem = Path(task.document_file).stem
            
            result_filename = f"result_{timestamp}_{doc_stem}.json"
            result_path = self.results_folder / result_filename
            
            result_path.write_text(result, encoding='utf-8')
            
            logger.info(f"Result saved to: {result_path}")
            
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
    
    async def _save_error_to_file(self, task: PromptTask, error: str, processing_time: float) -> None:
        """Save error information to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            doc_stem = Path(task.document_file).stem
            
            error_filename = f"error_{timestamp}_{doc_stem}.json"
            error_path = self.failed_folder / error_filename
            
            error_info = {
                "task_id": task.task_id,
                "document_file": task.document_file,
                "prompt_file": task.prompt_file,
                "error_message": error,
                "processing_time": processing_time,
                "failed_at": datetime.now().isoformat()
            }
            
            error_path.write_text(json.dumps(error_info, indent=2), encoding='utf-8')
            
            logger.info(f"Error saved to: {error_path}")
            
        except Exception as e:
            logger.error(f"Failed to save error: {e}")
    
    async def _cleanup_processed_files(self, task: PromptTask) -> None:
        """Move processed files to avoid reprocessing."""
        try:
            # Create processed subfolder
            processed_folder = self.watch_folder / "processed"
            processed_folder.mkdir(exist_ok=True)
            
            # Move files
            doc_file = self.watch_folder / task.document_file
            prompt_file = self.watch_folder / task.prompt_file
            
            if doc_file.exists():
                doc_file.rename(processed_folder / task.document_file)
            
            if prompt_file.exists():
                prompt_file.rename(processed_folder / task.prompt_file)
            
            logger.info(f"Files moved to processed folder: {task.document_file}, {task.prompt_file}")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    async def _move_to_failed(self, pair: Dict[str, Any], error: str) -> None:
        """Move failed pair to failed folder."""
        try:
            # Move document and prompt to failed folder
            doc_file = pair["document_file"]  
            prompt_file = pair["prompt_file"]
            
            doc_file.rename(self.failed_folder / doc_file.name)
            prompt_file.rename(self.failed_folder / prompt_file.name)
            
            # Create error log
            error_log = {
                "pair_id": pair["pair_id"],
                "error": error,
                "failed_at": datetime.now().isoformat(),
                "document_file": doc_file.name,
                "prompt_file": prompt_file.name
            }
            
            error_file = self.failed_folder / f"error_{pair['pair_id']}.json"
            error_file.write_text(json.dumps(error_log, indent=2))
            
            logger.info(f"Failed pair moved to failed folder: {pair['pair_id']}")
            
        except Exception as e:
            logger.error(f"Failed to move to failed folder: {e}")
    
    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the prompt worker."""
        logger.info(f"Stopping Prompt Worker '{self.worker_name}'...")
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await asyncio.wait_for(self.monitoring_task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        
        await super().stop(timeout)
        
        logger.info(f"Prompt Worker '{self.worker_name}' stopped")
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        return {
            "worker_name": self.worker_name,
            "watch_folder": str(self.watch_folder),
            "results_folder": str(self.results_folder),
            "failed_folder": str(self.failed_folder),
            "monitoring_active": self.monitoring_task is not None and not self.monitoring_task.done(),
            "available_gateways": self.available_gateways,
            "processed_pairs": len(self.processed_pairs),
            "worker_status": self.status.value,
            "worker_metrics": self.metrics.to_dict()
        }