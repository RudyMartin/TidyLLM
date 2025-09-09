#!/usr/bin/env python3
"""
Simple Prompt Processor - Any Dummy Can Use This!
=================================================

Ultra-simple document + prompt processing. Just:
1. Run this script
2. Drop document + prompt files in the 'documents/incoming' folder  
3. Get results in 'documents/results' folder

That's it! No complex setup, no configuration, no stages to manage.

Usage:
    python scripts/simple_prompt_processor.py
    
    # Or with custom folder
    python scripts/simple_prompt_processor.py --folder /path/to/my/documents
"""

import asyncio
import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from tidyllm.infrastructure.workers.prompt_worker import PromptWorker
    PROMPT_WORKER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Prompt worker not available: {e}")
    PROMPT_WORKER_AVAILABLE = False


def print_banner():
    """Print welcome banner."""
    print("🚀 TidyLLM Simple Prompt Processor")
    print("=" * 50)
    print("Any dummy can use this!")
    print()
    print("How it works:")
    print("1. Drop your document (PDF, Word, text) into 'documents/incoming'")
    print("2. Drop your prompt file (.md) into the same folder")
    print("3. Wait for magic to happen!")
    print("4. Check results in 'documents/results'")
    print()
    print("Example prompts you can use:")
    print("- analyst_report_prompts.md (from /prompts folder)")
    print("- section_view_prompts.md (from /prompts folder)")  
    print("- Or create your own .md file with custom prompts!")
    print()


def create_sample_prompts(base_folder: Path):
    """Create sample prompt files for dummies."""
    prompts_folder = base_folder / "sample_prompts"
    prompts_folder.mkdir(exist_ok=True)
    
    # Simple analysis prompt
    simple_prompt = prompts_folder / "simple_analysis.md"
    simple_prompt.write_text("""# Simple Document Analysis

```
Please analyze this document and tell me:

1. What type of document is this?
2. What are the main topics covered?
3. What are the key findings or important information?
4. Are there any potential issues or concerns?
5. What should I do next?

Document to analyze:
{document_content}

Please provide a clear, easy-to-understand analysis.
```
""")
    
    # Contract analysis prompt
    contract_prompt = prompts_folder / "contract_analysis.md"
    contract_prompt.write_text("""# Contract Analysis

```
Analyze this contract/agreement and provide:

1. **Contract Type**: What kind of agreement is this?
2. **Key Parties**: Who are the main parties involved?
3. **Key Terms**: What are the most important terms and conditions?
4. **Dates**: What are the important dates (start, end, renewal)?
5. **Financial Terms**: Any money, payments, or financial obligations?
6. **Risk Assessment**: What are the potential risks or concerns?
7. **Recommendations**: What should I pay attention to or negotiate?

Contract content:
{document_content}

Provide a business-focused analysis suitable for decision makers.
```
""")
    
    # Compliance check prompt
    compliance_prompt = prompts_folder / "compliance_check.md"
    compliance_prompt.write_text("""# Compliance Analysis

```
Review this document for compliance and regulatory considerations:

1. **Document Classification**: What type of regulatory document is this?
2. **Compliance Requirements**: What compliance standards apply?
3. **Risk Level**: High, Medium, or Low risk from compliance perspective?
4. **Missing Information**: What required information might be missing?
5. **Red Flags**: Any potential compliance violations or concerns?
6. **Recommendations**: What actions should be taken for compliance?
7. **Next Steps**: What follow-up is needed?

Document for compliance review:
{document_content}

Focus on practical compliance guidance and actionable recommendations.
```
""")
    
    print(f"✅ Sample prompts created in: {prompts_folder}")
    print("   - simple_analysis.md (general document analysis)")
    print("   - contract_analysis.md (contract review)")
    print("   - compliance_check.md (compliance analysis)")
    print()


def copy_existing_prompts(base_folder: Path):
    """Copy existing prompts from /prompts folder."""
    try:
        source_prompts = project_root / "prompts"
        if source_prompts.exists():
            dest_prompts = base_folder / "existing_prompts"
            dest_prompts.mkdir(exist_ok=True)
            
            copied = 0
            for prompt_file in source_prompts.glob("*.md"):
                dest_file = dest_prompts / prompt_file.name
                dest_file.write_text(prompt_file.read_text())
                copied += 1
            
            if copied > 0:
                print(f"✅ Copied {copied} existing prompts to: {dest_prompts}")
                print("   You can use these advanced prompts with your documents!")
                print()
    except Exception as e:
        print(f"⚠️  Could not copy existing prompts: {e}")


async def run_simple_processor(base_folder: Path):
    """Run the simple prompt processor."""
    
    if not PROMPT_WORKER_AVAILABLE:
        print("❌ Prompt worker not available. Check installation.")
        return False
    
    # Set up folders
    incoming = base_folder / "incoming"
    results = base_folder / "results"
    failed = base_folder / "failed"
    
    print(f"📁 Using folders:")
    print(f"   Drop files here: {incoming}")
    print(f"   Results here: {results}")
    print(f"   Failed items: {failed}")
    print()
    
    # Create sample prompts
    create_sample_prompts(base_folder)
    copy_existing_prompts(base_folder)
    
    # Initialize prompt worker
    try:
        worker = PromptWorker(
            watch_folder=str(incoming),
            results_folder=str(results),
            failed_folder=str(failed)
        )
        
        await worker.initialize()
        await worker.start()
        
        print("🎯 Prompt processor is now running!")
        print("=" * 40)
        print()
        print("Ready for your files:")
        print(f"1. Drop your document in: {incoming}")
        print(f"2. Drop a .md prompt file in: {incoming}")
        print(f"3. Check results in: {results}")
        print()
        print("Press Ctrl+C to stop")
        print()
        
        # Monitor and show status
        last_stats = None
        
        while True:
            try:
                # Show current status
                stats = worker.get_processing_status()
                
                if stats != last_stats:
                    print(f"[{time.strftime('%H:%M:%S')}] Status: {stats['worker_status']} | "
                          f"Processed: {stats['processed_pairs']} | "
                          f"Gateways: {', '.join(stats['available_gateways'])}")
                    last_stats = stats
                
                # Check for new results
                if results.exists():
                    result_files = list(results.glob("result_*.json"))
                    if result_files:
                        latest = max(result_files, key=lambda f: f.stat().st_mtime)
                        if hasattr(run_simple_processor, 'last_result_count'):
                            if len(result_files) > run_simple_processor.last_result_count:
                                print(f"🎉 New result available: {latest.name}")
                        else:
                            run_simple_processor.last_result_count = 0
                        run_simple_processor.last_result_count = len(result_files)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except KeyboardInterrupt:
                print("\n👋 Stopping prompt processor...")
                break
        
        await worker.stop()
        print("✅ Prompt processor stopped")
        return True
        
    except Exception as e:
        print(f"❌ Error running prompt processor: {e}")
        return False


def show_results_summary(results_folder: Path):
    """Show summary of processing results."""
    if not results_folder.exists():
        print("📂 No results folder found yet")
        return
    
    result_files = list(results_folder.glob("*.json"))
    
    if not result_files:
        print("📂 No results found yet")
        return
    
    print(f"\n📊 PROCESSING RESULTS SUMMARY")
    print("=" * 40)
    print(f"Total results: {len(result_files)}")
    
    for result_file in sorted(result_files)[-5:]:  # Show last 5
        try:
            import json
            with open(result_file) as f:
                data = json.load(f)
            
            doc_file = data.get("document_file", "unknown")
            processed_at = data.get("processed_at", "unknown")
            prompts = data.get("summary", {}).get("total_prompts", 0)
            
            print(f"  • {doc_file} ({prompts} prompts) - {processed_at}")
            
        except Exception as e:
            print(f"  • {result_file.name} - [could not read: {e}]")
    
    if len(result_files) > 5:
        print(f"  ... and {len(result_files) - 5} more results")
    
    print()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Prompt Processor for Any Dummy")
    parser.add_argument("--folder", help="Base folder for documents (default: documents)")
    parser.add_argument("--show-results", action="store_true", help="Show results summary and exit")
    parser.add_argument("--setup-only", action="store_true", help="Just set up folders and prompts")
    args = parser.parse_args()
    
    # Determine base folder
    if args.folder:
        base_folder = Path(args.folder)
    else:
        base_folder = project_root / "documents"
    
    base_folder.mkdir(exist_ok=True)
    
    print_banner()
    
    if args.show_results:
        show_results_summary(base_folder / "results")
        return 0
    
    if args.setup_only:
        create_sample_prompts(base_folder)
        copy_existing_prompts(base_folder)
        print("✅ Setup complete! Now run without --setup-only to start processing.")
        return 0
    
    try:
        success = await run_simple_processor(base_folder)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
        return 0
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))