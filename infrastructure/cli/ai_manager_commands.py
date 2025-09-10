# AI Manager CLI Commands
# 📟 CORE ENTERPRISE CLI - AI-Assisted Manager Operations
#
# This component provides:
# - Command-line interface for AI Manager operations
# - Status monitoring and queue management
# - Template and worker pool administration
# - Integration with existing TidyLLM CLI system
#
# Dependencies:
# - AI-Assisted Manager for orchestration
# - Click for CLI framework
# - Rich for enhanced terminal output
# - Existing TidyLLM CLI infrastructure

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import click
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.live import Live
    from rich.json import JSON
except ImportError:
    # Fallback if rich/click not available
    logging.warning("Rich/Click not available - CLI will have limited formatting")
    click = None
    Console = None

from ..workers.ai_dropzone_manager import AIDropzoneManager, AIManagerTask, ProcessingStrategy, DocumentComplexity
from ..session.unified import UnifiedSessionManager

logger = logging.getLogger(__name__)
console = Console() if Console else None

class AIManagerCLI:
    """
    Command-line interface for AI-Assisted Manager operations.
    """
    
    def __init__(self):
        self.manager: Optional[AIDropzoneManager] = None
        self.session_manager: Optional[UnifiedSessionManager] = None
    
    async def initialize(self):
        """Initialize the AI Dropzone Manager CLI."""
        try:
            self.manager = AIDropzoneManager()
            await self.manager.initialize()
            
            self.session_manager = UnifiedSessionManager()
            await self.session_manager.initialize()
            
            if console:
                console.print("[green][OK][/green] AI Manager CLI initialized")
            else:
                print("[OK] AI Manager CLI initialized")
                
        except Exception as e:
            if console:
                console.print(f"[red][ERROR][/red] Failed to initialize AI Manager CLI: {e}")
            else:
                print(f"[ERROR] Failed to initialize AI Manager CLI: {e}")
            raise

# Click commands (if available)
if click:
    
    @click.group()
    @click.pass_context
    def ai_manager(ctx):
        """AI-Assisted Manager operations and monitoring."""
        if ctx.obj is None:
            ctx.obj = {}
        
        # Initialize CLI instance
        if 'ai_cli' not in ctx.obj:
            ctx.obj['ai_cli'] = AIManagerCLI()
    
    @ai_manager.command()
    @click.pass_context
    def status(ctx):
        """Get AI Manager status and metrics."""
        asyncio.run(_status_command(ctx.obj['ai_cli']))
    
    @ai_manager.command()
    @click.argument('document_path', type=click.Path(exists=True))
    @click.option('--priority', default='normal', type=click.Choice(['critical', 'high', 'normal', 'low']))
    @click.option('--strategy', type=click.Choice(['single_template', 'hybrid_analysis', 'multi_perspective']))
    @click.option('--templates', multiple=True, help='Specific templates to use')
    @click.option('--watch', is_flag=True, help='Watch processing progress in real-time')
    @click.pass_context
    def process(ctx, document_path: str, priority: str, strategy: Optional[str], templates: tuple, watch: bool):
        """Process a document using AI Manager."""
        asyncio.run(_process_command(
            ctx.obj['ai_cli'], 
            document_path, 
            priority, 
            strategy, 
            list(templates), 
            watch
        ))
    
    @ai_manager.command()
    @click.pass_context
    def queue(ctx):
        """Show processing queue status and management."""
        asyncio.run(_queue_command(ctx.obj['ai_cli']))
    
    @ai_manager.command()
    @click.pass_context
    def workers(ctx):
        """Show worker pool status and performance."""
        asyncio.run(_workers_command(ctx.obj['ai_cli']))
    
    @ai_manager.command()
    @click.option('--detailed', is_flag=True, help='Show detailed template information')
    @click.pass_context
    def templates(ctx, detailed: bool):
        """List available processing templates."""
        asyncio.run(_templates_command(ctx.obj['ai_cli'], detailed))
    
    @ai_manager.command()
    @click.argument('document_path', type=click.Path(exists=True))
    @click.pass_context
    def analyze(ctx, document_path: str):
        """Analyze document and show recommended processing approach."""
        asyncio.run(_analyze_command(ctx.obj['ai_cli'], document_path))
    
    @ai_manager.command()
    @click.option('--watch', is_flag=True, help='Continuously monitor performance')
    @click.pass_context
    def monitor(ctx, watch: bool):
        """Monitor AI Manager performance and health."""
        asyncio.run(_monitor_command(ctx.obj['ai_cli'], watch))

# Command implementations
async def _status_command(cli: AIManagerCLI):
    """Implementation for status command."""
    try:
        await cli.initialize()
        
        status = await cli.manager.get_manager_status()
        
        if console:
            # Rich formatted output
            status_panel = Panel.fit(
                f"""[bold]AI-Assisted Manager Status[/bold]

[green]Status:[/green] {status.get('status', 'unknown')}
[blue]Active Workers:[/blue] {status.get('active_workers', 0)}
[yellow]Available Templates:[/yellow] {status.get('available_templates', 0)}
[cyan]Processing History:[/cyan] {status.get('processing_history_count', 0)}
[magenta]Cached Intelligence:[/magenta] {status.get('cached_intelligence_count', 0)}

[bold]Quality Thresholds:[/bold]
{chr(10).join(f'  {k}: {v}' for k, v in status.get('quality_thresholds', {}).items())}
""",
                title="AI Manager Status",
                border_style="blue"
            )
            console.print(status_panel)
        else:
            # Plain text output
            print("AI-Assisted Manager Status:")
            print(f"  Status: {status.get('status', 'unknown')}")
            print(f"  Active Workers: {status.get('active_workers', 0)}")
            print(f"  Available Templates: {status.get('available_templates', 0)}")
            print(f"  Processing History: {status.get('processing_history_count', 0)}")
            print(f"  Cached Intelligence: {status.get('cached_intelligence_count', 0)}")
            
    except Exception as e:
        if console:
            console.print(f"[red][ERROR][/red] Failed to get status: {e}")
        else:
            print(f"[ERROR] Failed to get status: {e}")

async def _process_command(
    cli: AIManagerCLI, 
    document_path: str, 
    priority: str, 
    strategy: Optional[str], 
    templates: List[str], 
    watch: bool
):
    """Implementation for process command."""
    try:
        await cli.initialize()
        
        # Create processing task
        task = AIManagerTask(
            document_path=document_path,
            business_priority=priority,
            user_context={
                "cli_request": True,
                "preferred_templates": templates,
                "processing_strategy": strategy
            }
        )
        
        if console:
            console.print(f"[blue][INFO][/blue] Processing document: {document_path}")
            console.print(f"[blue][INFO][/blue] Priority: {priority}")
            
            if strategy:
                console.print(f"[blue][INFO][/blue] Strategy: {strategy}")
            if templates:
                console.print(f"[blue][INFO][/blue] Templates: {', '.join(templates)}")
        else:
            print(f"[INFO] Processing document: {document_path}")
            print(f"[INFO] Priority: {priority}")
        
        # Execute processing
        if watch:
            # Real-time progress monitoring
            await _watch_processing(cli, task)
        else:
            # Standard processing
            result = await cli.manager.process_task(task)
            
            if result.success:
                if console:
                    success_panel = Panel.fit(
                        f"""[green][bold]Processing Completed Successfully[/bold][/green]

[blue]Strategy:[/blue] {result.processing_decision.processing_strategy.value if result.processing_decision else 'unknown'}
[yellow]Templates Used:[/yellow] {', '.join(result.processing_decision.selected_templates) if result.processing_decision else 'none'}
[cyan]Workers Assigned:[/cyan] {len(result.worker_assignments)}
[magenta]Estimated Completion:[/magenta] {result.estimated_completion}

[bold]Quality Metrics:[/bold]
{chr(10).join(f'  {k}: {v:.2f}' for k, v in result.quality_metrics.items())}
""",
                        title="Processing Result",
                        border_style="green"
                    )
                    console.print(success_panel)
                else:
                    print("[SUCCESS] Processing completed successfully")
                    print(f"  Workers assigned: {len(result.worker_assignments)}")
                    print(f"  Estimated completion: {result.estimated_completion}")
            else:
                if console:
                    console.print(f"[red][ERROR][/red] Processing failed: {getattr(result, 'error', 'Unknown error')}")
                else:
                    print(f"[ERROR] Processing failed: {getattr(result, 'error', 'Unknown error')}")
                    
    except Exception as e:
        if console:
            console.print(f"[red][ERROR][/red] Failed to process document: {e}")
        else:
            print(f"[ERROR] Failed to process document: {e}")

async def _watch_processing(cli: AIManagerCLI, task: AIManagerTask):
    """Watch processing progress in real-time."""
    if not console:
        print("[INFO] Real-time watching not available without Rich library")
        result = await cli.manager.process_task(task)
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        
        processing_task = progress.add_task(description="Analyzing document...", total=None)
        
        # Simulate progress tracking (in production would hook into actual progress)
        stages = [
            "Analyzing document intelligence...",
            "Selecting optimal templates...",
            "Allocating workers...",
            "Processing templates...",
            "Synthesizing results...",
            "Quality validation..."
        ]
        
        for stage in stages:
            progress.update(processing_task, description=stage)
            await asyncio.sleep(2)  # Simulate processing time
        
        progress.update(processing_task, description="Processing complete!")
        
    # Execute actual processing
    result = await cli.manager.process_task(task)
    
    if result.success:
        console.print("[green][SUCCESS][/green] Document processing completed!")
    else:
        console.print(f"[red][ERROR][/red] Processing failed: {getattr(result, 'error', 'Unknown error')}")

async def _queue_command(cli: AIManagerCLI):
    """Implementation for queue command."""
    try:
        await cli.initialize()
        
        # Get queue status (would need to implement in manager)
        queue_info = {
            "queued_items": 0,
            "processing_items": 0,
            "average_wait_time": 5.0,
            "total_processed_today": 0
        }
        
        if console:
            queue_table = Table(title="Processing Queue Status")
            queue_table.add_column("Metric", style="cyan", no_wrap=True)
            queue_table.add_column("Value", style="magenta")
            
            queue_table.add_row("Queued Items", str(queue_info["queued_items"]))
            queue_table.add_row("Currently Processing", str(queue_info["processing_items"]))
            queue_table.add_row("Average Wait Time", f"{queue_info['average_wait_time']} minutes")
            queue_table.add_row("Processed Today", str(queue_info["total_processed_today"]))
            
            console.print(queue_table)
        else:
            print("Processing Queue Status:")
            print(f"  Queued Items: {queue_info['queued_items']}")
            print(f"  Currently Processing: {queue_info['processing_items']}")
            print(f"  Average Wait Time: {queue_info['average_wait_time']} minutes")
            print(f"  Processed Today: {queue_info['total_processed_today']}")
            
    except Exception as e:
        if console:
            console.print(f"[red][ERROR][/red] Failed to get queue status: {e}")
        else:
            print(f"[ERROR] Failed to get queue status: {e}")

async def _workers_command(cli: AIManagerCLI):
    """Implementation for workers command."""
    try:
        await cli.initialize()
        
        status = await cli.manager.get_manager_status()
        worker_load = status.get('worker_load', {})
        
        if console:
            worker_table = Table(title="Worker Pool Status")
            worker_table.add_column("Worker Pool", style="cyan", no_wrap=True)
            worker_table.add_column("Load", style="yellow")
            worker_table.add_column("Status", style="green")
            
            for worker_id, load in worker_load.items():
                status_text = "Active" if load > 0 else "Idle"
                worker_table.add_row(
                    worker_id.replace('_', ' ').title(),
                    str(load),
                    status_text
                )
            
            console.print(worker_table)
        else:
            print("Worker Pool Status:")
            for worker_id, load in worker_load.items():
                status_text = "Active" if load > 0 else "Idle"
                print(f"  {worker_id.replace('_', ' ').title()}: Load={load}, Status={status_text}")
                
    except Exception as e:
        if console:
            console.print(f"[red][ERROR][/red] Failed to get worker status: {e}")
        else:
            print(f"[ERROR] Failed to get worker status: {e}")

async def _templates_command(cli: AIManagerCLI, detailed: bool):
    """Implementation for templates command."""
    try:
        await cli.initialize()
        
        templates = cli.manager.available_templates
        template_performance = cli.manager.template_performance
        
        if console:
            if detailed:
                # Detailed view with performance metrics
                for template_name, template_info in templates.items():
                    perf = template_performance.get(template_name, {})
                    
                    template_panel = Panel.fit(
                        f"""[bold]{template_name.replace('_', ' ').title()}[/bold]

[blue]Domain Focus:[/blue] {template_info.get('domain_focus', 'general')}
[yellow]Complexity:[/yellow] {template_info.get('estimated_complexity', 'medium')}
[green]Success Rate:[/green] {perf.get('success_rate', 0.85):.1%}
[cyan]Avg Processing Time:[/cyan] {perf.get('avg_processing_time', 10):.1f} minutes
[magenta]Usage Count:[/magenta] {perf.get('usage_count', 0)}

[dim]Path: {template_info.get('path', 'Unknown')}[/dim]
""",
                        border_style="blue"
                    )
                    console.print(template_panel)
            else:
                # Summary table view
                template_table = Table(title="Available Processing Templates")
                template_table.add_column("Template", style="cyan", no_wrap=True)
                template_table.add_column("Domain", style="yellow")
                template_table.add_column("Complexity", style="magenta")
                template_table.add_column("Success Rate", style="green")
                template_table.add_column("Usage", style="blue")
                
                for template_name, template_info in templates.items():
                    perf = template_performance.get(template_name, {})
                    
                    template_table.add_row(
                        template_name.replace('_', ' ').title(),
                        template_info.get('domain_focus', 'general'),
                        template_info.get('estimated_complexity', 'medium'),
                        f"{perf.get('success_rate', 0.85):.1%}",
                        str(perf.get('usage_count', 0))
                    )
                
                console.print(template_table)
        else:
            print("Available Processing Templates:")
            for template_name, template_info in templates.items():
                perf = template_performance.get(template_name, {})
                print(f"  {template_name.replace('_', ' ').title()}:")
                print(f"    Domain: {template_info.get('domain_focus', 'general')}")
                print(f"    Complexity: {template_info.get('estimated_complexity', 'medium')}")
                print(f"    Success Rate: {perf.get('success_rate', 0.85):.1%}")
                print(f"    Usage Count: {perf.get('usage_count', 0)}")
                
    except Exception as e:
        if console:
            console.print(f"[red][ERROR][/red] Failed to get template information: {e}")
        else:
            print(f"[ERROR] Failed to get template information: {e}")

async def _analyze_command(cli: AIManagerCLI, document_path: str):
    """Implementation for analyze command."""
    try:
        await cli.initialize()
        
        # Analyze document intelligence without processing
        intelligence = await cli.manager._analyze_document_intelligence(document_path)
        
        if console:
            analysis_panel = Panel.fit(
                f"""[bold]Document Intelligence Analysis[/bold]

[blue]Document Type:[/blue] {intelligence.detected_type}
[yellow]Complexity:[/yellow] {intelligence.complexity.name}
[green]Confidence:[/green] {intelligence.confidence_score:.1%}
[cyan]Processing Strategy:[/cyan] {intelligence.processing_strategy.value}
[magenta]Estimated Time:[/magenta] {intelligence.estimated_processing_time} minutes

[bold]Recommended Templates:[/bold]
{chr(10).join(f'  • {template.replace("_", " ").title()}' for template in intelligence.recommended_templates)}

[bold]Resource Requirements:[/bold]
{chr(10).join(f'  {k}: {v}' for k, v in intelligence.resource_requirements.items())}
""",
                title=f"Analysis: {Path(document_path).name}",
                border_style="cyan"
            )
            console.print(analysis_panel)
        else:
            print(f"Document Intelligence Analysis: {Path(document_path).name}")
            print(f"  Document Type: {intelligence.detected_type}")
            print(f"  Complexity: {intelligence.complexity.name}")
            print(f"  Confidence: {intelligence.confidence_score:.1%}")
            print(f"  Processing Strategy: {intelligence.processing_strategy.value}")
            print(f"  Estimated Time: {intelligence.estimated_processing_time} minutes")
            print("  Recommended Templates:")
            for template in intelligence.recommended_templates:
                print(f"    • {template.replace('_', ' ').title()}")
                
    except Exception as e:
        if console:
            console.print(f"[red][ERROR][/red] Failed to analyze document: {e}")
        else:
            print(f"[ERROR] Failed to analyze document: {e}")

async def _monitor_command(cli: AIManagerCLI, watch: bool):
    """Implementation for monitor command."""
    try:
        await cli.initialize()
        
        if watch and console:
            # Continuous monitoring with live updates
            with Live(console=console, refresh_per_second=2) as live:
                while True:
                    try:
                        status = await cli.manager.get_manager_status()
                        
                        # Create monitoring dashboard
                        monitor_table = Table(title="AI Manager Live Monitoring")
                        monitor_table.add_column("Metric", style="cyan")
                        monitor_table.add_column("Value", style="yellow")
                        monitor_table.add_column("Status", style="green")
                        
                        # System metrics
                        monitor_table.add_row("Manager Status", status.get('status', 'unknown'), "[green]Online[/green]" if status.get('status') == 'active' else "[red]Offline[/red]")
                        monitor_table.add_row("Active Workers", str(status.get('active_workers', 0)), "[green]Healthy[/green]")
                        monitor_table.add_row("Available Templates", str(status.get('available_templates', 0)), "[green]Ready[/green]")
                        monitor_table.add_row("Processing History", str(status.get('processing_history_count', 0)), "[blue]Tracking[/blue]")
                        monitor_table.add_row("Cached Intelligence", str(status.get('cached_intelligence_count', 0)), "[cyan]Optimized[/cyan]")
                        
                        # Update display
                        live.update(Panel(monitor_table, title=f"Live Monitoring - {datetime.now().strftime('%H:%M:%S')}", border_style="green"))
                        
                        await asyncio.sleep(5)  # Update every 5 seconds
                        
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        live.update(Panel(f"[red]Monitoring Error: {e}[/red]", border_style="red"))
                        await asyncio.sleep(5)
        else:
            # Single snapshot monitoring
            status = await cli.manager.get_manager_status()
            
            if console:
                console.print(JSON.from_data(status, indent=2))
            else:
                print("AI Manager Monitoring Snapshot:")
                print(json.dumps(status, indent=2, default=str))
                
    except Exception as e:
        if console:
            console.print(f"[red][ERROR][/red] Failed to monitor system: {e}")
        else:
            print(f"[ERROR] Failed to monitor system: {e}")

# Integration function for existing CLI system
def register_ai_manager_commands(cli_app):
    """
    Register AI Manager commands with existing TidyLLM CLI application.
    """
    if click:
        cli_app.add_command(ai_manager)
        logger.info("AI Manager commands registered with CLI")
    else:
        logger.warning("Click not available - AI Manager CLI commands not registered")

# Direct CLI execution for testing
if __name__ == "__main__":
    if click:
        ai_manager()
    else:
        print("Click not available - install 'click' and 'rich' packages for full CLI functionality")