#!/usr/bin/env python3
"""
Workflow Registry CLI - Command line interface for managing workflows
====================================================================

Usage:
    python workflow_cli.py list                    # List all workflows
    python workflow_cli.py list --active           # List only active workflows
    python workflow_cli.py stats                   # Show workflow statistics
    python workflow_cli.py info <workflow_id>      # Show detailed workflow info
    python workflow_cli.py structure               # Show folder structure
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from .registry import WorkflowRegistrySystem
except ImportError:
    try:
        # Fallback for standalone execution
        from workflow_registry_system import WorkflowRegistrySystem
    except ImportError:
        print("Error: Could not import WorkflowRegistrySystem")
        sys.exit(1)

def format_workflow_list(workflows, show_all=False, name_filter=None, type_filter=None, priority_filter=None):
    """Format workflow list for CLI display."""
    # Filter workflows
    if not show_all:
        workflows = [w for w in workflows if w["status"] == "active"]
    
    if name_filter:
        workflows = [w for w in workflows if name_filter.lower() in w["workflow_name"].lower() or name_filter.lower() in w["workflow_id"].lower()]
    
    if type_filter:
        workflows = [w for w in workflows if type_filter.lower() in w["workflow_type"].lower()]
    
    if priority_filter:
        workflows = [w for w in workflows if priority_filter.lower() == w["priority_level"].lower()]
    
    if not workflows:
        print("No workflows found matching criteria.")
        return
    
    # Header
    print(f"{'ID':<20} {'Name':<25} {'Status':<10} {'Priority':<10} {'Type':<15} {'Templates':<10}")
    print("-" * 90)
    
    # Workflows
    for workflow in workflows:
        status_icon = "*" if workflow["status"] == "active" else "o"
        priority_color = {
            "critical": "!",
            "high": "+", 
            "normal": "-",
            "low": "."
        }.get(workflow["priority_level"], "?")
        
        print(f"{workflow['workflow_id']:<20} "
              f"{workflow['workflow_name']:<25} "
              f"{status_icon} {workflow['status']:<8} "
              f"{priority_color} {workflow['priority_level']:<8} "
              f"{workflow['workflow_type']:<15} "
              f"{workflow['template_count']:<10}")

def format_stats(stats):
    """Format workflow statistics for CLI display."""
    print("Workflow Registry Statistics")
    print("=" * 40)
    print(f"Total Workflows: {stats['total_workflows']}")
    print(f"Active: {stats['active_workflows']}")
    print(f"Configured: {stats['configured_workflows']}")
    print(f"New Structure: {stats['new_structure_count']}")
    print(f"Legacy Structure: {stats['legacy_structure_count']}")
    print(f"Average Success Rate: {stats['average_success_rate']:.1%}")
    print(f"Total Executions: {stats['total_executions']}")
    
    print("\nBy Priority Level:")
    for priority, count in stats["priority_levels"].items():
        icon = {"critical": "🔴", "high": "🟡", "normal": "🟢", "low": "🔵"}.get(priority, "⚪")
        print(f"  {icon} {priority.title()}: {count}")
    
    print("\nBy Workflow Type:")
    for wf_type, count in stats["workflow_types"].items():
        print(f"  • {wf_type.title()}: {count}")

def format_workflow_info(workflow):
    """Format detailed workflow information."""
    print(f"Workflow: {workflow['workflow_name']} ({workflow['workflow_id']})")
    print("=" * 60)
    print(f"Description: {workflow['description']}")
    print(f"Type: {workflow['workflow_type']}")
    print(f"Status: {workflow['status']}")
    print(f"Priority: {workflow['priority_level']}")
    print(f"Processing Strategy: {workflow['processing_strategy']}")
    print(f"Template Count: {workflow['template_count']}")
    print(f"Success Rate: {workflow['success_rate']:.1%}")
    print(f"Total Executions: {workflow['total_executions']}")
    print(f"New Structure: {'Yes' if workflow['has_new_structure'] else 'No'}")
    print(f"Resources Available: {'Yes' if workflow['resources_available'] else 'No'}")
    
    if workflow['template_files']:
        print(f"\nTemplate Files:")
        for template in workflow['template_files']:
            print(f"  • {template}.md")
    
    if workflow['criteria_file']:
        print(f"\nCriteria File: {workflow['criteria_file']}")
    
    if workflow['flow_encoding']:
        print(f"\nFlow Encoding: {workflow['flow_encoding']}")

def show_structure():
    """Show the workflow folder structure."""
    registry_path = Path(__file__).parent / "definitions"
    
    def print_tree(path, prefix=""):
        """Print directory tree structure."""
        if not path.exists():
            return
            
        items = sorted([p for p in path.iterdir() if not p.name.startswith('.')])
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "+-- " if is_last else "+-- "
            print(f"{prefix}{current_prefix}{item.name}")

            if item.is_dir():
                extension = "    " if is_last else "|   "
                print_tree(item, prefix + extension)
    
    print("Workflow Registry Structure:")
    print("=" * 40)
    print("definitions/")
    print_tree(registry_path, "")

def main():
    parser = argparse.ArgumentParser(
        description="Workflow Registry CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List workflows (active by default)')
    list_parser.add_argument('--all', action='store_true', help='Show all workflows (including configured)')
    list_parser.add_argument('--name', help='Filter by workflow name or ID')
    list_parser.add_argument('--type', help='Filter by workflow type (e.g. mvr, financial)')
    list_parser.add_argument('--priority', choices=['critical', 'high', 'normal', 'low'], help='Filter by priority level')
    list_parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    # Stats command
    subparsers.add_parser('stats', help='Show workflow statistics')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show detailed workflow information')
    info_parser.add_argument('workflow_id', help='Workflow ID to show info for')
    
    # Structure command
    subparsers.add_parser('structure', help='Show folder structure')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        registry = WorkflowRegistrySystem()
    except Exception as e:
        print(f"Error initializing workflow registry: {e}")
        sys.exit(1)
    
    if args.command == 'list':
        workflows = registry.get_registered_workflows()
        if args.json:
            # Apply filters for JSON output too
            if not args.all:
                workflows = [w for w in workflows if w["status"] == "active"]
            if args.name:
                workflows = [w for w in workflows if args.name.lower() in w["workflow_name"].lower() or args.name.lower() in w["workflow_id"].lower()]
            if args.type:
                workflows = [w for w in workflows if args.type.lower() in w["workflow_type"].lower()]
            if args.priority:
                workflows = [w for w in workflows if args.priority.lower() == w["priority_level"].lower()]
            print(json.dumps(workflows, indent=2))
        else:
            format_workflow_list(workflows, args.all, args.name, args.type, args.priority)
    
    elif args.command == 'stats':
        stats = registry.get_workflow_stats()
        format_stats(stats)
    
    elif args.command == 'info':
        workflows = registry.get_registered_workflows()
        workflow = next((w for w in workflows if w['workflow_id'] == args.workflow_id), None)
        if workflow:
            format_workflow_info(workflow)
        else:
            print(f"Workflow '{args.workflow_id}' not found.")
            print("Available workflows:")
            for w in workflows:
                print(f"  • {w['workflow_id']}")
    
    elif args.command == 'structure':
        show_structure()

class WorkflowCLI:
    """Workflow CLI wrapper for TidyLLM integration."""

    def main(self, args=None):
        """Main entry point for workflow CLI."""
        # Save original sys.argv
        original_argv = sys.argv[:]

        try:
            # Set sys.argv for argparse
            if args is not None:
                sys.argv = ['workflow'] + args

            # Call the main function
            main()

        finally:
            # Restore original sys.argv
            sys.argv = original_argv

if __name__ == '__main__':
    main()