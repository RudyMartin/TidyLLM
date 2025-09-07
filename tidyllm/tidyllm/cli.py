#!/usr/bin/env python3
"""
TidyLLM Command Line Interface
Main entry point for tidyllm command with help and subcommands
"""

import argparse
import sys
from pathlib import Path
from . import __version__

def show_main_help():
    """Show main TidyLLM help with available commands."""
    help_text = f"""
TidyLLM - The Great Walled City of Enterprise AI
Version: {__version__}

USAGE:
    tidyllm <command> [options]

AVAILABLE COMMANDS:

Core Commands:
    help                Show this help message
    version             Show TidyLLM version information
    init                Initialize TidyLLM in current directory
    config              Show current configuration
    status              Show system status and health check

QA Processing:
    qa                  QA file processing commands
    qa-processor        Launch QA processor (alias for qa)
    chat-pdf           Interactive PDF chat mode
    
Testing & Validation:
    test                Run TidyLLM test suites
    test-runner         Launch QA test runner
    validate            Validate TidyLLM installation

Workflow Management:
    workflow            Workflow management commands
    demo                Launch demo interface
    
Development:
    debug               Debug and diagnostic commands
    admin               Administrative commands

EXAMPLES:
    tidyllm help                    # Show this help
    tidyllm qa --help              # QA processing help  
    tidyllm chat-pdf document.pdf  # Chat with PDF
    tidyllm test --all             # Run all tests
    tidyllm demo                   # Launch demo interface
    tidyllm status                 # System health check

For detailed help on any command:
    tidyllm <command> --help

Get started:
    tidyllm init                   # Initialize TidyLLM
    tidyllm status                 # Check system health
    tidyllm demo                   # Try the demo
    
Visit: https://docs.tidyllm.ai for full documentation
"""
    print(help_text)

def show_version():
    """Show version information."""
    print(f"TidyLLM version {__version__}")
    print("The Great Walled City of Enterprise AI")
    
    # Show component availability
    from . import GATEWAYS_AVAILABLE, KNOWLEDGE_SYSTEMS_AVAILABLE, KNOWLEDGE_SERVER_AVAILABLE
    print(f"\nComponents:")
    print(f"  Gateways: {'Available' if GATEWAYS_AVAILABLE else 'Not Available'}")
    print(f"  Knowledge Systems: {'Available' if KNOWLEDGE_SYSTEMS_AVAILABLE else 'Not Available'}")
    print(f"  Knowledge Server: {'Available' if KNOWLEDGE_SERVER_AVAILABLE else 'Not Available'}")

def init_tidyllm():
    """Initialize TidyLLM in current directory."""
    print("[INIT] Initializing TidyLLM in current directory...")
    
    # Create standard folders
    folders = ['qa_files', 'qa_reports', 'qa_config', 'workflows', 'knowledge']
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)
        print(f"  Created: {folder}/")
    
    # Create basic config
    config_content = """# TidyLLM Configuration
# Auto-generated configuration file

# QA Processing Settings
qa:
  watch_folder: './qa_files'
  output_folder: './qa_reports'
  config_folder: './qa_config'
  mlflow_enabled: true

# Model Settings  
model:
  default_provider: 'anthropic'
  default_model: 'claude-3-sonnet'
  experiment_prefix: 'tidyllm'

# Integration Settings
integrations:
  aws_enabled: true
  mlflow_enabled: true
  database_enabled: false
"""
    
    config_path = Path('tidyllm_config.yaml')
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"  Created: {config_path}")
    else:
        print(f"  Exists: {config_path}")
    
    print("\n[SUCCESS] TidyLLM initialized!")
    print("Next steps:")
    print("  tidyllm status     # Check system health")
    print("  tidyllm qa --help  # Learn about QA processing")
    print("  tidyllm demo       # Try the demo interface")

def show_status():
    """Show system status and health check."""
    print("[STATUS] TidyLLM System Health Check")
    print("=" * 40)
    
    # Check core imports
    try:
        from . import gateways
        print("[OK] Gateways: Available")
        
        # Try to init gateways
        gateway_registry = gateways.init_gateways()
        print("[OK] Gateway Registry: Initialized")
        
    except Exception as e:
        print(f"[ERROR] Gateways: Error - {e}")
    
    # Check MLflow
    try:
        import mlflow
        print(f"[OK] MLflow: Available ({mlflow.__version__})")
        print(f"  Tracking URI: {mlflow.get_tracking_uri()}")
    except ImportError:
        print("[ERROR] MLflow: Not available")
    
    # Check AWS
    try:
        import boto3
        print(f"[OK] AWS SDK: Available ({boto3.__version__})")
    except ImportError:
        print("[ERROR] AWS SDK: Not available")
    
    # Check key dependencies
    deps = {
        'pandas': 'pandas',
        'yaml': 'pyyaml', 
        'dspy': 'dspy-ai',
        'openai': 'openai',
        'anthropic': 'anthropic'
    }
    
    for import_name, package_name in deps.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"[OK] {package_name}: Available ({version})")
        except ImportError:
            print(f"[ERROR] {package_name}: Not available")
    
    print("\n[RECOMMENDATION]")
    print("If any components show as 'Not available':")
    print("  pip install -e .[all]")

def launch_qa_processor():
    """Launch QA processor with arguments."""
    print("[LAUNCH] Starting QA Processor...")
    
    # Import and run qa_processor
    try:
        # Try to import from the parent directory
        import sys
        from pathlib import Path
        
        # Add parent directory to path
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        # Import and run main from qa_processor
        import qa_processor
        qa_processor.main()
        
    except ImportError as e:
        print(f"[ERROR] Could not launch QA processor: {e}")
        print("Ensure qa_processor.py is available in the package directory")
        sys.exit(1)

def launch_test_runner():
    """Launch QA test runner with arguments.""" 
    print("[LAUNCH] Starting QA Test Runner...")
    
    try:
        import sys
        from pathlib import Path
        
        # Add parent directory to path
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        # Import and run main from qa_test_runner
        import qa_test_runner
        qa_test_runner.main()
        
    except ImportError as e:
        print(f"[ERROR] Could not launch test runner: {e}")
        print("Ensure qa_test_runner.py is available in the package directory")
        sys.exit(1)

def launch_demo():
    """Launch TidyLLM demo interface."""
    print("[LAUNCH] Starting TidyLLM Demo Interface...")
    
    try:
        from .demos import launch_demo
        launch_demo.main()
    except ImportError as e:
        print(f"[INFO] Demo interface not available: {e}")
        print("Install with: pip install -e .[web]")
        print("Or run: python -m streamlit run tidyllm/demos/visual_demo.py")

def show_config():
    """Show current TidyLLM configuration."""
    print("[CONFIG] TidyLLM Configuration")
    print("=" * 30)
    
    # Show version and components
    show_version()
    
    # Show config file if exists
    config_path = Path('tidyllm_config.yaml')
    if config_path.exists():
        print(f"\nConfiguration file: {config_path}")
        with open(config_path, 'r') as f:
            content = f.read()
        print("\nCurrent configuration:")
        print(content)
    else:
        print(f"\nNo configuration file found at: {config_path}")
        print("Run 'tidyllm init' to create default configuration")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='tidyllm',
        description='TidyLLM - The Great Walled City of Enterprise AI',
        add_help=False  # We'll handle help ourselves
    )
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        show_main_help()
        return
    
    command = sys.argv[1].lower()
    
    # Handle main commands
    if command in ['help', '--help', '-h']:
        show_main_help()
    elif command in ['version', '--version', '-v']:
        show_version()
    elif command == 'init':
        init_tidyllm()
    elif command == 'status':
        show_status()
    elif command == 'config':
        show_config()
    elif command in ['qa', 'qa-processor']:
        # Remove the command from sys.argv and launch QA processor
        sys.argv = ['qa_processor.py'] + sys.argv[2:]
        launch_qa_processor()
    elif command in ['test', 'test-runner']:
        # Remove the command from sys.argv and launch test runner
        sys.argv = ['qa_test_runner.py'] + sys.argv[2:]
        launch_test_runner()
    elif command == 'chat-pdf':
        # Handle chat-pdf command
        if len(sys.argv) < 3:
            print("[ERROR] chat-pdf requires a PDF file argument")
            print("Usage: tidyllm chat-pdf <file.pdf>")
            sys.exit(1)
        
        pdf_file = sys.argv[2]
        sys.argv = ['qa_processor.py', '--chat-pdf', pdf_file] + sys.argv[3:]
        launch_qa_processor()
    elif command == 'demo':
        launch_demo()
    elif command == 'workflow':
        print("[INFO] Workflow management commands coming soon!")
        print("For now, use: python -m tidyllm.workflows")
    elif command == 'debug':
        # Launch QA processor in debug mode
        sys.argv = ['qa_processor.py', '--debug-config'] + sys.argv[2:]
        launch_qa_processor()
    elif command == 'admin':
        print("[INFO] Admin commands coming soon!")
        print("For now, use: python -m tidyllm.admin")
    elif command == 'validate':
        # Run validation/health check
        show_status()
    else:
        print(f"[ERROR] Unknown command: '{command}'")
        print("Run 'tidyllm help' to see available commands")
        sys.exit(1)

if __name__ == '__main__':
    main()