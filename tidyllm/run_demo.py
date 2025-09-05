#!/usr/bin/env python3
"""
Command-Line Demo Runner for TidyLLM Enhanced Demo Systems

Provides a simple command-line interface for running the demo systems.
Useful for quick testing and when web interface is not available.

Usage:
    python run_demo.py                    # Run interactive demo
    python run_demo.py --sparse           # Run SPARSE agreements demo
    python run_demo.py --error-tracking   # Run error tracking demo
    python run_demo.py --protection       # Run protection demo
    python run_demo.py --all              # Run all demos
    python run_demo.py --web              # Launch web interface (if available)
"""

import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any

# Import our standalone demo systems
from sparse_agreements import SparseAgreementManager, execute_sparse_command
from error_tracker import PromptPipelineErrorTracker, track_dspy_error, ErrorSeverity
from demo_protection import DemoProtectionSystem, TransparentModeIndicator, protect_demo_input

def print_section(title: str, level: int = 1):
    """Print a formatted section header"""
    if level == 1:
        print(f"\n{'='*60}")
        print(f"TARGET: {title}")
        print(f"{'='*60}")
    elif level == 2:
        print(f"\n{'-'*50}")
        print(f"INFO: {title}")
        print(f"{'-'*50}")
    else:
        print(f"\nDETAIL: {title}")

def print_result(result: Dict[str, Any], title: str = "Result"):
    """Print a formatted result"""
    print(f"\nRESULT: {title}:")
    print(json.dumps(result, indent=2, default=str))

def demo_sparse_agreements():
    """Demo the SPARSE Agreements System"""
    print_section("SPARSE Agreements System Demo", 1)
    
    print("SUCCESS: SPARSE Agreements System Available")
    
    # Create SPARSE manager
    manager = SparseAgreementManager()
    
    # Show available agreements
    print_section("Available SPARSE Agreements", 2)
    agreements = manager.get_available_agreements()
    for i, agreement in enumerate(agreements, 1):
        print(f"{i}. {agreement}")
    
    # Demo SPARSE command execution
    print_section("SPARSE Command Execution", 2)
    
    # Test commands
    test_commands = [
        "[Performance Test]",
        "[Cost Analysis]", 
        "[Error Analysis]",
        "[Integration Test]",
        "[Scalability Test]",
        "[Security Test]"
    ]
    
    for command in test_commands:
        print(f"\n🔍 Executing: {command}")
        result = execute_sparse_command(command)
        print_result(result, f"SPARSE Result for {command}")
        
        # Show execution mode
        execution_mode = result.get('execution_mode', 'unknown')
        confidence = result.get('confidence', 0.0)
        
        if execution_mode == 'real':
            print("SUCCESS: Real implementation executed")
        elif execution_mode == 'simulated':
            print("🎭 Simulated implementation executed")
        else:
            print("❓ Unknown execution mode")
    
    # Show execution history
    print_section("SPARSE Execution History", 2)
    history = manager.get_execution_history()
    print(f"Total executions: {len(history)}")
    
    if history:
        recent_executions = history[-3:]  # Last 3 executions
        for execution in recent_executions:
            print(f"- {execution['trigger']} ({execution['execution_mode']}) - {execution['timestamp']}")

def demo_error_tracking():
    """Demo the Intelligent Error Tracking & Alerting System"""
    print_section("Intelligent Error Tracking & Alerting Demo", 1)
    
    print("SUCCESS: Error Tracking System Available")
    
    # Create error tracker
    tracker = PromptPipelineErrorTracker()
    
    # Demo error tracking
    print_section("Error Tracking Examples", 2)
    
    # Simulate different types of errors
    error_examples = [
        {
            'error_type': 'prompt_timeout',
            'error_message': 'DSPy operation timed out after 30 seconds',
            'agent_name': 'dspy_wrapper',
            'task_type': 'chain_of_thought',
            'response_time_ms': 35000,
            'user_facing': True
        },
        {
            'error_type': 'llm_api_failure',
            'error_message': 'OpenAI API rate limit exceeded',
            'agent_name': 'dspy_wrapper',
            'task_type': 'predict',
            'model_used': 'gpt-4',
            'user_facing': False
        },
        {
            'error_type': 'cost_exceeded',
            'error_message': 'Operation cost $150.00 exceeds budget limit',
            'agent_name': 'dspy_wrapper',
            'task_type': 'retrieve',
            'cost_usd': 150.0,
            'user_facing': True
        },
        {
            'error_type': 'polars_performance_degradation',
            'error_message': 'Polars operation taking longer than expected',
            'agent_name': 'polars_compat',
            'task_type': 'data_processing',
            'response_time_ms': 15000,
            'context': {'data_size_mb': 500}
        }
    ]
    
    for i, error_data in enumerate(error_examples, 1):
        print(f"\n🔍 Tracking Error {i}: {error_data['error_type']}")
        track_dspy_error(**error_data)
        print(f"SUCCESS: Error tracked: {error_data['error_message'][:50]}...")
    
    # Show error summary
    print_section("Error Summary (Last 24 Hours)", 2)
    summary = tracker.get_error_summary(hours=24)
    print_result(summary, "Error Summary")
    
    # Show alert history
    print_section("Alert History", 2)
    alerts = tracker.get_alert_history(hours=24)
    print(f"Alerts triggered: {len(alerts)}")
    
    if alerts:
        for alert in alerts[-3:]:  # Last 3 alerts
            print(f"- {alert.severity.value.upper()}: {alert.error_type} - {alert.timestamp}")

def demo_demo_protection():
    """Demo the Anti-Sabotage Demo Protection System"""
    print_section("Anti-Sabotage Demo Protection Demo", 1)
    
    print("SUCCESS: Demo Protection System Available")
    
    # Create protection system
    protection_system = DemoProtectionSystem(demo_mode=True)
    indicator = TransparentModeIndicator(protection_system)
    
    # Show current protection status
    print_section("Current Protection Status", 2)
    status = protection_system.get_protection_status()
    print_result(status, "Protection Status")
    
    # Show mode indicator
    print_section("Transparent Mode Indicator", 2)
    mode_info = indicator.get_mode_indicator()
    print_result(mode_info, "Current Mode")
    print(f"\nTARGET: {mode_info['user_message']}")
    
    # Demo input protection
    print_section("Input Protection Examples", 2)
    
    # Test different types of inputs
    test_inputs = [
        {
            'input': "This is a normal text input for DSPy processing",
            'type': 'text',
            'description': 'Normal text input'
        },
        {
            'input': "This contains script tags and executable code <script>alert('test')</script>",
            'type': 'text',
            'description': 'Text with suspicious content'
        },
        {
            'input': "This is a very large input that would normally cause issues " * 1000,
            'type': 'text',
            'description': 'Large text input'
        },
        {
            'input': {'estimated_cost': 75.0, 'operation': 'expensive_dspy_call'},
            'type': 'dict',
            'description': 'Expensive operation'
        }
    ]
    
    for test_input in test_inputs:
        print(f"\n🔍 Testing: {test_input['description']}")
        safe, result = protect_demo_input(test_input['input'], test_input['type'])
        
        if safe:
            print("SUCCESS: Input passed protection checks")
        else:
            print("FAILED: Input blocked by protection system")
        
        print(f"Action: {result['action_taken']}")
        print(f"Message: {result['user_friendly_message']}")
        
        if result['warnings']:
            print("WARNING: Warnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
    
    # Show protection history
    print_section("Protection History (Last 24 Hours)", 2)
    history = protection_system.get_protection_history(hours=24)
    print(f"Protection events: {len(history)}")
    
    if history:
        for event in history[-3:]:  # Last 3 events
            print(f"- {event.event_type}: {event.user_friendly_message}")

def demo_integrated_systems():
    """Demo how all systems work together"""
    print_section("Integrated Systems Demo", 1)
    
    print("SUCCESS: All Enhanced Demo Systems Available")
    
    # Create all systems
    sparse_manager = SparseAgreementManager()
    error_tracker = PromptPipelineErrorTracker()
    protection_system = DemoProtectionSystem()
    mode_indicator = TransparentModeIndicator(protection_system)
    
    # Simulate a complete demo workflow
    print_section("Complete Demo Workflow Simulation", 2)
    
    # Step 1: Check system status
    print("\n🔍 Step 1: System Status Check")
    mode_info = mode_indicator.get_mode_indicator()
    print(f"Mode: {mode_info['mode']} - {mode_info['user_message']}")
    
    # Step 2: Protect user input
    print("\n🔍 Step 2: Input Protection")
    user_input = "[Performance Test] with large dataset"
    safe, protection_result = protection_system.protect_input(user_input, "text")
    print(f"Input safe: {safe}")
    print(f"Protection action: {protection_result['action_taken']}")
    
    if not safe:
        print("FAILED: Input blocked by protection system")
        return
    
    # Step 3: Execute SPARSE command
    print("\n🔍 Step 3: SPARSE Command Execution")
    sparse_result = sparse_manager.find_agreement(user_input)
    if sparse_result:
        execution_result = sparse_manager.execute_agreement(sparse_result)
        print(f"SPARSE execution: {execution_result['execution_mode']}")
        print(f"Confidence: {execution_result['confidence']:.1%}")
    else:
        print("FAILED: No matching SPARSE agreement found")
        return
    
    # Step 4: Track any errors
    print("\n🔍 Step 4: Error Tracking")
    if execution_result['execution_mode'] == 'failed':
        error_data = {
            'error_type': 'dspy_wrapper_failure',
            'error_message': 'SPARSE command execution failed',
            'agent_name': 'dspy_wrapper',
            'task_type': 'sparse_execution',
            'user_facing': True
        }
        error_tracker.track_error(error_data)
        print("WARNING: Error tracked and alert sent")
    
    # Step 5: Show final status
    print("\n🔍 Step 5: Final Status")
    final_mode = mode_indicator.get_mode_indicator()
    print(f"Final mode: {final_mode['mode']}")
    
    # Show authenticity indicator
    authenticity = mode_indicator.show_analysis_authenticity(execution_result)
    print(f"Analysis authenticity: {authenticity['message']}")
    
    print("\nSUCCESS: Integrated demo workflow completed successfully!")

def launch_web_interface():
    """Launch the web interface if available"""
    try:
        import streamlit
        print("🚀 Launching web interface...")
        print("📝 Run this command in a separate terminal:")
        print("   streamlit run visual_demo.py")
        print("\n🌐 Or visit: http://localhost:8501")
    except ImportError:
        print("FAILED: Streamlit not available. Install with: pip install streamlit")
        print("📦 Then run: streamlit run visual_demo.py")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="TidyLLM Enhanced Demo Systems")
    parser.add_argument("--sparse", action="store_true", help="Run SPARSE agreements demo")
    parser.add_argument("--error-tracking", action="store_true", help="Run error tracking demo")
    parser.add_argument("--protection", action="store_true", help="Run protection demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    
    args = parser.parse_args()
    
    print("TARGET: TidyLLM Enhanced Demo Systems")
    print("   Command-line interface for demo team testing")
    
    if args.web:
        launch_web_interface()
    elif args.sparse:
        demo_sparse_agreements()
    elif args.error_tracking:
        demo_error_tracking()
    elif args.protection:
        demo_demo_protection()
    elif args.all:
        demo_sparse_agreements()
        demo_error_tracking()
        demo_demo_protection()
        demo_integrated_systems()
    else:
        # Interactive mode
        print("\nChoose a demo to run:")
        print("1. SPARSE Agreements System")
        print("2. Error Tracking & Alerting")
        print("3. Demo Protection System")
        print("4. Integrated Systems Demo")
        print("5. Launch Web Interface")
        print("6. Run All Demos")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == "1":
            demo_sparse_agreements()
        elif choice == "2":
            demo_error_tracking()
        elif choice == "3":
            demo_demo_protection()
        elif choice == "4":
            demo_integrated_systems()
        elif choice == "5":
            launch_web_interface()
        elif choice == "6":
            demo_sparse_agreements()
            demo_error_tracking()
            demo_demo_protection()
            demo_integrated_systems()
        elif choice == "0":
            print("👋 Goodbye!")
            return
        else:
            print("FAILED: Invalid choice. Please try again.")
    
    print_section("Demo Complete", 1)
    print("SUCCESS: Demo completed successfully!")
    print("🚀 Demo team can now test with sophisticated protection and transparency")

if __name__ == "__main__":
    main()
