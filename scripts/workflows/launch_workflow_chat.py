#!/usr/bin/env python3
"""
Launch TidyLLM Workflow Chat Interface
=====================================

Simple launcher for the chat-based workflow interface.
"""

import sys
import subprocess
from pathlib import Path

def check_streamlit():
    """Check if Streamlit is available."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install Streamlit if not available."""
    print("Installing Streamlit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    print("Streamlit installed successfully!")

def launch_interface():
    """Launch the workflow chat interface."""
    interface_path = Path(__file__).parent / "tidyllm" / "chat_workflow_interface.py"
    
    if not interface_path.exists():
        print(f"Error: Interface file not found at {interface_path}")
        return 1
    
    print("=" * 60)
    print("🚀 LAUNCHING TIDYLLM WORKFLOW CHAT")
    print("=" * 60)
    print("Interface Features:")
    print("  ✅ Flow Agreement Sidebar (25%)")
    print("  ✅ Chat Interface (75%)")
    print("  ✅ MVR Analysis Workflow")
    print("  ✅ Bracket Commands [mvr_analysis]")
    print("  ✅ 4-Stage Drop Zone Cascade")
    print("=" * 60)
    print()
    print("Opening in your browser...")
    print("Use Ctrl+C to stop the server")
    print()
    
    # Launch Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", str(interface_path)]
    subprocess.run(cmd)
    
    return 0

def show_demo_info():
    """Show information about the demo."""
    print("=" * 60)
    print("TIDYLLM WORKFLOW CHAT DEMO")
    print("=" * 60)
    
    print("\n🎯 INTERFACE LAYOUT:")
    print("  Left Sidebar (25%): Flow Agreement selection")
    print("  Main Area (75%): Chat interface")
    
    print("\n📋 AVAILABLE WORKFLOWS:")
    print("  • MVR Analysis - 4-stage document processing")
    print("  • Research Synthesis - Multi-document analysis")
    print("  • Compliance Review - Regulatory extraction")
    print("  • Document Classification - Auto-sorting")
    
    print("\n⚡ BRACKET COMMANDS:")
    print("  [mvr_analysis] - Start full MVR workflow")
    print("  [research_synth] - Research analysis")
    print("  [compliance_check] - Compliance review")
    print("  [doc_classify] - Document classification")
    
    print("\n🔄 MVR WORKFLOW STAGES:")
    print("  1. mvr_tag/ - Classification & metadata extraction")
    print("  2. mvr_qa/ - MVR vs VST comparison")
    print("  3. mvr_peer/ - Domain RAG peer review")
    print("  4. mvr_report/ - Final PDF/JSON generation")
    
    print("\n💬 CHAT EXAMPLES:")
    print("  'I need to analyze MVR documents'")
    print("  'Start the research synthesis workflow'") 
    print("  '[mvr_analysis]' - Direct command")
    print("  'Help me with compliance review'")
    
    print("\n📁 TIDYMART INTEGRATION:")
    print("  ✅ Automatic data storage")
    print("  ✅ Processing trail logging")
    print("  ✅ Metadata validation")
    print("  ✅ Quality assurance gates")

def main():
    """Main launcher function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--info":
        show_demo_info()
        return 0
    
    # Check dependencies
    if not check_streamlit():
        print("Streamlit not found. Installing...")
        try:
            install_streamlit()
        except Exception as e:
            print(f"Failed to install Streamlit: {e}")
            print("Please install manually: pip install streamlit")
            return 1
    
    # Launch interface
    return launch_interface()

if __name__ == "__main__":
    sys.exit(main())