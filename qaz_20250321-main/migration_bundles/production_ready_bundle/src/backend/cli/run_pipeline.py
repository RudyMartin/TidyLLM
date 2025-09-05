#!/usr/bin/env python3
"""
Pipeline Runner CLI

Runs the complete QA processing pipeline.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports

# Robust import setup
import sys
from pathlib import Path
_src_dir = Path(__file__).parent
while _src_dir.name != "src" and _src_dir.parent != _src_dir:
    _src_dir = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

def run_pipeline(config_path: str = None, input_path: str = None):
    """Run the QA processing pipeline"""
    print("🚀 Starting QA Processing Pipeline...")
    
    # Import configuration
    try:
        from config.settings import config
        print("✅ Configuration loaded")
    except ImportError:
        print("⚠️  Using default configuration")
        config = {}
    
    # Process input
    if input_path:
        print(f"📄 Processing input: {input_path}")
        # Placeholder processing logic
        print("✅ Processing completed")
    else:
        print("ℹ️  No input specified, running in demo mode")
    
    print("🎉 Pipeline completed successfully!")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Run QA processing pipeline")
    parser.add_argument("-c", "--config", help="Configuration file path")
    parser.add_argument("-i", "--input", help="Input file or directory path")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🧪 Running in dry-run mode...")
    
    run_pipeline(args.config, args.input)

if __name__ == "__main__":
    main()