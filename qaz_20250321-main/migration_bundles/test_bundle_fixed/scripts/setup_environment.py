#!/usr/bin/env python3
"""
Environment Setup Script

Sets up the environment for VectorQA Sage deployment.
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup the deployment environment"""
    # Add src to Python path
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Add backend to Python path
    backend_path = src_path / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    
    # Set environment variables
    os.environ['VECTORQA_ENV'] = 'aws'
    os.environ['LOG_LEVEL'] = 'INFO'
    
    print("✅ Environment setup complete")

if __name__ == "__main__":
    setup_environment()
