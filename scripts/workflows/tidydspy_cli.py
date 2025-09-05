#!/usr/bin/env python3
"""
TidyLLM CLI Entry Point

Entry point script for the TidyLLM command-line interface.
"""

import sys
import os

# Add src to path so we can import tidyllm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tidyllm.cli import main

if __name__ == "__main__":
    main()