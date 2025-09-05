#!/usr/bin/env python3
"""
Import Helper Utilities

Provides robust import patterns that work across different deployment environments.
This eliminates the need for try-except import blocks throughout the codebase.
"""

import sys
import os
from pathlib import Path
from typing import Any, Optional

class ImportManager:
    """Manages imports across different environments"""
    
    def __init__(self):
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup Python paths for consistent imports"""
        # Get the src directory
        src_dir = Path(__file__).parent.parent
        
        # Add src to path if not already there
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        
        # Add config to path for direct config imports
        config_dir = src_dir / "config"
        if config_dir.exists() and str(config_dir) not in sys.path:
            sys.path.insert(0, str(config_dir))
    
    def safe_import(self, module_path: str, fallback_paths: list = None) -> Optional[Any]:
        """
        Safely import a module with fallback paths
        
        Args:
            module_path: Primary module path to import
            fallback_paths: List of fallback paths to try
            
        Returns:
            Imported module or None if all imports fail
        """
        fallback_paths = fallback_paths or []
        all_paths = [module_path] + fallback_paths
        
        for path in all_paths:
            try:
                return __import__(path, fromlist=[''])
            except ImportError:
                continue
        
        raise ImportError(f"Could not import any of: {all_paths}")

# Global import manager instance
import_manager = ImportManager()

# Convenience functions for common imports
def import_config():
    """Import configuration with environment fallbacks"""
    return import_manager.safe_import(
        "config.settings",
        ["settings", "src.config.settings"]
    )

def import_core_module(module_name: str):
    """Import core module with fallbacks"""
    return import_manager.safe_import(
        f"core.{module_name}",
        [f"src.core.{module_name}", f"backend.core.{module_name}"]
    )

def import_mcp_module(module_name: str):
    """Import MCP module with fallbacks"""
    return import_manager.safe_import(
        f"mcp.{module_name}",
        [f"src.mcp.{module_name}", f"backend.mcp.{module_name}"]
    )
