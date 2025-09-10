"""
Path utilities for TidyLLM system.
Handles cross-platform path resolution and root folder configuration.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class PathManager:
    """Manages path resolution for TidyLLM system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize PathManager with configuration."""
        self.config = config or {}
        self._root_folder = None
        self._config_folder = None
        self._data_folder = None
        self._logs_folder = None
        
        # Initialize paths
        self._initialize_paths()
    
    def _initialize_paths(self):
        """Initialize all path configurations."""
        # Get system configuration
        system_config = self.config.get("system", {})
        
        # Set root path
        self._root_folder = system_config.get("root_path")
        if not self._root_folder:
            # Fallback: try to detect from current working directory
            self._root_folder = self._detect_root_folder()
        
        # Set relative folders
        self._config_folder = system_config.get("config_folder", "tidyllm/admin")
        self._data_folder = system_config.get("data_folder", "tidyllm/data")
        self._logs_folder = system_config.get("logs_folder", "tidyllm/logs")
        
        logger.info(f"PathManager initialized with root: {self._root_folder}")
    
    def _detect_root_folder(self) -> str:
        """Detect root folder from current working directory."""
        current_dir = os.getcwd()
        
        # Look for tidyllm directory in current path
        if "tidyllm" in current_dir:
            # Find the parent directory that contains tidyllm
            parts = Path(current_dir).parts
            for i, part in enumerate(parts):
                if part == "tidyllm":
                    root_parts = parts[:i]
                    return str(Path(*root_parts)) if root_parts else "."
        
        # Fallback to current directory
        return current_dir
    
    @property
    def root_folder(self) -> str:
        """Get the root folder path."""
        return self._root_folder or "."
    
    @property
    def config_folder(self) -> str:
        """Get the config folder path."""
        return os.path.join(self.root_folder, self._config_folder)
    
    @property
    def data_folder(self) -> str:
        """Get the data folder path."""
        return os.path.join(self.root_folder, self._data_folder)
    
    @property
    def logs_folder(self) -> str:
        """Get the logs folder path."""
        return os.path.join(self.root_folder, self._logs_folder)
    
    def get_config_path(self, filename: str) -> str:
        """Get full path to a config file."""
        return os.path.join(self.config_folder, filename)
    
    def get_data_path(self, filename: str) -> str:
        """Get full path to a data file."""
        return os.path.join(self.data_folder, filename)
    
    def get_logs_path(self, filename: str) -> str:
        """Get full path to a log file."""
        return os.path.join(self.logs_folder, filename)
    
    def ensure_folders_exist(self):
        """Ensure all required folders exist."""
        folders = [self.config_folder, self.data_folder, self.logs_folder]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            logger.debug(f"Ensured folder exists: {folder}")
    
    def get_relative_path(self, full_path: str) -> str:
        """Get relative path from root folder."""
        try:
            return os.path.relpath(full_path, self.root_folder)
        except ValueError:
            return full_path
    
    def is_absolute_path(self, path: str) -> bool:
        """Check if path is absolute."""
        return os.path.isabs(path)
    
    def normalize_path(self, path: str) -> str:
        """Normalize path for current platform."""
        return os.path.normpath(path)

# Global path manager instance
_path_manager: Optional[PathManager] = None

def get_path_manager(config: Optional[Dict[str, Any]] = None) -> PathManager:
    """Get global PathManager instance."""
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager(config)
    return _path_manager

def set_path_manager(path_manager: PathManager):
    """Set global PathManager instance."""
    global _path_manager
    _path_manager = path_manager

def get_config_path(filename: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Get config file path using global PathManager."""
    return get_path_manager(config).get_config_path(filename)

def get_data_path(filename: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Get data file path using global PathManager."""
    return get_path_manager(config).get_data_path(filename)

def get_logs_path(filename: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Get log file path using global PathManager."""
    return get_path_manager(config).get_logs_path(filename)
