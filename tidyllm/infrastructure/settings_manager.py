"""
Centralized Settings Manager for TidyLLM.
Loads settings.yaml once at startup and provides cached access to all configuration.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Lock

logger = logging.getLogger(__name__)

class SettingsManager:
    """
    Centralized settings manager that loads settings.yaml once and caches all configuration.
    Provides thread-safe access to all system configuration.
    """
    
    _instance: Optional['SettingsManager'] = None
    _lock = Lock()
    _settings: Dict[str, Any] = {}
    _settings_file: Optional[str] = None
    _loaded = False
    
    def __new__(cls):
        """Singleton pattern to ensure only one settings manager instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize settings manager (only called once due to singleton)."""
        if not self._loaded:
            self._load_settings()
            self._loaded = True
    
    def _load_settings(self):
        """Load settings from YAML file once at startup."""
        try:
            # Find settings file
            settings_file = self._find_settings_file()
            if not settings_file:
                logger.warning("No settings.yaml found, using defaults")
                self._settings = {}
                return
            
            # Load settings
            with open(settings_file, 'r', encoding='utf-8') as f:
                self._settings = yaml.safe_load(f) or {}
            
            self._settings_file = settings_file
            logger.info(f"Settings loaded from: {settings_file}")
            
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            self._settings = {}
    
    def _find_settings_file(self) -> Optional[str]:
        """Find the settings.yaml file using multiple search strategies."""
        # Strategy 1: Use root_path from environment or detect it
        root_path = os.environ.get('TIDYLLM_ROOT_PATH')
        if root_path:
            settings_path = Path(root_path) / "tidyllm" / "admin" / "settings.yaml"
            if settings_path.exists():
                return str(settings_path)
        
        # Strategy 2: Search from current directory upward
        current_dir = Path.cwd()
        for _ in range(5):
            potential_paths = [
                current_dir / "tidyllm" / "admin" / "settings.yaml",
                current_dir / "admin" / "settings.yaml",
                current_dir / "settings.yaml",
            ]
            
            for path in potential_paths:
                if path.exists():
                    return str(path)
            
            # Move up one directory
            parent = current_dir.parent
            if parent == current_dir:  # Reached root
                break
            current_dir = parent
        
        # Strategy 3: Try common locations
        common_paths = [
            Path.home() / "github" / "tidyllm" / "admin" / "settings.yaml",
            Path("C:/Users/marti/github/tidyllm/admin/settings.yaml"),
            Path("/opt/tidyllm/admin/settings.yaml"),
        ]
        
        for path in common_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def get_settings(self) -> Dict[str, Any]:
        """Get all settings."""
        return self._settings.copy()
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration section."""
        return self._settings.get("system", {})
    
    def get_aws_config(self) -> Dict[str, Any]:
        """Get AWS configuration section."""
        return self._settings.get("aws", {})
    
    def get_workflow_optimizer_config(self) -> Dict[str, Any]:
        """Get workflow optimizer configuration section."""
        return self._settings.get("workflow_optimizer", {})
    
    def get_gateways_config(self) -> Dict[str, Any]:
        """Get gateways configuration section."""
        return self._settings.get("gateways", {})
    
    def get_onboarding_config(self) -> Dict[str, Any]:
        """Get onboarding configuration section."""
        return self._settings.get("onboarding", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration section."""
        return self._settings.get("logging", {})
    
    def get_root_path(self) -> str:
        """Get the configured root path."""
        system_config = self.get_system_config()
        root_path = system_config.get("root_path")
        if root_path:
            return root_path
        
        # Fallback: detect from current directory
        current_dir = Path.cwd()
        if "tidyllm" in str(current_dir):
            parts = current_dir.parts
            for i, part in enumerate(parts):
                if part == "tidyllm":
                    root_parts = parts[:i]
                    return str(Path(*root_parts)) if root_parts else "."
        
        return str(current_dir)
    
    def get_config_path(self, filename: str) -> str:
        """Get full path to a config file."""
        root_path = self.get_root_path()
        config_folder = self.get_system_config().get("config_folder", "tidyllm/admin")
        return os.path.join(root_path, config_folder, filename)
    
    def get_data_path(self, filename: str) -> str:
        """Get full path to a data file."""
        root_path = self.get_root_path()
        data_folder = self.get_system_config().get("data_folder", "tidyllm/data")
        return os.path.join(root_path, data_folder, filename)
    
    def get_logs_path(self, filename: str) -> str:
        """Get full path to a log file."""
        root_path = self.get_root_path()
        logs_folder = self.get_system_config().get("logs_folder", "tidyllm/logs")
        return os.path.join(root_path, logs_folder, filename)
    
    def get_gateway_config(self, gateway_name: str) -> Dict[str, Any]:
        """Get configuration for a specific gateway."""
        gateways_config = self.get_gateways_config()
        return gateways_config.get(gateway_name, {})
    
    def is_gateway_enabled(self, gateway_name: str) -> bool:
        """Check if a gateway is enabled."""
        gateway_config = self.get_gateway_config(gateway_name)
        return gateway_config.get("enabled", True)
    
    def get_gateway_timeout(self, gateway_name: str) -> float:
        """Get timeout for a specific gateway."""
        gateway_config = self.get_gateway_config(gateway_name)
        return gateway_config.get("timeout", 30.0)
    
    def get_gateway_retry_attempts(self, gateway_name: str) -> int:
        """Get retry attempts for a specific gateway."""
        gateway_config = self.get_gateway_config(gateway_name)
        return gateway_config.get("retry_attempts", 3)
    
    def reload_settings(self):
        """Reload settings from file (useful for development)."""
        with self._lock:
            self._load_settings()
            logger.info("Settings reloaded from file")
    
    def refresh_settings(self):
        """Refresh settings from file and update cache."""
        with self._lock:
            if self._settings_file and os.path.exists(self._settings_file):
                try:
                    with open(self._settings_file, 'r', encoding='utf-8') as f:
                        new_settings = yaml.safe_load(f) or {}
                    
                    # Update cached settings
                    self._settings = new_settings
                    logger.info(f"Settings refreshed from: {self._settings_file}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to refresh settings: {e}")
                    return False
            else:
                logger.warning("No settings file found to refresh")
                return False
    
    @property
    def settings_file(self) -> Optional[str]:
        """Get the path to the settings file."""
        return self._settings_file
    
    @property
    def is_loaded(self) -> bool:
        """Check if settings are loaded."""
        return self._loaded

# Global settings manager instance
_settings_manager: Optional[SettingsManager] = None

def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager

def get_settings() -> Dict[str, Any]:
    """Get all settings (convenience function)."""
    return get_settings_manager().get_settings()

def get_system_config() -> Dict[str, Any]:
    """Get system configuration (convenience function)."""
    return get_settings_manager().get_system_config()

def get_workflow_optimizer_config() -> Dict[str, Any]:
    """Get workflow optimizer configuration (convenience function)."""
    return get_settings_manager().get_workflow_optimizer_config()

def get_root_path() -> str:
    """Get root path (convenience function)."""
    return get_settings_manager().get_root_path()

def get_config_path(filename: str) -> str:
    """Get config file path (convenience function)."""
    return get_settings_manager().get_config_path(filename)

def refresh_settings() -> bool:
    """Refresh settings from file (convenience function)."""
    return get_settings_manager().refresh_settings()

def reload_settings():
    """Reload settings from file (convenience function)."""
    get_settings_manager().reload_settings()
