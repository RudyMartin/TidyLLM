"""
################################################################################
# *** IMPORTANT: READ docs/2025-09-08/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md ***
# *** BEFORE PLANNING ANY CHANGES TO THIS FILE ***
################################################################################

TidyLLM Configuration Manager - JSON and YAML Configuration Management

Simplified configuration manager for handling JSON and YAML configuration files.
Supports dot notation for nested key access and auto-detects file format based on extension.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


class ConfigManager:
    """Simple configuration manager for JSON and YAML files with dot notation support"""
    
    def __init__(self, config_path: str):
        """
        Initialize ConfigManager with a configuration file path
        
        Auto-detects file format based on extension:
        - .json for JSON files
        - .yaml, .yml for YAML files
        
        Args:
            config_path: Path to the configuration file (.json, .yaml, or .yml)
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._file_type = self._detect_file_type()
        self._load_config()
    
    def _detect_file_type(self) -> str:
        """
        Detect file type based on extension
        
        Returns:
            'json' for .json files, 'yaml' for .yaml/.yml files
        """
        ext = os.path.splitext(self.config_path)[1].lower()
        if ext in ['.yaml', '.yml']:
            if yaml is None:
                raise ImportError("PyYAML is required for YAML file support. Install with: pip install pyyaml")
            return 'yaml'
        else:
            return 'json'
    
    def _load_config(self) -> None:
        """Load configuration from JSON or YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self._file_type == 'yaml':
                        self._config = yaml.safe_load(f) or {}
                    else:
                        self._config = json.load(f)
                logger.info(f"Loaded {self._file_type.upper()} configuration from {self.config_path}")
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                self._config = {}
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            logger.error(f"Invalid {self._file_type.upper()} in config file {self.config_path}: {e}")
            self._config = {}
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {e}")
            self._config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key, supporting dot notation for nested keys
        
        Args:
            key: Configuration key (supports dot notation like "nested.key")
            default: Default value to return if key is not found
            
        Returns:
            Configuration value or default if not found
            
        Examples:
            manager.get("test_key") -> "test_value"
            manager.get("nested.key") -> "value"
        """
        try:
            # Split the key by dots for nested access
            keys = key.split('.')
            current = self._config
            
            # Navigate through nested dictionaries
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            
            return current
            
        except Exception as e:
            logger.debug(f"Error getting config key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key, supporting dot notation for nested keys
        
        Args:
            key: Configuration key (supports dot notation like "nested.key")
            value: Value to set
        """
        try:
            keys = key.split('.')
            current = self._config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]
            
            # Set the final key
            current[keys[-1]] = value
            
        except Exception as e:
            logger.error(f"Error setting config key '{key}': {e}")
    
    def save(self) -> bool:
        """
        Save current configuration to JSON or YAML file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path) or ".", exist_ok=True)
            
            # Save to appropriate format with pretty formatting
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self._file_type == 'yaml':
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"{self._file_type.upper()} configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config to {self.config_path}: {e}")
            return False
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        try:
            for key, value in updates.items():
                self.set(key, value)
        except Exception as e:
            logger.error(f"Error updating config: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration data
        
        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()
    
    def has_key(self, key: str) -> bool:
        """
        Check if configuration key exists
        
        Args:
            key: Configuration key (supports dot notation)
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            keys = key.split('.')
            current = self._config
            
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return False
            
            return True
            
        except Exception:
            return False


# Utility functions
def load_config(config_path: str) -> ConfigManager:
    """Load configuration from JSON or YAML file (auto-detected by extension)"""
    return ConfigManager(config_path)