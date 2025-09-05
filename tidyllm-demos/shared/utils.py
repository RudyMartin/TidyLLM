"""
Shared utilities for TidyLLM demos
"""
import os
import yaml
import streamlit as st
from typing import Dict, Any, Optional
from pathlib import Path

def load_settings(settings_path: str = "settings.yaml") -> Dict[str, Any]:
    """Load settings from YAML file"""
    try:
        with open(settings_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Settings file not found: {settings_path}")
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error parsing settings file: {e}")
        return {}

def save_settings(settings: Dict[str, Any], settings_path: str = "settings.yaml") -> bool:
    """Save settings to YAML file"""
    try:
        with open(settings_path, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving settings: {e}")
        return False

def get_env_var(key: str, default: str = "") -> str:
    """Get environment variable with fallback"""
    return os.getenv(key, default)

def format_cost(cost: float, decimals: int = 4) -> str:
    """Format cost for display"""
    return f"${cost:.{decimals}f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage for display"""
    return f"{value:.{decimals}f}%"

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent

def create_backup(file_path: str, backup_count: int = 5) -> bool:
    """Create a backup of a file"""
    try:
        if not os.path.exists(file_path):
            return False
        
        base_path = Path(file_path)
        timestamp = int(os.path.getmtime(file_path))
        backup_path = base_path.parent / f"{base_path.stem}.backup.{timestamp}{base_path.suffix}"
        
        # Copy file
        import shutil
        shutil.copy2(file_path, backup_path)
        
        # Clean old backups
        pattern = f"{base_path.stem}.backup.*{base_path.suffix}"
        backups = sorted(base_path.parent.glob(pattern), key=os.path.getmtime, reverse=True)
        
        for backup in backups[backup_count:]:
            backup.unlink()
        
        return True
    except Exception as e:
        st.error(f"Error creating backup: {e}")
        return False


