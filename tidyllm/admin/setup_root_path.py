#!/usr/bin/env python3
"""
Setup script to configure root_path for TidyLLM system.
Run this script to automatically detect and set the root_path in settings.yaml.
"""

import os
import yaml
from pathlib import Path

def setup_root_path():
    """Setup root_path in settings.yaml."""
    print("TidyLLM Root Path Setup")
    print("=" * 40)
    
    # Get the directory containing this script (tidyllm/admin)
    script_dir = Path(__file__).parent
    settings_file = script_dir / "settings.yaml"
    
    if not settings_file.exists():
        print(f"❌ Settings file not found: {settings_file}")
        return False
    
    # Detect root path (parent of tidyllm directory)
    tidyllm_dir = script_dir.parent
    root_path = str(tidyllm_dir.parent)
    
    print(f"🔍 Detected root path: {root_path}")
    print(f"📁 TidyLLM directory: {tidyllm_dir}")
    print(f"⚙️  Settings file: {settings_file}")
    
    # Load current settings
    try:
        with open(settings_file, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"❌ Error loading settings: {e}")
        return False
    
    # Update system configuration
    if "system" not in settings:
        settings["system"] = {}
    
    current_root = settings["system"].get("root_path")
    if current_root == root_path:
        print(f"✅ Root path already correctly set: {root_path}")
        return True
    
    # Update root path
    settings["system"]["root_path"] = root_path
    settings["system"]["config_folder"] = "tidyllm/admin"
    settings["system"]["data_folder"] = "tidyllm/data"
    settings["system"]["logs_folder"] = "tidyllm/logs"
    
    # Save updated settings
    try:
        with open(settings_file, 'w', encoding='utf-8') as f:
            yaml.dump(settings, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Root path updated: {current_root} → {root_path}")
        print("✅ Settings saved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error saving settings: {e}")
        return False

def show_current_config():
    """Show current configuration."""
    print("\nCurrent Configuration:")
    print("-" * 40)
    
    script_dir = Path(__file__).parent
    settings_file = script_dir / "settings.yaml"
    
    if not settings_file.exists():
        print("❌ Settings file not found")
        return
    
    try:
        with open(settings_file, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f) or {}
        
        system_config = settings.get("system", {})
        print(f"Root Path: {system_config.get('root_path', 'Not set')}")
        print(f"Config Folder: {system_config.get('config_folder', 'tidyllm/admin')}")
        print(f"Data Folder: {system_config.get('data_folder', 'tidyllm/data')}")
        print(f"Logs Folder: {system_config.get('logs_folder', 'tidyllm/logs')}")
        
    except Exception as e:
        print(f"❌ Error reading settings: {e}")

if __name__ == "__main__":
    success = setup_root_path()
    show_current_config()
    
    if success:
        print("\n🎉 Setup complete! TidyLLM is now configured for this machine.")
        print("💡 To use on another machine, run this script again from that machine.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
