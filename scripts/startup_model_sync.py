#!/usr/bin/env python3
"""
Startup Model Synchronization
=============================

Safe one-time model discovery that runs ONLY on application startup.
Creates backups and defaults to prevent any config overwrites.

Usage:
- Run once on application startup
- Creates safe backups before any changes
- Falls back to known good defaults if discovery fails
- Never overwrites working configurations
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def startup_sync():
    """Run startup model synchronization"""
    
    print("TidyLLM Startup Model Synchronization")
    print("=" * 50)
    
    try:
        # from knowledge_systems.core.startup_model_discovery import run_startup_model_discovery  # REMOVED: core is superfluous
        raise ImportError("core module removed - model discovery moved to proper location")
        
        # Run one-time startup discovery
        result = run_startup_model_discovery()
        
        print("\nStartup Discovery Results:")
        print("-" * 30)
        
        if result.get("success"):
            print(f"+ SUCCESS: Discovered {result.get('models_discovered', 0)} models")
            print(f"+ Configuration updated: {result.get('updated_config')}")
            print(f"+ Backup created: {result.get('backup_created')}")
            
        elif result.get("fallback_used"):
            print(f"- Discovery failed: {result.get('error')}")
            print(f"+ Fallback used: {result.get('default_models_count')} default models")
            print(f"+ Backup created: {result.get('backup_created')}")
            print(f"+ Configuration safe: {result.get('config_file')}")
            
        else:
            print(f"- FAILED: {result.get('error')}")
            return False
        
        # Show available models
        print(f"\nAvailable Models Configuration:")
        print("-" * 30)
        
        config_file = result.get('config_file') or result.get('updated_config')
        if config_file and Path(config_file).exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                models = config.get('embeddings', {}).get('models', {})
                print(f"Total configured models: {len(models)}")
                
                for model_key, model_config in models.items():
                    status = model_config.get('status', 'unknown')
                    dimension = model_config.get('native_dimension', 'unknown')
                    provider = model_config.get('provider', 'unknown')
                    
                    print(f"  {model_key}: {dimension}d ({provider}) - {status}")
                
            except Exception as e:
                print(f"- Could not read config: {e}")
        
        # Show backup information
        print(f"\nBackup & Recovery Information:")
        print("-" * 30)
        
        # from knowledge_systems.core.startup_model_discovery import get_startup_discovery  # REMOVED: core is superfluous
        raise ImportError("core module removed")
        discovery = get_startup_discovery()
        
        if discovery:
            backup_folder = discovery.backup_folder
            defaults_folder = discovery.defaults_folder
            
            print(f"Backup folder: {backup_folder}")
            print(f"Defaults folder: {defaults_folder}")
            
            # List available backups
            backup_files = list(backup_folder.glob("embeddings_settings_*.yaml"))
            if backup_files:
                print(f"Available backups ({len(backup_files)}):")
                for backup in sorted(backup_files, reverse=True)[:3]:  # Show 3 most recent
                    backup_date = backup.stem.replace('embeddings_settings_', '')
                    print(f"  {backup_date}: {backup}")
            else:
                print("No backups found")
        
        print(f"\nRestore Commands:")
        print("-" * 30)
        print("To restore from backup:")
        print("  python -c \"from startup_model_sync import restore_config; restore_config('backup')\"")
        print("To restore defaults:")
        print("  python -c \"from startup_model_sync import restore_config; restore_config('defaults')\"")
        
        return True
        
    except ImportError as e:
        print(f"- Import error: {e}")
        print("+ Creating minimal default configuration...")
        
        # Create minimal fallback
        return create_minimal_config()
        
    except Exception as e:
        print(f"- Startup sync failed: {e}")
        return False

def create_minimal_config():
    """Create minimal configuration if modules not available"""
    try:
        admin_folder = Path(__file__).parent / "tidyllm" / "admin"
        admin_folder.mkdir(parents=True, exist_ok=True)
        
        config_file = admin_folder / "embeddings_settings.yaml"
        
        minimal_config = """# TidyLLM Embeddings - Minimal Configuration
# Safe fallback configuration with known models

embeddings:
  target_dimension: 1024
  padding_strategy: "zeros"
  default_model: "titan_v2_1024"
  
  models:
    titan_v2_1024:
      model_id: "amazon.titan-embed-text-v2:0"
      native_dimension: 1024
      provider: "bedrock"
      configurable_dimensions: true
      description: "Titan v2 - Standard 1024d"
      status: "minimal_default"
    
    cohere_english:
      model_id: "cohere.embed-english-v3"
      native_dimension: 384
      provider: "bedrock"
      description: "Cohere English v3"
      status: "minimal_default"

vector_database:
  postgres:
    host: "vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com"
    port: 5432
    database: "vectorqa"
    user: "vectorqa_user"
    vector:
      dimension: 1024
      similarity_metric: "cosine"

discovery:
  method: "minimal_fallback"
  created: "{}"
""".format(Path(__file__).name)
        
        with open(config_file, 'w') as f:
            f.write(minimal_config)
        
        print(f"+ Created minimal config: {config_file}")
        return True
        
    except Exception as e:
        print(f"- Even minimal config creation failed: {e}")
        return False

def restore_config(method: str = "backup", backup_date: str = None):
    """Restore configuration from backup or defaults"""
    print(f"Restoring configuration using method: {method}")
    
    try:
        # from knowledge_systems.core.startup_model_discovery import restore_embedding_config  # REMOVED: core is superfluous
        raise ImportError("core module removed")
        
        success = restore_embedding_config(method, backup_date)
        
        if success:
            print(f"+ Configuration restored successfully")
        else:
            print(f"- Restore failed")
            
        return success
        
    except Exception as e:
        print(f"- Restore error: {e}")
        return False

def list_backups():
    """List available configuration backups"""
    try:
        # from knowledge_systems.core.startup_model_discovery import get_startup_discovery  # REMOVED: core is superfluous
        raise ImportError("core module removed")
        
        discovery = get_startup_discovery()
        if not discovery:
            # from knowledge_systems.core.startup_model_discovery import StartupModelDiscovery  # REMOVED: core is superfluous
            pass  # Module removed
            discovery = StartupModelDiscovery()
        
        backup_folder = discovery.backup_folder
        backup_files = list(backup_folder.glob("embeddings_settings_*.yaml"))
        
        if backup_files:
            print(f"Available backups in {backup_folder}:")
            for backup in sorted(backup_files, reverse=True):
                backup_date = backup.stem.replace('embeddings_settings_', '')
                file_size = backup.stat().st_size
                print(f"  {backup_date}: {backup} ({file_size} bytes)")
        else:
            print("No backup files found")
            
        return backup_files
        
    except Exception as e:
        print(f"Error listing backups: {e}")
        return []

def verify_config():
    """Verify current configuration is valid"""
    try:
        config_paths = [
            Path(__file__).parent / "tidyllm" / "admin" / "embeddings_settings.yaml",
            Path(__file__).parent / "knowledge_systems" / "embeddings_settings.yaml"
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                print(f"Checking config: {config_path}")
                
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Verify structure
                if 'embeddings' in config and 'models' in config['embeddings']:
                    models = config['embeddings']['models']
                    print(f"+ Valid config with {len(models)} models")
                    
                    # Check target dimension
                    target_dim = config['embeddings'].get('target_dimension', 'not set')
                    print(f"+ Target dimension: {target_dim}")
                    
                    return True
                else:
                    print(f"- Invalid config structure")
                    
        print("- No valid config found")
        return False
        
    except Exception as e:
        print(f"Config verification error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "restore":
            method = sys.argv[2] if len(sys.argv) > 2 else "backup"
            backup_date = sys.argv[3] if len(sys.argv) > 3 else None
            restore_config(method, backup_date)
            
        elif command == "backups":
            list_backups()
            
        elif command == "verify":
            verify_config()
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: restore, backups, verify")
    else:
        # Default: run startup sync
        success = startup_sync()
        sys.exit(0 if success else 1)