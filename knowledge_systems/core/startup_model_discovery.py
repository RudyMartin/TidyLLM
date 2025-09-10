"""
Startup Model Discovery with Safe Backup System
===============================================

Runs model discovery ONLY on application startup to avoid permission issues.
Creates safe backups and default configurations to prevent blank overwrites.

Features:
- One-time startup discovery only
- Safe backup system with versioning  
- Default configuration restoration
- Graceful fallback if discovery fails
- No special permissions required after startup
"""

import json
import yaml
import logging
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

class StartupModelDiscovery:
    """Safe startup-only model discovery with backup system"""
    
    def __init__(self, admin_folder: str = None):
        """
        Initialize startup discovery
        
        Args:
            admin_folder: Path to admin folder containing configs
        """
        if admin_folder is None:
            admin_folder = Path(__file__).parent.parent.parent / "tidyllm" / "admin"
        
        self.admin_folder = Path(admin_folder)
        self.config_file = self.admin_folder / "embeddings_settings.yaml"
        self.backup_folder = self.admin_folder / "backups"
        self.defaults_folder = self.admin_folder / "defaults"
        
        # Ensure folders exist
        self.backup_folder.mkdir(exist_ok=True)
        self.defaults_folder.mkdir(exist_ok=True)
        
        # Default models (fallback if discovery fails)
        self.default_models = self._get_default_models()
        
        # Discovery results
        self.discovered_models = {}
        self.discovery_successful = False
        self.discovery_error = None
    
    def run_startup_discovery(self) -> Dict[str, Any]:
        """
        Run model discovery on startup only
        
        Returns:
            Discovery results and status
        """
        logger.info("Starting one-time model discovery on application startup...")
        
        # Create backup before any changes
        backup_path = self._create_backup()
        
        # Ensure default configuration exists
        self._ensure_default_config()
        
        try:
            # Attempt model discovery (may fail due to permissions)
            models = self._discover_bedrock_models()
            
            if models:
                self.discovered_models = models
                self.discovery_successful = True
                
                # Update configuration with discovered models
                self._update_config_safely(models)
                
                logger.info(f"✅ Startup discovery successful: {len(models)} models found")
                return {
                    "success": True,
                    "models_discovered": len(models),
                    "models": models,
                    "backup_created": str(backup_path),
                    "updated_config": str(self.config_file)
                }
            else:
                # No models discovered but no error
                logger.warning("⚠️ No embedding models discovered")
                return self._fallback_to_defaults(backup_path, "No models found")
                
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"⚠️ AWS permissions issue during startup discovery: {e}")
            return self._fallback_to_defaults(backup_path, str(e))
            
        except Exception as e:
            logger.error(f"❌ Startup discovery failed: {e}")
            self.discovery_error = str(e)
            return self._fallback_to_defaults(backup_path, str(e))
    
    def _create_backup(self) -> Path:
        """Create timestamped backup of current config"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_folder / f"embeddings_settings_{timestamp}.yaml"
        
        try:
            if self.config_file.exists():
                shutil.copy2(self.config_file, backup_path)
                logger.info(f"✅ Backup created: {backup_path}")
            else:
                logger.info("ℹ️ No existing config to backup")
                
        except Exception as e:
            logger.warning(f"⚠️ Backup creation failed: {e}")
            # Continue without backup - not critical
            
        return backup_path
    
    def _ensure_default_config(self):
        """Ensure default configuration file exists"""
        default_config_path = self.defaults_folder / "embeddings_settings_default.yaml"
        
        # Create default config if it doesn't exist
        if not default_config_path.exists():
            self._create_default_config(default_config_path)
        
        # If main config doesn't exist, copy from default
        if not self.config_file.exists():
            try:
                shutil.copy2(default_config_path, self.config_file)
                logger.info("✅ Created initial config from default")
            except Exception as e:
                logger.error(f"❌ Failed to create initial config: {e}")
    
    def _create_default_config(self, config_path: Path):
        """Create comprehensive default configuration"""
        default_config = {
            "# TidyLLM Embeddings Configuration": None,
            "# Auto-generated with safe defaults": None,
            "# Last updated": datetime.now().isoformat(),
            
            "embeddings": {
                "target_dimension": 1024,
                "padding_strategy": "zeros",
                "default_model": "titan_v2_1024",
                
                "models": {}
            },
            
            "vector_database": {
                "postgres": {
                    "host": "vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com",
                    "port": 5432,
                    "database": "vectorqa",
                    "user": "vectorqa_user",
                    "vector": {
                        "dimension": 1024,
                        "index_type": "ivfflat",
                        "similarity_metric": "cosine"
                    }
                }
            },
            
            "multi_model": {
                "strategy": {
                    "allow_mixed_models": True,
                    "dimension_handling": "standardize",
                    "track_model_metadata": True
                }
            },
            
            "discovery": {
                "last_startup_discovery": datetime.now().isoformat(),
                "discovery_method": "startup_only",
                "fallback_to_defaults": True
            }
        }
        
        # Add default models
        for model_key, model_config in self.default_models.items():
            default_config["embeddings"]["models"][model_key] = model_config
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"✅ Created default configuration: {config_path}")
        except Exception as e:
            logger.error(f"❌ Failed to create default config: {e}")
    
    def _get_default_models(self) -> Dict[str, Dict[str, Any]]:
        """Get hardcoded default models (known good models)"""
        return {
            "titan_v1": {
                "model_id": "amazon.titan-embed-text-v1",
                "native_dimension": 1536,
                "provider": "bedrock",
                "configurable_dimensions": False,
                "description": "Titan v1 - High dimensional embeddings",
                "status": "default"
            },
            "titan_v2_256": {
                "model_id": "amazon.titan-embed-text-v2:0",
                "native_dimension": 256,
                "provider": "bedrock",
                "configurable_dimensions": True,
                "supported_dimensions": [256, 512, 1024],
                "description": "Titan v2 - Compact 256d",
                "status": "default"
            },
            "titan_v2_512": {
                "model_id": "amazon.titan-embed-text-v2:0", 
                "native_dimension": 512,
                "provider": "bedrock",
                "configurable_dimensions": True,
                "supported_dimensions": [256, 512, 1024],
                "description": "Titan v2 - Medium 512d",
                "status": "default"
            },
            "titan_v2_1024": {
                "model_id": "amazon.titan-embed-text-v2:0",
                "native_dimension": 1024,
                "provider": "bedrock", 
                "configurable_dimensions": True,
                "supported_dimensions": [256, 512, 1024],
                "description": "Titan v2 - Standard 1024d",
                "status": "default"
            },
            "cohere_english": {
                "model_id": "cohere.embed-english-v3",
                "native_dimension": 384,
                "provider": "bedrock",
                "configurable_dimensions": False,
                "description": "Cohere English v3",
                "status": "default"
            },
            "cohere_multilingual": {
                "model_id": "cohere.embed-multilingual-v3", 
                "native_dimension": 384,
                "provider": "bedrock",
                "configurable_dimensions": False,
                "description": "Cohere Multilingual v3",
                "status": "default"
            }
        }
    
    def _discover_bedrock_models(self) -> Dict[str, Dict[str, Any]]:
        """Attempt to discover models from Bedrock (may fail)"""
        try:
            # AUDIT COMPLIANCE: Use UnifiedSessionManager instead of direct boto3
            try:
                from tidyllm.infrastructure.session.unified import UnifiedSessionManager
                session_manager = UnifiedSessionManager()
                bedrock = session_manager.get_bedrock_client()
            except ImportError:
                # NO FALLBACK - UnifiedSessionManager is required
                raise RuntimeError("StartupModelDiscovery: UnifiedSessionManager is required for Bedrock access")
            response = bedrock.list_foundation_models()
            
            discovered = {}
            
            for model in response.get('modelSummaries', []):
                model_id = model.get('modelId', '')
                
                # Only process known embedding models
                if self._is_known_embedding_model(model_id):
                    config_key = self._generate_config_key(model_id)
                    
                    discovered[config_key] = {
                        "model_id": model_id,
                        "model_name": model.get('modelName', model_id),
                        "native_dimension": self._get_known_dimension(model_id),
                        "provider": "bedrock",
                        "configurable_dimensions": self._is_configurable_model(model_id),
                        "supported_dimensions": self._get_supported_dimensions(model_id),
                        "description": f"{model.get('modelName', model_id)} - Auto-discovered",
                        "status": "discovered",
                        "discovered_at": datetime.now().isoformat()
                    }
            
            return discovered
            
        except Exception as e:
            logger.warning(f"Bedrock model discovery failed: {e}")
            return {}
    
    def _is_known_embedding_model(self, model_id: str) -> bool:
        """Check if this is a known embedding model"""
        embedding_patterns = [
            "amazon.titan-embed-text",
            "cohere.embed-english",
            "cohere.embed-multilingual"
        ]
        return any(pattern in model_id for pattern in embedding_patterns)
    
    def _generate_config_key(self, model_id: str) -> str:
        """Generate configuration key from model ID"""
        if "titan-embed-text-v1" in model_id:
            return "titan_v1"
        elif "titan-embed-text-v2" in model_id:
            return "titan_v2_1024"  # Default to 1024
        elif "cohere.embed-english" in model_id:
            return "cohere_english"
        elif "cohere.embed-multilingual" in model_id:
            return "cohere_multilingual"
        else:
            # Fallback key generation
            return model_id.replace(".", "_").replace(":", "_").replace("-", "_")
    
    def _get_known_dimension(self, model_id: str) -> int:
        """Get known dimension for model"""
        if "titan-embed-text-v1" in model_id:
            return 1536
        elif "titan-embed-text-v2" in model_id:
            return 1024
        elif "cohere.embed" in model_id:
            return 384
        else:
            return 1024  # Default
    
    def _is_configurable_model(self, model_id: str) -> bool:
        """Check if model supports configurable dimensions"""
        return "titan-embed-text-v2" in model_id
    
    def _get_supported_dimensions(self, model_id: str) -> List[int]:
        """Get supported dimensions for model"""
        if "titan-embed-text-v2" in model_id:
            return [256, 512, 1024]
        else:
            return [self._get_known_dimension(model_id)]
    
    def _update_config_safely(self, discovered_models: Dict[str, Dict[str, Any]]):
        """Safely update configuration with discovered models"""
        try:
            # Load existing config
            config = self._load_existing_config()
            
            # Ensure structure exists
            if 'embeddings' not in config:
                config['embeddings'] = {}
            if 'models' not in config['embeddings']:
                config['embeddings']['models'] = {}
            
            # Add/update discovered models
            updated_count = 0
            for model_key, model_config in discovered_models.items():
                if model_key not in config['embeddings']['models']:
                    config['embeddings']['models'][model_key] = model_config
                    updated_count += 1
                else:
                    # Update existing model with new discovery info
                    config['embeddings']['models'][model_key].update(model_config)
                    updated_count += 1
            
            # Update discovery metadata
            if 'discovery' not in config:
                config['discovery'] = {}
            
            config['discovery'].update({
                "last_startup_discovery": datetime.now().isoformat(),
                "models_discovered": len(discovered_models),
                "models_updated": updated_count,
                "discovery_successful": True
            })
            
            # Write updated config
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"✅ Configuration updated with {updated_count} models")
            
        except Exception as e:
            logger.error(f"❌ Failed to update configuration: {e}")
            raise
    
    def _load_existing_config(self) -> Dict[str, Any]:
        """Load existing configuration file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return yaml.safe_load(f) or {}
            else:
                return {}
        except Exception as e:
            logger.warning(f"Failed to load existing config: {e}")
            return {}
    
    def _fallback_to_defaults(self, backup_path: Path, error_message: str) -> Dict[str, Any]:
        """Fallback to default models when discovery fails"""
        logger.info("📋 Using default models configuration")
        
        try:
            # Ensure config exists with defaults
            config = self._load_existing_config()
            
            # Add defaults if models section is empty
            if not config.get('embeddings', {}).get('models'):
                if 'embeddings' not in config:
                    config['embeddings'] = {}
                if 'models' not in config['embeddings']:
                    config['embeddings']['models'] = {}
                
                # Add default models
                for model_key, model_config in self.default_models.items():
                    config['embeddings']['models'][model_key] = model_config
                
                # Update discovery metadata
                if 'discovery' not in config:
                    config['discovery'] = {}
                
                config['discovery'].update({
                    "last_startup_discovery": datetime.now().isoformat(),
                    "discovery_successful": False,
                    "discovery_error": error_message,
                    "fallback_to_defaults": True,
                    "default_models_count": len(self.default_models)
                })
                
                # Write config with defaults
                with open(self.config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                logger.info("✅ Configuration restored with default models")
            
            return {
                "success": False,
                "fallback_used": True,
                "error": error_message,
                "default_models_count": len(self.default_models),
                "backup_created": str(backup_path),
                "config_file": str(self.config_file)
            }
            
        except Exception as e:
            logger.error(f"❌ Even fallback to defaults failed: {e}")
            return {
                "success": False,
                "fallback_failed": True,
                "error": f"Original: {error_message}, Fallback: {str(e)}"
            }
    
    def restore_from_backup(self, backup_date: str = None) -> bool:
        """Restore configuration from backup"""
        try:
            if backup_date:
                backup_file = self.backup_folder / f"embeddings_settings_{backup_date}.yaml"
            else:
                # Find most recent backup
                backup_files = list(self.backup_folder.glob("embeddings_settings_*.yaml"))
                if not backup_files:
                    logger.error("No backup files found")
                    return False
                backup_file = max(backup_files, key=lambda x: x.stat().st_mtime)
            
            if backup_file.exists():
                shutil.copy2(backup_file, self.config_file)
                logger.info(f"✅ Restored configuration from {backup_file}")
                return True
            else:
                logger.error(f"Backup file not found: {backup_file}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Restore from backup failed: {e}")
            return False
    
    def restore_defaults(self) -> bool:
        """Restore to factory defaults"""
        try:
            default_config = self.defaults_folder / "embeddings_settings_default.yaml"
            
            if default_config.exists():
                shutil.copy2(default_config, self.config_file)
                logger.info("✅ Restored to factory defaults")
                return True
            else:
                # Create new defaults
                self._create_default_config(self.config_file)
                logger.info("✅ Created fresh default configuration")
                return True
                
        except Exception as e:
            logger.error(f"❌ Restore to defaults failed: {e}")
            return False

# Global instance for startup discovery
_startup_discovery = None

def run_startup_model_discovery(admin_folder: str = None) -> Dict[str, Any]:
    """
    Run one-time model discovery on application startup
    
    Args:
        admin_folder: Path to admin configuration folder
        
    Returns:
        Discovery results
    """
    global _startup_discovery
    
    _startup_discovery = StartupModelDiscovery(admin_folder)
    return _startup_discovery.run_startup_discovery()

def get_startup_discovery() -> Optional[StartupModelDiscovery]:
    """Get the startup discovery instance"""
    return _startup_discovery

def restore_embedding_config(method: str = "backup", backup_date: str = None) -> bool:
    """
    Restore embedding configuration
    
    Args:
        method: "backup" or "defaults"
        backup_date: Specific backup date (YYYYMMDD_HHMMSS)
        
    Returns:
        True if restore successful
    """
    global _startup_discovery
    
    if _startup_discovery is None:
        _startup_discovery = StartupModelDiscovery()
    
    if method == "backup":
        return _startup_discovery.restore_from_backup(backup_date)
    elif method == "defaults":
        return _startup_discovery.restore_defaults()
    else:
        logger.error(f"Unknown restore method: {method}")
        return False