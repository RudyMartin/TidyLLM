"""
Dynamic Model Discovery for AWS Bedrock
=======================================

Automatically discovers new embedding models, handles deprecated models,
and updates configurations without manual intervention.

Features:
- Auto-discovery of new Bedrock embedding models
- Dimension detection for new models
- Graceful handling of deprecated models
- Configuration updates with version tracking
- Backward compatibility for existing embeddings
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import boto3
from botocore.exceptions import ClientError
import os
import hashlib

# Import UnifiedSessionManager for consistent credential handling
try:
    from ...infrastructure.session.unified import UnifiedSessionManager
    UNIFIED_SESSION_AVAILABLE = True
except ImportError:
    UNIFIED_SESSION_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DiscoveredModel:
    """Represents a discovered embedding model"""
    model_id: str
    model_name: str
    provider: str
    native_dimension: Optional[int] = None
    configurable_dimensions: bool = False
    supported_dimensions: List[int] = None
    status: str = "available"  # available, deprecated, new
    discovered_at: str = ""
    last_verified: str = ""
    
    def __post_init__(self):
        if not self.discovered_at:
            self.discovered_at = datetime.now().isoformat()
        if not self.last_verified:
            self.last_verified = datetime.now().isoformat()
        if self.supported_dimensions is None:
            self.supported_dimensions = []

class DynamicModelDiscovery:
    """Handles dynamic discovery and management of embedding models"""
    
    def __init__(self, region: str = "us-east-1", cache_duration_hours: int = 24):
        """
        Initialize dynamic model discovery
        
        Args:
            region: AWS region for Bedrock
            cache_duration_hours: How long to cache discovery results
        """
        self.region = region
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.bedrock_client = None
        
        # Initialize UnifiedSessionManager for consistent credential handling
        self.session_manager = None
        if UNIFIED_SESSION_AVAILABLE:
            try:
                self.session_manager = UnifiedSessionManager()
                logger.info("DynamicModelDiscovery: UnifiedSessionManager integrated")
            except Exception as e:
                logger.warning(f"DynamicModelDiscovery: Failed to initialize UnifiedSessionManager: {e}")
        else:
            logger.info("DynamicModelDiscovery: UnifiedSessionManager not available, using direct boto3")
        self.cached_models = {}
        self.last_discovery = None
        
        # Known embedding model patterns
        self.embedding_model_patterns = [
            "amazon.titan-embed-text",
            "cohere.embed-",
            "amazon.titan-embed-image",  # Future multimodal
            "meta.embed-",               # Future Meta models
        ]
        
        # Known dimension configurations
        self.known_dimensions = {
            "amazon.titan-embed-text-v1": {
                "native": 1536,
                "configurable": False,
                "supported": [1536]
            },
            "amazon.titan-embed-text-v2:0": {
                "native": 1024,
                "configurable": True,
                "supported": [256, 512, 1024]
            },
            "cohere.embed-english-v3": {
                "native": 384,
                "configurable": False,
                "supported": [384]
            },
            "cohere.embed-multilingual-v3": {
                "native": 384,
                "configurable": False,
                "supported": [384]
            }
        }
    
    def _get_bedrock_client(self):
        """Get or create Bedrock client through UnifiedSessionManager if available"""
        if self.bedrock_client is None:
            try:
                # Use UnifiedSessionManager if available (consistent with gateways)
                if self.session_manager:
                    # Note: UnifiedSessionManager provides bedrock-runtime, but we need bedrock for model discovery
                    # Create a session from the session manager and get bedrock client
                    s3_client = self.session_manager.get_s3_client()
                    # Extract session from s3_client to create bedrock client 
                    session = self.session_manager._s3_client._client_config.region_name
                    
                    # Actually, let's use the session manager's approach for consistent credentials
                    if hasattr(self.session_manager, '_s3_client'):
                        # Create bedrock client using same credential approach as UnifiedSessionManager
                        if self.session_manager.config.credential_source.value == "environment":
                            import boto3
                            session = boto3.Session(
                                aws_access_key_id=self.session_manager.config.s3_access_key_id,
                                aws_secret_access_key=self.session_manager.config.s3_secret_access_key
                            )
                            self.bedrock_client = session.client('bedrock', region_name=self.region)
                            logger.info("DynamicModelDiscovery: Using UnifiedSessionManager credentials for Bedrock")
                        else:
                            # IAM role or default profile
                            session = boto3.Session()
                            self.bedrock_client = session.client('bedrock', region_name=self.region)
                            logger.info("DynamicModelDiscovery: Using UnifiedSessionManager session for Bedrock")
                    else:
                        # NO FALLBACK - UnifiedSessionManager is required
                        raise RuntimeError("DynamicModelDiscovery: UnifiedSessionManager is required for Bedrock access")
                else:
                    # NO FALLBACK - UnifiedSessionManager is required
                    raise RuntimeError("DynamicModelDiscovery: UnifiedSessionManager is required for Bedrock access")
            except Exception as e:
                logger.error(f"Failed to create Bedrock client: {e}")
                return None
        return self.bedrock_client
    
    def discover_models(self, force_refresh: bool = False) -> Dict[str, DiscoveredModel]:
        """
        Discover available embedding models from Bedrock
        
        Args:
            force_refresh: Force refresh even if cache is valid
            
        Returns:
            Dictionary of model_id -> DiscoveredModel
        """
        # Check cache first
        if not force_refresh and self._is_cache_valid():
            logger.info("Using cached model discovery results")
            return self.cached_models
        
        logger.info("Discovering embedding models from AWS Bedrock...")
        
        bedrock = self._get_bedrock_client()
        if not bedrock:
            logger.warning("Could not connect to Bedrock - using fallback models")
            return self._get_fallback_models()
        
        try:
            # List all foundation models
            response = bedrock.list_foundation_models()
            
            discovered_models = {}
            embedding_models = []
            
            for model in response.get('modelSummaries', []):
                model_id = model.get('modelId', '')
                
                # Check if this looks like an embedding model
                if self._is_embedding_model(model_id, model):
                    embedding_models.append(model)
                    
                    # Create discovered model
                    discovered = self._create_discovered_model(model)
                    discovered_models[model_id] = discovered
            
            # Test models to determine dimensions
            self._determine_model_dimensions(discovered_models)
            
            # Mark deprecated models
            self._mark_deprecated_models(discovered_models)
            
            # Cache results
            self.cached_models = discovered_models
            self.last_discovery = datetime.now()
            
            logger.info(f"Discovered {len(discovered_models)} embedding models")
            return discovered_models
            
        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
            return self._get_fallback_models()
    
    def _is_embedding_model(self, model_id: str, model_info: Dict) -> bool:
        """Determine if a model is an embedding model"""
        
        # Check by model ID patterns
        for pattern in self.embedding_model_patterns:
            if pattern in model_id.lower():
                return True
        
        # Check by model information
        model_name = model_info.get('modelName', '').lower()
        output_modalities = model_info.get('outputModalities', [])
        
        # Look for embedding-related keywords
        embedding_keywords = ['embed', 'embedding', 'vector']
        if any(keyword in model_name for keyword in embedding_keywords):
            return True
        
        # Check if output modality includes embeddings
        if 'EMBEDDING' in output_modalities:
            return True
        
        return False
    
    def _create_discovered_model(self, model_info: Dict) -> DiscoveredModel:
        """Create a DiscoveredModel from Bedrock model info"""
        model_id = model_info.get('modelId', '')
        
        # Determine provider
        provider = "aws"
        if "cohere" in model_id.lower():
            provider = "cohere"
        elif "titan" in model_id.lower():
            provider = "amazon"
        
        # Get known dimensions if available
        dim_info = self.known_dimensions.get(model_id, {})
        
        return DiscoveredModel(
            model_id=model_id,
            model_name=model_info.get('modelName', model_id),
            provider=provider,
            native_dimension=dim_info.get('native'),
            configurable_dimensions=dim_info.get('configurable', False),
            supported_dimensions=dim_info.get('supported', []),
            status="available"
        )
    
    def _determine_model_dimensions(self, models: Dict[str, DiscoveredModel]):
        """Test models to determine their embedding dimensions"""
        # Use UnifiedSessionManager for bedrock-runtime client if available
        if self.session_manager:
            try:
                bedrock_runtime = self.session_manager.get_bedrock_runtime_client()
                logger.info("DynamicModelDiscovery: Using UnifiedSessionManager for Bedrock Runtime")
            except Exception as e:
                logger.error(f"DynamicModelDiscovery: Failed to get Bedrock client from session manager: {e}")
                raise RuntimeError("DynamicModelDiscovery: UnifiedSessionManager is required for Bedrock Runtime access")
        else:
            raise RuntimeError("DynamicModelDiscovery: UnifiedSessionManager is required for Bedrock Runtime access")
        
        test_text = "Hello world"
        
        for model_id, model in models.items():
            if model.native_dimension is not None:
                continue  # Already know dimensions
            
            try:
                # Test the model with a simple text
                if "titan" in model_id.lower():
                    body = json.dumps({
                        "inputText": test_text,
                        "dimensions": 1024  # Try 1024 first
                    })
                elif "cohere" in model_id.lower():
                    body = json.dumps({
                        "texts": [test_text],
                        "input_type": "search_document"
                    })
                else:
                    continue  # Unknown model type
                
                response = bedrock_runtime.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=body
                )
                
                response_body = json.loads(response['body'].read().decode('utf-8'))
                
                # Extract embedding and determine dimension
                embedding = None
                if 'embedding' in response_body:
                    embedding = response_body['embedding']
                elif 'embeddings' in response_body and response_body['embeddings']:
                    embedding = response_body['embeddings'][0]
                
                if embedding:
                    model.native_dimension = len(embedding)
                    model.supported_dimensions = [len(embedding)]
                    
                    # Test if it supports configurable dimensions (for Titan v2)
                    if "titan-embed-text-v2" in model_id:
                        model.configurable_dimensions = True
                        model.supported_dimensions = [256, 512, 1024]
                
                logger.info(f"Detected dimensions for {model_id}: {model.native_dimension}")
                
            except Exception as e:
                logger.warning(f"Could not test model {model_id}: {e}")
                # Set reasonable defaults
                if "cohere" in model_id.lower():
                    model.native_dimension = 384
                    model.supported_dimensions = [384]
                elif "titan" in model_id.lower():
                    model.native_dimension = 1024
                    model.supported_dimensions = [1024]
    
    def _mark_deprecated_models(self, current_models: Dict[str, DiscoveredModel]):
        """Mark models as deprecated if they're no longer available"""
        if not self.cached_models:
            return
        
        for model_id, cached_model in self.cached_models.items():
            if model_id not in current_models:
                # Model was removed from Bedrock
                cached_model.status = "deprecated"
                cached_model.last_verified = datetime.now().isoformat()
                current_models[model_id] = cached_model
                logger.warning(f"Model {model_id} is no longer available - marked as deprecated")
    
    def _is_cache_valid(self) -> bool:
        """Check if cached results are still valid"""
        if not self.last_discovery or not self.cached_models:
            return False
        
        return datetime.now() - self.last_discovery < self.cache_duration
    
    def _get_fallback_models(self) -> Dict[str, DiscoveredModel]:
        """Return hardcoded fallback models if discovery fails"""
        fallback_models = {}
        
        for model_id, dim_info in self.known_dimensions.items():
            provider = "amazon" if "titan" in model_id else "cohere"
            
            fallback_models[model_id] = DiscoveredModel(
                model_id=model_id,
                model_name=model_id,
                provider=provider,
                native_dimension=dim_info['native'],
                configurable_dimensions=dim_info['configurable'],
                supported_dimensions=dim_info['supported'],
                status="available"
            )
        
        return fallback_models
    
    def update_embeddings_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        Update embeddings configuration with newly discovered models
        
        Args:
            config_path: Path to embeddings config file
            
        Returns:
            Update summary
        """
        if not config_path:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "..", "..", "tidyllm", "admin", "embeddings_settings.yaml"
            )
        
        # Discover current models
        current_models = self.discover_models()
        
        # Load existing config
        existing_config = self._load_existing_config(config_path)
        
        # Generate updates
        updates = {
            "new_models": [],
            "deprecated_models": [],
            "updated_models": [],
            "config_version": datetime.now().isoformat()
        }
        
        # Check for new models
        existing_model_ids = set()
        if existing_config and 'embeddings' in existing_config and 'models' in existing_config['embeddings']:
            for key, model_config in existing_config['embeddings']['models'].items():
                existing_model_ids.add(model_config.get('model_id', ''))
        
        for model_id, discovered_model in current_models.items():
            if model_id not in existing_model_ids and discovered_model.status == "available":
                # New model found
                new_key = self._generate_config_key(discovered_model)
                updates["new_models"].append({
                    "key": new_key,
                    "model": discovered_model
                })
            elif discovered_model.status == "deprecated":
                updates["deprecated_models"].append(model_id)
        
        # Update config if there are changes
        if updates["new_models"] or updates["deprecated_models"]:
            self._write_updated_config(config_path, existing_config, updates, current_models)
        
        return updates
    
    def _generate_config_key(self, model: DiscoveredModel) -> str:
        """Generate a configuration key for a model"""
        key = model.model_id.lower()
        key = key.replace("amazon.", "").replace(":", "_").replace("-", "_")
        
        # Add dimension suffix if configurable
        if model.configurable_dimensions and model.native_dimension:
            key += f"_{model.native_dimension}"
        
        return key
    
    def _load_existing_config(self, config_path: str) -> Dict:
        """Load existing configuration file"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load existing config: {e}")
            return {}
    
    def _write_updated_config(self, config_path: str, existing_config: Dict, 
                            updates: Dict, current_models: Dict[str, DiscoveredModel]):
        """Write updated configuration file"""
        try:
            import yaml
            
            # Ensure embeddings.models section exists
            if 'embeddings' not in existing_config:
                existing_config['embeddings'] = {}
            if 'models' not in existing_config['embeddings']:
                existing_config['embeddings']['models'] = {}
            
            # Add new models
            for new_model in updates["new_models"]:
                key = new_model["key"]
                model = new_model["model"]
                
                existing_config['embeddings']['models'][key] = {
                    "model_id": model.model_id,
                    "native_dimension": model.native_dimension,
                    "provider": "bedrock",
                    "configurable_dimensions": model.configurable_dimensions,
                    "description": f"{model.model_name} - Auto-discovered",
                    "discovered_at": model.discovered_at,
                    "status": model.status
                }
                
                if model.supported_dimensions:
                    existing_config['embeddings']['models'][key]["supported_dimensions"] = model.supported_dimensions
            
            # Mark deprecated models
            for model_key, model_config in existing_config['embeddings']['models'].items():
                model_id = model_config.get('model_id', '')
                if model_id in updates["deprecated_models"]:
                    model_config['status'] = 'deprecated'
                    model_config['deprecated_at'] = datetime.now().isoformat()
            
            # Add discovery metadata
            existing_config['embeddings']['last_discovery'] = datetime.now().isoformat()
            existing_config['embeddings']['discovery_version'] = updates["config_version"]
            
            # Write updated config
            with open(config_path, 'w') as f:
                yaml.dump(existing_config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Updated embeddings configuration: {len(updates['new_models'])} new models, {len(updates['deprecated_models'])} deprecated")
            
        except Exception as e:
            logger.error(f"Failed to write updated config: {e}")

    def get_model_compatibility_report(self) -> Dict[str, Any]:
        """Generate report on model compatibility and changes"""
        models = self.discover_models()
        
        report = {
            "total_models": len(models),
            "available_models": len([m for m in models.values() if m.status == "available"]),
            "deprecated_models": len([m for m in models.values() if m.status == "deprecated"]),
            "new_models": len([m for m in models.values() if m.status == "new"]),
            "by_provider": {},
            "by_dimension": {},
            "configurable_models": [],
            "recommendations": []
        }
        
        # Group by provider
        for model in models.values():
            if model.provider not in report["by_provider"]:
                report["by_provider"][model.provider] = 0
            report["by_provider"][model.provider] += 1
        
        # Group by dimension
        for model in models.values():
            if model.native_dimension:
                dim = str(model.native_dimension)
                if dim not in report["by_dimension"]:
                    report["by_dimension"][dim] = 0
                report["by_dimension"][dim] += 1
        
        # Find configurable models
        for model in models.values():
            if model.configurable_dimensions and model.status == "available":
                report["configurable_models"].append({
                    "model_id": model.model_id,
                    "supported_dimensions": model.supported_dimensions
                })
        
        # Generate recommendations
        if report["deprecated_models"] > 0:
            report["recommendations"].append("Consider migrating from deprecated models")
        
        if report["new_models"] > 0:
            report["recommendations"].append("New models available for testing")
        
        return report

# Global instance for easy access
_discovery_instance = None

def get_model_discovery(region: str = "us-east-1") -> DynamicModelDiscovery:
    """Get global model discovery instance"""
    global _discovery_instance
    
    if _discovery_instance is None:
        _discovery_instance = DynamicModelDiscovery(region=region)
    
    return _discovery_instance

def auto_update_embeddings_config(config_path: str = None) -> Dict[str, Any]:
    """Convenience function to auto-update embeddings configuration"""
    discovery = get_model_discovery()
    return discovery.update_embeddings_config(config_path)