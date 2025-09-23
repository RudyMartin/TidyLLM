"""
Centralized Embedding Configuration Manager
===========================================

Single source of truth for embedding configuration.
All services MUST use this instead of hardcoding values.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Embedding configuration from settings."""
    model_id: str
    dimensions: int
    batch_size: int
    max_chunk_size: int
    cache_enabled: bool
    normalize: bool
    timeout: int
    provider: str  # 'bedrock', 'openai', 'cohere', etc.


class EmbeddingConfigManager:
    """
    Singleton manager for embedding configuration.
    Ensures all services use the same config.
    """

    _instance = None
    _config: Optional[EmbeddingConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._config = self._load_config()

    def _load_config(self) -> EmbeddingConfig:
        """Load embedding configuration from settings.yaml."""
        try:
            from infrastructure.yaml_loader import get_settings_loader
            loader = get_settings_loader()
            settings = loader.settings

            # Try multiple locations for embedding config
            embedding_config = None

            # First try: credentials.bedrock_llm.embeddings
            if 'credentials' in settings and 'bedrock_llm' in settings['credentials']:
                embedding_config = settings['credentials']['bedrock_llm'].get('embeddings', {})

            # Second try: llm_models.bedrock.embeddings
            if not embedding_config and 'llm_models' in settings:
                if 'bedrock' in settings['llm_models']:
                    embedding_config = settings['llm_models']['bedrock'].get('embeddings', {})

            # Third try: services.embeddings
            if not embedding_config and 'services' in settings:
                embedding_config = settings['services'].get('embeddings', {})

            # Use config or defaults
            if embedding_config:
                logger.info(f"Loaded embedding config from settings: {embedding_config}")
                return EmbeddingConfig(
                    model_id=embedding_config.get('model_id', 'amazon.titan-embed-text-v2:0'),
                    dimensions=embedding_config.get('dimensions', 1024),
                    batch_size=embedding_config.get('batch_size', 25),
                    max_chunk_size=embedding_config.get('max_chunk_size', 2000),
                    cache_enabled=embedding_config.get('cache_enabled', True),
                    normalize=embedding_config.get('normalize', True),
                    timeout=embedding_config.get('timeout', 30),
                    provider=embedding_config.get('type', 'bedrock_embeddings')
                )
            else:
                logger.warning("No embedding config found in settings, using defaults")
                return self._get_default_config()

        except Exception as e:
            logger.error(f"Failed to load embedding config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> EmbeddingConfig:
        """Get default configuration."""
        return EmbeddingConfig(
            model_id='amazon.titan-embed-text-v2:0',
            dimensions=1024,
            batch_size=25,
            max_chunk_size=2000,
            cache_enabled=True,
            normalize=True,
            timeout=30,
            provider='bedrock_embeddings'
        )

    def get_config(self) -> EmbeddingConfig:
        """Get the current embedding configuration."""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def reload_config(self) -> EmbeddingConfig:
        """Force reload configuration from settings."""
        self._config = self._load_config()
        return self._config

    @property
    def model_id(self) -> str:
        return self.get_config().model_id

    @property
    def dimensions(self) -> int:
        return self.get_config().dimensions

    @property
    def batch_size(self) -> int:
        return self.get_config().batch_size

    @property
    def max_chunk_size(self) -> int:
        return self.get_config().max_chunk_size

    @property
    def cache_enabled(self) -> bool:
        return self.get_config().cache_enabled

    @property
    def normalize(self) -> bool:
        return self.get_config().normalize

    @property
    def timeout(self) -> int:
        return self.get_config().timeout

    @property
    def provider(self) -> str:
        return self.get_config().provider

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config = self.get_config()
        return {
            'model_id': config.model_id,
            'dimensions': config.dimensions,
            'batch_size': config.batch_size,
            'max_chunk_size': config.max_chunk_size,
            'cache_enabled': config.cache_enabled,
            'normalize': config.normalize,
            'timeout': config.timeout,
            'provider': config.provider
        }


# Global singleton instance
_embedding_config_manager = EmbeddingConfigManager()


def get_embedding_config() -> EmbeddingConfig:
    """Get the global embedding configuration."""
    return _embedding_config_manager.get_config()


def get_embedding_dimensions() -> int:
    """Quick accessor for dimensions."""
    return _embedding_config_manager.dimensions


def get_embedding_model_id() -> str:
    """Quick accessor for model ID."""
    return _embedding_config_manager.model_id


def reload_embedding_config():
    """Force reload the configuration."""
    return _embedding_config_manager.reload_config()


# For backward compatibility
def get_config_dict() -> Dict[str, Any]:
    """Get config as dictionary for legacy code."""
    return _embedding_config_manager.to_dict()


if __name__ == "__main__":
    # Test the config manager
    print("Testing Embedding Configuration Manager")
    print("=" * 50)

    config = get_embedding_config()
    print(f"Model ID: {config.model_id}")
    print(f"Dimensions: {config.dimensions}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Provider: {config.provider}")

    print("\nQuick accessors:")
    print(f"Dimensions: {get_embedding_dimensions()}")
    print(f"Model ID: {get_embedding_model_id()}")

    print("\nConfig as dict:")
    print(get_config_dict())