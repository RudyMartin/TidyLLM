#!/usr/bin/env python3
"""
Smart Embedding Service - Intelligent Embedding Orchestration
=============================================================

Coordinates existing embedding infrastructure for intelligent routing,
caching, and quality tracking. Built on top of existing components.

Phase 1 Implementation: Wrapper/Orchestrator Pattern
"""

import hashlib
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


# ==================== VALUE OBJECTS ====================

@dataclass
class EmbeddingRequest:
    """Request for embedding generation."""
    text: str
    model_preference: Optional[str] = None
    dimensions: Optional[int] = 1024
    use_cache: bool = True
    require_high_quality: bool = False
    source_context: Optional[str] = None  # 'rag', 'chat', 'search', etc.
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embedding: List[float]
    model_used: str
    dimensions: int
    source: str  # 'local', 'cache', 'gateway'
    quality_score: Optional[float] = None
    processing_time_ms: Optional[float] = None
    cached: bool = False
    cache_key: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingStrategy(Enum):
    """Strategy for embedding generation."""
    LOCAL_FAST = "local_fast"
    LOCAL_QUALITY = "local_quality"
    GATEWAY_BEDROCK = "gateway_bedrock"
    CACHE_ONLY = "cache_only"


# ==================== SMART EMBEDDING SERVICE ====================

class SmartEmbeddingService:
    """
    Intelligent embedding service that orchestrates existing components.

    This is a wrapper/coordinator that uses:
    - EmbeddingDelegate for local embeddings (SentenceTransformers)
    - CorporateLLMGateway for Bedrock embeddings
    - Future: PostgreSQL cache with pgvector
    """

    def __init__(self):
        """Initialize service using existing components."""
        # Initialize components lazily
        self._embedding_delegate = None
        self._corporate_gateway = None
        self._embedding_processor = None

        # Simple in-memory cache for Phase 1
        self._simple_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Configuration
        self.complexity_threshold_high = 0.7
        self.complexity_threshold_low = 0.3
        self.text_length_short = 200
        self.text_length_long = 1000

        # Get embedding config from settings
        self._embedding_config = self._load_embedding_config()

        logger.info("SmartEmbeddingService initialized")

    def _load_embedding_config(self):
        """Load embedding configuration from settings."""
        try:
            from infrastructure.yaml_loader import get_settings_loader
            loader = get_settings_loader()

            # Use get_config_value with explicit defaults - NO HARDCODING!
            config = {
                'model_id': loader.get_config_value('credentials.bedrock_llm.embeddings.model_id',
                                                   'amazon.titan-embed-text-v2:0'),
                'dimensions': loader.get_config_value('credentials.bedrock_llm.embeddings.dimensions',
                                                     1024),
                'batch_size': loader.get_config_value('credentials.bedrock_llm.embeddings.batch_size',
                                                     25),
                'cache_enabled': loader.get_config_value('credentials.bedrock_llm.embeddings.cache_enabled',
                                                        True),
                'normalize': loader.get_config_value('credentials.bedrock_llm.embeddings.normalize',
                                                    True)
            }

            logger.info(f"Loaded embedding config: model={config['model_id']}, dims={config['dimensions']}")
            return config

        except Exception as e:
            logger.warning(f"Could not load embedding config: {e}, using defaults")
            # Only used if settings loading completely fails
            return {
                'model_id': 'amazon.titan-embed-text-v2:0',
                'dimensions': 1024,
                'batch_size': 25,
                'cache_enabled': True,
                'normalize': True
            }

    def _get_embedding_delegate(self):
        """Lazy load EmbeddingDelegate for local embeddings."""
        if self._embedding_delegate is None:
            try:
                # Use PathManager for proper imports
                import sys
                from pathlib import Path

                # Get paths properly
                try:
                    from common.utilities.path_manager import PathManager
                    path_mgr = PathManager()
                    for path in path_mgr.get_python_paths():
                        if path not in sys.path:
                            sys.path.insert(0, path)
                except ImportError:
                    pass  # Continue without PathManager if not available

                from packages.tidyllm.infrastructure.delegates.embedding_delegate import EmbeddingDelegate
                self._embedding_delegate = EmbeddingDelegate()
                logger.info("EmbeddingDelegate loaded for local embeddings")
            except ImportError as e:
                logger.error(f"Failed to load EmbeddingDelegate: {e}")
        return self._embedding_delegate

    def _get_corporate_gateway(self):
        """Lazy load CorporateLLMGateway for Bedrock embeddings."""
        if self._corporate_gateway is None:
            try:
                # Use PathManager for proper imports
                import sys
                from pathlib import Path

                # Get paths properly
                try:
                    from common.utilities.path_manager import PathManager
                    path_mgr = PathManager()
                    for path in path_mgr.get_python_paths():
                        if path not in sys.path:
                            sys.path.insert(0, path)
                except ImportError:
                    pass  # Continue without PathManager if not available

                from packages.tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway
                self._corporate_gateway = CorporateLLMGateway()
                logger.info("CorporateLLMGateway loaded for Bedrock embeddings")
            except Exception as e:
                logger.error(f"Failed to load CorporateLLMGateway: {e}")
        return self._corporate_gateway

    def _get_embedding_processor(self):
        """Lazy load EmbeddingProcessor for TidyLLM-based embeddings."""
        if self._embedding_processor is None:
            try:
                # Use PathManager for proper imports
                import sys
                from pathlib import Path

                # Get paths properly
                try:
                    from common.utilities.path_manager import PathManager
                    path_mgr = PathManager()
                    for path in path_mgr.get_python_paths():
                        if path not in sys.path:
                            sys.path.insert(0, path)
                except ImportError:
                    pass  # Continue without PathManager if not available

                from packages.tidyllm.knowledge_systems.facades.embedding_processor import EmbeddingProcessor
                self._embedding_processor = EmbeddingProcessor(target_dimension=1024)
                logger.info("EmbeddingProcessor loaded for TidyLLM embeddings")
            except ImportError as e:
                logger.error(f"Failed to load EmbeddingProcessor: {e}")
        return self._embedding_processor

    def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResult:
        """
        Generate embedding using intelligent strategy selection.

        This is the main entry point that orchestrates different strategies.
        """
        start_time = time.time()

        # Generate cache key
        cache_key = self._generate_cache_key(request.text)

        # Step 1: Check cache if enabled
        if request.use_cache and cache_key in self._simple_cache:
            cached_result = self._simple_cache[cache_key]
            if self._is_cache_valid(cached_result, request):
                self._cache_hits += 1
                logger.debug(f"Cache hit for key {cache_key[:8]}...")
                cached_result.cached = True
                cached_result.processing_time_ms = (time.time() - start_time) * 1000
                return cached_result

        self._cache_misses += 1

        # Step 2: Determine strategy
        strategy = self._determine_strategy(request)
        logger.debug(f"Selected strategy: {strategy.value}")

        # Step 3: Execute strategy
        result = self._execute_strategy(strategy, request)
        result.cache_key = cache_key

        # Step 4: Update cache
        if request.use_cache and not result.cached:
            self._simple_cache[cache_key] = result
            logger.debug(f"Cached result for key {cache_key[:8]}...")

        # Add processing time
        result.processing_time_ms = (time.time() - start_time) * 1000

        return result

    def _determine_strategy(self, request: EmbeddingRequest) -> EmbeddingStrategy:
        """
        Determine optimal embedding strategy based on request and context.

        Decision tree:
        1. If high quality explicitly required -> Bedrock
        2. If text is simple/short -> Local fast
        3. If text is complex/long -> Bedrock or Local quality
        4. Default -> Local fast
        """
        # Explicit high quality requirement
        if request.require_high_quality:
            return EmbeddingStrategy.GATEWAY_BEDROCK

        # Analyze text complexity
        complexity = self._analyze_text_complexity(request.text)
        text_length = len(request.text)

        # Simple, short text - use fast local
        if complexity < self.complexity_threshold_low and text_length < self.text_length_short:
            return EmbeddingStrategy.LOCAL_FAST

        # Complex or long text
        if complexity > self.complexity_threshold_high or text_length > self.text_length_long:
            # For RAG contexts, prefer quality
            if request.source_context == 'rag':
                return EmbeddingStrategy.GATEWAY_BEDROCK
            else:
                return EmbeddingStrategy.LOCAL_QUALITY

        # Medium complexity - context-based decision
        if request.source_context in ['legal', 'medical', 'financial']:
            return EmbeddingStrategy.GATEWAY_BEDROCK

        # Default to fast local
        return EmbeddingStrategy.LOCAL_FAST

    def _execute_strategy(self, strategy: EmbeddingStrategy, request: EmbeddingRequest) -> EmbeddingResult:
        """Execute the selected embedding strategy."""

        if strategy == EmbeddingStrategy.LOCAL_FAST:
            return self._generate_local_embedding(request, quality_mode=False)

        elif strategy == EmbeddingStrategy.LOCAL_QUALITY:
            return self._generate_local_embedding(request, quality_mode=True)

        elif strategy == EmbeddingStrategy.GATEWAY_BEDROCK:
            return self._generate_bedrock_embedding(request)

        else:  # CACHE_ONLY or fallback
            # This should not happen as we check cache earlier
            return self._generate_local_embedding(request, quality_mode=False)

    def _generate_local_embedding(self, request: EmbeddingRequest, quality_mode: bool = False) -> EmbeddingResult:
        """Generate embedding using local models (EmbeddingDelegate)."""
        try:
            delegate = self._get_embedding_delegate()
            if not delegate:
                raise RuntimeError("EmbeddingDelegate not available")

            # Generate embedding
            import numpy as np
            embedding_array = delegate.embed_text(request.text)

            # Convert to list
            if hasattr(embedding_array, 'tolist'):
                embedding_list = embedding_array.tolist()
            elif isinstance(embedding_array, np.ndarray):
                embedding_list = embedding_array.tolist()
            else:
                embedding_list = list(embedding_array)

            # Estimate quality score
            quality_score = 0.85 if quality_mode else 0.7

            return EmbeddingResult(
                embedding=embedding_list,
                model_used="sentence-transformers/all-MiniLM-L6-v2",
                dimensions=len(embedding_list),
                source='local',
                quality_score=quality_score,
                cached=False,
                metadata=request.metadata
            )

        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            # Return zero embedding as fallback
            return EmbeddingResult(
                embedding=[0.0] * 384,
                model_used="fallback",
                dimensions=384,
                source='local',
                quality_score=0.0,
                cached=False,
                metadata={'error': str(e)}
            )

    def _generate_bedrock_embedding(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate embedding using Bedrock via Titan adapter."""
        try:
            # First try using the new Titan adapter for proper validation
            try:
                from packages.tidyllm.embedding.titan_adapter import (
                    titan_embed, get_titan_model, TITAN_MODELS
                )

                # Get bedrock runtime client
                import boto3
                from infrastructure.yaml_loader import get_settings_loader
                loader = get_settings_loader()
                aws_config = loader.get_aws_config()

                bedrock_runtime = boto3.client(
                    'bedrock-runtime',
                    region_name=aws_config.get('region', 'us-east-1'),
                    aws_access_key_id=aws_config['access_key_id'],
                    aws_secret_access_key=aws_config['secret_access_key']
                )

                # Determine model key from model ID
                model_id = request.model_preference or self._embedding_config.get('model_id')
                model_key = 'titan_v2'  # Default

                # Map model ID to model key
                if 'titan-embed-text-v1' in model_id:
                    model_key = 'titan_v1'
                elif 'titan-embed-text-v2' in model_id or 'titan-embed-v2' in model_id:
                    model_key = 'titan_v2'
                elif 'titan-embed-g1' in model_id:
                    model_key = 'titan_g1'

                # Generate embedding with validation
                embedding_array = titan_embed(
                    bedrock_runtime,
                    model_key,
                    request.text,
                    normalize=self._embedding_config.get('normalize', True),
                    validate=True  # Enable validation!
                )

                # Convert to list
                embedding_list = embedding_array.tolist() if hasattr(embedding_array, 'tolist') else list(embedding_array)

                # Get actual model info
                model_info = get_titan_model(model_key)

                return EmbeddingResult(
                    embedding=embedding_list,
                    model_used=model_info.model_id,
                    dimensions=model_info.dimensions,
                    source='bedrock_titan',
                    quality_score=0.95,
                    cached=False,
                    metadata=request.metadata
                )

            except ImportError:
                logger.info("Titan adapter not available, using gateway fallback")
                # Fall through to gateway approach

            # Fallback to gateway if Titan adapter not available
            gateway = self._get_corporate_gateway()
            if not gateway:
                # Fallback to local if gateway unavailable
                logger.warning("CorporateLLMGateway not available, falling back to local")
                return self._generate_local_embedding(request, quality_mode=True)

            # Create gateway request
            from packages.tidyllm.gateways.corporate_llm_gateway import LLMRequest

            # Use config for model ID and dimensions
            model_id = request.model_preference or self._embedding_config.get('model_id', 'amazon.titan-embed-text-v2:0')
            dimensions = request.dimensions or self._embedding_config.get('dimensions', 1024)

            llm_request = LLMRequest(
                prompt=request.text,
                model_id=model_id,
                is_embedding=True,
                dimensions=dimensions,
                user_id="smart_embedding_service",
                audit_reason=f"embedding_generation_{request.source_context or 'general'}"
            )

            # Process through gateway
            response = gateway.process_embedding_request(llm_request)

            if not response.success:
                raise Exception(f"Gateway embedding failed: {response.error}")

            # Parse embedding from response
            embedding_list = json.loads(response.content)

            return EmbeddingResult(
                embedding=embedding_list,
                model_used=response.model_used,
                dimensions=len(embedding_list),
                source='gateway',
                quality_score=0.95,  # High quality from Bedrock
                cached=False,
                metadata=request.metadata
            )

        except Exception as e:
            logger.error(f"Bedrock embedding generation failed: {e}")
            # Fallback to local quality mode
            logger.info("Falling back to local quality embedding")
            return self._generate_local_embedding(request, quality_mode=True)

    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _analyze_text_complexity(self, text: str) -> float:
        """
        Analyze text complexity (0-1 scale).

        Simple heuristics for now:
        - Word count
        - Average word length
        - Vocabulary diversity
        - Special characters
        """
        if not text:
            return 0.0

        words = text.split()
        word_count = len(words)

        if word_count == 0:
            return 0.0

        # Calculate metrics
        avg_word_length = sum(len(word) for word in words) / word_count
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / word_count
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)

        # Weighted complexity score
        complexity = min(1.0, (
            (word_count / 100) * 0.2 +  # Length factor
            (avg_word_length / 10) * 0.3 +  # Word complexity
            vocabulary_diversity * 0.3 +  # Vocabulary richness
            special_char_ratio * 0.2  # Technical content
        ))

        return complexity

    def _is_cache_valid(self, cached: EmbeddingResult, request: EmbeddingRequest) -> bool:
        """Check if cached embedding meets request requirements."""
        # Check quality requirement
        if request.require_high_quality:
            if not cached.quality_score or cached.quality_score < 0.9:
                return False

        # Check dimensions match
        if request.dimensions and cached.dimensions != request.dimensions:
            return False

        # Future: Check TTL, version, etc.

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "service": "SmartEmbeddingService",
            "stats": {
                "total_requests": total_requests,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self._simple_cache)
            },
            "components": {
                "embedding_delegate": self._embedding_delegate is not None,
                "corporate_gateway": self._corporate_gateway is not None,
                "embedding_processor": self._embedding_processor is not None
            },
            "configuration": {
                "complexity_threshold_high": self.complexity_threshold_high,
                "complexity_threshold_low": self.complexity_threshold_low,
                "text_length_short": self.text_length_short,
                "text_length_long": self.text_length_long
            }
        }


# ==================== FACTORY FUNCTION ====================

def create_smart_embedding_service() -> SmartEmbeddingService:
    """Factory function to create SmartEmbeddingService."""
    return SmartEmbeddingService()


# ==================== INTEGRATION HELPER ====================

class SmartEmbeddingAdapter:
    """
    Adapter to make SmartEmbeddingService compatible with existing interfaces.

    This allows RAG adapters to use SmartEmbeddingService without modification.
    """

    def __init__(self):
        """Initialize adapter with SmartEmbeddingService."""
        self.service = SmartEmbeddingService()

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text (compatible with EmbeddingDelegate interface).

        Args:
            text: Input text

        Returns:
            Embedding vector as list
        """
        request = EmbeddingRequest(text=text)
        result = self.service.generate_embedding(request)
        return result.embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            request = EmbeddingRequest(text=text)
            result = self.service.generate_embedding(request)
            embeddings.append(result.embedding)
        return embeddings

    def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embedding (compatible with RAGDelegateProtocol).

        Args:
            text: Input text
            model: Optional model preference

        Returns:
            Embedding vector as list
        """
        request = EmbeddingRequest(text=text, model_preference=model)
        result = self.service.generate_embedding(request)
        return result.embedding


if __name__ == "__main__":
    # Example usage
    import asyncio

    def test_service():
        """Test the SmartEmbeddingService."""
        service = SmartEmbeddingService()

        # Test different types of text
        test_cases = [
            ("Simple short text", False),
            ("This is a much longer and more complex text that discusses various technical topics including machine learning, natural language processing, and artificial intelligence systems that require deeper understanding and analysis.", True),
            ("Legal document: The party of the first part shall indemnify and hold harmless the party of the second part.", True),
        ]

        for text, expect_high_quality in test_cases:
            print(f"\n{'='*60}")
            print(f"Text: {text[:50]}...")

            request = EmbeddingRequest(
                text=text,
                source_context='test'
            )

            result = service.generate_embedding(request)

            print(f"Model used: {result.model_used}")
            print(f"Source: {result.source}")
            print(f"Dimensions: {result.dimensions}")
            print(f"Quality score: {result.quality_score}")
            print(f"Processing time: {result.processing_time_ms:.2f}ms")
            print(f"Cached: {result.cached}")

        # Show stats
        print(f"\n{'='*60}")
        print("Service Statistics:")
        stats = service.get_stats()
        print(json.dumps(stats, indent=2))

    # Run test
    test_service()