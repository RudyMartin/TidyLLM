"""
Embedding Worker - Vector Embedding Generation Agent
===================================================

Dedicated worker for vector embedding generation and processing operations.
Extracts functionality from existing embedding components and organizes
it as a scalable agent worker.

Capabilities:
- Text-to-vector embedding generation
- Batch embedding processing
- Multi-model embedding support (TidyLLM providers)
- Embedding standardization and normalization
- Vector dimensionality management
- Embedding caching and optimization
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import time

from .base_worker import BaseWorker, TaskPriority
from ..session.unified import UnifiedSessionManager

logger = logging.getLogger("embedding_worker")


@dataclass
class EmbeddingRequest:
    """Request for embedding generation."""
    text_id: str  # Unique identifier for the text
    text_content: str  # Text to embed
    model_provider: str = "default"  # TidyLLM provider name
    target_dimension: int = 1024  # Target embedding dimension
    normalize: bool = True  # Whether to normalize the embedding
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchEmbeddingRequest:
    """Request for batch embedding generation."""
    batch_id: str
    texts: List[Dict[str, str]]  # List of {"id": str, "content": str}
    model_provider: str = "default"
    target_dimension: int = 1024
    normalize: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    text_id: str
    embedding: List[float]  # The vector embedding
    embedding_dimension: int
    model_used: str
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text_id": self.text_id,
            "embedding": self.embedding,
            "embedding_dimension": self.embedding_dimension,
            "model_used": self.model_used,
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }


@dataclass
class BatchEmbeddingResult:
    """Result from batch embedding generation."""
    batch_id: str
    embeddings: List[EmbeddingResult]
    total_processed: int
    successful_embeddings: int
    failed_embeddings: int
    total_processing_time: Optional[float] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "embeddings": [emb.to_dict() for emb in self.embeddings],
            "total_processed": self.total_processed,
            "successful_embeddings": self.successful_embeddings,
            "failed_embeddings": self.failed_embeddings,
            "total_processing_time": self.total_processing_time,
            "errors": self.errors
        }


class EmbeddingWorker(BaseWorker[Union[EmbeddingRequest, BatchEmbeddingRequest], 
                                Union[EmbeddingResult, BatchEmbeddingResult]]):
    """
    Worker for vector embedding generation operations.
    
    Integrates existing embedding capabilities:
    - EmbeddingProcessor for standardized embedding generation
    - TidyLLM provider system for multi-model support
    - Automatic dimensionality standardization
    - Batch processing optimization
    
    Task Types:
    - generate_embedding: Generate single text embedding
    - generate_batch_embeddings: Generate multiple embeddings efficiently
    - standardize_embedding: Normalize existing embedding to target dimension
    """
    
    def __init__(self,
                 worker_name: str = "embedding_worker",
                 default_target_dimension: int = 1024,
                 batch_size: int = 32,
                 **kwargs):
        """
        Initialize Embedding Worker.
        
        Args:
            worker_name: Worker identifier
            default_target_dimension: Default embedding dimension
            batch_size: Maximum batch size for batch processing
        """
        super().__init__(worker_name, **kwargs)
        
        self.default_target_dimension = default_target_dimension
        self.batch_size = batch_size
        
        # Embedding backends
        self.embedding_processor = None
        self.tidyllm_providers = {}
        self.session_manager = None
        
        # Model availability tracking
        self.available_models = []
        self.default_provider = None
        
        logger.info(f"Embedding Worker '{worker_name}' configured for {default_target_dimension}d embeddings")
    
    async def _initialize_worker(self) -> None:
        """Initialize embedding backends and providers."""
        try:
            # Initialize UnifiedSessionManager
            try:
                self.session_manager = UnifiedSessionManager()
                logger.info("Embedding Worker: UnifiedSessionManager initialized")
            except Exception as e:
                logger.warning(f"Embedding Worker: UnifiedSessionManager not available: {e}")
            
            # Initialize EmbeddingProcessor
            try:
                from ...knowledge_systems.facades.embedding_processor import EmbeddingProcessor
                self.embedding_processor = EmbeddingProcessor(
                    target_dimension=self.default_target_dimension
                )
                logger.info("Embedding Worker: EmbeddingProcessor initialized")
            except ImportError as e:
                logger.warning(f"Embedding Worker: EmbeddingProcessor not available: {e}")
            
            # Initialize TidyLLM providers
            await self._initialize_providers()
            
            if not (self.embedding_processor or self.tidyllm_providers):
                raise RuntimeError("No embedding backends available")
                
        except Exception as e:
            logger.error(f"Embedding Worker initialization failed: {e}")
            raise
    
    async def _initialize_providers(self) -> None:
        """Initialize CorporateLLMGateway for compliant embedding generation."""
        try:
            # ONLY use CorporateLLMGateway for ALL Bedrock operations (compliance requirement)
            from ...gateways.corporate_llm_gateway import CorporateLLMGateway

            self.corporate_gateway = CorporateLLMGateway()
            logger.info("Embedding Worker: Using CorporateLLMGateway for compliant embeddings")

            # Mark available models based on gateway capabilities
            self.available_models = ["titan-embed-v1", "titan-embed-v2", "cohere-embed"]
            self.default_provider = "titan-embed-v2"  # Default embedding model

        except ImportError as e:
            logger.error(f"Embedding Worker: CorporateLLMGateway not available: {e}")
            self.corporate_gateway = None
    
    def validate_input(self, task_input: Any) -> bool:
        """Validate embedding request input."""
        if isinstance(task_input, EmbeddingRequest):
            return bool(task_input.text_id and task_input.text_content.strip())
        elif isinstance(task_input, BatchEmbeddingRequest):
            return bool(task_input.batch_id and task_input.texts)
        return False
    
    async def process_task(self, task_input: Union[EmbeddingRequest, BatchEmbeddingRequest]) -> Union[EmbeddingResult, BatchEmbeddingResult]:
        """Process embedding generation request."""
        if isinstance(task_input, EmbeddingRequest):
            return await self._process_single_embedding(task_input)
        elif isinstance(task_input, BatchEmbeddingRequest):
            return await self._process_batch_embeddings(task_input)
        else:
            raise ValueError(f"Unsupported task input type: {type(task_input)}")
    
    async def _process_single_embedding(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Process single embedding request."""
        start_time = time.time()
        
        try:
            logger.info(f"Generating embedding for text '{request.text_id}' using '{request.model_provider}'")
            
            # Generate embedding
            embedding, model_used = await self._generate_embedding(
                text=request.text_content,
                provider=request.model_provider,
                target_dimension=request.target_dimension,
                normalize=request.normalize
            )
            
            processing_time = time.time() - start_time
            
            result = EmbeddingResult(
                text_id=request.text_id,
                embedding=embedding,
                embedding_dimension=len(embedding),
                model_used=model_used,
                processing_time=processing_time,
                metadata=request.metadata.copy()
            )
            
            logger.info(f"Embedding generated for '{request.text_id}': "
                       f"{len(embedding)}d vector in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Single embedding generation failed for '{request.text_id}': {e}")
            raise
    
    async def _process_batch_embeddings(self, request: BatchEmbeddingRequest) -> BatchEmbeddingResult:
        """Process batch embedding request."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing batch '{request.batch_id}' with {len(request.texts)} texts")
            
            embeddings = []
            errors = []
            successful = 0
            failed = 0
            
            # Process texts in batches
            for i in range(0, len(request.texts), self.batch_size):
                batch_texts = request.texts[i:i + self.batch_size]
                
                # Process each text in the current batch
                batch_tasks = []
                for text_info in batch_texts:
                    text_id = text_info.get("id")
                    text_content = text_info.get("content", "")
                    
                    if not text_id or not text_content.strip():
                        errors.append(f"Invalid text info: {text_info}")
                        failed += 1
                        continue
                    
                    # Create embedding task
                    task = self._generate_embedding(
                        text=text_content,
                        provider=request.model_provider,
                        target_dimension=request.target_dimension,
                        normalize=request.normalize
                    )
                    batch_tasks.append((text_id, task))
                
                # Execute batch concurrently
                if batch_tasks:
                    results = await asyncio.gather(*[task for _, task in batch_tasks], return_exceptions=True)
                    
                    for (text_id, _), result in zip(batch_tasks, results):
                        if isinstance(result, Exception):
                            errors.append(f"Text '{text_id}': {str(result)}")
                            failed += 1
                        else:
                            embedding, model_used = result
                            embeddings.append(EmbeddingResult(
                                text_id=text_id,
                                embedding=embedding,
                                embedding_dimension=len(embedding),
                                model_used=model_used,
                                metadata={"batch_id": request.batch_id}
                            ))
                            successful += 1
            
            total_processing_time = time.time() - start_time
            
            result = BatchEmbeddingResult(
                batch_id=request.batch_id,
                embeddings=embeddings,
                total_processed=len(request.texts),
                successful_embeddings=successful,
                failed_embeddings=failed,
                total_processing_time=total_processing_time,
                errors=errors
            )
            
            logger.info(f"Batch '{request.batch_id}' completed: "
                       f"{successful}/{len(request.texts)} successful in {total_processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed for '{request.batch_id}': {e}")
            raise
    
    async def _generate_embedding(self,
                                text: str,
                                provider: str = "default",
                                target_dimension: int = None,
                                normalize: bool = True) -> tuple[List[float], str]:
        """Generate embedding using available backends."""
        target_dimension = target_dimension or self.default_target_dimension

        try:
            # ALWAYS use CorporateLLMGateway for Bedrock embeddings (compliance)
            if hasattr(self, 'corporate_gateway') and self.corporate_gateway:
                return await self._generate_with_gateway(text, provider, target_dimension, normalize)

            # Try EmbeddingProcessor as fallback (for non-Bedrock embeddings)
            if self.embedding_processor:
                logger.warning("Using EmbeddingProcessor fallback - should use CorporateLLMGateway")
                return await self._generate_with_processor(text, provider, target_dimension, normalize)

            # Final fallback - mock embedding for development
            logger.warning("No embedding backends available, generating mock embedding")
            return await self._generate_mock_embedding(text, target_dimension), "mock"

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def _generate_with_processor(self, 
                                     text: str,
                                     provider: str,
                                     target_dimension: int,
                                     normalize: bool) -> tuple[List[float], str]:
        """Generate embedding using EmbeddingProcessor."""
        try:
            # Get TidyLLM provider
            tidyllm_provider = self.tidyllm_providers.get(provider, self.default_provider)
            
            if not tidyllm_provider:
                raise ValueError(f"Provider '{provider}' not available")
            
            # Generate embedding
            embedding = await asyncio.get_event_loop().run_in_executor(
                None,
                self.embedding_processor.embed,
                text,
                tidyllm_provider
            )
            
            # Ensure it's a list
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                embedding = list(embedding)
            
            # Standardize dimension if needed
            if len(embedding) != target_dimension:
                embedding = self._standardize_dimension(embedding, target_dimension)
            
            # Normalize if requested
            if normalize:
                embedding = self._normalize_embedding(embedding)
            
            return embedding, f"embedding_processor_{provider}"
            
        except Exception as e:
            logger.error(f"EmbeddingProcessor generation failed: {e}")
            raise
    
    async def _generate_with_gateway(self,
                                   text: str,
                                   provider: str,
                                   target_dimension: int,
                                   normalize: bool) -> tuple[List[float], str]:
        """Generate embedding using CorporateLLMGateway (compliant)."""
        try:
            from ...gateways.corporate_llm_gateway import LLMRequest

            if not self.corporate_gateway:
                raise ValueError("CorporateLLMGateway not initialized")

            # Map provider to model ID
            model_map = {
                "bedrock": "titan-embed-v2",
                "titan": "titan-embed-v2",
                "cohere": "cohere-embed",
                "default": "titan-embed-v2"
            }
            model_id = model_map.get(provider, provider)

            # Create embedding request with dimensions
            request = LLMRequest(
                prompt=text,
                model_id=model_id,
                is_embedding=True,
                dimensions=target_dimension,  # IMPORTANT: Pass dimensions for Titan v2
                user_id="embedding_worker",
                audit_reason="vector_embedding_generation"
            )

            # Process through gateway (tracked and compliant)
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.corporate_gateway.process_embedding_request,
                request
            )

            if not response.success:
                raise Exception(f"Gateway embedding failed: {response.error}")

            # Parse embedding from response
            import json
            embedding = json.loads(response.content)

            # Ensure it's a list
            if not isinstance(embedding, list):
                embedding = list(embedding)

            # Standardize dimension if needed (should already match)
            if len(embedding) != target_dimension:
                embedding = self._standardize_dimension(embedding, target_dimension)

            # Normalize if requested
            if normalize:
                embedding = self._normalize_embedding(embedding)

            return embedding, f"gateway_{model_id}"

        except Exception as e:
            logger.error(f"TidyLLM embedding generation failed: {e}")
            raise
    
    async def _generate_mock_embedding(self, text: str, dimension: int) -> List[float]:
        """Generate mock embedding for testing/fallback."""
        import random
        import hashlib
        
        # Use text hash as seed for reproducible mock embeddings
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        random.seed(seed)
        
        # Generate random embedding
        embedding = [random.gauss(0, 1) for _ in range(dimension)]
        
        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def _standardize_dimension(self, embedding: List[float], target_dimension: int) -> List[float]:
        """Standardize embedding to target dimension."""
        current_dim = len(embedding)
        
        if current_dim == target_dimension:
            return embedding
        elif current_dim > target_dimension:
            # Truncate
            return embedding[:target_dimension]
        else:
            # Pad with zeros
            return embedding + [0.0] * (target_dimension - current_dim)
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """L2 normalize the embedding vector."""
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            return [x / norm for x in embedding]
        return embedding
    
    # Task submission convenience methods
    async def generate_embedding(self,
                                text_id: str,
                                text_content: str,
                                model_provider: str = "default",
                                target_dimension: int = None,
                                normalize: bool = True,
                                priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        Submit single embedding generation task.
        
        Returns:
            Task ID for tracking
        """
        request = EmbeddingRequest(
            text_id=text_id,
            text_content=text_content,
            model_provider=model_provider,
            target_dimension=target_dimension or self.default_target_dimension,
            normalize=normalize
        )
        
        task = await self.submit_task(
            task_type="generate_embedding",
            task_input=request,
            priority=priority
        )
        
        return task.task_id
    
    async def generate_batch_embeddings(self,
                                      batch_id: str,
                                      texts: List[Dict[str, str]],
                                      model_provider: str = "default",
                                      target_dimension: int = None,
                                      normalize: bool = True,
                                      priority: TaskPriority = TaskPriority.HIGH) -> str:
        """
        Submit batch embedding generation task.
        
        Args:
            batch_id: Unique batch identifier
            texts: List of {"id": str, "content": str} dictionaries
            
        Returns:
            Task ID for tracking
        """
        request = BatchEmbeddingRequest(
            batch_id=batch_id,
            texts=texts,
            model_provider=model_provider,
            target_dimension=target_dimension or self.default_target_dimension,
            normalize=normalize
        )
        
        task = await self.submit_task(
            task_type="generate_batch_embeddings",
            task_input=request,
            priority=priority
        )
        
        return task.task_id