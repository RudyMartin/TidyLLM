"""
RAG2DAG Configuration with Bedrock Model Settings
=================================================
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


class BedrockModel(str, Enum):
    """Available Bedrock models for RAG2DAG operations."""
    
    # Claude Models (Reasoning & Analysis)
    CLAUDE_3_5_SONNET = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    
    # Titan Models (Embeddings & Text)
    TITAN_TEXT_G1_LARGE = "amazon.titan-text-lite-v1"
    TITAN_TEXT_G1_EXPRESS = "amazon.titan-text-express-v1"
    TITAN_EMBEDDINGS_G1 = "amazon.titan-embed-text-v1"
    
    # Cohere Models (Retrieval & Ranking)
    COHERE_COMMAND_LIGHT = "cohere.command-light-text-v14"
    COHERE_COMMAND_R = "cohere.command-r-v1:0"
    COHERE_EMBED_ENGLISH = "cohere.embed-english-v3"
    
    # Llama Models (Fast Processing)
    LLAMA2_70B = "meta.llama2-70b-chat-v1"
    LLAMA2_13B = "meta.llama2-13b-chat-v1"


@dataclass
class BedrockModelConfig:
    """Configuration for specific Bedrock model usage."""
    model_id: BedrockModel
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 250
    stop_sequences: List[str] = None
    
    # Model-specific optimizations
    use_streaming: bool = False
    batch_size: int = 1
    timeout_seconds: int = 300
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


@dataclass 
class RAG2DAGConfig:
    """Configuration for RAG2DAG converter and execution."""
    
    # Primary model for workflow orchestration
    orchestrator_model: BedrockModelConfig
    
    # Specialized models for different DAG operations
    retrieval_model: BedrockModelConfig      # For searching/finding content
    extraction_model: BedrockModelConfig     # For extracting data from docs
    synthesis_model: BedrockModelConfig      # For combining results
    generation_model: BedrockModelConfig     # For final output generation
    embedding_model: BedrockModelConfig      # For vector operations
    
    # Workflow optimization settings
    max_parallel_nodes: int = 5
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    
    # Performance tuning
    optimization_level: str = "balanced"  # "speed", "balanced", "quality"
    enable_streaming_results: bool = True
    max_workflow_depth: int = 10
    
    # AWS/Bedrock settings
    aws_region: str = "us-east-1"
    bedrock_runtime_endpoint: Optional[str] = None
    
    @classmethod
    def create_default_config(cls) -> "RAG2DAGConfig":
        """Create default configuration optimized for most RAG workflows."""
        return cls(
            # Claude 3.5 Sonnet for complex orchestration and synthesis
            orchestrator_model=BedrockModelConfig(
                model_id=BedrockModel.CLAUDE_3_5_SONNET,
                max_tokens=8192,
                temperature=0.1
            ),
            
            # Cohere Command-R for retrieval and ranking
            retrieval_model=BedrockModelConfig(
                model_id=BedrockModel.COHERE_COMMAND_R,
                max_tokens=2048,
                temperature=0.0
            ),
            
            # Claude 3 Haiku for fast extraction (cheaper)
            extraction_model=BedrockModelConfig(
                model_id=BedrockModel.CLAUDE_3_HAIKU,
                max_tokens=4096,
                temperature=0.0,
                use_streaming=True
            ),
            
            # Claude 3.5 Sonnet for synthesis (best reasoning)
            synthesis_model=BedrockModelConfig(
                model_id=BedrockModel.CLAUDE_3_5_SONNET,
                max_tokens=8192,
                temperature=0.2
            ),
            
            # Claude 3.5 Sonnet for final generation
            generation_model=BedrockModelConfig(
                model_id=BedrockModel.CLAUDE_3_5_SONNET,
                max_tokens=8192,
                temperature=0.3
            ),
            
            # Titan embeddings for vector operations
            embedding_model=BedrockModelConfig(
                model_id=BedrockModel.TITAN_EMBEDDINGS_G1,
                max_tokens=512,
                temperature=0.0,
                batch_size=25  # Titan supports batching
            ),
            
            # Optimization settings
            max_parallel_nodes=3,  # Conservative default
            optimization_level="balanced",
            enable_streaming_results=True
        )
    
    @classmethod
    def create_speed_config(cls) -> "RAG2DAGConfig":
        """Create configuration optimized for speed over quality."""
        return cls(
            # Haiku for orchestration (fastest)
            orchestrator_model=BedrockModelConfig(
                model_id=BedrockModel.CLAUDE_3_HAIKU,
                max_tokens=4096,
                temperature=0.0,
                use_streaming=True
            ),
            
            # Cohere Command Light for retrieval (faster)
            retrieval_model=BedrockModelConfig(
                model_id=BedrockModel.COHERE_COMMAND_LIGHT,
                max_tokens=1024,
                temperature=0.0
            ),
            
            # Haiku for extraction (fastest)
            extraction_model=BedrockModelConfig(
                model_id=BedrockModel.CLAUDE_3_HAIKU,
                max_tokens=2048,
                temperature=0.0,
                use_streaming=True,
                batch_size=3
            ),
            
            # Titan for synthesis (fast, cheap)
            synthesis_model=BedrockModelConfig(
                model_id=BedrockModel.TITAN_TEXT_G1_EXPRESS,
                max_tokens=4096,
                temperature=0.1
            ),
            
            # Haiku for generation (fast)
            generation_model=BedrockModelConfig(
                model_id=BedrockModel.CLAUDE_3_HAIKU,
                max_tokens=4096,
                temperature=0.2,
                use_streaming=True
            ),
            
            # Cohere embeddings (faster than Titan for some operations)
            embedding_model=BedrockModelConfig(
                model_id=BedrockModel.COHERE_EMBED_ENGLISH,
                max_tokens=512,
                temperature=0.0,
                batch_size=50
            ),
            
            # Aggressive parallel processing
            max_parallel_nodes=8,
            optimization_level="speed",
            enable_streaming_results=True
        )
    
    @classmethod 
    def create_quality_config(cls) -> "RAG2DAGConfig":
        """Create configuration optimized for quality over speed."""
        return cls(
            # Claude 3.5 Sonnet for everything (highest quality)
            orchestrator_model=BedrockModelConfig(
                model_id=BedrockModel.CLAUDE_3_5_SONNET,
                max_tokens=8192,
                temperature=0.0
            ),
            
            retrieval_model=BedrockModelConfig(
                model_id=BedrockModel.CLAUDE_3_5_SONNET,
                max_tokens=4096,
                temperature=0.0
            ),
            
            extraction_model=BedrockModelConfig(
                model_id=BedrockModel.CLAUDE_3_5_SONNET,
                max_tokens=8192,
                temperature=0.0
            ),
            
            synthesis_model=BedrockModelConfig(
                model_id=BedrockModel.CLAUDE_3_5_SONNET,
                max_tokens=8192,
                temperature=0.1
            ),
            
            generation_model=BedrockModelConfig(
                model_id=BedrockModel.CLAUDE_3_5_SONNET,
                max_tokens=8192,
                temperature=0.2
            ),
            
            embedding_model=BedrockModelConfig(
                model_id=BedrockModel.TITAN_EMBEDDINGS_G1,
                max_tokens=512,
                temperature=0.0,
                batch_size=10  # Smaller batches for quality
            ),
            
            # Conservative parallel processing for quality
            max_parallel_nodes=2,
            optimization_level="quality",
            enable_streaming_results=False  # Wait for complete results
        )
    
    def get_model_for_operation(self, operation: str) -> BedrockModelConfig:
        """Get the appropriate model for a specific operation."""
        operation_map = {
            "orchestrate": self.orchestrator_model,
            "retrieve": self.retrieval_model,
            "search": self.retrieval_model,
            "extract": self.extraction_model,
            "parse": self.extraction_model,
            "synthesize": self.synthesis_model,
            "merge": self.synthesis_model,
            "generate": self.generation_model,
            "create": self.generation_model,
            "embed": self.embedding_model,
            "vectorize": self.embedding_model
        }
        
        return operation_map.get(operation, self.orchestrator_model)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "orchestrator_model": {
                "model_id": self.orchestrator_model.model_id.value,
                "max_tokens": self.orchestrator_model.max_tokens,
                "temperature": self.orchestrator_model.temperature,
                "use_streaming": self.orchestrator_model.use_streaming
            },
            "retrieval_model": {
                "model_id": self.retrieval_model.model_id.value,
                "max_tokens": self.retrieval_model.max_tokens,
                "temperature": self.retrieval_model.temperature
            },
            "extraction_model": {
                "model_id": self.extraction_model.model_id.value,
                "max_tokens": self.extraction_model.max_tokens,
                "temperature": self.extraction_model.temperature,
                "use_streaming": self.extraction_model.use_streaming
            },
            "synthesis_model": {
                "model_id": self.synthesis_model.model_id.value,
                "max_tokens": self.synthesis_model.max_tokens,
                "temperature": self.synthesis_model.temperature
            },
            "generation_model": {
                "model_id": self.generation_model.model_id.value,
                "max_tokens": self.generation_model.max_tokens,
                "temperature": self.generation_model.temperature
            },
            "embedding_model": {
                "model_id": self.embedding_model.model_id.value,
                "batch_size": self.embedding_model.batch_size
            },
            "workflow_settings": {
                "max_parallel_nodes": self.max_parallel_nodes,
                "optimization_level": self.optimization_level,
                "enable_streaming_results": self.enable_streaming_results,
                "aws_region": self.aws_region
            }
        }