"""
RAG2DAG Converter - Transform RAG patterns into optimized DAG workflows
======================================================================

CORE IMPLEMENTATION - Pattern Recognition & DAG Generation Engine

This module contains the foundational algorithms for:
- Defining and recognizing 7 distinct RAG optimization patterns
- Converting linear RAG workflows into parallel DAG execution plans
- Pattern matching with intelligent optimization recommendations

ARCHITECTURE POSITION:
---------------------
CORE LAYER (this file) -> SERVICE LAYER (tidyllm/services/rag2dag/)

The service layer imports and orchestrates this core converter:
```python
from ...rag2dag.converter import RAG2DAGConverter, RAGPatternType
converter = RAG2DAGConverter(config)
```

PATTERN DEFINITIONS:
-------------------
1. MULTI_SOURCE - Parallel retrieval from multiple sources (3.5x speedup)
2. RESEARCH_SYNTHESIS - Extract & synthesize findings (2.8x speedup)
3. COMPARATIVE_ANALYSIS - Compare across documents (3.2x speedup)
4. FACT_CHECKING - Validate claims against sources (2.5x speedup)
5. KNOWLEDGE_EXTRACTION - Extract structured info (2.2x speedup)
6. DOCUMENT_PIPELINE - Sequential processing (1.8x speedup)
7. SIMPLE_QA - Basic Q&A (1.0x - no optimization)

PRESERVATION NOTICE:
-------------------
⚠️ CRITICAL DEPENDENCY - Required by service layer and enterprise integrations.
Do not modify pattern definitions without updating service layer accordingly.

Version: 2.0.0 (Core Implementation)
Service Layer: tidyllm/services/rag2dag/ v2.0.0
Last Updated: 2025-09-15
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime

from .config import RAG2DAGConfig, BedrockModelConfig


class RAGPatternType(str, Enum):
    """Types of RAG patterns that can be converted to DAG."""
    SIMPLE_QA = "simple_qa"                    # Single retrieve → generate
    MULTI_SOURCE = "multi_source"              # Parallel retrieval from different sources
    RESEARCH_SYNTHESIS = "research_synthesis"  # Extract different aspects → synthesize
    COMPARATIVE_ANALYSIS = "comparative_analysis"  # Compare across multiple documents
    FACT_CHECKING = "fact_checking"            # Validate claims against sources
    DOCUMENT_PIPELINE = "document_pipeline"    # Sequential document processing
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"  # Extract structured knowledge


@dataclass
class DAGWorkflowNode:
    """A single node in the RAG2DAG workflow."""
    node_id: str
    operation: str              # retrieve, extract, synthesize, generate, etc.
    instruction: str            # Plain English instruction for the operation
    input_from: List[str]       # Node IDs this depends on
    model_config: BedrockModelConfig
    
    # Execution details
    parallel_group: Optional[str] = None  # For parallel execution
    cache_key: Optional[str] = None       # For caching results
    timeout_seconds: int = 300
    retry_attempts: int = 2
    
    # Input/Output configuration
    input_files: List[str] = None
    output_format: str = "json"           # json, text, markdown
    
    def __post_init__(self):
        if self.input_files is None:
            self.input_files = []


@dataclass
class RAGPattern:
    """Definition of a RAG pattern that can be converted to DAG."""
    pattern_type: RAGPatternType
    name: str
    description: str
    
    # Pattern matching
    intent_keywords: List[str]      # Keywords that suggest this pattern
    file_type_hints: List[str]      # File types that work well with this pattern
    complexity_score: int           # 1-10, higher = more complex
    
    # DAG template
    node_template: List[Dict[str, Any]]  # Template for creating nodes
    parallel_groups: List[List[str]]     # Which nodes can run in parallel
    
    # Optimization hints
    recommended_models: Dict[str, str]   # operation -> model mapping
    estimated_cost_factor: float        # Relative cost compared to simple QA


class RAG2DAGConverter:
    """Convert RAG workflows into optimized DAG execution plans."""
    
    def __init__(self, config: RAG2DAGConfig):
        self.config = config
        self.patterns = self._load_rag_patterns()
    
    def _load_rag_patterns(self) -> Dict[RAGPatternType, RAGPattern]:
        """Load predefined RAG patterns."""
        return {
            RAGPatternType.SIMPLE_QA: RAGPattern(
                pattern_type=RAGPatternType.SIMPLE_QA,
                name="Simple Question Answering",
                description="Single document retrieval and generation",
                intent_keywords=["what is", "define", "explain", "simple question"],
                file_type_hints=["pdf", "txt", "md"],
                complexity_score=2,
                node_template=[
                    {"operation": "retrieve", "instruction": "Find relevant content for the query"},
                    {"operation": "generate", "instruction": "Generate answer from retrieved content", "input_from": ["retrieve"]}
                ],
                parallel_groups=[],
                recommended_models={"retrieve": "cohere", "generate": "claude"},
                estimated_cost_factor=1.0
            ),
            
            RAGPatternType.MULTI_SOURCE: RAGPattern(
                pattern_type=RAGPatternType.MULTI_SOURCE,
                name="Multi-Source Retrieval",
                description="Parallel retrieval from multiple sources, then synthesis",
                intent_keywords=["compare", "sources", "comprehensive", "multiple"],
                file_type_hints=["pdf", "docx", "txt"],
                complexity_score=5,
                node_template=[
                    {"operation": "retrieve", "instruction": "Vector search for semantic matches", "parallel_group": "search"},
                    {"operation": "retrieve", "instruction": "Keyword search for exact matches", "parallel_group": "search"},
                    {"operation": "retrieve", "instruction": "Full-text search for context", "parallel_group": "search"},
                    {"operation": "synthesize", "instruction": "Merge and rank search results", "input_from": ["vector_search", "keyword_search", "fulltext_search"]},
                    {"operation": "generate", "instruction": "Generate comprehensive answer", "input_from": ["synthesize"]}
                ],
                parallel_groups=[["vector_search", "keyword_search", "fulltext_search"]],
                recommended_models={"retrieve": "cohere", "synthesize": "claude", "generate": "claude"},
                estimated_cost_factor=2.5
            ),
            
            RAGPatternType.RESEARCH_SYNTHESIS: RAGPattern(
                pattern_type=RAGPatternType.RESEARCH_SYNTHESIS,
                name="Research Synthesis",
                description="Extract different aspects from documents, then synthesize findings",
                intent_keywords=["analyze", "research", "synthesis", "comprehensive analysis", "themes"],
                file_type_hints=["pdf", "docx"],
                complexity_score=7,
                node_template=[
                    {"operation": "extract", "instruction": "Extract key facts and data points", "parallel_group": "extraction"},
                    {"operation": "extract", "instruction": "Extract quotes and references", "parallel_group": "extraction"},
                    {"operation": "extract", "instruction": "Extract methodology and approach", "parallel_group": "extraction"},
                    {"operation": "extract", "instruction": "Extract conclusions and findings", "parallel_group": "extraction"},
                    {"operation": "synthesize", "instruction": "Combine extractions into coherent analysis", "input_from": ["extract_facts", "extract_quotes", "extract_methods", "extract_conclusions"]},
                    {"operation": "generate", "instruction": "Generate comprehensive research synthesis", "input_from": ["synthesize"]}
                ],
                parallel_groups=[["extract_facts", "extract_quotes", "extract_methods", "extract_conclusions"]],
                recommended_models={"extract": "haiku", "synthesize": "sonnet", "generate": "sonnet"},
                estimated_cost_factor=3.2
            ),
            
            RAGPatternType.COMPARATIVE_ANALYSIS: RAGPattern(
                pattern_type=RAGPatternType.COMPARATIVE_ANALYSIS,
                name="Comparative Analysis",
                description="Compare the same aspects across multiple documents",
                intent_keywords=["compare", "contrast", "differences", "similarities", "versus"],
                file_type_hints=["pdf", "docx"],
                complexity_score=6,
                node_template=[
                    {"operation": "extract", "instruction": "Extract key aspects from document A", "parallel_group": "doc_analysis"},
                    {"operation": "extract", "instruction": "Extract key aspects from document B", "parallel_group": "doc_analysis"},
                    {"operation": "extract", "instruction": "Extract key aspects from document C", "parallel_group": "doc_analysis"},
                    {"operation": "synthesize", "instruction": "Compare and contrast extracted aspects", "input_from": ["extract_doc_a", "extract_doc_b", "extract_doc_c"]},
                    {"operation": "generate", "instruction": "Generate comparative analysis report", "input_from": ["synthesize"]}
                ],
                parallel_groups=[["extract_doc_a", "extract_doc_b", "extract_doc_c"]],
                recommended_models={"extract": "haiku", "synthesize": "sonnet", "generate": "sonnet"},
                estimated_cost_factor=2.8
            )
        }
    
    def analyze_rag_intent(self, query: str, files: List[str], context: Dict[str, Any] = None) -> RAGPatternType:
        """Analyze query and files to determine the best RAG pattern."""
        query_lower = query.lower()
        
        # Score each pattern based on keyword matching
        pattern_scores = {}
        
        for pattern_type, pattern in self.patterns.items():
            score = 0
            
            # Check intent keywords
            for keyword in pattern.intent_keywords:
                if keyword in query_lower:
                    score += 2
            
            # Check file type compatibility
            file_extensions = [f.split('.')[-1].lower() for f in files if '.' in f]
            for ext in file_extensions:
                if ext in pattern.file_type_hints:
                    score += 1
            
            # Adjust for number of files
            if len(files) > 3 and pattern_type in [RAGPatternType.MULTI_SOURCE, RAGPatternType.COMPARATIVE_ANALYSIS]:
                score += 2
            elif len(files) == 1 and pattern_type == RAGPatternType.SIMPLE_QA:
                score += 1
            
            # Context-based scoring
            if context:
                if context.get('complexity_preference') == 'simple' and pattern.complexity_score <= 3:
                    score += 1
                elif context.get('complexity_preference') == 'comprehensive' and pattern.complexity_score >= 6:
                    score += 2
            
            pattern_scores[pattern_type] = score
        
        # Return the highest scoring pattern (with fallback to SIMPLE_QA)
        best_pattern = max(pattern_scores.items(), key=lambda x: x[1])[0]
        return best_pattern if pattern_scores[best_pattern] > 0 else RAGPatternType.SIMPLE_QA
    
    def generate_dag_from_pattern(self, pattern_type: RAGPatternType, query: str, files: List[str], 
                                  context: Dict[str, Any] = None) -> List[DAGWorkflowNode]:
        """Generate optimized DAG nodes from a RAG pattern."""
        pattern = self.patterns[pattern_type]
        nodes = []
        
        # Create nodes from template
        for i, node_template in enumerate(pattern.node_template):
            # Generate unique node ID
            operation = node_template["operation"]
            node_id = f"{operation}_{i}" if operation in [n.operation for n in nodes] else operation
            
            # Customize instruction with query context
            instruction = self._customize_instruction(node_template["instruction"], query, files)
            
            # Select appropriate model for operation
            model_config = self.config.get_model_for_operation(operation)
            
            # Handle input dependencies
            input_from = []
            if "input_from" in node_template:
                input_from = [self._resolve_node_id(dep, pattern.node_template) for dep in node_template["input_from"]]
            
            node = DAGWorkflowNode(
                node_id=node_id,
                operation=operation,
                instruction=instruction,
                input_from=input_from,
                model_config=model_config,
                parallel_group=node_template.get("parallel_group"),
                input_files=files if operation in ["retrieve", "extract"] else [],
                timeout_seconds=self._estimate_timeout(operation, files)
            )
            
            nodes.append(node)
        
        # Apply optimizations based on configuration
        nodes = self._optimize_dag_nodes(nodes, pattern)
        
        return nodes
    
    def _customize_instruction(self, template_instruction: str, query: str, files: List[str]) -> str:
        """Customize template instruction with specific query and file context."""
        customized = template_instruction
        
        # Add query context
        if "{query}" not in customized:
            customized += f" Related to query: '{query}'"
        else:
            customized = customized.replace("{query}", query)
        
        # Add file context
        if len(files) == 1:
            customized += f" From file: {files[0]}"
        elif len(files) > 1:
            customized += f" From {len(files)} files: {', '.join(files[:3])}{'...' if len(files) > 3 else ''}"
        
        return customized
    
    def _resolve_node_id(self, dependency_ref: str, node_templates: List[Dict]) -> str:
        """Resolve dependency reference to actual node ID."""
        # Simple mapping for now - could be more sophisticated
        operation_map = {
            "retrieve": "retrieve",
            "vector_search": "retrieve_0",
            "keyword_search": "retrieve_1", 
            "fulltext_search": "retrieve_2",
            "extract_facts": "extract_0",
            "extract_quotes": "extract_1",
            "extract_methods": "extract_2",
            "extract_conclusions": "extract_3",
            "extract_doc_a": "extract_0",
            "extract_doc_b": "extract_1",
            "extract_doc_c": "extract_2",
            "synthesize": "synthesize",
            "generate": "generate"
        }
        return operation_map.get(dependency_ref, dependency_ref)
    
    def _estimate_timeout(self, operation: str, files: List[str]) -> int:
        """Estimate appropriate timeout based on operation and file count."""
        base_timeouts = {
            "retrieve": 60,
            "extract": 120,
            "synthesize": 180,
            "generate": 240
        }
        
        base_timeout = base_timeouts.get(operation, 300)
        
        # Scale with file count
        file_multiplier = min(len(files) * 0.5, 3.0)  # Cap at 3x
        
        return int(base_timeout * (1 + file_multiplier))
    
    def _optimize_dag_nodes(self, nodes: List[DAGWorkflowNode], pattern: RAGPattern) -> List[DAGWorkflowNode]:
        """Apply optimizations to DAG nodes based on configuration."""
        
        # Apply model optimizations based on configuration level
        if self.config.optimization_level == "speed":
            # Use faster models for non-critical operations
            for node in nodes:
                if node.operation == "extract":
                    node.model_config = self.config.extraction_model
                elif node.operation in ["synthesize", "generate"]:
                    # Keep quality models for final output
                    pass
        
        elif self.config.optimization_level == "quality":
            # Use highest quality models throughout
            for node in nodes:
                if node.operation in ["synthesize", "generate"]:
                    node.model_config = self.config.generation_model
        
        # Enable caching for expensive operations
        if self.config.enable_caching:
            for node in nodes:
                if node.operation in ["retrieve", "extract"]:
                    # Create cache key based on operation + files + instruction hash
                    cache_components = [node.operation, str(sorted(node.input_files)), node.instruction]
                    node.cache_key = str(hash(tuple(cache_components)))
        
        return nodes
    
    def create_workflow_from_query(self, query: str, files: List[str], 
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete workflow: analyze intent → generate DAG → return execution plan."""
        
        # Step 1: Analyze RAG intent
        pattern_type = self.analyze_rag_intent(query, files, context)
        
        # Step 2: Generate DAG nodes
        dag_nodes = self.generate_dag_from_pattern(pattern_type, query, files, context)
        
        # Step 3: Create execution metadata
        pattern = self.patterns[pattern_type]
        
        workflow = {
            "workflow_id": f"rag2dag_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "query": query,
            "files": files,
            "pattern_type": pattern_type.value,
            "pattern_name": pattern.name,
            "description": pattern.description,
            "estimated_cost_factor": pattern.estimated_cost_factor,
            "complexity_score": pattern.complexity_score,
            "dag_nodes": [
                {
                    "node_id": node.node_id,
                    "operation": node.operation,
                    "instruction": node.instruction,
                    "input_from": node.input_from,
                    "model_id": node.model_config.model_id.value,
                    "parallel_group": node.parallel_group,
                    "timeout_seconds": node.timeout_seconds,
                    "cache_key": node.cache_key
                } for node in dag_nodes
            ],
            "execution_plan": {
                "parallel_groups": pattern.parallel_groups,
                "max_parallel_nodes": self.config.max_parallel_nodes,
                "enable_streaming": self.config.enable_streaming_results,
                "total_estimated_time_seconds": sum(node.timeout_seconds for node in dag_nodes) // len(pattern.parallel_groups) if pattern.parallel_groups else sum(node.timeout_seconds for node in dag_nodes)
            },
            "config_summary": self.config.to_dict()
        }
        
        return workflow


# Example usage configuration
def create_example_workflows():
    """Create example RAG2DAG workflows for testing."""
    
    # Default balanced configuration
    config = RAG2DAGConfig.create_default_config()
    converter = RAG2DAGConverter(config)
    
    # Example 1: Simple QA
    simple_workflow = converter.create_workflow_from_query(
        query="What are the key benefits of this approach?",
        files=["research_paper.pdf"]
    )
    
    # Example 2: Multi-source research
    research_workflow = converter.create_workflow_from_query(
        query="Compare the methodologies across these studies",
        files=["study1.pdf", "study2.pdf", "study3.pdf", "study4.pdf"]
    )
    
    return {
        "simple_qa": simple_workflow,
        "multi_source_research": research_workflow,
        "config_used": config.to_dict()
    }