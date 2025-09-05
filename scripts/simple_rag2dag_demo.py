#!/usr/bin/env python3
"""
Simple RAG2DAG Bedrock Model Demo
=================================
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tidyllm.rag2dag import RAG2DAGConverter, RAG2DAGConfig

def main():
    print("=" * 60)
    print("RAG2DAG BEDROCK MODEL CONFIGURATION")
    print("=" * 60)
    
    # Create default configuration
    config = RAG2DAGConfig.create_default_config()
    
    print("\nBEDROCK MODEL ASSIGNMENTS:")
    print(f"  Orchestrator:  {config.orchestrator_model.model_id.value}")
    print(f"  Retrieval:     {config.retrieval_model.model_id.value}")
    print(f"  Extraction:    {config.extraction_model.model_id.value}")
    print(f"  Synthesis:     {config.synthesis_model.model_id.value}")
    print(f"  Generation:    {config.generation_model.model_id.value}")
    print(f"  Embeddings:    {config.embedding_model.model_id.value}")
    
    print(f"\nCONFIGURATION SETTINGS:")
    print(f"  Max Parallel Nodes: {config.max_parallel_nodes}")
    print(f"  Optimization Level: {config.optimization_level}")
    print(f"  Streaming Results:  {config.enable_streaming_results}")
    print(f"  AWS Region:         {config.aws_region}")
    
    # Test workflow generation
    print("\n" + "=" * 60)
    print("SAMPLE WORKFLOW GENERATION")
    print("=" * 60)
    
    converter = RAG2DAGConverter(config)
    
    # Create sample workflow
    workflow = converter.create_workflow_from_query(
        query="What are the main findings in these research papers?",
        files=["paper1.pdf", "paper2.pdf", "paper3.pdf"]
    )
    
    print(f"\nDetected Pattern: {workflow['pattern_name']}")
    print(f"Complexity Score: {workflow['complexity_score']}/10")
    print(f"Cost Factor: {workflow['estimated_cost_factor']}x")
    
    print(f"\nDAG Workflow Nodes ({len(workflow['dag_nodes'])}):")
    for i, node in enumerate(workflow['dag_nodes'], 1):
        model_short = node['model_id'].split('.')[-1].replace('-v1:0', '').replace('-20241022-v2:0', '')
        print(f"  {i}. {node['operation'].upper()}: {model_short}")
        print(f"     \"{node['instruction'][:50]}...\"")
        if node['input_from']:
            print(f"     Dependencies: {', '.join(node['input_from'])}")
    
    print(f"\nExecution Plan:")
    exec_plan = workflow['execution_plan']
    print(f"  Estimated Time: {exec_plan['total_estimated_time_seconds']}s")
    print(f"  Max Parallel: {exec_plan['max_parallel_nodes']} nodes")
    print(f"  Streaming: {'Yes' if exec_plan['enable_streaming'] else 'No'}")
    
    # Show different optimization levels
    print("\n" + "=" * 60)
    print("OPTIMIZATION LEVEL COMPARISON")
    print("=" * 60)
    
    configs = {
        "Speed": RAG2DAGConfig.create_speed_config(),
        "Balanced": RAG2DAGConfig.create_default_config(),
        "Quality": RAG2DAGConfig.create_quality_config()
    }
    
    print(f"\n{'Level':<10} {'Orchestrator':<25} {'Generation':<25} {'Parallel'}")
    print("-" * 70)
    
    for level, cfg in configs.items():
        orch_model = cfg.orchestrator_model.model_id.value.split('.')[-1].replace('-v1:0', '').replace('-20241022-v2:0', '')[:20]
        gen_model = cfg.generation_model.model_id.value.split('.')[-1].replace('-v1:0', '').replace('-20241022-v2:0', '')[:20]
        print(f"{level:<10} {orch_model:<25} {gen_model:<25} {cfg.max_parallel_nodes}")

if __name__ == "__main__":
    try:
        main()
        print("\nRAG2DAG Demo Complete - Bedrock models configured!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()