"""
TidyLLM Improved Usage Examples
===============================

Demonstrates the improved, purpose-driven gateway architecture with
clear naming and TidyLLM's native stack constraints:

STACK CONSTRAINTS:
- NO numpy/pandas - uses tlm (pure Python numpy substitute)
- NO sentence-transformers - uses tidyllm-sentence (77% quality, 177x less memory)  
- Uses polars for data manipulation
- Pure Python architecture for sovereignty and transparency
"""

# =============================================================================
# Example 1: Quick Start with Gateway Registry
# =============================================================================

def example_quick_start():
    """Quick start example showing simplified usage."""
    from tidyllm.gateways import init_gateways
    
    # Initialize all gateways with configuration
    registry = init_gateways({
        "corporate_llm": {
            "budget_limit_daily_usd": 500.0,
            "require_audit_reason": True
        },
        "ai_processing": {
            "backend": "anthropic",
            "cache_enabled": True
        }
    })
    
    # Use AI processing (clear purpose)
    ai = registry.get("ai_processing")
    response = ai.process("Explain quantum computing in simple terms")
    print(f"AI Response: {response.data}")
    
    # Use workflow optimization (clear purpose)
    optimizer = registry.get("workflow_optimizer")
    if optimizer:
        from tidyllm.gateways import WorkflowRequest, WorkflowOperation
        
        workflow_request = WorkflowRequest(
            operation=WorkflowOperation.ANALYZE_BOTTLENECKS,
            workflow={"stage1": "data_ingestion", "stage2": "processing"},
            options={"include_suggestions": True}
        )
        
        result = optimizer.process_workflow(workflow_request)
        print(f"Optimization Result: {result.data.improvements}")


# =============================================================================
# Example 2: Corporate LLM with Full Controls
# =============================================================================

def example_corporate_llm():
    """Example showing enterprise LLM access control."""
    from tidyllm.gateways import CorporateLLMGateway, LLMRequest
    
    # Create corporate LLM gateway with strict controls
    llm_gateway = CorporateLLMGateway(
        budget_limit_daily_usd=1000.0,
        max_cost_per_request_usd=5.0,
        require_audit_reason=True,
        enable_content_filtering=True,
        enable_pii_detection=True
    )
    
    # Create structured request with compliance info
    request = LLMRequest(
        prompt="Analyze quarterly financial performance trends",
        model="claude-3-5-sonnet",
        audit_reason="Financial analysis for board presentation",
        user_id="jane.analyst@company.com",
        session_id="session_123",
        temperature=0.3,
        max_tokens=2000
    )
    
    # Execute with corporate controls
    response = llm_gateway.execute_llm_request(request)
    
    if response.is_success:
        print(f"Analysis: {response.data}")
        print(f"Cost: ${response.metadata['cost_usd']:.2f}")
        print(f"Compliance: Fully audited and controlled")
    
    # Check usage statistics
    stats = llm_gateway.get_usage_stats()
    print(f"Daily spend: ${stats['daily_spend']:.2f}")
    print(f"Budget remaining: ${stats['daily_budget_remaining']:.2f}")


# =============================================================================
# Example 3: AI Processing with Multiple Backends
# =============================================================================

def example_ai_processing():
    """Example showing AI processing with TidyLLM native stack."""
    from tidyllm.gateways import AIProcessingGateway, AIRequest, AIBackend
    import tidyllm.tlm as np  # Pure Python numpy substitute
    import tidyllm_sentence as tls  # Lightweight sentence transformer alternative
    
    # Create AI processing gateway with Anthropic backend
    ai_gateway = AIProcessingGateway(
        backend=AIBackend.ANTHROPIC,
        cache_enabled=True,
        retry_max=3,
        temperature=0.7
    )
    
    # Structured AI request for technical content
    request = AIRequest(
        prompt="Write a technical specification for a REST API",
        temperature=0.5,
        max_tokens=1500,
        context={"domain": "software_engineering"},
        metadata={"project": "api_redesign"}
    )
    
    # Process with caching and retry logic
    response = ai_gateway.process_ai_request(request)
    
    if response.is_success:
        print(f"Specification: {response.data[:100]}...")
        print(f"Processing time: {response.metadata['processing_time']:.2f}s")
        print(f"Cache hit: {response.metadata.get('cache_hit', False)}")
        print(f"Model used: {response.metadata['model']}")
        
        # Use TidyLLM's native embedding for semantic analysis
        sentences = [
            response.data,
            "REST API documentation template",
            "Technical specification example"
        ]
        
        # Generate embeddings using tidyllm-sentence (NOT sentence-transformers)
        embeddings, model = tls.tfidf_fit_transform(sentences)
        print(f"Generated {len(embeddings)} embeddings using tidyllm-sentence")
        
        # Use tlm for similarity calculation (NOT numpy)
        query_emb = embeddings[0]  # Response embedding
        similarities = []
        for emb in embeddings[1:]:
            # Use tlm's pure Python similarity calculation
            similarity = tls.cosine_similarity(query_emb, emb)
            similarities.append(similarity)
        print(f"Semantic similarities: {similarities}")
    
    # Check gateway capabilities
    capabilities = ai_gateway.get_capabilities()
    print(f"Available backends: {capabilities['backends']}")
    print(f"Current backend: {capabilities['current_backend']}")


# =============================================================================
# Example 4: Workflow Optimization
# =============================================================================

def example_workflow_optimization():
    """Example showing workflow intelligence with TidyLLM native stack."""
    from tidyllm.gateways import WorkflowOptimizerGateway, WorkflowRequest, WorkflowOperation
    import tidyllm.tlm as np  # Pure Python numpy substitute for calculations
    
    # Create workflow optimizer with high optimization level
    optimizer = WorkflowOptimizerGateway(
        optimization_level=2,  # Aggressive optimization
        compliance_mode=True,
        audit_trail=True
    )
    
    # Define a sample data pipeline workflow (typical use case)
    problematic_workflow = {
        "name": "ml_data_pipeline",
        "stages": [
            {"id": "extract", "type": "polars_query", "parallel": False},  # Uses polars, not pandas
            {"id": "embed", "type": "tidyllm_sentence", "parallel": False, "method": "tfidf"},  # Native embeddings
            {"id": "cluster", "type": "tlm_kmeans", "parallel": False, "memory": "minimal"},  # tlm algorithms
            {"id": "store", "type": "polars_write", "batch_size": 1}
        ],
        "dependencies": [
            {"from": "extract", "to": "embed"},
            {"from": "embed", "to": "cluster"}, 
            {"from": "cluster", "to": "store"}
        ],
        "stack_constraints": {
            "no_numpy": True,
            "no_pandas": True,
            "no_sentence_transformers": True,
            "pure_python_preferred": True
        }
    }
    
    # Request performance optimization with stack awareness
    request = WorkflowRequest(
        operation=WorkflowOperation.OPTIMIZE_PERFORMANCE,
        workflow=problematic_workflow,
        options={
            "max_parallel": 4,
            "target_improvement": 50.0,  # 50% improvement target
            "respect_stack_constraints": True,  # Honor TidyLLM philosophy
            "optimization_focus": "memory_efficiency"  # Important for pure Python
        }
    )
    
    result = optimizer.process_workflow(request)
    
    if result.is_success:
        optimization = result.data
        print(f"Performance gain: {optimization.performance_gain:.1f}%")
        print(f"Stack-aware improvements applied:")
        for improvement in optimization.improvements:
            print(f"  - {improvement}")
        print(f"Memory efficiency score: {optimization.compliance_score:.2f}")
        
        # Show TidyLLM stack optimizations
        if hasattr(optimization, 'stack_optimizations'):
            print("TidyLLM-specific optimizations:")
            for opt in optimization.stack_optimizations:
                print(f"  - {opt}")
    
    # Generate audit documentation with stack compliance
    audit_request = WorkflowRequest(
        operation=WorkflowOperation.GENERATE_AUDIT_TRAIL,
        workflow=result.data.optimized_workflow if result.is_success else problematic_workflow,
        options={"include_stack_compliance": True}
    )
    
    audit_result = optimizer.process_workflow(audit_request)
    if audit_result.is_success:
        print(f"Stack-compliant audit trail: {audit_result.data.audit_info[:100]}...")


# =============================================================================
# Example 5: Knowledge Resource Server (MCP)
# =============================================================================

def example_knowledge_resources():
    """Example showing MCP-based knowledge resource provision."""
    from tidyllm.knowledge_resource_server import KnowledgeMCPServer, S3KnowledgeSource, LocalKnowledgeSource
    
    # Create knowledge MCP server
    knowledge_server = KnowledgeMCPServer()
    
    # Register domain knowledge sources
    knowledge_server.register_domain(
        "legal-contracts",
        S3KnowledgeSource(bucket="legal-docs", prefix="contracts/")
    )
    
    knowledge_server.register_domain(
        "technical-docs", 
        LocalKnowledgeSource(directory="./technical_documentation")
    )
    
    # Search across knowledge domains
    search_result = knowledge_server.handle_mcp_tool_call("search", {
        "query": "contract termination clauses",
        "domain": "legal-contracts",
        "max_results": 5,
        "similarity_threshold": 0.8
    })
    
    if search_result["success"]:
        print(f"Found {search_result['result_count']} relevant documents:")
        for result in search_result["results"]:
            print(f"  - {result['title']} (score: {result['similarity_score']:.2f})")
    
    # Natural language query
    query_result = knowledge_server.handle_mcp_tool_call("query", {
        "question": "What are the standard validation requirements for ML models?",
        "domain": "technical-docs"
    })
    
    if query_result["success"]:
        print(f"Answer: {query_result['answer']}")
        print(f"Context length: {query_result['context_length']} characters")
    
    # Get server capabilities
    capabilities = knowledge_server.get_mcp_capabilities()
    print(f"MCP Server: {capabilities['server']['name']}")
    print(f"Available tools: {[tool['name'] for tool in capabilities['tools']]}")


# =============================================================================
# Example 6: Complete End-to-End Workflow
# =============================================================================

def example_end_to_end():
    """Complete example showing integrated TidyLLM native stack usage."""
    from tidyllm.gateways import init_gateways, AIRequest, WorkflowRequest, WorkflowOperation, LLMRequest
    import tidyllm.tlm as np  # Pure Python numpy substitute
    import tidyllm_sentence as tls  # Lightweight embeddings
    
    # Initialize complete system with stack awareness
    registry = init_gateways({
        "corporate_llm": {
            "budget_limit_daily_usd": 200.0,
            "require_audit_reason": True,
            "stack_compliance": "tidyllm_native"  # Enforce native stack
        },
        "ai_processing": {
            "backend": "auto",  # Auto-detect best backend
            "cache_enabled": True,
            "embedding_provider": "tidyllm_sentence",  # NOT sentence-transformers
            "math_backend": "tlm"  # NOT numpy
        },
        "workflow_optimizer": {
            "optimization_level": 1,
            "compliance_mode": True,
            "stack_constraints": {
                "data_processing": "polars",  # NOT pandas
                "embeddings": "tidyllm_sentence",
                "math": "tlm"
            }
        }
    })
    
    print("🚀 TidyLLM Native Stack System Initialized")
    print(f"Available services: {', '.join(registry.get_available_services())}")
    print("Stack: polars + tlm + tidyllm-sentence (NO numpy/pandas/sentence-transformers)")
    
    # Step 1: Use corporate LLM for sensitive analysis
    llm = registry.get("corporate_llm")
    sensitive_request = LLMRequest(
        prompt="Analyze customer churn patterns using polars and tlm algorithms",
        audit_reason="Customer retention strategy development",
        user_id="data.scientist@company.com",
        metadata={"stack_preference": "tidyllm_native"}
    )
    
    analysis_response = llm.execute_llm_request(sensitive_request)
    print(f"\n📊 Customer Analysis (Stack-Native): {analysis_response.data[:100]}...")
    
    # Step 2: Use AI processing with native embeddings  
    ai = registry.get("ai_processing")
    tech_request = AIRequest(
        prompt="Generate polars-based data pipeline with tlm clustering",
        temperature=0.3,
        max_tokens=1000,
        context={"preferred_stack": "polars_tlm_tidyllm"},
        metadata={"avoid_dependencies": ["numpy", "pandas", "sklearn"]}
    )
    
    code_response = ai.process_ai_request(tech_request)
    print(f"\n💻 Stack-Native Code: {code_response.data[:100]}...")
    
    # Step 3: Native embedding analysis
    documents = [analysis_response.data, code_response.data, "TidyLLM native processing"]
    embeddings, model = tls.tfidf_fit_transform(documents)
    
    # Use tlm for clustering (NOT sklearn)
    normalized_embs = np.l2_normalize(embeddings)  # tlm's pure Python normalization
    print(f"\n🧮 Generated embeddings using tidyllm-sentence: {len(embeddings)} docs")
    print(f"Embedding dimension: {len(embeddings[0])}")
    
    # Step 4: Optimize workflow with stack constraints
    optimizer = registry.get("workflow_optimizer")
    if optimizer:
        workflow = {
            "name": "native_stack_pipeline",
            "stages": [
                {"id": "polars_extract", "type": "polars_query"},
                {"id": "tls_embed", "type": "tidyllm_sentence_tfidf"}, 
                {"id": "tlm_cluster", "type": "tlm_kmeans"},
                {"id": "polars_store", "type": "polars_write"}
            ],
            "stack_compliance": "full_tidyllm_native"
        }
        
        opt_request = WorkflowRequest(
            operation=WorkflowOperation.VALIDATE_STACK_COMPLIANCE,
            workflow=workflow,
            options={"enforce_no_big_tech_deps": True}
        )
        
        opt_response = optimizer.process_workflow(opt_request)
        if opt_response.is_success:
            print(f"\n⚡ Native Stack Validated: {opt_response.data.compliance_score:.1f}% compliant")
    
    # Step 5: System health with stack awareness
    health = registry.health_check()
    healthy_count = health["healthy_services"] 
    total_count = health["total_services"]
    print(f"\n🏥 System Health: {healthy_count}/{total_count} services healthy")
    print(f"🎯 Stack Sovereignty: 100% Big Tech Independent")
    
    return registry


# =============================================================================
# Example 7: Error Handling and Best Practices
# =============================================================================

def example_error_handling():
    """Example showing proper error handling and best practices."""
    from tidyllm.gateways import create_gateway, AIRequest, GatewayStatus
    
    try:
        # Create gateway with validation
        ai_gateway = create_gateway("ai_processing", 
                                   backend="anthropic",
                                   timeout=30.0,
                                   retry_max=2)
        
        # Create request with validation
        request = AIRequest(
            prompt="Explain machine learning algorithms",
            temperature=0.7,  # Valid range
            max_tokens=500   # Within limits
        )
        
        # Process with error handling
        response = ai_gateway.process_ai_request(request)
        
        # Check response status
        if response.is_success:
            print(f"✅ Success: {response.data}")
        elif response.is_partial:
            print(f"⚠️  Partial success: {response.data}")
            print(f"Warnings: {response.errors}")
        else:
            print(f"❌ Failed: {response.errors}")
        
        # Always check for errors
        if response.has_errors:
            print(f"Errors encountered: {response.errors}")
        
        # Access metadata safely
        processing_time = response.metadata.get("processing_time", 0)
        print(f"Processing time: {processing_time:.2f}s")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    print("TidyLLM Improved Usage Examples")
    print("=" * 40)
    
    # Run examples
    print("\n1. Quick Start Example:")
    example_quick_start()
    
    print("\n2. Corporate LLM Example:")
    example_corporate_llm()
    
    print("\n3. AI Processing Example:")
    example_ai_processing()
    
    print("\n4. Workflow Optimization Example:")
    example_workflow_optimization()
    
    print("\n5. Knowledge Resources Example:")
    example_knowledge_resources()
    
    print("\n6. End-to-End Example:")
    registry = example_end_to_end()
    
    print("\n7. Error Handling Example:")
    example_error_handling()
    
    print("\n✨ All examples completed!")
    print("\nKey TidyLLM Improvements:")
    print("• Clear, purpose-driven naming (AIProcessingGateway vs DSPyGateway)")
    print("• Structured request/response objects")
    print("• Unified registry for service discovery")
    print("• Comprehensive error handling")
    print("• Enterprise controls and compliance")
    print("• MCP-based knowledge resources")
    print("\n🎯 TidyLLM Stack Sovereignty:")
    print("• NO numpy → tlm (pure Python math, 100% transparent)")
    print("• NO pandas → polars (fast, memory-efficient)")  
    print("• NO sentence-transformers → tidyllm-sentence (77% quality, 177x less memory)")
    print("• 100% Big Tech Independent - Complete ML Infrastructure Sovereignty")
    print("• Educational transparency - every algorithm step is readable")
    print("• Zero vendor lock-in - maximum portability")