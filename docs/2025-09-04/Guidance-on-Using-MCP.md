# Guidance on Using MCP (Model Context Protocol) with TidyLLM
**Document Version:** 1.0  
**Date:** 2025-09-05  
**Purpose:** Comprehensive guide for implementing and using MCP in TidyLLM

---

## 🧠 **What is MCP (Model Context Protocol)?**

**Model Context Protocol (MCP)** is a standardized protocol for providing structured context and resources to Large Language Models (LLMs). In TidyLLM, the **KnowledgeMCPServer** implements this protocol to serve as a centralized knowledge resource provider.

### **Key Benefits:**
- **Standardized Interface**: Consistent API for knowledge access across all gateways
- **Resource Management**: Organized knowledge domains and document collections  
- **Tool Integration**: Built-in tools for search, retrieval, embedding, and extraction
- **Scalable Architecture**: Independent service that can serve multiple clients

---

## 🏗️ **TidyLLM MCP Architecture**

```
┌─────────────────────────────────────────────┐
│              TidyLLM Gateway System         │
├─────────────────────────────────────────────┤
│ AIProcessingGateway                         │
│ CorporateLLMGateway        ─────────────┐   │
│ WorkflowOptimizerGateway               │   │
└────────────────────────────────────────┼───┘
                                         │
                    MCP Protocol         │
                         │               │
┌────────────────────────▼───────────────▼───┐
│           KnowledgeMCPServer               │
├─────────────────────────────────────────────┤
│ MCP Resources:                              │
│ • domains/{domain_name}                     │
│ • documents/{doc_id}                        │
│ • contexts/{context_id}                     │
│ • embeddings/{embedding_id}                 │
├─────────────────────────────────────────────┤
│ MCP Tools:                                  │
│ • search    • retrieve   • embed            │
│ • extract   • query                         │
└─────────────────────────────────────────────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
    ┌─────────▼─┐  ┌─────▼─────┐  ┌─▼────────┐
    │ S3 Sources │  │ Databases │  │ Local FS │
    └───────────┘  └───────────┘  └──────────┘
```

---

## 🚀 **Quick Start: Basic MCP Usage**

### **1. Initialize TidyLLM with Knowledge Server**

```python
import tidyllm
from tidyllm.knowledge_resource_server import S3KnowledgeSource

# Initialize the gateway registry (includes KnowledgeMCPServer)
registry = tidyllm.init_gateways()

# Get the knowledge server
knowledge_server = registry.get('knowledge_resources')
print(f"Knowledge server type: {type(knowledge_server).__name__}")
# Output: Knowledge server type: KnowledgeMCPServer
```

### **2. Register Knowledge Domains**

```python
# Register a legal documents domain from S3
knowledge_server.register_domain(
    "legal-docs",
    S3KnowledgeSource(
        bucket="company-legal-docs",
        prefix="contracts/"
    )
)

# Register a model validation domain
knowledge_server.register_domain(
    "model-validation", 
    S3KnowledgeSource(
        bucket="ml-validation-docs",
        prefix="validation-reports/"
    )
)

# Register a technical documentation domain
knowledge_server.register_domain(
    "tech-docs",
    S3KnowledgeSource(
        bucket="technical-documentation",
        prefix="api-specs/"
    )
)

print("Registered domains:", knowledge_server.registered_domains.keys())
# Output: Registered domains: dict_keys(['legal-docs', 'model-validation', 'tech-docs'])
```

---

## 🔍 **MCP Tools: Comprehensive Examples**

### **1. Semantic Search Tool**

```python
# Basic search across all domains
search_results = knowledge_server.handle_mcp_tool_call("search", {
    "query": "contract termination clauses",
    "max_results": 5,
    "similarity_threshold": 0.7
})

print("Search Results:")
for result in search_results["results"]:
    print(f"- {result['title']} (Score: {result['similarity_score']:.3f})")
    print(f"  Domain: {result['domain']}")
    print(f"  Preview: {result['content'][:100]}...")
    print()

# Domain-specific search
legal_search = knowledge_server.handle_mcp_tool_call("search", {
    "query": "liability and indemnification",
    "domain": "legal-docs",
    "max_results": 3,
    "similarity_threshold": 0.8
})

print("Legal Search Results:")
if legal_search["success"]:
    print(f"Found {legal_search['result_count']} results")
    for result in legal_search["results"]:
        print(f"- {result['title']}")
        print(f"  Content: {result['content'][:150]}...")
else:
    print(f"Search failed: {legal_search['error']}")
```

### **2. Document Retrieval Tool**

```python
# Retrieve by document ID
doc_result = knowledge_server.handle_mcp_tool_call("retrieve", {
    "document_id": "contract_2024_001"
})

if doc_result["success"]:
    document = doc_result["document"]
    print(f"Document: {document['title']}")
    print(f"Domain: {document['domain']}")
    print(f"Content length: {len(document['content'])} chars")
    print(f"Metadata: {document['metadata']}")
else:
    print(f"Retrieval failed: {doc_result['error']}")

# Retrieve by criteria
criteria_results = knowledge_server.handle_mcp_tool_call("retrieve", {
    "domain": "model-validation",
    "criteria": {
        "document_type": "validation_report",
        "model_version": "v2.1"
    }
})

if criteria_results["success"]:
    print(f"Found {criteria_results['result_count']} validation reports")
    for doc in criteria_results["documents"]:
        print(f"- {doc['title']} (Updated: {doc['last_updated']})")
```

### **3. Embedding Generation Tool**

```python
# Generate embeddings for text
embed_result = knowledge_server.handle_mcp_tool_call("embed", {
    "text": "Risk assessment criteria for machine learning models",
    "model": "sentence-transformers"
})

if embed_result["success"]:
    embedding = embed_result["embedding"]
    print(f"Generated embedding with {len(embedding)} dimensions")
    print(f"Text length: {embed_result['text_length']} characters")
    print(f"Model used: {embed_result['model']}")
    print(f"First 5 dimensions: {embedding[:5]}")
else:
    print(f"Embedding failed: {embed_result['error']}")

# Use embeddings for similarity comparison
text1 = "Model validation requirements"
text2 = "Validation criteria for AI models"

embedding1 = knowledge_server.handle_mcp_tool_call("embed", {"text": text1})
embedding2 = knowledge_server.handle_mcp_tool_call("embed", {"text": text2})

if embedding1["success"] and embedding2["success"]:
    # Calculate cosine similarity (simplified)
    import numpy as np
    
    vec1 = np.array(embedding1["embedding"])
    vec2 = np.array(embedding2["embedding"])
    
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    print(f"Similarity between texts: {similarity:.3f}")
```

### **4. Structured Data Extraction Tool**

```python
# Extract structured data from documents
extraction_result = knowledge_server.handle_mcp_tool_call("extract", {
    "document_id": "contract_2024_001",
    "extraction_type": "contract_terms"
})

if extraction_result["success"]:
    extracted_data = extraction_result["extracted_data"]
    print("Extracted Contract Terms:")
    print(f"- Parties: {extracted_data.get('parties', [])}")
    print(f"- Term Length: {extracted_data.get('term_length', 'N/A')}")
    print(f"- Payment Terms: {extracted_data.get('payment_terms', 'N/A')}")
    print(f"- Termination Clauses: {len(extracted_data.get('termination_clauses', []))}")
else:
    print(f"Extraction failed: {extraction_result['error']}")

# Extract metadata from validation reports
validation_extraction = knowledge_server.handle_mcp_tool_call("extract", {
    "document_id": "validation_report_2024_q2",
    "extraction_type": "validation_metrics"
})

if validation_extraction["success"]:
    metrics = validation_extraction["extracted_data"]
    print("Validation Metrics:")
    print(f"- Accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"- Precision: {metrics.get('precision', 'N/A')}")
    print(f"- Recall: {metrics.get('recall', 'N/A')}")
    print(f"- F1 Score: {metrics.get('f1_score', 'N/A')}")
```

### **5. Natural Language Query Tool**

```python
# Natural language query with context
query_result = knowledge_server.handle_mcp_tool_call("query", {
    "question": "What are the key validation requirements for machine learning models in production?",
    "domain": "model-validation",
    "context_length": 2000
})

if query_result["success"]:
    print("Query:", query_result["question"])
    print("Domain:", query_result["domain"])
    print("\nAnswer:")
    print(query_result["answer"])
    print(f"\nContext used ({query_result['context_length']} chars):")
    print(query_result["context"][:500] + "..." if len(query_result["context"]) > 500 else query_result["context"])
else:
    print(f"Query failed: {query_result['error']}")

# Multi-domain query
general_query = knowledge_server.handle_mcp_tool_call("query", {
    "question": "What should be included in a data processing agreement?",
    # No domain specified - searches all domains
    "context_length": 1500
})

if general_query["success"]:
    print("Cross-domain query results:")
    print(general_query["answer"])
```

---

## 🏢 **Advanced MCP Integration with Gateways**

### **1. AI Processing Gateway Integration**

```python
# Get both AI processing and knowledge services
ai_gateway = registry.get('ai_processing')
knowledge_server = registry.get('knowledge_resources')

def enhanced_ai_processing_with_context(user_query, domain=None):
    """AI processing enhanced with knowledge context."""
    
    # Step 1: Get relevant context from knowledge server
    context_result = knowledge_server.handle_mcp_tool_call("query", {
        "question": user_query,
        "domain": domain,
        "context_length": 1500
    })
    
    if not context_result["success"]:
        print(f"Warning: Could not retrieve context - {context_result['error']}")
        context = ""
    else:
        context = context_result["context"]
    
    # Step 2: Process with AI gateway using retrieved context
    enhanced_prompt = f"""
    Context from knowledge base:
    {context}
    
    User question: {user_query}
    
    Please provide a comprehensive answer using the provided context.
    """
    
    # Use AI gateway to process the enhanced prompt
    if ai_gateway:
        try:
            ai_response = ai_gateway.process(enhanced_prompt)
            return {
                "success": True,
                "answer": ai_response.data,
                "context_used": len(context) > 0,
                "context_length": len(context)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    else:
        return {"success": False, "error": "AI gateway not available"}

# Example usage
result = enhanced_ai_processing_with_context(
    "What are the compliance requirements for AI model deployment?",
    domain="model-validation"
)

if result["success"]:
    print("Enhanced AI Response:")
    print(result["answer"])
    print(f"\nContext used: {result['context_used']} ({result['context_length']} chars)")
```

### **2. Workflow Optimizer Integration**

```python
# Get workflow optimizer service
workflow_optimizer = registry.get('workflow_optimizer')

def knowledge_enhanced_workflow_analysis(workflow_description):
    """Analyze workflows with knowledge-base enhancement."""
    
    # Step 1: Search for relevant workflow patterns
    pattern_search = knowledge_server.handle_mcp_tool_call("search", {
        "query": f"workflow optimization best practices {workflow_description}",
        "domain": "tech-docs",
        "max_results": 3
    })
    
    best_practices = []
    if pattern_search["success"]:
        for result in pattern_search["results"]:
            best_practices.append({
                "practice": result["title"],
                "relevance": result["similarity_score"],
                "content": result["content"][:200]
            })
    
    # Step 2: Get compliance requirements
    compliance_search = knowledge_server.handle_mcp_tool_call("search", {
        "query": f"compliance requirements workflow {workflow_description}",
        "domain": "legal-docs",
        "max_results": 2
    })
    
    compliance_requirements = []
    if compliance_search["success"]:
        for result in compliance_search["results"]:
            compliance_requirements.append({
                "requirement": result["title"],
                "content": result["content"][:200]
            })
    
    # Step 3: Use workflow optimizer with enhanced context
    if workflow_optimizer:
        try:
            optimization_result = workflow_optimizer.analyze_workflow({
                "description": workflow_description,
                "best_practices": best_practices,
                "compliance_requirements": compliance_requirements
            })
            
            return {
                "success": True,
                "optimization": optimization_result,
                "knowledge_enhancement": {
                    "best_practices_found": len(best_practices),
                    "compliance_requirements_found": len(compliance_requirements)
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    else:
        return {"success": False, "error": "Workflow optimizer not available"}

# Example usage
workflow_result = knowledge_enhanced_workflow_analysis(
    "data processing pipeline for customer analytics"
)

if workflow_result["success"]:
    print("Knowledge-Enhanced Workflow Analysis:")
    print(f"Best practices found: {workflow_result['knowledge_enhancement']['best_practices_found']}")
    print(f"Compliance requirements found: {workflow_result['knowledge_enhancement']['compliance_requirements_found']}")
```

---

## 📊 **MCP Server Management and Monitoring**

### **1. Server Status and Health Monitoring**

```python
# Get server status
status = knowledge_server.get_server_status()
print("Knowledge Server Status:")
print(f"- Name: {status['server_name']}")
print(f"- Version: {status['version']}")
print(f"- Status: {status['status']}")
print(f"- Registered Domains: {status['registered_domains']}")
print(f"- Total Resources: {status['total_resources']}")
print(f"- Last Updated: {status['last_updated']}")

# Get MCP capabilities
capabilities = knowledge_server.get_mcp_capabilities()
print("\nMCP Capabilities:")
print(f"- Server: {capabilities['server']['name']} v{capabilities['server']['version']}")
print(f"- Resources: {capabilities['resources']['resource_count']} domains")
print(f"- Search Support: {capabilities['resources']['supports_search']}")
print(f"- Tools Available: {len(capabilities['tools'])}")

for tool in capabilities['tools']:
    print(f"  - {tool['name']}: {tool['description']}")
```

### **2. Resource Information and Statistics**

```python
# List all available resources
resources = knowledge_server.list_resources()
print("Available Resources:")
for resource in resources:
    print(f"- URI: {resource['uri']}")
    print(f"  Type: {resource['type']}")
    print(f"  Name: {resource['name']}")
    print(f"  Capabilities: {', '.join(resource['capabilities'])}")
    if 'statistics' in resource:
        stats = resource['statistics']
        print(f"  Documents: {stats.get('document_count', 'N/A')}")
        print(f"  Last Indexed: {stats.get('last_indexed', 'N/A')}")
    print()

# Get specific resource information
legal_info = knowledge_server.get_resource_info("domains/legal-docs")
if legal_info:
    print("Legal Documents Domain Info:")
    print(f"- Name: {legal_info['name']}")
    print(f"- Type: {legal_info['type']}")
    print(f"- Source: {legal_info['source']}")
    print(f"- Capabilities: {', '.join(legal_info['capabilities'])}")
    print(f"- Last Updated: {legal_info['last_updated']}")
```

### **3. Gateway Registry Integration**

```python
# Check knowledge server availability in registry
registry_status = registry.health_check()
print("Gateway Registry Health Check:")

for service_name, health_info in registry_status["services"].items():
    print(f"- {service_name}: {health_info['status']}")
    if health_info.get("healthy"):
        print(f"  ✅ Healthy")
        if "details" in health_info:
            print(f"  Details: {health_info['details']}")
    else:
        print(f"  ❌ Unhealthy: {health_info.get('error', 'Unknown error')}")

print(f"\nOverall System Health: {'✅ Healthy' if registry_status['overall_healthy'] else '❌ Unhealthy'}")
print(f"Services: {registry_status['healthy_services']}/{registry_status['total_services']} healthy")
```

---

## 🛠️ **Custom Knowledge Sources**

### **1. Creating Custom Knowledge Sources**

```python
from tidyllm.knowledge_resource_server.sources import LocalKnowledgeSource, DatabaseKnowledgeSource

# Register local file system source
knowledge_server.register_domain(
    "local-docs",
    LocalKnowledgeSource(
        base_path="/path/to/local/documents",
        file_patterns=["*.md", "*.txt", "*.pdf"]
    )
)

# Register database source
knowledge_server.register_domain(
    "database-knowledge",
    DatabaseKnowledgeSource(
        connection_string="postgresql://user:pass@localhost/knowledge_db",
        table_name="documents",
        content_column="content",
        metadata_columns=["title", "author", "created_at"]
    )
)

print("Custom sources registered successfully")
```

### **2. Implementing Custom Knowledge Source**

```python
from tidyllm.knowledge_resource_server.sources import KnowledgeSource
from typing import List, Dict, Any

class APIKnowledgeSource(KnowledgeSource):
    """Custom knowledge source that fetches from external API."""
    
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from API."""
        import requests
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(f"{self.api_endpoint}/documents", headers=headers)
        
        if response.status_code == 200:
            api_docs = response.json()
            
            # Convert API format to knowledge format
            documents = []
            for doc in api_docs:
                documents.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "metadata": {
                        "source": "api",
                        "api_endpoint": self.api_endpoint,
                        "created_at": doc.get("created_at"),
                        "tags": doc.get("tags", [])
                    }
                })
            
            return documents
        else:
            raise Exception(f"API request failed: {response.status_code}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get source information."""
        return {
            "type": "api",
            "endpoint": self.api_endpoint,
            "authenticated": bool(self.api_key)
        }

# Register custom API source
api_source = APIKnowledgeSource(
    api_endpoint="https://api.company.com/knowledge",
    api_key="your-api-key-here"
)

knowledge_server.register_domain("api-docs", api_source)
print("API knowledge source registered")
```

---

## 🔒 **Security and Best Practices**

### **1. Secure Configuration**

```python
from tidyllm.knowledge_resource_server.mcp_server import MCPServerConfig

# Secure MCP server configuration
secure_config = MCPServerConfig(
    server_name="company-knowledge-server",
    server_version="1.0.0",
    
    # Security settings
    require_authentication=True,
    allowed_clients=["ai-gateway", "workflow-optimizer", "admin-console"],
    
    # Performance settings
    max_search_results=10,
    default_similarity_threshold=0.75,
    max_concurrent_requests=5,
    request_timeout=30.0,
    
    # Caching settings
    enable_caching=True,
    cache_ttl=1800  # 30 minutes
)

# Initialize with secure configuration
secure_knowledge_server = KnowledgeMCPServer(config=secure_config)
```

### **2. Error Handling and Logging**

```python
import logging

# Set up logging for knowledge server
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_client")

def safe_mcp_call(knowledge_server, tool_name, parameters):
    """Safely call MCP tools with comprehensive error handling."""
    try:
        result = knowledge_server.handle_mcp_tool_call(tool_name, parameters)
        
        if result["success"]:
            logger.info(f"MCP tool '{tool_name}' executed successfully")
            return result
        else:
            logger.error(f"MCP tool '{tool_name}' failed: {result['error']}")
            return {"success": False, "error": result["error"]}
            
    except Exception as e:
        logger.exception(f"Exception calling MCP tool '{tool_name}': {str(e)}")
        return {"success": False, "error": f"Exception: {str(e)}"}

# Example safe usage
result = safe_mcp_call(knowledge_server, "search", {
    "query": "data privacy compliance",
    "domain": "legal-docs",
    "max_results": 5
})

if result["success"]:
    print(f"Found {len(result['results'])} results")
else:
    print(f"Search failed: {result['error']}")
```

### **3. Performance Optimization**

```python
# Batch processing for multiple queries
def batch_knowledge_queries(knowledge_server, queries, domain=None):
    """Process multiple queries efficiently."""
    results = []
    
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
        
        result = knowledge_server.handle_mcp_tool_call("search", {
            "query": query,
            "domain": domain,
            "max_results": 3,
            "similarity_threshold": 0.7
        })
        
        results.append({
            "query": query,
            "result": result,
            "success": result.get("success", False)
        })
    
    return results

# Example batch processing
queries = [
    "contract termination procedures",
    "liability and indemnification clauses",
    "data processing agreements",
    "service level agreements"
]

batch_results = batch_knowledge_queries(knowledge_server, queries, domain="legal-docs")

print("Batch Processing Results:")
for i, batch_result in enumerate(batch_results):
    status = "✅" if batch_result["success"] else "❌"
    print(f"{i+1}. {status} {batch_result['query']}")
    if batch_result["success"]:
        result_count = batch_result["result"].get("result_count", 0)
        print(f"   Found {result_count} results")
```

---

## 📚 **Complete Working Example**

```python
#!/usr/bin/env python3
"""
Complete MCP Knowledge Server Example
====================================

This example demonstrates a full knowledge management workflow using
TidyLLM's KnowledgeMCPServer with multiple domains and tools.
"""

import tidyllm
from tidyllm.knowledge_resource_server import S3KnowledgeSource

def main():
    print("🚀 Initializing TidyLLM Knowledge Management System...")
    
    # Initialize TidyLLM with all gateways
    registry = tidyllm.init_gateways()
    knowledge_server = registry.get('knowledge_resources')
    
    if not knowledge_server:
        print("❌ Knowledge server not available")
        return
    
    print("✅ Knowledge server initialized")
    
    # Register knowledge domains
    print("\n📚 Registering knowledge domains...")
    
    domains = [
        ("legal-docs", "company-legal", "contracts/"),
        ("model-validation", "ml-docs", "validation/"),
        ("technical-specs", "tech-docs", "specifications/"),
        ("policies", "company-policies", "hr-policies/")
    ]
    
    for domain_name, bucket, prefix in domains:
        try:
            knowledge_server.register_domain(
                domain_name,
                S3KnowledgeSource(bucket=bucket, prefix=prefix)
            )
            print(f"✅ Registered domain: {domain_name}")
        except Exception as e:
            print(f"❌ Failed to register {domain_name}: {e}")
    
    # Demonstrate all MCP tools
    print("\n🔍 Demonstrating MCP Tools...")
    
    # 1. Search demonstration
    print("\n1. Semantic Search:")
    search_queries = [
        ("What are contract termination procedures?", "legal-docs"),
        ("Model validation best practices", "model-validation"),
        ("API authentication requirements", "technical-specs")
    ]
    
    for query, domain in search_queries:
        result = knowledge_server.handle_mcp_tool_call("search", {
            "query": query,
            "domain": domain,
            "max_results": 2
        })
        
        print(f"\n   Query: {query}")
        print(f"   Domain: {domain}")
        if result["success"]:
            print(f"   Results: {result['result_count']} found")
            for r in result["results"][:1]:  # Show first result only
                print(f"   - {r['title']} (Score: {r.get('similarity_score', 'N/A')})")
        else:
            print(f"   Error: {result['error']}")
    
    # 2. Natural Language Query
    print("\n2. Natural Language Query:")
    query_result = knowledge_server.handle_mcp_tool_call("query", {
        "question": "What are the key compliance requirements for data processing?",
        "context_length": 1000
    })
    
    if query_result["success"]:
        print(f"   Question: {query_result['question']}")
        print(f"   Answer: {query_result['answer'][:200]}...")
        print(f"   Context Length: {query_result['context_length']} chars")
    else:
        print(f"   Query failed: {query_result['error']}")
    
    # 3. Embedding generation
    print("\n3. Embedding Generation:")
    embed_result = knowledge_server.handle_mcp_tool_call("embed", {
        "text": "Machine learning model validation requirements"
    })
    
    if embed_result["success"]:
        print(f"   Text: {embed_result['text_length']} characters")
        print(f"   Embedding: {len(embed_result['embedding'])} dimensions")
        print(f"   First 3 values: {embed_result['embedding'][:3]}")
    else:
        print(f"   Embedding failed: {embed_result['error']}")
    
    # 4. Server status
    print("\n📊 Server Status:")
    status = knowledge_server.get_server_status()
    print(f"   Server: {status['server_name']} v{status['version']}")
    print(f"   Status: {status['status']}")
    print(f"   Domains: {status['registered_domains']}")
    print(f"   Resources: {status['total_resources']}")
    
    # 5. Gateway integration example
    print("\n🤝 Gateway Integration Example:")
    ai_gateway = registry.get('ai_processing')
    
    if ai_gateway:
        # Get context from knowledge server
        context_result = knowledge_server.handle_mcp_tool_call("search", {
            "query": "compliance requirements",
            "max_results": 1
        })
        
        if context_result["success"] and context_result["results"]:
            context = context_result["results"][0]["content"][:500]
            
            print(f"   Context retrieved: {len(context)} characters")
            print("   ✅ Ready for AI processing with knowledge context")
        else:
            print("   ❌ No context available")
    else:
        print("   ❌ AI gateway not available")
    
    print("\n🎉 MCP Knowledge Server demonstration complete!")

if __name__ == "__main__":
    main()
```

---

## 📝 **Summary**

The **KnowledgeMCPServer** in TidyLLM provides a powerful, standardized way to manage and access knowledge resources. Key takeaways:

### **✅ Key Features:**
- **5 MCP Tools**: search, retrieve, embed, extract, query
- **Multiple Knowledge Sources**: S3, Local FS, Databases, Custom APIs  
- **Domain Organization**: Separate knowledge by domain (legal, technical, etc.)
- **Gateway Integration**: Seamless integration with AI and workflow gateways
- **Enterprise Ready**: Security, monitoring, and performance optimization

### **🚀 Best Practices:**
1. **Organize by Domain**: Separate different types of knowledge into distinct domains
2. **Use Appropriate Tools**: Match the right MCP tool to your use case
3. **Handle Errors Gracefully**: Always check `result["success"]` before processing
4. **Monitor Performance**: Use server status and health checks regularly
5. **Secure Configuration**: Enable authentication and set appropriate limits

### **📖 Next Steps:**
- Implement custom knowledge sources for your specific data
- Integrate with your AI processing workflows
- Set up monitoring and alerting for knowledge server health
- Explore advanced features like custom extractors and analyzers

The **KnowledgeMCPServer** is now fully integrated and ready to power intelligent knowledge management across your TidyLLM system! 🧠✨
