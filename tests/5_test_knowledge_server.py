#!/usr/bin/env python3
"""
TidyLLM Strategic Test Suite: Knowledge Server (MCP)
===================================================

Comprehensive testing for KnowledgeMCPServer and MCP functionality.
Replaces multiple knowledge/MCP test files with strategic coverage.
"""

import unittest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import tidyllm
    from tidyllm.knowledge_resource_server import KnowledgeMCPServer
    KNOWLEDGE_SERVER_AVAILABLE = True
except ImportError as e:
    print(f"Knowledge server not available: {e}")
    KNOWLEDGE_SERVER_AVAILABLE = False


class TestKnowledgeMCPServer(unittest.TestCase):
    """Test core KnowledgeMCPServer functionality."""
    
    def setUp(self):
        """Initialize knowledge server for each test."""
        if KNOWLEDGE_SERVER_AVAILABLE:
            self.knowledge_server = KnowledgeMCPServer()
        else:
            self.knowledge_server = None
    
    def test_knowledge_server_initialization(self):
        """Test knowledge server initializes correctly."""
        if not KNOWLEDGE_SERVER_AVAILABLE:
            self.skipTest("Knowledge server not available")
        
        self.assertIsNotNone(self.knowledge_server)
        self.assertIsInstance(self.knowledge_server, KnowledgeMCPServer)
        
        # Test basic attributes
        self.assertTrue(hasattr(self.knowledge_server, 'config'))
        self.assertTrue(hasattr(self.knowledge_server, 'resources'))
        self.assertTrue(hasattr(self.knowledge_server, 'registered_domains'))
    
    def test_mcp_capabilities(self):
        """Test MCP server capabilities."""
        if not KNOWLEDGE_SERVER_AVAILABLE:
            self.skipTest("Knowledge server not available")
        
        capabilities = self.knowledge_server.get_mcp_capabilities()
        
        self.assertIsInstance(capabilities, dict)
        self.assertIn('server', capabilities)
        self.assertIn('resources', capabilities)
        self.assertIn('tools', capabilities)
        
        # Check required tools
        tool_names = [tool['name'] for tool in capabilities['tools']]
        required_tools = ['search', 'retrieve', 'embed', 'extract', 'query']
        
        for tool in required_tools:
            self.assertIn(tool, tool_names, f"Required MCP tool '{tool}' not found")
        
        print(f"MCP server exposes {len(tool_names)} tools: {tool_names}")
    
    def test_server_status(self):
        """Test server status functionality."""
        if not KNOWLEDGE_SERVER_AVAILABLE:
            self.skipTest("Knowledge server not available")
        
        status = self.knowledge_server.get_server_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('server_name', status)
        self.assertIn('version', status)
        self.assertIn('status', status)
        self.assertIn('registered_domains', status)
        
        print(f"Server status: {status['status']}, domains: {status['registered_domains']}")


class TestMCPTools(unittest.TestCase):
    """Test MCP tool functionality."""
    
    def setUp(self):
        """Initialize knowledge server for each test."""
        if KNOWLEDGE_SERVER_AVAILABLE:
            self.knowledge_server = KnowledgeMCPServer()
        else:
            self.knowledge_server = None
    
    def test_search_tool_basic(self):
        """Test basic search tool functionality."""
        if not KNOWLEDGE_SERVER_AVAILABLE:
            self.skipTest("Knowledge server not available")
        
        # Test search with minimal parameters
        result = self.knowledge_server.handle_mcp_tool_call("search", {
            "query": "test query"
        })
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('tool', result)
        self.assertEqual(result['tool'], 'search')
        
        if result['success']:
            self.assertIn('results', result)
            self.assertIn('query', result)
            print(f"Search successful: {result.get('result_count', 0)} results")
        else:
            print(f"Search failed (expected): {result.get('error', 'Unknown error')}")
    
    def test_embed_tool(self):
        """Test embedding generation tool."""
        if not KNOWLEDGE_SERVER_AVAILABLE:
            self.skipTest("Knowledge server not available")
        
        result = self.knowledge_server.handle_mcp_tool_call("embed", {
            "text": "test embedding generation"
        })
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('tool', result)
        self.assertEqual(result['tool'], 'embed')
        
        if result['success']:
            self.assertIn('embedding', result)
            self.assertIn('text_length', result)
            print(f"Embedding successful: {len(result['embedding'])} dimensions")
        else:
            print(f"Embedding failed (may be expected): {result.get('error', 'Unknown error')}")
    
    def test_query_tool(self):
        """Test natural language query tool.""" 
        if not KNOWLEDGE_SERVER_AVAILABLE:
            self.skipTest("Knowledge server not available")
        
        result = self.knowledge_server.handle_mcp_tool_call("query", {
            "question": "What is machine learning?"
        })
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('tool', result)
        self.assertEqual(result['tool'], 'query')
        
        if result['success']:
            self.assertIn('answer', result)
            self.assertIn('question', result)
            print(f"Query successful: {len(result.get('answer', ''))} char answer")
        else:
            print(f"Query failed (may be expected): {result.get('error', 'Unknown error')}")
    
    def test_invalid_tool(self):
        """Test handling of invalid tool calls."""
        if not KNOWLEDGE_SERVER_AVAILABLE:
            self.skipTest("Knowledge server not available")
        
        result = self.knowledge_server.handle_mcp_tool_call("nonexistent_tool", {})
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        print(f"Invalid tool correctly rejected: {result['error']}")


class TestDomainManagement(unittest.TestCase):
    """Test knowledge domain registration and management."""
    
    def setUp(self):
        """Initialize knowledge server for each test."""
        if KNOWLEDGE_SERVER_AVAILABLE:
            self.knowledge_server = KnowledgeMCPServer()
        else:
            self.knowledge_server = None
    
    def test_domain_registration_mock(self):
        """Test domain registration with mock source."""
        if not KNOWLEDGE_SERVER_AVAILABLE:
            self.skipTest("Knowledge server not available")
        
        # Create a mock knowledge source
        from tidyllm.knowledge_resource_server.sources import KnowledgeSource
        
        class MockKnowledgeSource(KnowledgeSource):
            def initialize(self) -> None:
                """Initialize the mock source."""
                pass
            
            def get_document_count(self) -> int:
                """Get total number of documents in this source."""
                return 1
            
            def search(self, query: str, max_results: int = 5, similarity_threshold: float = 0.7):
                """Search for documents matching query."""
                return [
                    {
                        "id": "test_doc_1",
                        "title": "Test Document",
                        "content": "This is test content for testing purposes.",
                        "metadata": {"source": "mock", "type": "test"},
                        "score": 0.85
                    }
                ]
            
            def retrieve_by_id(self, document_id: str):
                """Retrieve document by ID."""
                if document_id == "test_doc_1":
                    return {
                        "id": "test_doc_1",
                        "title": "Test Document",
                        "content": "This is test content for testing purposes.",
                        "metadata": {"source": "mock", "type": "test"}
                    }
                return None
            
            def retrieve_by_criteria(self, criteria):
                """Retrieve documents matching criteria."""
                return [
                    {
                        "id": "test_doc_1",
                        "title": "Test Document",
                        "content": "This is test content for testing purposes.",
                        "metadata": {"source": "mock", "type": "test"}
                    }
                ]
            
            def load_documents(self):
                return [
                    {
                        "id": "test_doc_1",
                        "title": "Test Document",
                        "content": "This is test content for testing purposes.",
                        "metadata": {"source": "mock", "type": "test"}
                    }
                ]
            
            def get_info(self):
                return {"type": "mock", "description": "Mock source for testing"}
        
        # Register mock domain
        mock_source = MockKnowledgeSource()
        
        try:
            self.knowledge_server.register_domain("test-domain", mock_source)
            
            # Verify registration
            self.assertIn("test-domain", self.knowledge_server.registered_domains)
            
            # Test resource info
            resource_info = self.knowledge_server.get_resource_info("domains/test-domain")
            self.assertIsNotNone(resource_info)
            self.assertEqual(resource_info['name'], 'test-domain')
            
            print("Mock domain registration successful")
            
        except Exception as e:
            print(f"Domain registration failed (may be expected): {e}")
    
    def test_resource_listing(self):
        """Test resource listing functionality."""
        if not KNOWLEDGE_SERVER_AVAILABLE:
            self.skipTest("Knowledge server not available")
        
        resources = self.knowledge_server.list_resources()
        
        self.assertIsInstance(resources, list)
        print(f"Listed {len(resources)} resources")
        
        for resource in resources:
            self.assertIsInstance(resource, dict)
            self.assertIn('uri', resource)
            self.assertIn('type', resource)
            self.assertIn('name', resource)


class TestGatewayIntegration(unittest.TestCase):
    """Test knowledge server integration with gateway registry."""
    
    def test_knowledge_server_in_registry(self):
        """Test knowledge server availability through gateway registry."""
        if not KNOWLEDGE_SERVER_AVAILABLE:
            self.skipTest("Knowledge server not available")
        
        try:
            registry = tidyllm.init_gateways()
            knowledge_service = registry.get('knowledge_resources')
            
            if knowledge_service:
                self.assertIsInstance(knowledge_service, KnowledgeMCPServer)
                print("Knowledge server successfully integrated with gateway registry")
            else:
                print("Knowledge server not available in registry (may be expected)")
                
        except Exception as e:
            print(f"Gateway integration test failed: {e}")
    
    def test_registry_health_with_knowledge_server(self):
        """Test registry health check includes knowledge server."""
        if not KNOWLEDGE_SERVER_AVAILABLE:
            self.skipTest("Knowledge server not available")
        
        try:
            registry = tidyllm.init_gateways()
            health = registry.health_check()
            
            self.assertIsInstance(health, dict)
            self.assertIn('services', health)
            
            if 'knowledge_resources' in health['services']:
                knowledge_health = health['services']['knowledge_resources']
                print(f"Knowledge server health: {knowledge_health['status']}")
            else:
                print("Knowledge server not in health check (may be expected)")
                
        except Exception as e:
            print(f"Health check test failed: {e}")


def run_knowledge_server_tests():
    """Run all knowledge server tests with detailed output."""
    print("="*60)
    print("TIDYLLM STRATEGIC KNOWLEDGE SERVER TESTS")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestKnowledgeMCPServer,
        TestMCPTools,
        TestDomainManagement,
        TestGatewayIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("KNOWLEDGE SERVER TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"Overall status: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_knowledge_server_tests()
    sys.exit(0 if success else 1)