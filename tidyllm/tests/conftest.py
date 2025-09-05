"""
Pytest configuration and fixtures for TidyLLM testing
"""

import pytest
import sys
import os

# Add the tidyllm package to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def tidyllm_module():
    """Fixture to provide tidyllm module for tests"""
    try:
        import tidyllm
        return tidyllm
    except ImportError as e:
        pytest.skip(f"tidyllm module not available: {e}")


@pytest.fixture  
def sample_data():
    """Fixture providing sample data for testing"""
    return {
        'numbers': [1, 2, 3, 4, 5],
        'text': "This is a sample text for testing",
        'sentences': [
            "First test sentence.",
            "Second test sentence.",  
            "Third test sentence."
        ],
        'document_content': "Sample document content for processing tests."
    }


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests (may be slow or require dependencies)"
    )
    config.addinivalue_line(
        "markers",
        "dependencies: marks tests that require specific external dependencies"
    )
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle dependency-based skipping"""
    
    # Try to import tidyllm to check availability
    try:
        import tidyllm
        tidyllm_available = True
    except Exception:
        tidyllm_available = False
        
    for item in items:
        # Skip integration tests if tidyllm is not available
        if "integration" in item.keywords and not tidyllm_available:
            item.add_marker(pytest.mark.skip(reason="tidyllm not available"))
            
        # Mark tests that require specific dependencies
        if "dependencies" in item.keywords:
            item.add_marker(pytest.mark.dependencies)