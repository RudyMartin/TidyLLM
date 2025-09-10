#!/usr/bin/env python3
"""
Simple TidyLLM API Test - Direct Implementation
==============================================
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_direct_api():
    """Test TidyLLM API with direct implementation."""
    
    print("="*50)
    print("DIRECT TIDYLLM API TEST")
    print("="*50)
    
    # Try direct gateway creation
    try:
        from tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway, CorporateLLMConfig, LLMRequest
        
        print("\n1. Creating Corporate LLM Gateway:")
        print("-" * 30)
        
        # Create config without problematic parameters
        config = CorporateLLMConfig(
            budget_limit_daily_usd=100.0,
            log_all_requests=True,
            require_audit_reason=True
        )
        
        gateway = CorporateLLMGateway(config=config)
        print("[SUCCESS] Corporate LLM Gateway created")
        
        # Test LLM request
        print("\n2. Testing LLM Request:")
        print("-" * 30)
        
        request = LLMRequest(
            prompt="Hello! What is TidyLLM?",
            model="claude-3-sonnet",
            audit_reason="api_test",
            user_id="test_user"
        )
        
        response = gateway.process_llm_request(request)
        print(f"Response Status: {response.status}")
        print(f"Response Content: {response.content[:100]}...")
        
    except Exception as e:
        print(f"[ERROR] Gateway test failed: {e}")
        
    # Test basic API without gateways
    print("\n3. Testing Basic API:")
    print("-" * 30)
    
    try:
        import tidyllm
        
        # Test basic functions
        models = tidyllm.list_models()
        print(f"Available models: {len(models)}")
        
        status = tidyllm.status()
        print(f"API Status: {status['architecture']}")
        print(f"AWS Integration: {status['has_aws_key']}")
        
        print("[SUCCESS] Basic API working")
        
    except Exception as e:
        print(f"[ERROR] Basic API failed: {e}")
    
    print("\n" + "="*50)
    print("DIRECT API TEST COMPLETE")
    print("="*50)

if __name__ == "__main__":
    test_direct_api()