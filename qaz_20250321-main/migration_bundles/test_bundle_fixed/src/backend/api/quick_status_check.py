#!/usr/bin/env python3
"""
Quick Status Check for VectorQA Sage APIs
Fast overview of what's working
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_status_check():
    """Quick check of all API providers."""
    print("⚡ Quick API Status Check")
    print("=" * 40)
    
    from backend.core.llm_manager import LLMManager
    from backend.config.credential_manager import credential_manager
    
    # Check credentials first
    print("\n🔑 Credentials Status:")
    providers = {
        "Cohere": credential_manager.get_cohere_config(),
        "Gemini": credential_manager.get_google_config(),
        "OpenAI": credential_manager.get_openai_config(),
        "HuggingFace": credential_manager.get_huggingface_config(),
        "Anthropic": credential_manager.get_anthropic_config()
    }
    
    for name, config in providers.items():
        status = "✅" if config.get("api_key") else "❌"
        print(f"  {status} {name}")
    
    # Quick API tests
    print("\n🤖 API Status:")
    api_providers = ["cohere", "google", "openai", "huggingface", "anthropic"]
    
    working = []
    for provider in api_providers:
        try:
            llm = LLMManager(provider=provider)
            response = llm.generate_response("Hi", max_tokens=3)
            
            if response.startswith("Error") or response.startswith("Rate limit"):
                print(f"  ❌ {provider.upper()}")
            else:
                print(f"  ✅ {provider.upper()}")
                working.append(provider)
                
        except Exception as e:
            print(f"  ❌ {provider.upper()}")
    
    # Summary
    print(f"\n📊 Summary: {len(working)}/{len(api_providers)} providers working")
    
    if working:
        print(f"✅ Working: {', '.join(working).upper()}")
        print(f"🎯 Primary: {working[0].upper()}")
        if len(working) > 1:
            print(f"🔄 Fallbacks: {', '.join(working[1:]).upper()}")
    
    return working

if __name__ == "__main__":
    quick_status_check()
