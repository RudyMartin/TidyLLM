#!/usr/bin/env python3
"""
API Status Dashboard for VectorQA Sage
Shows the status of all LLM providers and their capabilities
"""

import sys
import os
import time
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ..core.llm_manager import LLMManager
from ..config.credential_manager import credential_manager

class APIStatusDashboard:
    """Dashboard for monitoring API status across all providers."""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
    
    def test_provider(self, provider_name: str, test_name: str = "Basic Response"):
        """Test a specific provider."""
        try:
            print(f"🔍 Testing {provider_name}...", end=" ")
            
            llm = LLMManager(provider=provider_name)
            response = llm.generate_response(
                "Say 'Hello' in one word.",
                max_tokens=5
            )
            
            # Check if response is successful
            if response.startswith("Error") or response.startswith("Rate limit"):
                status = "❌ Failed"
                details = response
            else:
                status = "✅ Working"
                details = response[:50] + "..." if len(response) > 50 else response
            
            self.results[provider_name] = {
                "status": status,
                "details": details,
                "test_name": test_name,
                "timestamp": datetime.now()
            }
            
            print(status)
            return status == "✅ Working"
            
        except Exception as e:
            status = "❌ Error"
            details = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
            
            self.results[provider_name] = {
                "status": status,
                "details": details,
                "test_name": test_name,
                "timestamp": datetime.now()
            }
            
            print(status)
            return False
    
    def test_all_providers(self):
        """Test all configured providers."""
        print("🚀 Starting API Status Dashboard")
        print("=" * 60)
        
        providers = [
            ("cohere", "Basic Response"),
            ("google", "Basic Response"), 
            ("openai", "Basic Response"),
            ("huggingface", "Basic Response"),
            ("anthropic", "Basic Response")
        ]
        
        working_count = 0
        total_count = len(providers)
        
        for provider, test_name in providers:
            if self.test_provider(provider, test_name):
                working_count += 1
        
        return working_count, total_count
    
    def test_advanced_capabilities(self):
        """Test advanced capabilities for working providers."""
        print("\n🔬 Testing Advanced Capabilities")
        print("=" * 40)
        
        advanced_tests = []
        
        # Test Cohere
        if self.results.get("cohere", {}).get("status") == "✅ Working":
            print("\n🔵 Testing Cohere Advanced Features:")
            try:
                cohere_llm = LLMManager(provider="cohere")
                
                # Test QA
                qa_response = cohere_llm.answer_question(
                    "What is the capital of France?",
                    "Paris is the capital of France."
                )
                advanced_tests.append(("Cohere QA", "✅ Working", qa_response[:50]))
                
                # Test validation
                from backend.core.validator import ValidationModule
                validator = ValidationModule()
                validator.llm_manager = cohere_llm
                val_result = validator.forward("test", "test", "test")
                advanced_tests.append(("Cohere Validation", "✅ Working", val_result))
                
            except Exception as e:
                advanced_tests.append(("Cohere Advanced", "❌ Error", str(e)[:50]))
        
        # Test Gemini
        if self.results.get("google", {}).get("status") == "✅ Working":
            print("\n🟡 Testing Gemini Advanced Features:")
            try:
                gemini_llm = LLMManager(provider="google")
                
                # Test QA
                qa_response = gemini_llm.answer_question(
                    "What is AI?",
                    "Artificial Intelligence is computer systems that can perform tasks."
                )
                advanced_tests.append(("Gemini QA", "✅ Working", qa_response[:50]))
                
                # Test summarization
                summary = gemini_llm.summarize_document(
                    "VectorQA Sage is an AI-powered document analysis tool."
                )
                advanced_tests.append(("Gemini Summary", "✅ Working", summary[:50]))
                
            except Exception as e:
                advanced_tests.append(("Gemini Advanced", "❌ Error", str(e)[:50]))
        
        return advanced_tests
    
    def check_credentials(self):
        """Check credential status for all providers."""
        print("\n🔐 Checking Credentials")
        print("=" * 30)
        
        credentials = {}
        
        # Check each provider's credentials
        providers = {
            "OpenAI": credential_manager.get_openai_config(),
            "Cohere": credential_manager.get_cohere_config(),
            "Google": credential_manager.get_google_config(),
            "Hugging Face": credential_manager.get_huggingface_config(),
            "Anthropic": credential_manager.get_anthropic_config()
        }
        
        for provider, config in providers.items():
            if config.get("api_key"):
                credentials[provider] = "✅ Configured"
            else:
                credentials[provider] = "❌ Missing"
        
        return credentials
    
    def generate_report(self):
        """Generate a comprehensive status report."""
        print("\n📊 API Status Dashboard Report")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Duration: {(datetime.now() - self.start_time).total_seconds():.1f} seconds")
        
        # Provider Status Summary
        print("\n🤖 Provider Status Summary")
        print("-" * 30)
        
        working_count = 0
        for provider, result in self.results.items():
            status = result["status"]
            if status == "✅ Working":
                working_count += 1
            print(f"{provider.upper():15} {status}")
        
        print(f"\n📈 Success Rate: {working_count}/{len(self.results)} ({working_count/len(self.results)*100:.1f}%)")
        
        # Credentials Status
        credentials = self.check_credentials()
        print("\n🔑 Credentials Status")
        print("-" * 20)
        for provider, status in credentials.items():
            print(f"{provider:15} {status}")
        
        # Advanced Capabilities
        advanced_tests = self.test_advanced_capabilities()
        if advanced_tests:
            print("\n🔬 Advanced Capabilities")
            print("-" * 25)
            for test_name, status, details in advanced_tests:
                print(f"{test_name:20} {status}")
        
        # Recommendations
        print("\n💡 Recommendations")
        print("-" * 15)
        
        if working_count >= 2:
            print("✅ Excellent! You have multiple working providers.")
            print("✅ Your system is robust and production-ready.")
        elif working_count == 1:
            print("⚠️  You have one working provider. Consider adding more for redundancy.")
        else:
            print("❌ No providers are working. Check your credentials and network.")
        
        # Working Provider Details
        working_providers = [p for p, r in self.results.items() if r["status"] == "✅ Working"]
        if working_providers:
            print(f"\n🎯 Recommended Primary Provider: {working_providers[0].upper()}")
            print(f"🔄 Fallback Providers: {', '.join(working_providers[1:]).upper() if len(working_providers) > 1 else 'None'}")
        
        return {
            "working_count": working_count,
            "total_count": len(self.results),
            "working_providers": working_providers,
            "credentials": credentials,
            "results": self.results
        }

def main():
    """Run the API status dashboard."""
    dashboard = APIStatusDashboard()
    
    # Test all providers
    working_count, total_count = dashboard.test_all_providers()
    
    # Generate comprehensive report
    report = dashboard.generate_report()
    
    print("\n🎉 Dashboard Complete!")
    print("=" * 60)
    
    return report

if __name__ == "__main__":
    main()
