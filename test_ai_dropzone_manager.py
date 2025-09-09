#!/usr/bin/env python3
"""
Quick Test of AI Dropzone Manager
=================================

Test the AI Dropzone Manager we just built to see if it actually works.
"""

import asyncio
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent / "tidyllm"))

async def test_ai_dropzone_manager():
    """Test the AI Dropzone Manager functionality."""
    
    print("[TEST] Testing AI Dropzone Manager")
    print("=" * 50)
    
    try:
        # Import the AI Dropzone Manager
        from tidyllm.infrastructure.workers.ai_dropzone_manager import AIDropzoneManager, AIManagerTask
        
        print("[OK] Successfully imported AI Dropzone Manager")
        
        # Initialize the manager
        print("\n[INIT] Initializing AI Dropzone Manager...")
        manager = AIDropzoneManager()
        
        # Test initialization
        await manager.initialize()
        print("[OK] AI Dropzone Manager initialized successfully")
        
        # Get manager status
        print("\n[STATUS] Getting manager status...")
        status = await manager.get_manager_status()
        
        print("Manager Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Test document intelligence (without actual file)
        print("\n[INTEL] Testing document intelligence...")
        sample_content = """
        FINANCIAL STATEMENT
        
        Revenue: $1,250,000
        Net Income: $180,000  
        Total Assets: $2,500,000
        Total Liabilities: $1,200,000
        
        This quarterly financial report shows strong performance
        with revenue growth of 15% year-over-year.
        """
        
        intelligence = await manager._analyze_document_intelligence("sample_financial.pdf")
        
        print("Document Intelligence Results:")
        print(f"  Document Type: {intelligence.detected_type}")
        print(f"  Complexity: {intelligence.complexity.name}")
        print(f"  Confidence: {intelligence.confidence_score:.1%}")
        print(f"  Processing Strategy: {intelligence.processing_strategy.value}")
        print(f"  Recommended Templates: {intelligence.recommended_templates}")
        print(f"  Estimated Time: {intelligence.estimated_processing_time} minutes")
        
        print("\n✅ AI Dropzone Manager test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   Make sure you're running from the correct directory")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

async def test_flow_integration_manager():
    """Test the Flow Integration Manager functionality."""
    
    print("\n🧪 Testing Flow Integration Manager")
    print("=" * 50)
    
    try:
        from tidyllm.infrastructure.workers.flow_integration_manager import FlowIntegrationManager
        
        print("✅ Successfully imported Flow Integration Manager")
        
        # Initialize the manager
        print("\n🔧 Initializing Flow Integration Manager...")
        flow_manager = FlowIntegrationManager()
        
        await flow_manager.initialize()
        print("✅ Flow Integration Manager initialized successfully")
        
        # Get available bracket commands
        print("\n📋 Available bracket commands:")
        commands = flow_manager.get_available_bracket_commands()
        
        for i, command in enumerate(commands[:5], 1):  # Show first 5
            print(f"  {i}. {command}")
        
        print(f"  ... and {len(commands) - 5} more commands")
        
        # Test flow mapping lookup
        print(f"\n🔍 Testing flow mapping for '[Process MVR]'...")
        mapping = flow_manager.get_flow_mapping("[Process MVR]")
        
        if mapping:
            print("Flow Mapping Details:")
            print(f"  Templates: {mapping.template_names}")
            print(f"  Strategy: {mapping.processing_strategy.value}")
            print(f"  Priority: {mapping.priority_level}")
            print(f"  Validation Rules: {mapping.validation_rules}")
        
        # Get integration status
        print(f"\n📊 Integration status:")
        status = await flow_manager.get_integration_status()
        
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Flow Integration Manager test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

async def test_bracket_registry():
    """Test the Bracket Commands Registry."""
    
    print("\n🧪 Testing Bracket Commands Registry")
    print("=" * 50)
    
    try:
        from tidyllm.flow.examples.bracket_registry import BracketRegistry, get_all_bracket_commands
        
        print("✅ Successfully imported Bracket Registry")
        
        # Initialize registry
        registry = BracketRegistry()
        
        # Get all commands
        all_commands = registry.get_all_commands()
        print(f"📋 Found {len(all_commands)} bracket commands")
        
        # Test command validation
        test_commands = ["[Process MVR]", "[Invalid Command]", "[Financial Analysis]"]
        
        print("\n🔍 Testing command validation:")
        for cmd in test_commands:
            is_valid = registry.validate_command(cmd)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"  {cmd}: {status}")
        
        # Get help for a command
        print(f"\n📖 Help for '[Quality Check]':")
        help_text = registry.get_command_help("[Quality Check]")
        print(help_text[:200] + "..." if len(help_text) > 200 else help_text)
        
        # Get registry stats
        print(f"\n📊 Registry statistics:")
        stats = registry.get_registry_stats()
        
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        print("\n✅ Bracket Commands Registry test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

async def main():
    """Run all tests."""
    print("🚀 AI Dropzone Manager Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test AI Dropzone Manager
    results.append(await test_ai_dropzone_manager())
    
    # Test Flow Integration Manager  
    results.append(await test_flow_integration_manager())
    
    # Test Bracket Registry
    results.append(await test_bracket_registry())
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("\n🎉 AI Dropzone Manager is working correctly!")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total} passed)")
        print("\n🔧 Check the error messages above for details")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)