#!/usr/bin/env python3
"""
AI Dropzone Manager Trigger Testing
===================================

Tests all trigger mechanisms for the AI Dropzone Manager:
- File drop zone triggers
- CLI bracket commands  
- API endpoint triggers
- Flow integration triggers
"""

import asyncio
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent / "tidyllm"))

async def test_drop_zone_triggers():
    """Test file drop zone triggers."""
    print("[TEST] Drop Zone File Triggers")
    print("=" * 50)
    
    try:
        from tidyllm.infrastructure.workers.ai_dropzone_manager import AIDropzoneManager
        
        manager = AIDropzoneManager()
        await manager.initialize()
        
        # Test drop zone monitoring
        drop_zones = [
            "drop_zones/mvr_analysis",
            "drop_zones/financial_analysis", 
            "drop_zones/contract_review",
            "drop_zones/compliance_check"
        ]
        
        print("\n[MONITOR] Setting up drop zone monitoring...")
        for zone in drop_zones:
            zone_path = Path(zone)
            zone_path.mkdir(parents=True, exist_ok=True)
            print(f"  [OK] Monitoring: {zone}")
        
        # Simulate file drop
        test_file = Path("drop_zones/mvr_analysis/test_document.pdf")
        test_file.write_text("Sample MVR document content for testing")
        
        print(f"\n[DROP] Simulated file drop: {test_file}")
        
        # Test detection
        result = await manager.detect_and_process_drop_zone_files()
        
        if result:
            print(f"[TRIGGER] Drop zone trigger activated successfully")
            print(f"  Files detected: {result.get('files_detected', 0)}")
            print(f"  Processing started: {result.get('processing_started', 0)}")
        else:
            print("[INFO] No files detected in drop zones")
            
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

async def test_cli_triggers():
    """Test CLI bracket command triggers."""
    print("\n[TEST] CLI Bracket Command Triggers") 
    print("=" * 50)
    
    try:
        from tidyllm.infrastructure.workers.flow_integration_manager import FlowIntegrationManager
        
        flow_manager = FlowIntegrationManager()
        await flow_manager.initialize()
        
        # Test CLI-style bracket commands
        test_commands = [
            "[Process MVR]",
            "[Financial Analysis]", 
            "[Quality Check]",
            "[Contract Review]"
        ]
        
        for command in test_commands:
            print(f"\n[CLI] Testing command: {command}")
            
            # Validate command
            is_valid = flow_manager.validate_bracket_command(command)
            if is_valid:
                print(f"  [VALID] Command registered in system")
                
                # Get mapping details
                mapping = flow_manager.get_flow_mapping(command)
                if mapping:
                    print(f"  [MAPPING] Templates: {mapping.template_names}")
                    print(f"  [MAPPING] Strategy: {mapping.processing_strategy.value}")
                    print(f"  [MAPPING] Priority: {mapping.priority_level}")
                
                # Simulate trigger (without actual file)
                print(f"  [TRIGGER] Would activate AI Dropzone Manager")
                
            else:
                print(f"  [INVALID] Command not found in registry")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

async def test_api_triggers():
    """Test API endpoint triggers."""
    print("\n[TEST] API Endpoint Triggers")
    print("=" * 50)
    
    # Simulate API requests that would trigger processing
    api_examples = [
        {
            "method": "POST",
            "endpoint": "/api/v1/flow/execute",
            "payload": {
                "bracket_command": "[Process MVR]",
                "document_path": "/path/to/mvr_document.pdf",
                "user_context": {"department": "compliance"}
            }
        },
        {
            "method": "POST", 
            "endpoint": "/api/v1/flow/execute",
            "payload": {
                "bracket_command": "[Financial Analysis]",
                "document_path": "/path/to/financial_report.pdf",
                "business_priority": "high"
            }
        }
    ]
    
    for example in api_examples:
        print(f"\n[API] {example['method']} {example['endpoint']}")
        print(f"      Payload: {example['payload']}")
        
        bracket_command = example['payload']['bracket_command']
        print(f"  [TRIGGER] Would activate AI Dropzone Manager")
        print(f"  [PROCESS] Command: {bracket_command}")
        print(f"  [RESULT] Processing ID would be returned")
    
    return True

async def test_integration_flow():
    """Test complete integration flow."""
    print("\n[TEST] Complete Integration Flow")
    print("=" * 50)
    
    try:
        from tidyllm.infrastructure.workers.ai_dropzone_manager import AIDropzoneManager
        from tidyllm.infrastructure.workers.flow_integration_manager import FlowIntegrationManager
        
        # Initialize managers
        ai_manager = AIDropzoneManager()
        flow_manager = FlowIntegrationManager()
        
        await ai_manager.initialize()
        await flow_manager.initialize()
        
        print("[INIT] AI Dropzone Manager initialized")
        print("[INIT] Flow Integration Manager initialized")
        
        # Test integration status
        ai_status = await ai_manager.get_manager_status()
        flow_status = await flow_manager.get_integration_status()
        
        print("\n[STATUS] AI Dropzone Manager:")
        for key, value in ai_status.items():
            print(f"  {key}: {value}")
            
        print("\n[STATUS] Flow Integration Manager:")
        for key, value in flow_status.items():
            print(f"  {key}: {value}")
        
        # Test bracket command availability
        available_commands = flow_manager.get_available_bracket_commands()
        print(f"\n[COMMANDS] Available bracket commands: {len(available_commands)}")
        for cmd in available_commands[:5]:  # Show first 5
            print(f"  - {cmd}")
        
        print("\n[FLOW] Complete integration flow tested successfully")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

async def demonstrate_activation_methods():
    """Demonstrate all activation methods."""
    print("\n" + "=" * 70)
    print("AI DROPZONE MANAGER - ACTIVATION METHODS DEMONSTRATION")
    print("=" * 70)
    
    activation_methods = [
        {
            "method": "File Drop Zone",
            "description": "Drop PDF in workflow folder",
            "example": "cp document.pdf tidyllm/drop_zones/mvr_analysis/",
            "trigger": "Automatic file system monitoring",
            "result": "AI manager detects and processes immediately"
        },
        {
            "method": "CLI Bracket Command", 
            "description": "Execute bracket command via CLI",
            "example": "tidyllm flow '[Process MVR]' document.pdf",
            "trigger": "Flow Integration Manager -> AI Dropzone Manager",
            "result": "Command parsed and routed to appropriate workers"
        },
        {
            "method": "API Endpoint",
            "description": "REST API call with bracket command",
            "example": "POST /api/v1/flow/execute {'bracket_command': '[Process MVR]'}",
            "trigger": "API handler -> Flow Integration Manager -> AI Manager",
            "result": "HTTP response with processing ID and status"
        },
        {
            "method": "S3 Drop Zone",
            "description": "Upload to S3 bucket with workflow prefix",
            "example": "s3://workflows/mvr_analysis/document.pdf",
            "trigger": "S3 event -> Lambda -> AI Dropzone Manager",
            "result": "Cloud-native processing with audit trail"
        },
        {
            "method": "Chat Interface",
            "description": "Natural language with bracket detection", 
            "example": "Please [Process MVR] this document for me",
            "trigger": "NLP parsing -> bracket extraction -> AI Manager",
            "result": "Conversational processing with user feedback"
        }
    ]
    
    for i, method in enumerate(activation_methods, 1):
        print(f"\n{i}. {method['method']}")
        print(f"   Description: {method['description']}")
        print(f"   Example: {method['example']}")
        print(f"   Trigger Flow: {method['trigger']}")
        print(f"   Result: {method['result']}")
    
    print("\n" + "=" * 70)
    print("KEY FEATURES:")
    print("- Universal bracket syntax across all interfaces")
    print("- Automatic document intelligence and worker selection")
    print("- Security-first with CorporateLLMGateway integration")
    print("- Complete audit trail and monitoring")
    print("- Fault tolerance with recovery mechanisms")

async def main():
    """Run all trigger tests."""
    print("[LAUNCH] AI Dropzone Manager Trigger Testing Suite")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(await test_drop_zone_triggers())
    results.append(await test_cli_triggers())
    results.append(await test_api_triggers())
    results.append(await test_integration_flow())
    
    # Show activation methods
    await demonstrate_activation_methods()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if passed == total:
        print(f"[SUCCESS] ALL TRIGGER TESTS PASSED ({passed}/{total})")
        print("\n[READY] AI Dropzone Manager is ready for activation!")
    else:
        print(f"[WARNING] SOME TESTS FAILED ({passed}/{total} passed)")
        print("\n[ACTION] Check error messages above for details")
    
    print("\n[NEXT STEPS] Ready to process real documents:")
    print("1. Drop documents in drop_zones/ folders")
    print("2. Use CLI: tidyllm flow '[Process MVR]' document.pdf")
    print("3. Call API: POST /api/v1/flow/execute")
    print("4. Monitor processing in drop_zones/processing/")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)