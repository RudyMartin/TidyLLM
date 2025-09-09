#!/usr/bin/env python3
"""
Test 06: Flow System & Bracket Commands
========================================
Test bracket command registry and flow system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_26_bracket_registry_import():
    """Test BracketRegistry can be imported."""
    print("[TEST 26] Testing BracketRegistry import...")
    
    from tidyllm.flow.examples.bracket_registry import BracketRegistry
    
    assert BracketRegistry is not None, "FAIL: BracketRegistry import failed"
    print("  [PASS] BracketRegistry imports")
    return True

def test_27_bracket_registry_commands():
    """Test BracketRegistry has commands."""
    print("[TEST 27] Testing bracket command registry...")
    
    from tidyllm.flow.examples.bracket_registry import BracketRegistry
    
    registry = BracketRegistry()
    commands = registry.get_all_commands()
    
    assert len(commands) > 0, "FAIL: No bracket commands in registry"
    
    # Check specific commands exist
    expected_commands = ["[Process MVR]", "[Financial Analysis]", "[Contract Review]"]
    for cmd in expected_commands:
        assert cmd in commands, f"FAIL: Missing command {cmd}"
    
    print(f"  [PASS] Registry has {len(commands)} commands")
    return True

def test_28_command_validation():
    """Test command validation works."""
    print("[TEST 28] Testing command validation...")
    
    from tidyllm.flow.examples.bracket_registry import BracketRegistry
    
    registry = BracketRegistry()
    
    # Valid command
    assert registry.validate_command("[Process MVR]"), "FAIL: Valid command not recognized"
    
    # Invalid command
    assert not registry.validate_command("[Invalid Command]"), "FAIL: Invalid command accepted"
    
    print("  [PASS] Command validation works")
    return True

def test_29_flow_mappings():
    """Test flow mappings file exists."""
    print("[TEST 29] Testing flow mappings...")
    
    mappings_path = Path("prompts/flow_mappings.json")
    assert mappings_path.exists(), f"FAIL: flow_mappings.json not found at {mappings_path}"
    
    import json
    with open(mappings_path) as f:
        mappings = json.load(f)
    
    assert len(mappings) > 0, "FAIL: No mappings in flow_mappings.json"
    assert "[Process MVR]" in mappings, "FAIL: [Process MVR] not in mappings"
    
    print(f"  [PASS] Flow mappings has {len(mappings)} entries")
    return True

def test_30_bracket_command_details():
    """Test getting command details."""
    print("[TEST 30] Testing command details...")
    
    from tidyllm.flow.examples.bracket_registry import BracketRegistry
    
    registry = BracketRegistry()
    details = registry.get_command_details("[Process MVR]")
    
    assert details is not None, "FAIL: No details for [Process MVR]"
    assert hasattr(details, 'templates'), "FAIL: Command details missing templates"
    assert hasattr(details, 'priority'), "FAIL: Command details missing priority"
    
    print("  [PASS] Command details available")
    return True

def run_all_tests():
    """Run all flow system tests."""
    print("\n" + "="*60)
    print("FLOW SYSTEM & BRACKET COMMAND TESTS")
    print("="*60)
    
    tests = [
        test_26_bracket_registry_import,
        test_27_bracket_registry_commands,
        test_28_command_validation,
        test_29_flow_mappings,
        test_30_bracket_command_details
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1
    
    print("\n" + "-"*60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)