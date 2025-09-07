#!/usr/bin/env python3
"""
Unified FLOW Interface
=====================

Single, consistent API for all FLOW Agreement operations.
Provides unified access to both Clean and Original FLOW systems.

This is the recommended interface for:
- CLI applications
- API endpoints  
- UI components
- Programmatic access

Usage:
    from flow_unified import UnifiedFlowInterface
    
    flow = UnifiedFlowInterface()
    result = flow.execute("[Integration Test]")
    
    # Or use as CLI
    python flow_unified.py "[Integration Test]"
"""

import sys
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FlowExecutionResult:
    """Standardized result format for all FLOW operations."""
    success: bool
    command: str
    system_used: str
    execution_mode: str
    confidence: float
    result: Dict[str, Any]
    timestamp: str
    error: Optional[str] = None
    fallback_chain: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class UnifiedFlowInterface:
    """
    Unified interface for all FLOW Agreement operations.
    
    Provides consistent API regardless of underlying system implementation.
    Automatically selects best available system for each operation.
    """
    
    def __init__(self, prefer_system: str = "auto"):
        """
        Initialize unified FLOW interface.
        
        Args:
            prefer_system: "auto", "clean", "original", or "bridge"
        """
        self.prefer_system = prefer_system
        self._bridge = None
        self._clean_manager = None
        self._original_manager = None
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize available FLOW systems."""
        try:
            # Initialize bridge system
            from flow_bridge import HybridFlowBridge
            self._bridge = HybridFlowBridge()
            logger.info("Hybrid FLOW Bridge initialized")
        except Exception as e:
            logger.warning(f"Bridge system unavailable: {e}")
        
        try:
            # Initialize clean system directly
            from flow_clean import CleanFlowManager
            self._clean_manager = CleanFlowManager()
            logger.info("Clean FLOW system initialized")
        except Exception as e:
            logger.warning(f"Clean system unavailable: {e}")
        
        try:
            # Initialize original system directly
            from tidyllm.flow.flow_agreements import FlowAgreementManager
            self._original_manager = FlowAgreementManager()
            logger.info("Original FLOW system initialized")
        except Exception as e:
            logger.warning(f"Original system unavailable: {e}")
    
    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> FlowExecutionResult:
        """
        Execute FLOW command using best available system.
        
        Args:
            command: FLOW command (e.g., "[Integration Test]")
            context: Optional context for execution
            
        Returns:
            FlowExecutionResult with standardized response
        """
        context = context or {}
        start_time = datetime.now()
        
        # Add unified interface metadata
        context.update({
            'unified_interface': True,
            'interface_version': '1.0.0',
            'timestamp': start_time.isoformat()
        })
        
        # Determine system to use
        if self.prefer_system == "clean" and self._clean_manager:
            result = self._execute_with_clean(command, context)
        elif self.prefer_system == "original" and self._original_manager:
            result = self._execute_with_original(command, context)
        elif self.prefer_system == "bridge" and self._bridge:
            result = self._execute_with_bridge(command, context)
        else:
            # Auto selection - prefer bridge for best experience
            if self._bridge:
                result = self._execute_with_bridge(command, context)
            elif self._clean_manager:
                result = self._execute_with_clean(command, context)
            elif self._original_manager:
                result = self._execute_with_original(command, context)
            else:
                result = self._create_error_result(command, "No FLOW systems available")
        
        # Add execution timing
        execution_time = (datetime.now() - start_time).total_seconds()
        if result.metadata:
            result.metadata['execution_time_seconds'] = execution_time
        else:
            result.metadata = {'execution_time_seconds': execution_time}
        
        return result
    
    def _execute_with_bridge(self, command: str, context: Dict[str, Any]) -> FlowExecutionResult:
        """Execute using hybrid bridge system."""
        try:
            bridge_result = self._bridge.execute_flow_command(command, context)
            
            return FlowExecutionResult(
                success=bridge_result.get('success', True),
                command=command,
                system_used=f"bridge-{bridge_result.get('system_used', 'unknown')}",
                execution_mode=bridge_result.get('result', {}).get('execution_mode', 'unknown'),
                confidence=bridge_result.get('result', {}).get('confidence', 0.5),
                result=bridge_result.get('result', {}),
                timestamp=bridge_result.get('timestamp', datetime.now().isoformat()),
                error=bridge_result.get('error'),
                fallback_chain=bridge_result.get('fallback_chain'),
                metadata={
                    'bridge_version': bridge_result.get('bridge_version'),
                    'system_status': self._bridge.get_system_status()
                }
            )
        except Exception as e:
            return self._create_error_result(command, f"Bridge execution failed: {e}")
    
    def _execute_with_clean(self, command: str, context: Dict[str, Any]) -> FlowExecutionResult:
        """Execute using clean system."""
        try:
            from flow_clean import execute_clean_flow_command
            
            clean_result = execute_clean_flow_command(command, context)
            
            if clean_result.get('error'):
                return FlowExecutionResult(
                    success=False,
                    command=command,
                    system_used="clean",
                    execution_mode="error",
                    confidence=0.0,
                    result={},
                    timestamp=datetime.now().isoformat(),
                    error=clean_result['error']
                )
            
            return FlowExecutionResult(
                success=True,
                command=command,
                system_used="clean",
                execution_mode=clean_result.get('execution_mode', 'unknown'),
                confidence=clean_result.get('confidence', 0.8),
                result=clean_result.get('result', {}),
                timestamp=clean_result.get('timestamp', datetime.now().isoformat()),
                metadata={'clean_system': True}
            )
        except Exception as e:
            return self._create_error_result(command, f"Clean system execution failed: {e}")
    
    def _execute_with_original(self, command: str, context: Dict[str, Any]) -> FlowExecutionResult:
        """Execute using original system."""
        try:
            from tidyllm.flow.flow_agreements import execute_flow_command
            
            original_result = execute_flow_command(command, context)
            
            return FlowExecutionResult(
                success=True,
                command=command,
                system_used="original",
                execution_mode=original_result.get('execution_mode', 'simulation'),
                confidence=original_result.get('confidence', 0.7),
                result=original_result,
                timestamp=datetime.now().isoformat(),
                metadata={'original_system': True}
            )
        except Exception as e:
            return self._create_error_result(command, f"Original system execution failed: {e}")
    
    def _create_error_result(self, command: str, error: str) -> FlowExecutionResult:
        """Create standardized error result."""
        return FlowExecutionResult(
            success=False,
            command=command,
            system_used="none",
            execution_mode="error",
            confidence=0.0,
            result={},
            timestamp=datetime.now().isoformat(),
            error=error
        )
    
    def list_available_commands(self) -> Dict[str, List[str]]:
        """List all available FLOW commands from all systems."""
        commands = {}
        
        if self._clean_manager:
            commands['clean'] = self._clean_manager.get_available_agreements()
        
        if self._original_manager:
            commands['original'] = self._original_manager.get_available_agreements()
        
        if self._bridge:
            bridge_status = self._bridge.get_system_status()
            commands['bridge'] = {
                'clean_commands': bridge_status['systems']['clean'].get('capabilities', 0),
                'original_commands': bridge_status['systems']['original'].get('capabilities', 0)
            }
        
        return commands
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all FLOW systems."""
        status = {
            'unified_interface': {
                'version': '1.0.0',
                'initialized': True,
                'prefer_system': self.prefer_system
            },
            'systems': {}
        }
        
        if self._bridge:
            status['systems']['bridge'] = self._bridge.get_system_status()
        
        if self._clean_manager:
            status['systems']['clean'] = {
                'available': True,
                'commands': len(self._clean_manager.get_available_agreements())
            }
        
        if self._original_manager:
            status['systems']['original'] = {
                'available': True,
                'commands': len(self._original_manager.get_available_agreements())
            }
        
        return status
    
    def execute_batch(self, commands: List[str], context: Optional[Dict[str, Any]] = None) -> List[FlowExecutionResult]:
        """Execute multiple FLOW commands in batch."""
        results = []
        
        for command in commands:
            result = self.execute(command, context)
            results.append(result)
        
        return results
    
    def to_dict(self, result: FlowExecutionResult) -> Dict[str, Any]:
        """Convert FlowExecutionResult to dictionary."""
        return asdict(result)
    
    def to_json(self, result: FlowExecutionResult) -> str:
        """Convert FlowExecutionResult to JSON string."""
        return json.dumps(self.to_dict(result), indent=2, default=str)

def main():
    """Main CLI interface for unified FLOW system."""
    
    if len(sys.argv) < 2:
        print("=" * 60)
        print("UNIFIED FLOW INTERFACE")
        print("=" * 60)
        print("Single, consistent API for all FLOW Agreement operations")
        print()
        print("Usage:")
        print('  python flow_unified.py "[Integration Test]"     # Execute command')
        print('  python flow_unified.py --commands              # List all commands')
        print('  python flow_unified.py --status                # System status')
        print('  python flow_unified.py --batch "[cmd1]" "[cmd2]" # Batch execution')
        print()
        
        # Quick system overview
        flow = UnifiedFlowInterface()
        status = flow.get_system_status()
        
        print("System Overview:")
        print("-" * 30)
        systems_available = len([s for s in status['systems'].keys() if status['systems'][s].get('available', False)])
        print(f"Available Systems: {systems_available}")
        print(f"Preferred System: {status['unified_interface']['prefer_system']}")
        print()
        return
    
    flow = UnifiedFlowInterface()
    
    if sys.argv[1] == "--commands":
        print("=" * 60)
        print("AVAILABLE FLOW COMMANDS")
        print("=" * 60)
        
        commands = flow.list_available_commands()
        for system_name, system_commands in commands.items():
            print(f"\n{system_name.upper()} SYSTEM:")
            print("-" * 30)
            if isinstance(system_commands, list):
                for i, cmd in enumerate(system_commands, 1):
                    print(f"{i:2d}. {cmd}")
            else:
                for key, value in system_commands.items():
                    print(f"  {key}: {value}")
        
    elif sys.argv[1] == "--status":
        print("=" * 60)
        print("UNIFIED FLOW SYSTEM STATUS")
        print("=" * 60)
        
        status = flow.get_system_status()
        print(json.dumps(status, indent=2, default=str))
        
    elif sys.argv[1] == "--batch":
        print("=" * 60)
        print("BATCH FLOW EXECUTION")
        print("=" * 60)
        
        batch_commands = sys.argv[2:]
        if not batch_commands:
            print("Error: No commands provided for batch execution")
            return
        
        print(f"Executing {len(batch_commands)} commands...")
        print("-" * 60)
        
        results = flow.execute_batch(batch_commands)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result.command}")
            print(f"    System: {result.system_used}")
            print(f"    Success: {'YES' if result.success else 'NO'}")
            print(f"    Mode: {result.execution_mode}")
            print(f"    Confidence: {result.confidence}")
            if result.error:
                print(f"    Error: {result.error}")
        
        print(f"\n\nBatch Summary: {sum(1 for r in results if r.success)}/{len(results)} successful")
        
    else:
        # Execute single command
        command = sys.argv[1]
        
        print("=" * 60)
        print("UNIFIED FLOW INTERFACE")
        print("=" * 60)
        print(f"Executing: {command}")
        print("-" * 60)
        
        result = flow.execute(command, context={'cli': True})
        
        print(f"System Used: {result.system_used.upper()}")
        print(f"Success: {'YES' if result.success else 'NO'}")
        print(f"Execution Mode: {result.execution_mode}")
        print(f"Confidence: {result.confidence:.2f}")
        
        if result.fallback_chain:
            print(f"Fallback Chain: {' → '.join(result.fallback_chain)}")
        
        if result.error:
            print(f"Error: {result.error}")
        else:
            print("\nResult Details:")
            for key, value in result.result.items():
                if key in ['action', 'status', 'execution_mode', 'real_implementation']:
                    print(f"  {key}: {value}")
        
        if result.metadata and result.metadata.get('execution_time_seconds'):
            print(f"\nExecution Time: {result.metadata['execution_time_seconds']:.3f}s")
        
        print("\nJSON Output:")
        print("-" * 30)
        print(flow.to_json(result))

if __name__ == "__main__":
    main()