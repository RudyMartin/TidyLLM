#!/usr/bin/env python3
"""
FLOW Bridge - Hybrid Migration System
====================================

Provides unified interface to both clean FLOW system and original FLOW core.
Enables gradual migration from scattered implementations to unified system.

Strategy:
1. Try clean system first (working, real mode)
2. Fallback to original system (if import issues resolved)  
3. Graceful degradation to simulation
4. Migration tracking and reporting

Usage:
    python flow_bridge.py "[Integration Test]"    # Uses best available system
    python flow_bridge.py --status                # Show system availability  
    python flow_bridge.py --migrate               # Migration analysis
"""

import sys
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Suppress Unicode errors for status symbols
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class FlowSystemType(Enum):
    """Available FLOW system types."""
    CLEAN = "clean"           # flow_clean.py - working, real mode
    ORIGINAL = "original"     # tidyllm/flow/flow_agreements.py - has import issues
    SIMULATION = "simulation" # Pure simulation fallback

@dataclass
class FlowSystemStatus:
    """Status of a FLOW system."""
    system_type: FlowSystemType
    available: bool = False
    import_success: bool = False
    error_message: str = ""
    capabilities: List[str] = None
    confidence: float = 0.0

class HybridFlowBridge:
    """Bridge between clean and original FLOW systems."""
    
    def __init__(self):
        self.system_status: Dict[FlowSystemType, FlowSystemStatus] = {}
        self.preferred_system = None
        self._assess_available_systems()
    
    def _assess_available_systems(self):
        """Assess which FLOW systems are available."""
        
        # Test Clean System
        self._test_clean_system()
        
        # Test Original System  
        self._test_original_system()
        
        # Determine preferred system
        self._determine_preferred_system()
    
    def _test_clean_system(self):
        """Test clean FLOW system availability."""
        try:
            # Try importing clean flow system
            from flow_clean import execute_clean_flow_command, CleanFlowManager
            
            # Test basic execution
            test_result = execute_clean_flow_command("[Integration Test]", context={"test": True})
            
            # Determine capabilities
            manager = CleanFlowManager()
            capabilities = manager.get_available_agreements()
            
            self.system_status[FlowSystemType.CLEAN] = FlowSystemStatus(
                system_type=FlowSystemType.CLEAN,
                available=True,
                import_success=True,
                capabilities=capabilities,
                confidence=0.95 if test_result.get('execution_mode') == 'real' else 0.8,
                error_message=""
            )
            
        except Exception as e:
            self.system_status[FlowSystemType.CLEAN] = FlowSystemStatus(
                system_type=FlowSystemType.CLEAN,
                available=False,
                import_success=False,
                error_message=str(e),
                confidence=0.0
            )
    
    def _test_original_system(self):
        """Test original FLOW system availability."""
        try:
            # Try importing original FLOW system (expect this to fail initially)
            from tidyllm.flow.flow_agreements import execute_flow_command, FlowAgreementManager
            
            # Test basic execution
            test_result = execute_flow_command("[Integration Test]", context={"test": True})
            
            # Determine capabilities
            manager = FlowAgreementManager()
            capabilities = manager.get_available_agreements()
            
            self.system_status[FlowSystemType.ORIGINAL] = FlowSystemStatus(
                system_type=FlowSystemType.ORIGINAL,
                available=True,
                import_success=True,
                capabilities=capabilities,
                confidence=0.9,  # Original system when working
                error_message=""
            )
            
        except Exception as e:
            self.system_status[FlowSystemType.ORIGINAL] = FlowSystemStatus(
                system_type=FlowSystemType.ORIGINAL,
                available=False,
                import_success=False,
                error_message=str(e)[:100] + "..." if len(str(e)) > 100 else str(e),
                confidence=0.0
            )
    
    def _determine_preferred_system(self):
        """Determine which system to use as preferred."""
        
        # Priority order: Clean (real mode) > Original > Clean (simulation)
        
        clean_status = self.system_status.get(FlowSystemType.CLEAN)
        original_status = self.system_status.get(FlowSystemType.ORIGINAL)
        
        if clean_status and clean_status.available and clean_status.confidence >= 0.9:
            self.preferred_system = FlowSystemType.CLEAN
        elif original_status and original_status.available:
            self.preferred_system = FlowSystemType.ORIGINAL  
        elif clean_status and clean_status.available:
            self.preferred_system = FlowSystemType.CLEAN
        else:
            self.preferred_system = FlowSystemType.SIMULATION
    
    def execute_flow_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute FLOW command using best available system."""
        
        execution_record = {
            'command': command,
            'timestamp': datetime.now().isoformat(),
            'bridge_version': '1.0.0',
            'system_used': None,
            'fallback_chain': [],
            'result': None,
            'error': None
        }
        
        # Try preferred system first
        if self.preferred_system == FlowSystemType.CLEAN:
            result = self._execute_with_clean_system(command, context, execution_record)
            if result:
                return result
                
        elif self.preferred_system == FlowSystemType.ORIGINAL:
            result = self._execute_with_original_system(command, context, execution_record)
            if result:
                return result
        
        # Fallback chain
        fallback_systems = [FlowSystemType.CLEAN, FlowSystemType.ORIGINAL, FlowSystemType.SIMULATION]
        
        for system in fallback_systems:
            if system == self.preferred_system:
                continue  # Already tried
                
            execution_record['fallback_chain'].append(system.value)
            
            if system == FlowSystemType.CLEAN:
                result = self._execute_with_clean_system(command, context, execution_record)
                if result:
                    return result
            elif system == FlowSystemType.ORIGINAL:
                result = self._execute_with_original_system(command, context, execution_record)
                if result:
                    return result
            elif system == FlowSystemType.SIMULATION:
                result = self._execute_with_simulation(command, context, execution_record)
                return result  # Simulation should always work
        
        # If all else fails
        execution_record.update({
            'system_used': 'none',
            'error': 'All FLOW systems unavailable',
            'result': {'error': 'FLOW bridge failure - no systems available'}
        })
        
        return execution_record
    
    def _execute_with_clean_system(self, command: str, context: Optional[Dict[str, Any]], record: Dict) -> Optional[Dict]:
        """Execute with clean FLOW system."""
        try:
            from flow_clean import execute_clean_flow_command
            
            result = execute_clean_flow_command(command, context)
            
            record.update({
                'system_used': 'clean',
                'result': result,
                'success': True
            })
            
            return record
            
        except Exception as e:
            logger.warning(f"Clean system execution failed: {e}")
            return None
    
    def _execute_with_original_system(self, command: str, context: Optional[Dict[str, Any]], record: Dict) -> Optional[Dict]:
        """Execute with original FLOW system."""
        try:
            from tidyllm.flow.flow_agreements import execute_flow_command
            
            result = execute_flow_command(command, context)
            
            record.update({
                'system_used': 'original', 
                'result': result,
                'success': True
            })
            
            return record
            
        except Exception as e:
            logger.warning(f"Original system execution failed: {e}")
            return None
    
    def _execute_with_simulation(self, command: str, context: Optional[Dict[str, Any]], record: Dict) -> Dict:
        """Execute with pure simulation."""
        
        simulation_result = {
            'action': 'simulation_fallback',
            'command': command,
            'result': f"Simulated execution of {command}",
            'execution_mode': 'bridge_simulation',
            'confidence': 0.5,
            'note': 'Bridge fallback simulation - no FLOW systems available'
        }
        
        record.update({
            'system_used': 'simulation',
            'result': simulation_result, 
            'success': True
        })
        
        return record
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all FLOW systems."""
        return {
            'bridge_status': 'operational',
            'preferred_system': self.preferred_system.value if self.preferred_system else 'none',
            'systems': {
                system.value: {
                    'available': status.available,
                    'import_success': status.import_success,
                    'capabilities': len(status.capabilities or []),
                    'confidence': status.confidence,
                    'error': status.error_message if status.error_message else None
                }
                for system, status in self.system_status.items()
            },
            'migration_recommendation': self._get_migration_recommendation()
        }
    
    def _get_migration_recommendation(self) -> str:
        """Get migration recommendation based on system status."""
        
        clean_status = self.system_status.get(FlowSystemType.CLEAN)
        original_status = self.system_status.get(FlowSystemType.ORIGINAL)
        
        if clean_status and clean_status.available and clean_status.confidence >= 0.9:
            if original_status and not original_status.available:
                return "RECOMMENDED: Use clean system (real mode working). Fix original system imports when time permits."
            else:
                return "RECOMMENDED: Migrate to clean system for better reliability and real mode capabilities."
        
        elif original_status and original_status.available:
            return "RECOMMENDED: Original system working. Consider migrating to clean system for better reliability."
        
        elif clean_status and clean_status.available:
            return "RECOMMENDED: Use clean system (simulation mode). Fix AWS credentials for real mode."
        
        else:
            return "CRITICAL: All FLOW systems have issues. Fix clean system first as it has fewer dependencies."

def main():
    """Main entry point for hybrid FLOW bridge."""
    
    bridge = HybridFlowBridge()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            # Show system status
            print("=" * 60)
            print("HYBRID FLOW BRIDGE - SYSTEM STATUS")
            print("=" * 60)
            
            status = bridge.get_system_status()
            
            print(f"Bridge Status: {status['bridge_status']}")
            print(f"Preferred System: {status['preferred_system']}")
            print()
            
            print("SYSTEM AVAILABILITY:")
            print("-" * 40)
            for system_name, system_info in status['systems'].items():
                availability = "AVAILABLE" if system_info['available'] else "UNAVAILABLE"
                print(f"{system_name.upper():<12}: {availability}")
                if system_info['available']:
                    print(f"              Capabilities: {system_info['capabilities']}")
                    print(f"              Confidence: {system_info['confidence']:.1f}")
                else:
                    error = system_info['error'][:60] + "..." if system_info['error'] and len(system_info['error']) > 60 else system_info['error']
                    print(f"              Error: {error}")
                print()
            
            print("MIGRATION RECOMMENDATION:")
            print("-" * 40)
            print(status['migration_recommendation'])
            
        elif sys.argv[1] == "--migrate":
            # Migration analysis
            print("=" * 60)
            print("FLOW MIGRATION ANALYSIS")
            print("=" * 60)
            
            status = bridge.get_system_status()
            
            print("CURRENT STATE:")
            clean_available = status['systems']['clean']['available']
            original_available = status['systems']['original']['available']
            
            if clean_available and original_available:
                print("[OK] Both systems working - can choose optimal system per use case")
            elif clean_available:
                print("[OK] Clean system working - [WARN] Original system needs fixes")
            elif original_available:
                print("[WARN] Original system working - [FAIL] Clean system has issues")  
            else:
                print("[FAIL] Both systems have issues - need immediate attention")
            
            print(f"\nRECOMMENDATION: {status['migration_recommendation']}")
            
        else:
            # Execute FLOW command
            command = sys.argv[1]
            
            print("=" * 60)
            print("HYBRID FLOW BRIDGE")
            print("=" * 60)
            print(f"Executing: {command}")
            print("-" * 60)
            
            result = bridge.execute_flow_command(command, context={"bridge": True})
            
            print(f"System Used: {result['system_used'].upper()}")
            print(f"Success: {'YES' if result.get('success') else 'NO'}")
            
            if result.get('fallback_chain'):
                print(f"Fallback Chain: {' → '.join(result['fallback_chain'])}")
            
            if result.get('result'):
                flow_result = result['result']
                if isinstance(flow_result, dict):
                    print("\nResult:")
                    for key, value in flow_result.items():
                        if key in ['action', 'execution_mode', 'confidence', 'real_implementation']:
                            print(f"  {key}: {value}")
                
            if result.get('error'):
                print(f"Error: {result['error']}")
    
    else:
        # Show usage
        print("=" * 60)
        print("HYBRID FLOW BRIDGE")
        print("=" * 60)
        print("Unified interface for FLOW Agreement systems")
        print()
        print("Usage:")
        print('  python flow_bridge.py "[Integration Test]"    # Execute command')
        print('  python flow_bridge.py --status               # System status')
        print('  python flow_bridge.py --migrate              # Migration analysis')
        print()
        
        # Quick status
        status = bridge.get_system_status()
        print(f"Preferred System: {status['preferred_system']}")
        print(f"Systems Available: {sum(1 for s in status['systems'].values() if s['available'])}/2")

if __name__ == "__main__":
    main()