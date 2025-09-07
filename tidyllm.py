#!/usr/bin/env python3
"""
TidyLLM - The Great Walled City of Enterprise AI
===============================================

Welcome to TidyLLM, a great ancient walled city protecting enterprise AI workflows.
Like any great city, there are THREE GATES for entry, each offering different paths
to explore the powerful FLOW Agreement systems within.

**The Three Gates:**

**CORPORATE GATE** (tidyllm.py) - *The Grand Entrance*
   - Guided tours with expert recommendations
   - Executive-friendly interface with enterprise reporting
   - Full audit trails and compliance documentation
   - Choose your guide: Clean, Original, or Bridge systems

**EXPLORER'S GATE** (flow_bridge.py) - *The Adventurer's Choice*
   - Hybrid exploration with system migration tools
   - Choose your own path between different implementations
   - Real-time system health and migration guidance
   - For those who want control over their journey

**ARTISAN'S GATE** (flow_clean.py) - *The Direct Path*
   - Direct access to clean, presentation-ready workflows
   - Zero dependencies, perfect for demonstrations
   - Real-time AWS infrastructure connections
   - For craftsmen who appreciate elegant simplicity

**Within the City Walls:**
   - FLOW Agreement temples (audit-compliant workflows)
   - PostgreSQL archives (7-year retention halls)
   - AWS infrastructure districts (S3, Bedrock, RDS)
   - Performance amphitheaters and security bastions

Usage:
    # Main entrance - Corporate Gate (Recommended for executives)
    python tidyllm.py "[Integration Test]"
    
    # Explorer's path - Hybrid Bridge
    python flow_bridge.py "[Integration Test]" 
    
    # Direct artisan path - Clean system
    python flow_clean.py "[Integration Test]"
    
    # Python API - Program your own city tour
    from tidyllm import TidyLLMInterface
    city_guide = TidyLLMInterface()
    tour_result = city_guide.execute("[Integration Test]")
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

class TidyLLMInterface:
    """
    TidyLLM Enterprise AI Workflow Platform Interface.
    
    Provides enterprise-grade FLOW Agreement execution with:
    - Audit compliance and 7-year retention
    - Real-time AWS infrastructure integration  
    - Multi-system fallback and reliability
    - Corporate-friendly terminology and reporting
    """
    
    def __init__(self, prefer_system: str = "auto"):
        """
        Initialize TidyLLM Enterprise AI Workflow Platform.
        
        Args:
            prefer_system: "auto", "clean", "original", or "bridge"
                         "auto" selects optimal system automatically
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
        Execute enterprise FLOW Agreement with full audit compliance.
        
        Args:
            command: FLOW Agreement command (e.g., "[Integration Test]")
            context: Optional execution context for audit trail
            
        Returns:
            FlowExecutionResult with enterprise metadata and audit information
        """
        context = context or {}
        start_time = datetime.now()
        
        # Add TidyLLM enterprise metadata
        context.update({
            'tidyllm_platform': True,
            'platform_version': '1.0.0',
            'enterprise_mode': True,
            'audit_timestamp': start_time.isoformat()
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
    """Main CLI interface for TidyLLM Enterprise AI Workflow Platform."""
    
    if len(sys.argv) < 2:
        print("=" * 60)
        print("TidyLLM - THE GREAT WALLED CITY OF ENTERPRISE AI")
        print("=" * 60)
        print("Welcome to the CORPORATE GATE - The Grand Entrance")
        print("Your guided tour of enterprise AI workflows awaits")
        print()
        print("The Three Gates of TidyLLM:")
        print("  CORPORATE GATE  (tidyllm.py)     - Executive tours with expert guides")  
        print("  EXPLORER'S GATE (flow_bridge.py) - Adventure paths with migration tools")
        print("  ARTISAN'S GATE  (flow_clean.py)  - Direct craftsmanship access")
        print()
        print("Corporate Gate Services:")
        print('  python tidyllm.py "[Integration Test]"          # Guided FLOW Agreement tour')
        print('  python tidyllm.py --commands                   # Map of city districts')
        print('  python tidyllm.py --status                     # City walls status')
        print('  python tidyllm.py --batch "[cmd1]" "[cmd2]"    # Group expedition')
        print()
        
        # City status overview
        tidyllm = TidyLLMInterface()
        status = tidyllm.get_system_status()
        
        print("City Defenses Status:")
        print("-" * 22)
        systems_available = len([s for s in status['systems'].keys() if status['systems'][s].get('available', False)])
        print(f"Gates Open: {systems_available}/3")
        print(f"Recommended Guide: {status['unified_interface']['prefer_system']}")
        print(f"City Charter: v{status['unified_interface']['version']}")
        print()
        print("Choose your gate wisely, traveler. Each path leads to the same")
        print("powerful FLOW Agreement temples, but offers a different journey.")
        print()
        return
    
    tidyllm = TidyLLMInterface()
    
    if sys.argv[1] == "--commands":
        print("=" * 60)
        print("TidyLLM CITY MAP - FLOW AGREEMENT DISTRICTS")
        print("=" * 60)
        print("Ancient temples and districts within the city walls")
        print("Each district offers specialized services for travelers")
        print()
        
        commands = tidyllm.list_available_commands()
        for system_name, system_commands in commands.items():
            if system_name == "clean":
                print(f"\nARTISAN DISTRICT ({system_name.upper()} system):")
                print("-" * 40)
                print("Elegant, direct access to core services")
            elif system_name == "original": 
                print(f"\nHISTORIC QUARTER ({system_name.upper()} system):")
                print("-" * 40)
                print("Traditional services with full heritage")
            else:
                print(f"\nGUIDE SERVICES ({system_name.upper()} system):")
                print("-" * 40)
                print("Hybrid tours combining all districts")
            if isinstance(system_commands, list):
                for i, cmd in enumerate(system_commands, 1):
                    print(f"{i:2d}. {cmd}")
            else:
                for key, value in system_commands.items():
                    print(f"  {key}: {value}")
        
    elif sys.argv[1] == "--status":
        print("=" * 60)
        print("TidyLLM - ENTERPRISE PLATFORM STATUS")
        print("=" * 60)
        print("Real-time status of all enterprise AI workflow systems")
        print()
        
        status = tidyllm.get_system_status()
        print(json.dumps(status, indent=2, default=str))
        
    elif sys.argv[1] == "--batch":
        print("=" * 60)
        print("TidyLLM - BATCH FLOW EXECUTION")
        print("=" * 60)
        print("Enterprise batch processing with full audit trail")
        print()
        
        batch_commands = sys.argv[2:]
        if not batch_commands:
            print("Error: No FLOW Agreement commands provided for batch execution")
            return
        
        print(f"Executing {len(batch_commands)} enterprise FLOW Agreement(s)...")
        print("-" * 60)
        
        results = tidyllm.execute_batch(batch_commands)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result.command}")
            print(f"    System: {result.system_used}")
            print(f"    Success: {'YES' if result.success else 'NO'}")
            print(f"    Mode: {result.execution_mode}")
            print(f"    Confidence: {result.confidence}")
            if result.error:
                print(f"    Error: {result.error}")
        
        print(f"\n\nEnterprise Batch Summary: {sum(1 for r in results if r.success)}/{len(results)} FLOW Agreements executed successfully")
        
    else:
        # Execute single FLOW Agreement
        command = sys.argv[1]
        
        print("=" * 60)
        print("TidyLLM - GUIDED TOUR IN PROGRESS")
        print("=" * 60)
        print(f"Your guide is leading you to: {command}")
        print("Passing through the city gates...")
        print("-" * 60)
        
        result = tidyllm.execute(command, context={'cli': True, 'enterprise_mode': True})
        
        guide_name = result.system_used.upper().replace('BRIDGE-', 'BRIDGE -> ')
        print(f"Guide Selected: {guide_name}")
        print(f"Tour Success: {'YES - Temple reached!' if result.success else 'NO - Path blocked'}")
        print(f"Journey Mode: {result.execution_mode}")
        print(f"Guide Confidence: {result.confidence:.2f}")
        
        if result.fallback_chain:
            print(f"Alternative Paths Taken: {' -> '.join(result.fallback_chain)}")
        
        if result.error:
            print(f"Journey Issue: {result.error}")
        else:
            print("\nTemple Services Accessed:")
            for key, value in result.result.items():
                if key in ['action', 'status', 'execution_mode', 'real_implementation']:
                    print(f"  {key}: {value}")
        
        if result.metadata and result.metadata.get('execution_time_seconds'):
            print(f"\nTour Duration: {result.metadata['execution_time_seconds']:.3f} seconds")
        
        print("\nCity Archives Record (JSON):")
        print("-" * 30)
        print(tidyllm.to_json(result))

if __name__ == "__main__":
    main()