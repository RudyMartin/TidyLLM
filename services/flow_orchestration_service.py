"""
Flow Orchestration Service (Hexagonal Architecture Refactor)
===========================================================

Replaces UnifiedFlowManager with proper hexagonal architecture.
This service acts as an application layer that uses the domain service
and integrates with the existing portal layer.

Architecture: Portal → FlowOrchestrationService → WorkflowService → Infrastructure
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import domain service and ports
try:
    from domain.services.workflow_service import WorkflowService
    from domain.ports.outbound.workflow_port import WorkflowSystemType, WorkflowStatus
    from adapters.secondary.workflow.workflow_dependencies_adapter import get_workflow_dependencies_adapter
    DOMAIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Domain services not available: {e}")
    DOMAIN_AVAILABLE = False

# Import existing infrastructure for backward compatibility
try:
    from tidyllm.infrastructure.session.unified import UnifiedSessionManager
    from tidyllm.services.unified_rag_manager import UnifiedRAGManager, RAGSystemType
    INFRASTRUCTURE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TidyLLM infrastructure not available: {e}")
    INFRASTRUCTURE_AVAILABLE = False


class FlowOrchestrationService:
    """
    Application service for flow orchestration following hexagonal architecture.

    This service:
    1. Uses the domain WorkflowService for business logic
    2. Provides backward compatibility with existing portal code
    3. Acts as a clean interface between portals and domain logic
    """

    def __init__(self, auto_load_credentials: bool = True):
        """Initialize Flow Orchestration Service."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("+ Initializing Flow Orchestration Service (Hexagonal Architecture)...")

        # Initialize domain service with proper dependency injection
        if DOMAIN_AVAILABLE:
            dependencies_adapter = get_workflow_dependencies_adapter()
            self.workflow_service = WorkflowService(dependencies_adapter)
            self.logger.info("+ Domain workflow service initialized")
        else:
            self.workflow_service = None
            self.logger.warning("+ Domain services not available, using fallback mode")

        # Backward compatibility fields for existing code
        self.usm = None
        self.rag_manager = None

        # Initialize backward compatibility connections if needed
        if auto_load_credentials and INFRASTRUCTURE_AVAILABLE:
            self._initialize_backward_compatibility()

        self.logger.info("+ Flow Orchestration Service initialized")

    def _initialize_backward_compatibility(self):
        """Initialize backward compatibility connections for existing code."""
        try:
            # These are only for backward compatibility with existing portal code
            self.usm = UnifiedSessionManager()
            self.rag_manager = UnifiedRAGManager(auto_load_credentials=False)
            self.logger.info("+ Backward compatibility infrastructure initialized")
        except Exception as e:
            self.logger.warning(f"Backward compatibility initialization failed: {e}")

    # ==================== WORKFLOW SYSTEM AVAILABILITY ====================

    def is_system_available(self, system_type: WorkflowSystemType) -> bool:
        """Check if a workflow system type is available."""
        if self.workflow_service:
            return self.workflow_service.is_system_available(system_type)
        else:
            # Fallback for when domain service is not available
            return system_type in [WorkflowSystemType.MVR_ANALYSIS, WorkflowSystemType.DOMAIN_RAG]

    def get_available_systems(self) -> Dict[str, bool]:
        """Get availability status for all workflow systems."""
        if self.workflow_service:
            return self.workflow_service.get_available_systems()
        else:
            # Fallback availability
            return {
                "mvr_analysis": True,
                "domain_rag": True,
                "financial_analysis": False,
                "contract_review": False,
                "compliance_check": False,
                "quality_check": False,
                "peer_review": False,
                "data_extraction": False,
                "hybrid_analysis": False,
                "code_review": False,
                "research_synthesis": False,
                "classification": False,
                "custom_workflow": False
            }

    # ==================== WORKFLOW LIFECYCLE MANAGEMENT ====================

    def create_workflow(self, system_type: WorkflowSystemType, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new workflow instance."""
        if self.workflow_service:
            return self.workflow_service.create_workflow(system_type, config)
        else:
            # Fallback implementation
            return {
                "success": False,
                "error": "Domain workflow service not available"
            }

    def deploy_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Deploy a workflow for execution."""
        if self.workflow_service:
            return self.workflow_service.deploy_workflow(workflow_id)
        else:
            return {"success": False, "error": "Domain workflow service not available"}

    def execute_workflow(self, workflow_id: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a deployed workflow."""
        if self.workflow_service:
            return self.workflow_service.execute_workflow(workflow_id, inputs)
        else:
            return {"success": False, "error": "Domain workflow service not available"}

    def get_workflow_status(self, workflow_id: str) -> str:
        """Get current workflow execution status (backward compatible)."""
        if self.workflow_service:
            # Get WorkflowStatus enum and return its value
            status_enum = self.workflow_service.get_workflow_details(workflow_id)
            if status_enum and isinstance(status_enum, dict):
                return status_enum.get("current_status", "unknown")
        return "unknown"

    def pause_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Pause a running workflow."""
        if self.workflow_service:
            return self.workflow_service.pause_workflow(workflow_id)
        else:
            return {"success": False, "error": "Domain workflow service not available"}

    def resume_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Resume a paused workflow."""
        if self.workflow_service:
            return self.workflow_service.resume_workflow(workflow_id)
        else:
            return {"success": False, "error": "Domain workflow service not available"}

    def cancel_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Cancel a workflow execution."""
        if self.workflow_service:
            return self.workflow_service.cancel_workflow(workflow_id)
        else:
            return {"success": False, "error": "Domain workflow service not available"}

    # ==================== FLOW MACRO OPERATIONS ====================

    def execute_flow_macro(self, macro_command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a Flow Macro command."""
        if self.workflow_service:
            return self.workflow_service.execute_flow_macro(macro_command, context)
        else:
            return {"success": False, "error": "Domain workflow service not available"}

    def get_available_flow_macros(self) -> List[Dict[str, Any]]:
        """Get list of available Flow Macros."""
        if self.workflow_service:
            # Get standard flow macros
            standard_macros = self.workflow_service.get_available_flow_macros()

            # Get business signature tools
            business_tools = self.workflow_service.list_business_signature_tools()

            # Convert business tools to flow macro format
            business_macros = []
            for tool in business_tools:
                business_macros.append({
                    "name": f"BUSINESS_TOOL:{tool.get('signature_id', 'unknown')}",
                    "description": f"{tool.get('business_name', 'Business Tool')} - {tool.get('business_purpose', '')[:100]}",
                    "type": "business_signature",
                    "business_context": tool.get('business_context', ''),
                    "input_schema": tool.get('input_schema', {}),
                    "output_schema": tool.get('output_schema', {})
                })

            # Combine standard and business macros
            return standard_macros + business_macros
        else:
            # Fallback list
            return [
                {"name": "MVR_ANALYSIS", "description": "4-stage MVR document analysis"},
                {"name": "DOMAIN_RAG", "description": "Create domain-specific RAG systems"}
            ]

    # ==================== WORKFLOW MANAGEMENT ====================

    def list_workflows(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List workflows with optional filters."""
        if self.workflow_service:
            return self.workflow_service.list_workflows(filters)
        else:
            return []

    def get_workflow_details(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed workflow information."""
        if self.workflow_service:
            return self.workflow_service.get_workflow_details(workflow_id)
        else:
            return None

    def delete_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Delete a workflow."""
        if self.workflow_service:
            # Delegate to storage through domain service
            dependencies = self.workflow_service.dependencies
            storage_service = dependencies.get_storage_service()
            return storage_service.delete_workflow(workflow_id)
        else:
            return {"success": False, "error": "Domain workflow service not available"}

    def archive_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Archive a workflow."""
        if self.workflow_service:
            dependencies = self.workflow_service.dependencies
            storage_service = dependencies.get_storage_service()
            return storage_service.archive_workflow(workflow_id)
        else:
            return {"success": False, "error": "Domain workflow service not available"}

    # ==================== HEALTH AND MONITORING ====================

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        if self.workflow_service:
            return self.workflow_service.health_check()
        else:
            return {
                "overall_healthy": False,
                "error": "Domain workflow service not available",
                "components": {
                    "domain_service": {"healthy": False, "details": "Not available"}
                }
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get workflow system performance metrics."""
        if self.workflow_service:
            return self.workflow_service.get_performance_metrics()
        else:
            return {"error": "Domain workflow service not available"}

    # ==================== BACKWARD COMPATIBILITY METHODS ====================
    # These methods maintain compatibility with existing portal code

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Backward compatible method for getting workflow."""
        return self.get_workflow_details(workflow_id)

    def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatible method for updating workflow."""
        if self.workflow_service:
            dependencies = self.workflow_service.dependencies
            storage_service = dependencies.get_storage_service()

            # Load existing workflow
            workflow_data = storage_service.load_workflow(workflow_id)
            if not workflow_data:
                return {"success": False, "error": f"Workflow {workflow_id} not found"}

            # Apply updates
            workflow_data.update(updates)
            workflow_data["updated_at"] = datetime.now().isoformat()

            # Save updated workflow
            return storage_service.save_workflow(workflow_id, workflow_data)
        else:
            return {"success": False, "error": "Domain workflow service not available"}

    def get_workflows_by_type(self, system_type: str) -> List[Dict[str, Any]]:
        """Backward compatible method for getting workflows by type."""
        return self.list_workflows({"system_type": system_type})

    def get_running_workflows(self) -> List[Dict[str, Any]]:
        """Backward compatible method for getting running workflows."""
        return self.list_workflows({"status": "running"})

    # ==================== BUSINESS SIGNATURE INTEGRATION ====================

    def execute_business_tool_macro(self, tool_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a business tool through Flow Macro syntax."""
        if self.workflow_service:
            return self.workflow_service.execute_business_tool_macro(tool_id, inputs)
        else:
            return {"success": False, "error": "Domain workflow service not available"}

    def list_business_signature_tools(self) -> List[Dict[str, Any]]:
        """List all available business signature tools."""
        if self.workflow_service:
            return self.workflow_service.list_business_signature_tools()
        else:
            return []

    def create_business_signature_workflow(self, signature_id: str, business_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create and execute a workflow based on a business signature."""
        if self.workflow_service:
            return self.workflow_service.create_business_signature_workflow(signature_id, business_inputs)
        else:
            return {"success": False, "error": "Domain workflow service not available"}

    # ==================== LEGACY UNIFIED FLOW MANAGER COMPATIBILITY ====================
    # These properties maintain compatibility with code expecting UnifiedFlowManager

    @property
    def workflow_cache(self) -> Dict[str, Any]:
        """Backward compatible workflow cache access."""
        if self.workflow_service:
            dependencies = self.workflow_service.dependencies
            execution_service = dependencies.get_execution_service()
            if hasattr(execution_service, 'workflow_cache'):
                return execution_service.workflow_cache
        return {}

    @property
    def health_cache(self) -> Dict[str, Any]:
        """Backward compatible health cache access."""
        if self.workflow_service:
            dependencies = self.workflow_service.dependencies
            monitoring_service = dependencies.get_monitoring_service()
            if hasattr(monitoring_service, 'health_cache'):
                return monitoring_service.health_cache
        return {}

    @property
    def last_health_check(self) -> Dict[str, Any]:
        """Backward compatible last health check access."""
        if self.workflow_service:
            dependencies = self.workflow_service.dependencies
            monitoring_service = dependencies.get_monitoring_service()
            if hasattr(monitoring_service, 'last_health_check'):
                return monitoring_service.last_health_check
        return {}


# ==================== FACTORY FUNCTIONS ====================

def create_flow_orchestration_service(auto_load_credentials: bool = True) -> FlowOrchestrationService:
    """Factory function to create Flow Orchestration Service."""
    return FlowOrchestrationService(auto_load_credentials)


# Backward compatibility alias
UnifiedFlowManager = FlowOrchestrationService


def get_unified_flow_manager(auto_load_credentials: bool = True) -> FlowOrchestrationService:
    """
    Backward compatible factory function.
    Returns the new FlowOrchestrationService but maintains the old interface.
    """
    return FlowOrchestrationService(auto_load_credentials)