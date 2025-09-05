"""
MCP Layer Communication Contracts

Defines communication contracts and interfaces between MCP layers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum


class LayerType(Enum):
    """MCP layer types"""
    ORCHESTRATOR = "orchestrator"
    COORDINATOR = "coordinator"
    WORKER = "worker"
    TOOL = "tool"


class ContractType(Enum):
    """Contract types for layer communication"""
    TASK_DELEGATION = "task_delegation"
    RESULT_COLLECTION = "result_collection"
    STATUS_REPORTING = "status_reporting"
    CONTEXT_SHARING = "context_sharing"
    ERROR_HANDLING = "error_handling"


class LayerContract(ABC):
    """Base contract for MCP layer communication"""
    
    def __init__(self, layer_type: LayerType, contract_type: ContractType):
        self.layer_type = layer_type
        self.contract_type = contract_type
        self.required_fields: List[str] = []
        self.optional_fields: List[str] = []
        self.validation_rules: Dict[str, Any] = {}

    @abstractmethod
    def validate_contract(self, data: Dict[str, Any]) -> bool:
        """Validate contract data"""
        pass

    @abstractmethod
    def get_contract_schema(self) -> Dict[str, Any]:
        """Get contract schema definition"""
        pass


class TaskDelegationContract(LayerContract):
    """Contract for task delegation between layers"""
    
    def __init__(self, from_layer: LayerType, to_layer: LayerType):
        super().__init__(from_layer, ContractType.TASK_DELEGATION)
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.required_fields = ["task_id", "task_type", "task_data", "context", "constraints"]
        self.optional_fields = ["priority", "timeout", "retry_policy"]

    def validate_contract(self, data: Dict[str, Any]) -> bool:
        """Validate task delegation contract"""
        for field in self.required_fields:
            if field not in data:
                return False
        
        # Validate task_type is supported by target layer
        if not self._is_task_type_supported(data.get("task_type")):
            return False
            
        return True

    def get_contract_schema(self) -> Dict[str, Any]:
        """Get task delegation contract schema"""
        return {
            "contract_type": self.contract_type.value,
            "from_layer": self.from_layer.value,
            "to_layer": self.to_layer.value,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "validation_rules": {
                "task_type": "must be supported by target layer",
                "context": "must be valid context object",
                "constraints": "must be valid constraints object"
            }
        }

    def _is_task_type_supported(self, task_type: str) -> bool:
        """Check if task type is supported by target layer"""
        # This would be implemented based on layer capabilities
        supported_tasks = {
            LayerType.ORCHESTRATOR: ["plan_workflow", "allocate_resources", "coordinate_tasks"],
            LayerType.COORDINATOR: ["decompose_task", "assign_workers", "monitor_progress"],
            LayerType.WORKER: ["process_document", "analyze_content", "generate_report"],
            LayerType.TOOL: ["embed_text", "summarize_content", "check_compliance"]
        }
        return task_type in supported_tasks.get(self.to_layer, [])


class ResultCollectionContract(LayerContract):
    """Contract for result collection between layers"""
    
    def __init__(self, from_layer: LayerType, to_layer: LayerType):
        super().__init__(from_layer, ContractType.RESULT_COLLECTION)
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.required_fields = ["task_id", "result", "status", "metadata"]
        self.optional_fields = ["error_message", "quality_score", "processing_time"]

    def validate_contract(self, data: Dict[str, Any]) -> bool:
        """Validate result collection contract"""
        for field in self.required_fields:
            if field not in data:
                return False
        
        # Validate status is valid
        valid_statuses = ["success", "error", "partial", "timeout"]
        if data.get("status") not in valid_statuses:
            return False
            
        return True

    def get_contract_schema(self) -> Dict[str, Any]:
        """Get result collection contract schema"""
        return {
            "contract_type": self.contract_type.value,
            "from_layer": self.from_layer.value,
            "to_layer": self.to_layer.value,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "validation_rules": {
                "status": "must be one of: success, error, partial, timeout",
                "result": "must be valid result object",
                "metadata": "must be valid metadata object"
            }
        }


class StatusReportingContract(LayerContract):
    """Contract for status reporting between layers"""
    
    def __init__(self, from_layer: LayerType, to_layer: LayerType):
        super().__init__(from_layer, ContractType.STATUS_REPORTING)
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.required_fields = ["node_id", "status", "timestamp"]
        self.optional_fields = ["details", "metrics", "health_score"]

    def validate_contract(self, data: Dict[str, Any]) -> bool:
        """Validate status reporting contract"""
        for field in self.required_fields:
            if field not in data:
                return False
        
        # Validate status is valid
        valid_statuses = ["healthy", "degraded", "unhealthy", "offline"]
        if data.get("status") not in valid_statuses:
            return False
            
        return True

    def get_contract_schema(self) -> Dict[str, Any]:
        """Get status reporting contract schema"""
        return {
            "contract_type": self.contract_type.value,
            "from_layer": self.from_layer.value,
            "to_layer": self.to_layer.value,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "validation_rules": {
                "status": "must be one of: healthy, degraded, unhealthy, offline",
                "timestamp": "must be valid ISO timestamp",
                "details": "must be valid details object"
            }
        }


class ContextSharingContract(LayerContract):
    """Contract for context sharing between layers"""
    
    def __init__(self, from_layer: LayerType, to_layer: LayerType):
        super().__init__(from_layer, ContractType.CONTEXT_SHARING)
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.required_fields = ["context_id", "context_data", "update_type"]
        self.optional_fields = ["source_layer", "metadata", "expiry_time"]

    def validate_contract(self, data: Dict[str, Any]) -> bool:
        """Validate context sharing contract"""
        for field in self.required_fields:
            if field not in data:
                return False
        
        # Validate update_type is valid
        valid_update_types = ["add", "modify", "remove", "replace"]
        if data.get("update_type") not in valid_update_types:
            return False
            
        return True

    def get_contract_schema(self) -> Dict[str, Any]:
        """Get context sharing contract schema"""
        return {
            "contract_type": self.contract_type.value,
            "from_layer": self.from_layer.value,
            "to_layer": self.to_layer.value,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "validation_rules": {
                "update_type": "must be one of: add, modify, remove, replace",
                "context_data": "must be valid context data object",
                "metadata": "must be valid metadata object"
            }
        }


class ErrorHandlingContract(LayerContract):
    """Contract for error handling between layers"""
    
    def __init__(self, from_layer: LayerType, to_layer: LayerType):
        super().__init__(from_layer, ContractType.ERROR_HANDLING)
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.required_fields = ["error_type", "error_message", "timestamp"]
        self.optional_fields = ["error_code", "stack_trace", "recovery_suggestion"]

    def validate_contract(self, data: Dict[str, Any]) -> bool:
        """Validate error handling contract"""
        for field in self.required_fields:
            if field not in data:
                return False
        
        # Validate error_type is valid
        valid_error_types = ["validation_error", "processing_error", "timeout_error", "system_error"]
        if data.get("error_type") not in valid_error_types:
            return False
            
        return True

    def get_contract_schema(self) -> Dict[str, Any]:
        """Get error handling contract schema"""
        return {
            "contract_type": self.contract_type.value,
            "from_layer": self.from_layer.value,
            "to_layer": self.to_layer.value,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "validation_rules": {
                "error_type": "must be one of: validation_error, processing_error, timeout_error, system_error",
                "error_message": "must be non-empty string",
                "timestamp": "must be valid ISO timestamp"
            }
        }


class ContractRegistry:
    """Registry for managing layer communication contracts"""
    
    def __init__(self):
        self.contracts: Dict[str, LayerContract] = {}

    def register_contract(self, contract: LayerContract, contract_id: str):
        """Register a contract"""
        self.contracts[contract_id] = contract

    def get_contract(self, contract_id: str) -> Optional[LayerContract]:
        """Get a contract by ID"""
        return self.contracts.get(contract_id)

    def validate_message(self, contract_id: str, data: Dict[str, Any]) -> bool:
        """Validate message against contract"""
        contract = self.get_contract(contract_id)
        if not contract:
            return False
        return contract.validate_contract(data)

    def get_contract_schema(self, contract_id: str) -> Optional[Dict[str, Any]]:
        """Get contract schema by ID"""
        contract = self.get_contract(contract_id)
        if not contract:
            return None
        return contract.get_contract_schema()
