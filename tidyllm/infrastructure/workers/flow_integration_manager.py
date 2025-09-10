# Flow Integration Manager - Bridge between FLOW System and AI Dropzone Manager
# 🔄 FLOW INTEGRATION LAYER - Bracket Command to Template Processing Bridge
#
# INTEGRATION SCOPE:
# ==================
# This component bridges the existing FLOW (Flexible Logic Operations Workflows) 
# bracket command system with the new AI Dropzone Manager architecture:
#
# FLOW System Integration:
# - FlowAgreement bracket commands: [Command Name]
# - QA Control Flows: [Process MVR], [Check MVS Compliance]
# - Existing workflow definitions in tidyllm/flow/
# - Drop zone staged processing with purgatory recovery
#
# AI Dropzone Manager Integration:
# - Template-based processing in prompts/templates/
# - Worker orchestration through approved registry
# - Security-controlled LLM operations via CorporateLLMGateway
# - Quality monitoring and feedback loops
#
# SECURITY CONSTRAINTS:
# ====================
# 1. BRACKET COMMAND VALIDATION: Only approved bracket commands from flow registry
# 2. TEMPLATE MAPPING: Flow commands must map to approved templates only
# 3. NO DYNAMIC EXECUTION: No arbitrary code execution from bracket commands
# 4. AUDIT TRAIL: All flow-to-template mappings logged and auditable
# 5. WORKER REGISTRY: Only approved workers execute flow-mapped processes
#
# Dependencies:
# - AI Dropzone Manager for orchestration
# - FLOW system (tidyllm.flow) for bracket command parsing
# - CorporateLLMGateway for all LLM operations
# - Approved template library in prompts/templates/

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum

try:
    from ...tidyllm.flow import FlowAgreement, FlowAgreementManager, execute_flow_command
    FLOW_SYSTEM_AVAILABLE = True
except ImportError:
    try:
        from tidyllm.flow import FlowAgreement, FlowAgreementManager, execute_flow_command
        FLOW_SYSTEM_AVAILABLE = True
    except ImportError:
        logger.warning("FLOW system not available - bracket command integration disabled")
        FLOW_SYSTEM_AVAILABLE = False

from .ai_dropzone_manager import AIDropzoneManager, AIManagerTask, ProcessingStrategy, DocumentComplexity
from .base_worker import BaseWorker, TaskInput, TaskResult

logger = logging.getLogger(__name__)

@dataclass
class FlowToTemplateMapping:
    """Maps FLOW bracket commands to approved processing templates."""
    bracket_command: str  # e.g., "[Process MVR]"
    flow_encoding: str    # e.g., "@mvr#process!extract@compliance_data"
    template_names: List[str]  # Mapped to approved templates
    processing_strategy: ProcessingStrategy
    priority_level: str   # critical, high, normal, low
    validation_rules: List[str]  # Additional validation requirements
    
@dataclass
class FlowIntegrationTask:
    bracket_command: str
    document_path: str
    flow_context: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None

@dataclass  
class FlowIntegrationResult:
    bracket_command: str
    mapped_templates: List[str]
    ai_manager_result: Any
    flow_execution_log: Dict[str, Any]

class FlowIntegrationManager(BaseWorker[FlowIntegrationTask, FlowIntegrationResult]):
    """
    Flow Integration Manager - Bridge between FLOW bracket commands and AI Dropzone Manager.
    
    SECURITY: This manager enforces strict validation of bracket commands and only
    allows execution of approved flow-to-template mappings.
    
    This manager provides:
    1. Bracket Command Validation - validates commands against approved registry
    2. Flow-to-Template Mapping - maps bracket commands to approved processing templates
    3. AI Dropzone Integration - routes validated flows to AI Dropzone Manager
    4. Staged Processing Support - handles drop zone file movement and purgatory recovery
    """
    
    def __init__(
        self,
        ai_dropzone_manager: Optional[AIDropzoneManager] = None,
        flow_manager: Optional[FlowAgreementManager] = None,
        mapping_config_path: str = "C:/Users/marti/github/prompts/flow_mappings.json"
    ):
        super().__init__(worker_name="flow_integration_manager")
        
        # Core component integration
        self.ai_dropzone_manager = ai_dropzone_manager or AIDropzoneManager()
        
        # FLOW system integration
        if FLOW_SYSTEM_AVAILABLE:
            self.flow_manager = flow_manager or FlowAgreementManager()
        else:
            self.flow_manager = None
            logger.warning("FLOW system not available - bracket command processing disabled")
        
        # SECURITY: Flow-to-template mapping registry
        self.mapping_config_path = Path(mapping_config_path)
        self.approved_flow_mappings: Dict[str, FlowToTemplateMapping] = {}
        self.bracket_command_registry: Set[str] = set()
        
        # Drop zone staged processing
        self.drop_zones = {
            "input": Path("drop_zones/input"),
            "processing": Path("drop_zones/processing"), 
            "completed": Path("drop_zones/completed"),
            "purgatory": Path("drop_zones/purgatory"),
            "failed": Path("drop_zones/failed")
        }
        
        # Audit and monitoring
        self.flow_execution_history: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize Flow Integration Manager with security validation."""
        await super().initialize()
        
        # Initialize AI Dropzone Manager
        await self.ai_dropzone_manager.initialize()
        
        # SECURITY: Load and validate approved flow mappings
        await self._load_flow_mappings()
        
        # Validate drop zone structure
        await self._initialize_drop_zones()
        
        logger.info("Flow Integration Manager initialized successfully")
        logger.info(f"[SECURITY] Approved bracket commands: {len(self.bracket_command_registry)}")
    
    def validate_input(self, task_input: FlowIntegrationTask) -> bool:
        """
        Validate input task for Flow Integration Manager.
        
        Args:
            task_input: FlowIntegrationTask to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check if task_input is the correct type
            if not isinstance(task_input, FlowIntegrationTask):
                logger.error(f"Invalid task input type: {type(task_input)}")
                return False
            
            # Validate required fields
            if not task_input.bracket_command:
                logger.error("Bracket command is required")
                return False
            
            if not task_input.document_path:
                logger.error("Document path is required")
                return False
            
            # Validate bracket command format
            if not task_input.bracket_command.startswith('[') or not task_input.bracket_command.endswith(']'):
                logger.error(f"Invalid bracket command format: {task_input.bracket_command}")
                return False
            
            # Validate bracket command is in approved registry
            if task_input.bracket_command not in self.bracket_command_registry:
                logger.error(f"Bracket command not in approved registry: {task_input.bracket_command}")
                return False
            
            logger.debug(f"Task input validation successful for: {task_input.bracket_command}")
            return True
            
        except Exception as e:
            logger.error(f"Task input validation failed: {e}")
            return False
    
    async def process_task(self, task: FlowIntegrationTask) -> FlowIntegrationResult:
        """
        Process bracket command through flow-to-template integration.
        """
        try:
            logger.info(f"Processing bracket command: {task.bracket_command}")
            
            # SECURITY: Validate bracket command is approved
            if not await self._validate_bracket_command(task.bracket_command):
                raise SecurityError(f"Unauthorized bracket command: {task.bracket_command}")
            
            # Get flow mapping
            mapping = self.approved_flow_mappings.get(task.bracket_command)
            if not mapping:
                raise ValueError(f"No approved mapping found for: {task.bracket_command}")
            
            # Create AI Manager task from flow mapping
            ai_manager_task = AIManagerTask(
                document_path=task.document_path,
                user_context={
                    **(task.user_context or {}),
                    "flow_command": task.bracket_command,
                    "flow_encoding": mapping.flow_encoding,
                    "flow_templates": mapping.template_names
                },
                business_priority=mapping.priority_level
            )
            
            # Execute through AI Dropzone Manager
            ai_result = await self.ai_dropzone_manager.process_task(ai_manager_task)
            
            # Log flow execution
            execution_log = await self._log_flow_execution(task, mapping, ai_result)
            
            # Handle staged processing if needed
            await self._handle_staged_processing(task.document_path, ai_result.success)
            
            return FlowIntegrationResult(
                bracket_command=task.bracket_command,
                mapped_templates=mapping.template_names,
                ai_manager_result=ai_result,
                flow_execution_log=execution_log,
                success=ai_result.success
            )
            
        except Exception as e:
            logger.error(f"Flow integration failed for {task.bracket_command}: {e}")
            return FlowIntegrationResult(
                bracket_command=task.bracket_command,
                mapped_templates=[],
                ai_manager_result=None,
                flow_execution_log={"error": str(e)},
                success=False,
                error=str(e)
            )
    
    async def _validate_bracket_command(self, bracket_command: str) -> bool:
        """
        SECURITY: Validate bracket command against approved registry.
        """
        try:
            # Check if command is in approved registry
            if bracket_command not in self.bracket_command_registry:
                logger.warning(f"[SECURITY] Unauthorized bracket command rejected: {bracket_command}")
                return False
            
            # Validate bracket syntax
            if not (bracket_command.startswith('[') and bracket_command.endswith(']')):
                logger.warning(f"[SECURITY] Invalid bracket syntax: {bracket_command}")
                return False
            
            # Check for injection patterns
            dangerous_patterns = ['exec', 'eval', 'import', '__', 'system']
            command_lower = bracket_command.lower()
            if any(pattern in command_lower for pattern in dangerous_patterns):
                logger.warning(f"[SECURITY] Dangerous pattern detected in command: {bracket_command}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[SECURITY] Bracket command validation failed: {e}")
            return False
    
    async def _load_flow_mappings(self):
        """
        SECURITY: Load approved flow-to-template mappings from configuration.
        """
        try:
            # Try to load from configuration file
            if self.mapping_config_path.exists():
                with open(self.mapping_config_path, 'r') as f:
                    mapping_data = json.load(f)
                self._parse_flow_mappings(mapping_data)
            else:
                # Load default mappings
                await self._load_default_flow_mappings()
            
            logger.info(f"[SECURITY] Loaded {len(self.approved_flow_mappings)} approved flow mappings")
            
        except Exception as e:
            logger.warning(f"Failed to load flow mappings: {e}, using defaults")
            await self._load_default_flow_mappings()
    
    async def _load_default_flow_mappings(self):
        """
        Load default flow-to-template mappings for common operations.
        """
        default_mappings = {
            "[Process MVR]": FlowToTemplateMapping(
                bracket_command="[Process MVR]",
                flow_encoding="@mvr#process!extract@compliance_data",
                template_names=["mvr_analysis", "qa_control"],
                processing_strategy=ProcessingStrategy.MULTI_PERSPECTIVE,
                priority_level="high",
                validation_rules=["mvr_document_type", "compliance_standards"]
            ),
            
            "[Financial Analysis]": FlowToTemplateMapping(
                bracket_command="[Financial Analysis]",
                flow_encoding="@financial#analysis!assess@risk_metrics",
                template_names=["financial_analysis", "qa_control"],
                processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
                priority_level="normal",
                validation_rules=["financial_document_type"]
            ),
            
            "[Contract Review]": FlowToTemplateMapping(
                bracket_command="[Contract Review]",
                flow_encoding="@contract#review!validate@legal_terms",
                template_names=["contract_analysis", "compliance_review"],
                processing_strategy=ProcessingStrategy.MULTI_PERSPECTIVE,
                priority_level="high",
                validation_rules=["legal_document_type", "contract_complexity"]
            ),
            
            "[Quality Check]": FlowToTemplateMapping(
                bracket_command="[Quality Check]",
                flow_encoding="@quality#check!validate@standards",
                template_names=["qa_control"],
                processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
                priority_level="normal",
                validation_rules=["quality_standards"]
            ),
            
            "[Peer Review]": FlowToTemplateMapping(
                bracket_command="[Peer Review]",
                flow_encoding="@peer#review!validate@expert_opinion",
                template_names=["peer_review", "qa_control"],
                processing_strategy=ProcessingStrategy.MULTI_PERSPECTIVE,
                priority_level="critical",
                validation_rules=["expert_review_required"]
            ),
            
            "[Compliance Check]": FlowToTemplateMapping(
                bracket_command="[Compliance Check]",
                flow_encoding="@compliance#check!validate@regulations",
                template_names=["compliance_review", "qa_control"],
                processing_strategy=ProcessingStrategy.MULTI_PERSPECTIVE,
                priority_level="high",
                validation_rules=["regulatory_requirements"]
            ),
            
            "[Data Extraction]": FlowToTemplateMapping(
                bracket_command="[Data Extraction]",
                flow_encoding="@data#extraction!extract@structured_data",
                template_names=["data_extraction"],
                processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
                priority_level="normal",
                validation_rules=["data_structure_validation"]
            ),
            
            "[Hybrid Analysis]": FlowToTemplateMapping(
                bracket_command="[Hybrid Analysis]",
                flow_encoding="@hybrid#analysis!synthesize@multi_framework",
                template_names=["hybrid_analysis", "qa_control"],
                processing_strategy=ProcessingStrategy.HYBRID_ANALYSIS,
                priority_level="high",
                validation_rules=["multi_framework_applicable"]
            )
        }
        
        self.approved_flow_mappings = default_mappings
        self.bracket_command_registry = set(default_mappings.keys())
        
        # Save default mappings for future reference
        await self._save_flow_mappings()
    
    def _parse_flow_mappings(self, mapping_data: Dict[str, Any]):
        """Parse flow mappings from configuration data."""
        for command, data in mapping_data.items():
            try:
                mapping = FlowToTemplateMapping(
                    bracket_command=command,
                    flow_encoding=data.get("flow_encoding", ""),
                    template_names=data.get("template_names", []),
                    processing_strategy=ProcessingStrategy(data.get("processing_strategy", "single_template")),
                    priority_level=data.get("priority_level", "normal"),
                    validation_rules=data.get("validation_rules", [])
                )
                
                self.approved_flow_mappings[command] = mapping
                self.bracket_command_registry.add(command)
                
            except Exception as e:
                logger.warning(f"Failed to parse mapping for {command}: {e}")
    
    async def _save_flow_mappings(self):
        """Save current flow mappings to configuration file."""
        try:
            mapping_data = {}
            for command, mapping in self.approved_flow_mappings.items():
                mapping_data[command] = {
                    "flow_encoding": mapping.flow_encoding,
                    "template_names": mapping.template_names,
                    "processing_strategy": mapping.processing_strategy.value,
                    "priority_level": mapping.priority_level,
                    "validation_rules": mapping.validation_rules
                }
            
            self.mapping_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.mapping_config_path, 'w') as f:
                json.dump(mapping_data, f, indent=2)
                
            logger.debug("Flow mappings saved to configuration file")
            
        except Exception as e:
            logger.warning(f"Failed to save flow mappings: {e}")
    
    async def _initialize_drop_zones(self):
        """Initialize drop zone directories for staged processing."""
        try:
            for zone_name, zone_path in self.drop_zones.items():
                zone_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Initialized drop zone: {zone_name} -> {zone_path}")
            
            logger.info("Drop zones initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize drop zones: {e}")
    
    async def _handle_staged_processing(self, document_path: str, processing_success: bool):
        """
        Handle drop zone staged processing - move documents based on processing results.
        """
        try:
            doc_path = Path(document_path)
            
            if processing_success:
                # Move to completed zone
                target_path = self.drop_zones["completed"] / doc_path.name
                if doc_path.exists():
                    doc_path.rename(target_path)
                    logger.info(f"Document moved to completed zone: {doc_path.name}")
            else:
                # Move to failed zone for manual review
                target_path = self.drop_zones["failed"] / doc_path.name
                if doc_path.exists():
                    doc_path.rename(target_path)
                    logger.warning(f"Failed document moved to failed zone: {doc_path.name}")
                    
        except Exception as e:
            logger.error(f"Failed to handle staged processing: {e}")
    
    async def _log_flow_execution(
        self, 
        task: FlowIntegrationTask, 
        mapping: FlowToTemplateMapping, 
        ai_result: Any
    ) -> Dict[str, Any]:
        """Log flow execution for audit trail."""
        execution_log = {
            "timestamp": datetime.now().isoformat(),
            "bracket_command": task.bracket_command,
            "flow_encoding": mapping.flow_encoding,
            "document_path": task.document_path,
            "mapped_templates": mapping.template_names,
            "processing_strategy": mapping.processing_strategy.value,
            "priority_level": mapping.priority_level,
            "success": ai_result.success if ai_result else False,
            "user_context": task.user_context,
            "flow_context": task.flow_context
        }
        
        self.flow_execution_history.append(execution_log)
        
        # Limit history size
        if len(self.flow_execution_history) > 1000:
            self.flow_execution_history = self.flow_execution_history[-500:]
        
        return execution_log
    
    async def execute_bracket_command(
        self, 
        bracket_command: str, 
        document_path: str, 
        **kwargs
    ) -> FlowIntegrationResult:
        """
        Convenience method to execute bracket commands directly.
        """
        task = FlowIntegrationTask(
            bracket_command=bracket_command,
            document_path=document_path,
            user_context=kwargs.get('user_context'),
            flow_context=kwargs.get('flow_context')
        )
        
        return await self.process_task(task)
    
    def get_available_bracket_commands(self) -> List[str]:
        """Get list of available bracket commands."""
        return list(self.bracket_command_registry)
    
    def get_flow_mapping(self, bracket_command: str) -> Optional[FlowToTemplateMapping]:
        """Get flow mapping for a specific bracket command."""
        return self.approved_flow_mappings.get(bracket_command)
    
    async def register_flow_mapping(
        self, 
        bracket_command: str, 
        mapping: FlowToTemplateMapping
    ) -> bool:
        """
        SECURITY: Register a new flow mapping (requires validation).
        """
        try:
            # Validate bracket command format
            if not await self._validate_bracket_command(bracket_command):
                return False
            
            # Validate templates exist in approved library
            available_templates = set(self.ai_dropzone_manager.available_templates.keys())
            invalid_templates = set(mapping.template_names) - available_templates
            
            if invalid_templates:
                logger.error(f"[SECURITY] Invalid templates in mapping: {invalid_templates}")
                return False
            
            # Register the mapping
            self.approved_flow_mappings[bracket_command] = mapping
            self.bracket_command_registry.add(bracket_command)
            
            # Save updated mappings
            await self._save_flow_mappings()
            
            logger.info(f"[SECURITY] Flow mapping registered: {bracket_command}")
            return True
            
        except Exception as e:
            logger.error(f"[SECURITY] Flow mapping registration failed: {e}")
            return False
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status for monitoring."""
        return {
            "flow_system_available": FLOW_SYSTEM_AVAILABLE,
            "ai_dropzone_manager_status": "active" if self.ai_dropzone_manager else "unavailable",
            "approved_bracket_commands": len(self.bracket_command_registry),
            "available_templates": len(self.ai_dropzone_manager.available_templates) if self.ai_dropzone_manager else 0,
            "drop_zones_initialized": all(zone.exists() for zone in self.drop_zones.values()),
            "execution_history_count": len(self.flow_execution_history),
            "recent_commands": [log["bracket_command"] for log in self.flow_execution_history[-5:]]
        }