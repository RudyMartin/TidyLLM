"""
MCP Worker - Operational Execution Layer

The Worker is the bottom-level node in the MCP hierarchy that executes specific tasks
using specialized capabilities and tools.
"""

from typing import Any, Dict, Optional
from .base import MCPNode, MCPRole, MCPContext, MCPProtocol
from .protocol import MCPMessageProtocol, MCPMessage
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Worker(MCPNode):
    """Bottom-level worker that executes specific tasks"""
    
    def __init__(self, name: str, specialization: str, model_config: Dict[str, Any]):
        super().__init__(MCPRole.WORKER, model_config)
        self.name = name
        self.specialization = specialization
        self.task_history = []
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{name}")
    
    def _define_protocol(self) -> MCPProtocol:
        """Define the protocol for worker communication"""
        return MCPProtocol(
            message_format="json",
            validation_schema={
                "type": "object",
                "required": ["task"],
                "properties": {
                    "task": {"type": "string"},
                    "input_data": {"type": "object"},
                    "constraints": {"type": "object"},
                    "priority": {"type": "string"}
                }
            },
            error_handling={
                "retry_count": 1,
                "fallback_strategy": "simple_response",
                "error_threshold": 0.7
            },
            retry_policy={
                "max_retries": 0,
                "backoff_factor": 1.0,
                "retry_delay": 0.0
            }
        )
    
    def process(self, context: MCPContext, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific worker task"""
        
        self.logger.info(f"Processing task: {task.get('task', 'Unknown')[:100]}...")
        
        try:
            # Validate task against worker specialization
            if not self._can_handle_task(task):
                raise ValueError(f"Worker {self.name} cannot handle task: {task.get('task', 'Unknown')}")
            
            # Validate task against protocol
            if not self.validate_input(task):
                raise ValueError("Invalid task format")
            
            # Execute task using LLM
            execution_start = datetime.now()
            execution_prompt = self._create_execution_prompt(task, context)
            result = self.llm_manager.generate_response(execution_prompt)
            execution_duration = (datetime.now() - execution_start).total_seconds()
            
            # Format result according to protocol
            formatted_result = self._format_result(result, task, execution_duration)
            
            # Record task execution
            self.task_history.append({
                "task": task,
                "result": formatted_result,
                "execution_duration": execution_duration,
                "timestamp": datetime.now().isoformat()
            })
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"Error in worker process: {e}")
            return self.handle_error(e, context)
    
    def _can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Check if worker can handle the given task"""
        task_description = task.get("task", "").lower()
        specialization = self.specialization.lower()
        
        # Simple keyword matching - can be enhanced with more sophisticated logic
        specialization_keywords = specialization.split()
        task_keywords = task_description.split()
        
        # Check for keyword overlap
        overlap = set(specialization_keywords) & set(task_keywords)
        
        # Check for semantic similarity
        semantic_matches = [
            ("retriev" in specialization and any(word in task_description for word in ["find", "search", "retrieve", "get"])),
            ("analyz" in specialization and any(word in task_description for word in ["analyze", "examine", "study", "evaluate"])),
            ("writ" in specialization and any(word in task_description for word in ["write", "generate", "create", "compose"])),
            ("summariz" in specialization and any(word in task_description for word in ["summarize", "summarise", "brief", "overview"])),
            ("classif" in specialization and any(word in task_description for word in ["classify", "categorize", "sort", "organize"]))
        ]
        
        return len(overlap) > 0 or any(semantic_matches)
    
    def _create_execution_prompt(self, task: Dict[str, Any], context: MCPContext) -> str:
        """Create execution prompt for the worker"""
        
        task_description = task.get("task", "")
        input_data = task.get("input_data", {})
        constraints = task.get("constraints", {})
        
        prompt = f"""
        As a {self.specialization} worker, execute this specific task:
        
        TASK: {task_description}
        
        INPUT DATA: {json.dumps(input_data, indent=2)}
        
        CONSTRAINTS: {json.dumps(constraints, indent=2)}
        
        CONTEXT: {json.dumps(context.session_data, indent=2)}
        
        EXECUTION INSTRUCTIONS:
        1. Focus specifically on the task description
        2. Use the provided input data effectively
        3. Respect all constraints and requirements
        4. Provide a high-quality, focused result
        5. Ensure the result is relevant to the task
        
        Execute the task and provide a clear, well-structured response.
        """
        
        return prompt
    
    def _format_result(self, result: str, task: Dict[str, Any], execution_duration: float) -> Dict[str, Any]:
        """Format result according to worker protocol"""
        
        # Extract key information from result
        result_metadata = self._extract_result_metadata(result, task)
        
        return {
            "worker": self.name,
            "specialization": self.specialization,
            "task": task.get("task"),
            "success": True,
            "result": result,
            "execution_metadata": {
                "duration": execution_duration,
                "timestamp": datetime.now().isoformat(),
                "confidence": result_metadata.get("confidence", "medium"),
                "result_type": result_metadata.get("result_type", "text"),
                "word_count": len(result.split()),
                "quality_score": result_metadata.get("quality_score", 0.8)
            },
            "task_metadata": {
                "input_data_keys": list(task.get("input_data", {}).keys()),
                "constraints_count": len(task.get("constraints", {})),
                "priority": task.get("priority", "medium")
            }
        }
    
    def _extract_result_metadata(self, result: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from the result"""
        metadata = {
            "confidence": "medium",
            "result_type": "text",
            "quality_score": 0.8
        }
        
        # Analyze result length and structure
        word_count = len(result.split())
        char_count = len(result)
        
        # Adjust confidence based on result characteristics
        if word_count > 100:
            metadata["confidence"] = "high"
            metadata["quality_score"] = 0.9
        elif word_count > 50:
            metadata["confidence"] = "medium"
            metadata["quality_score"] = 0.8
        else:
            metadata["confidence"] = "low"
            metadata["quality_score"] = 0.6
        
        # Detect result type
        if any(keyword in result.lower() for keyword in ["json", "xml", "yaml"]):
            metadata["result_type"] = "structured"
        elif any(keyword in result.lower() for keyword in ["list:", "1.", "2.", "3."]):
            metadata["result_type"] = "list"
        elif any(keyword in result.lower() for keyword in ["summary", "overview", "brief"]):
            metadata["result_type"] = "summary"
        
        return metadata
    
    def get_task_history(self, limit: int = 10) -> list:
        """Get recent task execution history"""
        return self.task_history[-limit:] if self.task_history else []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get worker performance metrics"""
        if not self.task_history:
            return {
                "total_tasks": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "specialization": self.specialization
            }
        
        total_tasks = len(self.task_history)
        successful_tasks = len([t for t in self.task_history if t["result"].get("success", True)])
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        durations = [t["execution_duration"] for t in self.task_history]
        average_duration = sum(durations) / len(durations) if durations else 0.0
        
        return {
            "total_tasks": total_tasks,
            "success_rate": success_rate,
            "average_duration": average_duration,
            "specialization": self.specialization,
            "last_execution": self.task_history[-1]["timestamp"] if self.task_history else None
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get worker capabilities and specializations"""
        return {
            "name": self.name,
            "specialization": self.specialization,
            "capabilities": self._get_specialization_capabilities(),
            "protocol": {
                "message_format": self.protocol.message_format,
                "retry_policy": self.protocol.retry_policy
            }
        }
    
    def _get_specialization_capabilities(self) -> list:
        """Get capabilities based on specialization"""
        capabilities = []
        
        specialization_lower = self.specialization.lower()
        
        if "retriev" in specialization_lower:
            capabilities.extend([
                "Document search and retrieval",
                "Information gathering",
                "Vector similarity search",
                "Content indexing"
            ])
        
        if "analyz" in specialization_lower:
            capabilities.extend([
                "Content analysis",
                "Sentiment analysis",
                "Key point extraction",
                "Data processing",
                "Pattern recognition"
            ])
        
        if "writ" in specialization_lower:
            capabilities.extend([
                "Content generation",
                "Report writing",
                "Text composition",
                "Document creation"
            ])
        
        if "summariz" in specialization_lower:
            capabilities.extend([
                "Text summarization",
                "Content condensation",
                "Key information extraction",
                "Brief generation"
            ])
        
        if "classif" in specialization_lower:
            capabilities.extend([
                "Content classification",
                "Categorization",
                "Tagging",
                "Organization"
            ])
        
        return capabilities if capabilities else ["General task execution"]
