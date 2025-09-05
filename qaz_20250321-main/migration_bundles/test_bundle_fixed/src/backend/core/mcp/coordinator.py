"""
MCP Coordinator - Tactical Coordination Layer

The Coordinator is the mid-level node in the MCP hierarchy that manages specific domains
and coordinates the execution of tasks through specialized workers.
"""

from typing import List, Dict, Any, Optional
from .base import MCPNode, MCPRole, MCPContext, MCPProtocol
from .protocol import MCPMessageProtocol, MCPMessage
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Coordinator(MCPNode):
    """Mid-level coordinator that manages specific domains"""
    
    def __init__(self, name: str, domain: str, model_config: Dict[str, Any]):
        super().__init__(MCPRole.COORDINATOR, model_config)
        self.name = name
        self.domain = domain
        self.workers = {}
        self.task_history = []
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{name}")
    
    def _define_protocol(self) -> MCPProtocol:
        """Define the protocol for coordinator communication"""
        return MCPProtocol(
            message_format="json",
            validation_schema={
                "type": "object",
                "required": ["task"],
                "properties": {
                    "task": {"type": "string"},
                    "priority": {"type": "string"},
                    "constraints": {"type": "object"},
                    "input_data": {"type": "object"},
                    "dependencies": {"type": "array"}
                }
            },
            error_handling={
                "retry_count": 2,
                "fallback_strategy": "worker_fallback",
                "error_threshold": 0.5
            },
            retry_policy={
                "max_retries": 1,
                "backoff_factor": 1.0,
                "retry_delay": 0.5
            }
        )
    
    def process(self, context: MCPContext, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process task by delegating to appropriate workers"""
        
        self.logger.info(f"Processing task: {task.get('task', 'Unknown')[:100]}...")
        
        try:
            # Validate task against protocol
            if not self.validate_input(task):
                raise ValueError("Invalid task format")
            
            # Analyze task and determine required workers
            worker_tasks = self._decompose_task(task, context)
            
            if not worker_tasks:
                raise ValueError("No worker tasks generated from task decomposition")
            
            # Execute worker tasks
            worker_results = {}
            execution_start = datetime.now()
            
            for worker_name, worker_task in worker_tasks.items():
                if worker_name in self.workers:
                    worker = self.workers[worker_name]
                    
                    # Create worker-specific context
                    worker_context = context.enrich({
                        "task": worker_task,
                        "coordinator": self.name,
                        "domain": self.domain
                    })
                    
                    # Execute worker task
                    task_start = datetime.now()
                    result = worker.process(worker_context, worker_task)
                    task_duration = (datetime.now() - task_start).total_seconds()
                    
                    # Add execution metadata
                    result["worker_metadata"] = {
                        "worker_name": worker_name,
                        "coordinator": self.name,
                        "duration": task_duration,
                        "start_time": task_start.isoformat(),
                        "end_time": datetime.now().isoformat()
                    }
                    
                    worker_results[worker_name] = result
                    
                else:
                    self.logger.warning(f"Worker '{worker_name}' not found")
                    worker_results[worker_name] = {
                        "error": f"Worker '{worker_name}' not registered",
                        "success": False
                    }
            
            # Synthesize worker results
            final_result = self._synthesize_results(worker_results, task, context)
            
            # Record task execution
            execution_duration = (datetime.now() - execution_start).total_seconds()
            self.task_history.append({
                "task": task,
                "worker_results": worker_results,
                "final_result": final_result,
                "execution_duration": execution_duration,
                "timestamp": datetime.now().isoformat()
            })
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in coordinator process: {e}")
            return self.handle_error(e, context)
    
    def _decompose_task(self, task: Dict[str, Any], context: MCPContext) -> Dict[str, Any]:
        """Decompose coordinator task into worker tasks"""
        
        task_description = task.get("task", "")
        available_workers = list(self.workers.keys())
        
        decomposition_prompt = f"""
        As a {self.domain} coordinator, decompose this task into worker tasks:
        
        TASK: {task_description}
        TASK CONSTRAINTS: {json.dumps(task.get('constraints', {}), indent=2)}
        INPUT DATA: {json.dumps(task.get('input_data', {}), indent=2)}
        
        AVAILABLE WORKERS: {available_workers}
        
        WORKER CAPABILITIES:
        """
        
        # Add worker descriptions based on domain
        worker_descriptions = self._get_worker_descriptions()
        for worker_name in available_workers:
            description = worker_descriptions.get(worker_name, "Specialized worker")
            decomposition_prompt += f"- {worker_name}: {description}\n"
        
        decomposition_prompt += f"""
        DECOMPOSITION INSTRUCTIONS:
        1. Break down the task into logical subtasks
        2. Assign each subtask to the most appropriate worker
        3. Consider task dependencies and order
        4. Ensure all aspects of the task are covered
        
        Create a JSON structure mapping worker names to their specific tasks:
        {{
            "worker_name": {{
                "task": "specific worker task description",
                "priority": "high/medium/low",
                "input_data": {{}},
                "constraints": {{}}
            }}
        }}
        
        Ensure the decomposition is logical and executable.
        """
        
        try:
            decomposition_response = self.llm_manager.generate_response(decomposition_prompt)
            return self._parse_decomposition(decomposition_response)
            
        except Exception as e:
            self.logger.error(f"Task decomposition error: {e}")
            # Fallback to simple decomposition
            return self._simple_decomposition(task, available_workers)
    
    def _parse_decomposition(self, decomposition_response: str) -> Dict[str, Any]:
        """Parse LLM response into worker task decomposition"""
        try:
            # Extract JSON from response
            json_start = decomposition_response.find('{')
            json_end = decomposition_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in decomposition response")
            
            decomposition_json = decomposition_response[json_start:json_end]
            decomposition = json.loads(decomposition_json)
            
            return decomposition
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error in decomposition: {e}")
            raise ValueError(f"Failed to parse decomposition JSON: {e}")
        except Exception as e:
            self.logger.error(f"Decomposition parsing error: {e}")
            raise ValueError(f"Failed to parse decomposition: {e}")
    
    def _simple_decomposition(self, task: Dict[str, Any], available_workers: List[str]) -> Dict[str, Any]:
        """Simple fallback decomposition strategy"""
        task_description = task.get("task", "")
        
        # Basic keyword-based assignment
        decomposition = {}
        
        for worker_name in available_workers:
            if "retriev" in worker_name.lower() and any(word in task_description.lower() for word in ["find", "search", "retrieve", "get"]):
                decomposition[worker_name] = {
                    "task": f"Retrieve information for: {task_description}",
                    "priority": "high",
                    "input_data": task.get("input_data", {}),
                    "constraints": task.get("constraints", {})
                }
            elif "analyz" in worker_name.lower() and any(word in task_description.lower() for word in ["analyze", "examine", "study", "evaluate"]):
                decomposition[worker_name] = {
                    "task": f"Analyze content for: {task_description}",
                    "priority": "high",
                    "input_data": task.get("input_data", {}),
                    "constraints": task.get("constraints", {})
                }
            elif "writ" in worker_name.lower() and any(word in task_description.lower() for word in ["write", "generate", "create", "compose"]):
                decomposition[worker_name] = {
                    "task": f"Generate content for: {task_description}",
                    "priority": "high",
                    "input_data": task.get("input_data", {}),
                    "constraints": task.get("constraints", {})
                }
        
        # If no specific assignment, assign to first available worker
        if not decomposition and available_workers:
            decomposition[available_workers[0]] = {
                "task": task_description,
                "priority": "medium",
                "input_data": task.get("input_data", {}),
                "constraints": task.get("constraints", {})
            }
        
        return decomposition
    
    def _get_worker_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available workers based on domain"""
        descriptions = {
            "retriever": "Document retrieval and search operations",
            "analyzer": "Content analysis and data processing",
            "writer": "Content generation and writing tasks"
        }
        
        # Domain-specific descriptions
        if self.domain == "document retrieval":
            descriptions.update({
                "retriever": "Vector search and document retrieval",
                "analyzer": "Document content analysis and relevance scoring"
            })
        elif self.domain == "content analysis":
            descriptions.update({
                "analyzer": "Sentiment analysis, key point extraction, and content evaluation",
                "writer": "Analysis report generation and summarization"
            })
        elif self.domain == "report writing":
            descriptions.update({
                "writer": "Report generation, content creation, and document writing",
                "analyzer": "Content validation and quality checking"
            })
        
        return descriptions
    
    def _synthesize_results(self, worker_results: Dict[str, Any], task: Dict[str, Any], context: MCPContext) -> Dict[str, Any]:
        """Synthesize worker results into coordinator response"""
        
        # Prepare results summary for synthesis
        results_summary = {}
        for worker_name, result in worker_results.items():
            if result.get("success", True):
                results_summary[worker_name] = {
                    "status": "success",
                    "result": result.get("result", result.get("worker_response", "No result"))
                }
            else:
                results_summary[worker_name] = {
                    "status": "error",
                    "error": result.get("error", "Unknown error")
                }
        
        synthesis_prompt = f"""
        As a {self.domain} coordinator, synthesize these worker results:
        
        ORIGINAL TASK: {task.get('task')}
        TASK CONSTRAINTS: {json.dumps(task.get('constraints', {}), indent=2)}
        
        WORKER RESULTS:
        {json.dumps(results_summary, indent=2)}
        
        SYNTHESIS INSTRUCTIONS:
        1. Combine all successful worker results into a coherent response
        2. Address any worker failures appropriately
        3. Ensure the response is domain-specific and relevant
        4. Maintain consistency with the original task requirements
        5. Provide a comprehensive, well-structured response
        
        Provide a domain-specific response that synthesizes all worker contributions.
        """
        
        try:
            synthesized_response = self.llm_manager.generate_response(synthesis_prompt)
            
            return {
                "coordinator": self.name,
                "domain": self.domain,
                "success": True,
                "synthesized_response": synthesized_response,
                "worker_results": worker_results,
                "synthesis_metadata": {
                    "total_workers": len(worker_results),
                    "successful_workers": len([r for r in worker_results.values() if r.get("success", True)]),
                    "failed_workers": len([r for r in worker_results.values() if not r.get("success", True)]),
                    "synthesis_method": "llm_coordination"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in result synthesis: {e}")
            return {
                "coordinator": self.name,
                "domain": self.domain,
                "success": False,
                "error": f"Synthesis failed: {e}",
                "worker_results": worker_results
            }
    
    def register_worker(self, name: str, worker: 'Worker'):
        """Register a worker for task execution"""
        self.workers[name] = worker
        self.logger.info(f"Registered worker: {name}")
    
    def get_worker_status(self) -> Dict[str, Any]:
        """Get status of all registered workers"""
        status = {}
        for name, worker in self.workers.items():
            status[name] = {
                "registered": True,
                "status": "healthy",
                "specialization": worker.specialization if hasattr(worker, 'specialization') else "Unknown"
            }
        return status
    
    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task execution history"""
        return self.task_history[-limit:] if self.task_history else []
