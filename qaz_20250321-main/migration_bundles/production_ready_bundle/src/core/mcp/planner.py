"""
MCP Planner - Strategic Orchestration Layer

The Planner is the top-level node in the MCP hierarchy that creates execution plans
and orchestrates the entire workflow through coordinators and workers.
"""

from typing import List, Dict, Any, Optional
from .base import MCPNode, MCPRole, MCPContext, MCPProtocol
from .protocol import MCPMessageProtocol, MCPMessage
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Planner(MCPNode):
    """Top-level planner that orchestrates the entire workflow"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(MCPRole.PLANNER, model_config)
        self.coordinators = {}
        self.execution_history = []
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def _define_protocol(self) -> MCPProtocol:
        """Define the protocol for planner communication"""
        return MCPProtocol(
            message_format="json",
            validation_schema={
                "type": "object",
                "required": ["goal", "subtasks"],
                "properties": {
                    "goal": {"type": "string"},
                    "constraints": {"type": "object"},
                    "subtasks": {"type": "array"},
                    "priority": {"type": "string"},
                    "estimated_duration": {"type": "number"}
                }
            },
            error_handling={
                "retry_count": 3,
                "fallback_strategy": "simplify",
                "error_threshold": 0.3
            },
            retry_policy={
                "max_retries": 2,
                "backoff_factor": 1.5,
                "retry_delay": 1.0
            }
        )
    
    def process(self, context: MCPContext, user_request: str) -> Dict[str, Any]:
        """Process user request and create execution plan"""
        
        self.logger.info(f"Processing request: {user_request[:100]}...")
        
        try:
            # Generate plan using LLM
            plan_prompt = self._create_planning_prompt(context, user_request)
            plan_response = self.llm_manager.generate_response(plan_prompt)
            
            # Parse plan into structured format
            execution_plan = self._parse_plan(plan_response)
            
            # Validate plan against protocol
            if not self.validate_input(execution_plan):
                raise ValueError("Invalid execution plan generated")
            
            # Add metadata to plan
            execution_plan["metadata"] = {
                "created_at": datetime.now().isoformat(),
                "planner_version": "1.0",
                "context_id": context.context_id,
                "estimated_duration": self._estimate_plan_duration(execution_plan)
            }
            
            self.logger.info(f"Execution plan created with {len(execution_plan.get('subtasks', []))} subtasks")
            
            return execution_plan
            
        except Exception as e:
            self.logger.error(f"Error in planner process: {e}")
            return self.handle_error(e, context)
    
    def _create_planning_prompt(self, context: MCPContext, user_request: str) -> str:
        """Create comprehensive planning prompt"""
        
        available_coordinators = list(self.coordinators.keys())
        coordinator_descriptions = {
            "retrieval": "Handles document retrieval, search, and information gathering",
            "analysis": "Handles content analysis, reasoning, and data processing",
            "writer": "Handles report generation, writing, and content creation"
        }
        
        prompt = f"""
        As a strategic planner, create an execution plan for the following request:
        
        USER REQUEST: {user_request}
        
        CONTEXT CONSTRAINTS: {json.dumps(context.constraints, indent=2)}
        
        SESSION DATA: {json.dumps(context.session_data, indent=2)}
        
        AVAILABLE COORDINATORS:
        """
        
        for coord_name in available_coordinators:
            description = coordinator_descriptions.get(coord_name, "Specialized coordination")
            prompt += f"- {coord_name.upper()} COORDINATOR: {description}\n"
        
        prompt += f"""
        PLANNING INSTRUCTIONS:
        1. Analyze the request and break it down into logical subtasks
        2. Assign each subtask to the most appropriate coordinator
        3. Consider dependencies between subtasks
        4. Set appropriate priorities (high/medium/low)
        5. Estimate duration for each subtask
        
        Create a JSON plan with this exact structure:
        {{
            "goal": "clear, specific objective",
            "subtasks": [
                {{
                    "id": "unique_task_id",
                    "coordinator": "coordinator_name",
                    "task": "specific task description",
                    "priority": "high/medium/low",
                    "dependencies": ["other_task_ids"],
                    "estimated_duration": 30,
                    "constraints": {{}},
                    "input_data": {{}}
                }}
            ],
            "constraints": {json.dumps(context.constraints, indent=2)},
            "total_estimated_duration": 0,
            "parallel_tasks": []
        }}
        
        Ensure the plan is comprehensive, logical, and executable.
        """
        
        return prompt
    
    def _parse_plan(self, plan_response: str) -> Dict[str, Any]:
        """Parse LLM response into structured plan"""
        try:
            # Extract JSON from response
            json_start = plan_response.find('{')
            json_end = plan_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            plan_json = plan_response[json_start:json_end]
            plan = json.loads(plan_json)
            
            # Validate required fields
            if "goal" not in plan:
                raise ValueError("Plan missing 'goal' field")
            if "subtasks" not in plan:
                raise ValueError("Plan missing 'subtasks' field")
            
            # Add task IDs if missing
            for i, subtask in enumerate(plan["subtasks"]):
                if "id" not in subtask:
                    subtask["id"] = f"task_{i+1}"
            
            return plan
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            raise ValueError(f"Failed to parse plan JSON: {e}")
        except Exception as e:
            self.logger.error(f"Plan parsing error: {e}")
            raise ValueError(f"Failed to parse plan: {e}")
    
    def _estimate_plan_duration(self, plan: Dict[str, Any]) -> int:
        """Estimate total duration of the plan"""
        total_duration = 0
        subtasks = plan.get("subtasks", [])
        
        for subtask in subtasks:
            duration = subtask.get("estimated_duration", 30)
            total_duration += duration
        
        return total_duration
    
    def register_coordinator(self, name: str, coordinator: 'Coordinator'):
        """Register a coordinator for delegation"""
        self.coordinators[name] = coordinator
        self.logger.info(f"Registered coordinator: {name}")
    
    def execute_plan(self, plan: Dict[str, Any], context: MCPContext) -> Dict[str, Any]:
        """Execute the plan by delegating to coordinators"""
        
        self.logger.info(f"Executing plan with {len(plan.get('subtasks', []))} subtasks")
        
        results = {}
        execution_start = datetime.now()
        
        try:
            # Execute subtasks
            for subtask in plan["subtasks"]:
                task_id = subtask["id"]
                coordinator_name = subtask["coordinator"]
                
                self.logger.info(f"Executing task {task_id} with coordinator {coordinator_name}")
                
                if coordinator_name in self.coordinators:
                    coordinator = self.coordinators[coordinator_name]
                    
                    # Create enriched context for coordinator
                    coordinator_context = context.enrich({
                        "task": subtask,
                        "plan": plan,
                        "task_id": task_id
                    })
                    
                    # Execute subtask
                    task_start = datetime.now()
                    result = coordinator.process(coordinator_context, subtask)
                    task_duration = (datetime.now() - task_start).total_seconds()
                    
                    # Add execution metadata
                    result["execution_metadata"] = {
                        "task_id": task_id,
                        "coordinator": coordinator_name,
                        "duration": task_duration,
                        "start_time": task_start.isoformat(),
                        "end_time": datetime.now().isoformat()
                    }
                    
                    results[task_id] = result
                    
                else:
                    error_msg = f"Coordinator '{coordinator_name}' not found"
                    self.logger.error(error_msg)
                    results[task_id] = {
                        "error": error_msg,
                        "success": False,
                        "execution_metadata": {
                            "task_id": task_id,
                            "coordinator": coordinator_name,
                            "error": True
                        }
                    }
            
            # Aggregate results
            final_result = self._aggregate_results(results, plan, context)
            
            # Record execution
            execution_duration = (datetime.now() - execution_start).total_seconds()
            self.execution_history.append({
                "plan": plan,
                "results": results,
                "final_result": final_result,
                "execution_duration": execution_duration,
                "timestamp": datetime.now().isoformat()
            })
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error during plan execution: {e}")
            return self.handle_error(e, context)
    
    def _aggregate_results(self, results: Dict[str, Any], plan: Dict[str, Any], context: MCPContext) -> Dict[str, Any]:
        """Aggregate results from all coordinators"""
        
        # Prepare results for aggregation
        results_summary = {}
        for task_id, result in results.items():
            if result.get("success", True):
                results_summary[task_id] = {
                    "status": "success",
                    "result": result.get("result", result.get("synthesized_response", "No result"))
                }
            else:
                results_summary[task_id] = {
                    "status": "error",
                    "error": result.get("error", "Unknown error")
                }
        
        # Create aggregation prompt
        aggregation_prompt = f"""
        Aggregate the following results into a final comprehensive response:
        
        ORIGINAL GOAL: {plan['goal']}
        
        USER REQUEST: {context.user_request}
        
        EXECUTION RESULTS:
        {json.dumps(results_summary, indent=2)}
        
        INSTRUCTIONS:
        1. Synthesize all successful results into a coherent response
        2. Address any errors or failures appropriately
        3. Ensure the response directly addresses the original goal
        4. Maintain professional tone and structure
        5. Include key insights and findings from all subtasks
        
        Provide a comprehensive, well-structured response that addresses the original goal.
        """
        
        try:
            final_response = self.llm_manager.generate_response(aggregation_prompt)
            
            return {
                "success": True,
                "final_response": final_response,
                "intermediate_results": results,
                "plan": plan,
                "aggregation_metadata": {
                    "total_tasks": len(results),
                    "successful_tasks": len([r for r in results.values() if r.get("success", True)]),
                    "failed_tasks": len([r for r in results.values() if not r.get("success", True)]),
                    "aggregation_method": "llm_synthesis"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in result aggregation: {e}")
            return {
                "success": False,
                "error": f"Aggregation failed: {e}",
                "intermediate_results": results,
                "plan": plan
            }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:] if self.execution_history else []
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get status of all registered coordinators"""
        status = {}
        for name, coordinator in self.coordinators.items():
            status[name] = {
                "registered": True,
                "status": "healthy",
                "workers": list(coordinator.workers.keys()) if hasattr(coordinator, 'workers') else []
            }
        return status
