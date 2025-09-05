"""
MCP Orchestrator - Complete Hierarchical LLM System

The MCP Orchestrator manages the complete hierarchical LLM system, setting up
the Planner, Coordinators, and Workers, and providing a unified interface for
processing requests through the MCP framework.
"""

from typing import Dict, Any, List, Optional
from .planner import Planner
from .coordinator import Coordinator
from .workers.retrieval_worker import RetrievalWorker
from .workers.analysis_worker import AnalysisWorker
from .workers.writer_worker import WriterWorker
from .base import MCPContext
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MCPOrchestrator:
    """Main orchestrator for MCP hierarchical LLM system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.planner = None
        self.coordinators = {}
        self.workers = {}
        self.execution_history = []
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        self.setup_hierarchy()
        self.logger.info("MCP Hierarchical LLM System initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the MCP system"""
        return {
            "planner_config": {
                "model": "gpt-4o",
                "max_tokens": 2000,
                "temperature": 0.1,
                "provider": "openai"
            },
            "coordinator_config": {
                "model": "gpt-4o-mini",
                "max_tokens": 1500,
                "temperature": 0.2,
                "provider": "openai"
            },
            "worker_config": {
                "model": "gpt-3.5-turbo",
                "max_tokens": 1000,
                "temperature": 0.3,
                "provider": "openai"
            },
            "system_config": {
                "enable_logging": True,
                "enable_analytics": True,
                "max_execution_time": 300,  # 5 minutes
                "retry_attempts": 2
            }
        }
    
    def setup_hierarchy(self):
        """Setup the complete MCP hierarchy"""
        
        self.logger.info("Setting up MCP hierarchy...")
        
        # Initialize planner
        self.planner = Planner(self.config.get("planner_config", {}))
        self.logger.info("✓ Planner initialized")
        
        # Initialize coordinators
        self.coordinators["retrieval"] = Coordinator(
            "retrieval_coordinator", "document retrieval", self.config.get("coordinator_config", {})
        )
        self.coordinators["analysis"] = Coordinator(
            "analysis_coordinator", "content analysis", self.config.get("coordinator_config", {})
        )
        self.coordinators["writer"] = Coordinator(
            "writer_coordinator", "report writing", self.config.get("coordinator_config", {})
        )
        self.logger.info("✓ Coordinators initialized")
        
        # Initialize workers
        self.workers["retriever"] = RetrievalWorker(self.config.get("worker_config", {}))
        self.workers["analyzer"] = AnalysisWorker(self.config.get("worker_config", {}))
        self.workers["writer"] = WriterWorker(self.config.get("worker_config", {}))
        self.logger.info("✓ Workers initialized")
        
        # Register coordinators with planner
        for name, coordinator in self.coordinators.items():
            self.planner.register_coordinator(name, coordinator)
        self.logger.info("✓ Coordinators registered with planner")
        
        # Register workers with coordinators
        self.coordinators["retrieval"].register_worker("retriever", self.workers["retriever"])
        self.coordinators["analysis"].register_worker("analyzer", self.workers["analyzer"])
        self.coordinators["writer"].register_worker("writer", self.workers["writer"])
        self.logger.info("✓ Workers registered with coordinators")
        
        self.logger.info("MCP hierarchy setup completed successfully")
    
    def process_request(self, user_request: str, constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user request through MCP hierarchy"""
        
        self.logger.info(f"Processing request: {user_request[:100]}...")
        
        execution_start = datetime.now()
        
        try:
            # Create initial context
            context = MCPContext(
                user_request=user_request,
                constraints=constraints or {},
                session_data={
                    "request_timestamp": execution_start.isoformat(),
                    "request_id": f"req_{execution_start.timestamp()}"
                }
            )
            
            # Step 1: Planner creates execution plan
            self.logger.info("Step 1: Creating execution plan...")
            execution_plan = self.planner.process(context, user_request)
            
            if not execution_plan or "subtasks" not in execution_plan:
                raise ValueError("Failed to create valid execution plan")
            
            # Step 2: Execute plan through coordinators and workers
            self.logger.info("Step 2: Executing plan...")
            results = self.planner.execute_plan(execution_plan, context)
            
            # Step 3: Record execution
            execution_duration = (datetime.now() - execution_start).total_seconds()
            execution_record = {
                "request": user_request,
                "constraints": constraints,
                "execution_plan": execution_plan,
                "results": results,
                "execution_duration": execution_duration,
                "timestamp": execution_start.isoformat(),
                "success": results.get("success", False)
            }
            
            self.execution_history.append(execution_record)
            
            # Step 4: Return comprehensive result
            final_result = {
                "success": results.get("success", False),
                "final_response": results.get("final_response", "No response generated"),
                "execution_plan": execution_plan,
                "intermediate_results": results.get("intermediate_results", {}),
                "execution_metadata": {
                    "total_execution_time": execution_duration,
                    "hierarchy_levels_used": ["planner", "coordinators", "workers"],
                    "request_id": context.session_data.get("request_id"),
                    "timestamp": execution_start.isoformat()
                },
                "system_status": self.get_system_status()
            }
            
            if not results.get("success", False):
                final_result["error"] = results.get("error", "Unknown error occurred")
                final_result["error_type"] = results.get("error_type", "ExecutionError")
            
            self.logger.info(f"Request processed successfully in {execution_duration:.2f} seconds")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            
            # Record failed execution
            execution_duration = (datetime.now() - execution_start).total_seconds()
            execution_record = {
                "request": user_request,
                "constraints": constraints,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_duration": execution_duration,
                "timestamp": execution_start.isoformat(),
                "success": False
            }
            
            self.execution_history.append(execution_record)
            
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_metadata": {
                    "total_execution_time": execution_duration,
                    "request_id": f"req_{execution_start.timestamp()}",
                    "timestamp": execution_start.isoformat()
                },
                "system_status": self.get_system_status()
            }
    
    def process_qa_request(self, question: str, context_documents: List[str] = None) -> Dict[str, Any]:
        """Process QA request using MCP hierarchy"""
        
        # Prepare user request
        user_request = f"Answer the following question: {question}"
        
        # Add context if provided
        if context_documents:
            user_request += f"\n\nContext documents:\n" + "\n\n".join(context_documents)
        
        # Add QA-specific constraints
        constraints = {
            "task_type": "question_answering",
            "require_accuracy": True,
            "require_citations": bool(context_documents),
            "response_format": "comprehensive_answer"
        }
        
        return self.process_request(user_request, constraints)
    
    def generate_report(self, topic: str, data_sources: List[str]) -> Dict[str, Any]:
        """Generate comprehensive report using MCP hierarchy"""
        
        user_request = f"Generate a comprehensive report on: {topic}"
        
        constraints = {
            "task_type": "report_generation",
            "data_sources": data_sources,
            "report_type": "comprehensive",
            "target_audience": "professional",
            "require_executive_summary": True,
            "require_recommendations": True
        }
        
        return self.process_request(user_request, constraints)
    
    def analyze_documents(self, documents: List[str], analysis_type: str) -> Dict[str, Any]:
        """Analyze documents using MCP hierarchy"""
        
        user_request = f"Analyze the following documents for: {analysis_type}"
        
        constraints = {
            "task_type": "document_analysis",
            "documents": documents,
            "analysis_type": analysis_type,
            "output_format": "structured",
            "require_insights": True,
            "require_action_items": True
        }
        
        return self.process_request(user_request, constraints)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Get planner status
        planner_status = {
            "status": "healthy" if self.planner else "not_initialized",
            "registered_coordinators": list(self.coordinators.keys()),
            "execution_history_count": len(self.planner.execution_history) if self.planner else 0
        }
        
        # Get coordinator status
        coordinator_status = {}
        for name, coordinator in self.coordinators.items():
            coordinator_status[name] = {
                "status": "healthy",
                "domain": coordinator.domain,
                "registered_workers": list(coordinator.workers.keys()),
                "task_history_count": len(coordinator.task_history)
            }
        
        # Get worker status
        worker_status = {}
        for name, worker in self.workers.items():
            metrics = worker.get_performance_metrics()
            worker_status[name] = {
                "status": "healthy",
                "specialization": worker.specialization,
                "total_tasks": metrics["total_tasks"],
                "success_rate": metrics["success_rate"],
                "average_duration": metrics["average_duration"]
            }
        
        # Get overall system metrics
        total_executions = len(self.execution_history)
        successful_executions = len([e for e in self.execution_history if e.get("success", False)])
        success_rate = successful_executions / total_executions if total_executions > 0 else 0.0
        
        avg_execution_time = 0.0
        if self.execution_history:
            execution_times = [e.get("execution_duration", 0) for e in self.execution_history]
            avg_execution_time = sum(execution_times) / len(execution_times)
        
        return {
            "system_overview": {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate": success_rate,
                "average_execution_time": avg_execution_time,
                "last_execution": self.execution_history[-1]["timestamp"] if self.execution_history else None
            },
            "planner": planner_status,
            "coordinators": coordinator_status,
            "workers": worker_status,
            "configuration": {
                "planner_model": self.config.get("planner_config", {}).get("model", "unknown"),
                "coordinator_model": self.config.get("coordinator_config", {}).get("model", "unknown"),
                "worker_model": self.config.get("worker_config", {}).get("model", "unknown")
            }
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:] if self.execution_history else []
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics for the MCP system"""
        
        # Basic system analytics
        system_analytics = {
            "total_requests": len(self.execution_history),
            "success_rate": 0.0,
            "average_execution_time": 0.0,
            "request_types": {},
            "error_types": {},
            "performance_trends": {}
        }
        
        if self.execution_history:
            # Calculate success rate
            successful_requests = len([e for e in self.execution_history if e.get("success", False)])
            system_analytics["success_rate"] = successful_requests / len(self.execution_history)
            
            # Calculate average execution time
            execution_times = [e.get("execution_duration", 0) for e in self.execution_history]
            system_analytics["average_execution_time"] = sum(execution_times) / len(execution_times)
            
            # Analyze request types
            for execution in self.execution_history:
                request = execution.get("request", "")
                if "question" in request.lower() or "answer" in request.lower():
                    system_analytics["request_types"]["qa"] = system_analytics["request_types"].get("qa", 0) + 1
                elif "report" in request.lower() or "generate" in request.lower():
                    system_analytics["request_types"]["report"] = system_analytics["request_types"].get("report", 0) + 1
                elif "analyze" in request.lower() or "analysis" in request.lower():
                    system_analytics["request_types"]["analysis"] = system_analytics["request_types"].get("analysis", 0) + 1
                else:
                    system_analytics["request_types"]["general"] = system_analytics["request_types"].get("general", 0) + 1
            
            # Analyze error types
            for execution in self.execution_history:
                if not execution.get("success", False):
                    error_type = execution.get("error_type", "Unknown")
                    system_analytics["error_types"][error_type] = system_analytics["error_types"].get(error_type, 0) + 1
        
        # Worker-specific analytics
        worker_analytics = {}
        for name, worker in self.workers.items():
            if hasattr(worker, 'get_search_analytics'):
                worker_analytics[name] = worker.get_search_analytics()
            elif hasattr(worker, 'get_analysis_analytics'):
                worker_analytics[name] = worker.get_analysis_analytics()
            elif hasattr(worker, 'get_writing_analytics'):
                worker_analytics[name] = worker.get_writing_analytics()
            else:
                worker_analytics[name] = worker.get_performance_metrics()
        
        return {
            "system_analytics": system_analytics,
            "worker_analytics": worker_analytics,
            "hierarchy_analytics": {
                "planner_executions": len(self.planner.execution_history) if self.planner else 0,
                "coordinator_tasks": sum(len(c.task_history) for c in self.coordinators.values()),
                "worker_tasks": sum(len(w.task_history) for w in self.workers.values())
            }
        }
    
    def reset_system(self):
        """Reset the MCP system (clear history, reinitialize components)"""
        self.logger.info("Resetting MCP system...")
        
        # Clear execution history
        self.execution_history.clear()
        
        # Clear component histories
        if self.planner:
            self.planner.execution_history.clear()
        
        for coordinator in self.coordinators.values():
            coordinator.task_history.clear()
        
        for worker in self.workers.values():
            worker.task_history.clear()
        
        self.logger.info("MCP system reset completed")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update system configuration"""
        self.logger.info("Updating MCP system configuration...")
        
        # Update config
        self.config.update(new_config)
        
        # Reinitialize components with new config
        self.setup_hierarchy()
        
        self.logger.info("Configuration updated successfully")
