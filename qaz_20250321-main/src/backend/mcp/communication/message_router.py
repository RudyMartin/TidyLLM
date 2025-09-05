#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Message Router - Real-Time Message Routing for MCP System

Provides real-time message routing between all MCP components with:
- Priority-based message queuing
- Load balancing across orchestrators
- Message validation and error handling
- Performance monitoring and metrics
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue, Queue
import threading
from collections import defaultdict

from ..protocol.message_protocol import MCPMessage, MessageType, TaskType, Priority


class RouteType(Enum):
    """Types of message routes"""
    PLANNER_TO_COORDINATOR = "planner_to_coordinator"
    COORDINATOR_TO_WORKER = "coordinator_to_worker"
    WORKER_TO_COORDINATOR = "worker_to_coordinator"
    COORDINATOR_TO_PLANNER = "coordinator_to_planner"
    BROADCAST = "broadcast"


@dataclass
class RouteConfig:
    """Configuration for message routing"""
    route_type: RouteType
    source_component: str
    target_component: str
    priority: Priority = Priority.MEDIUM
    timeout_seconds: float = 30.0
    retry_count: int = 3
    load_balancing: bool = True


@dataclass
class RoutingMetrics:
    """Metrics for message routing performance"""
    total_messages: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    average_routing_time: float = 0.0
    total_routing_time: float = 0.0
    last_route_time: Optional[datetime] = None
    component_loads: Dict[str, int] = field(default_factory=dict)


class MessageRouter:
    """Real-time message router for MCP system"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Message queues
        self.priority_queue = PriorityQueue()
        self.broadcast_queue = Queue()
        
        # Routing configuration
        self.routes: Dict[str, RouteConfig] = {}
        self.component_handlers: Dict[str, Callable] = {}
        self.load_balancers: Dict[str, List[str]] = defaultdict(list)
        
        # Performance metrics
        self.metrics = RoutingMetrics()
        
        # Threading
        self.running = False
        self.router_thread = None
        self.lock = threading.Lock()
        
        # Initialize default routes
        self._initialize_default_routes()
        
        self.logger.info("Message Router initialized")
    
    def _initialize_default_routes(self):
        """Initialize default routing configuration"""
        default_routes = [
            RouteConfig(
                route_type=RouteType.PLANNER_TO_COORDINATOR,
                source_component="enhanced_planner",
                target_component="document_coordinator",
                priority=Priority.HIGH
            ),
            RouteConfig(
                route_type=RouteType.COORDINATOR_TO_WORKER,
                source_component="document_coordinator",
                target_component="pdf_processor_worker",
                priority=Priority.MEDIUM
            ),
            RouteConfig(
                route_type=RouteType.WORKER_TO_COORDINATOR,
                source_component="pdf_processor_worker",
                target_component="document_coordinator",
                priority=Priority.MEDIUM
            ),
            RouteConfig(
                route_type=RouteType.COORDINATOR_TO_PLANNER,
                source_component="document_coordinator",
                target_component="enhanced_planner",
                priority=Priority.HIGH
            )
        ]
        
        for route in default_routes:
            self.add_route(route)
    
    def add_route(self, route_config: RouteConfig):
        """Add a new routing configuration"""
        route_key = f"{route_config.source_component}->{route_config.target_component}"
        self.routes[route_key] = route_config
        
        # Add to load balancer if enabled
        if route_config.load_balancing:
            self.load_balancers[route_config.target_component].append(route_config.source_component)
        
        self.logger.info(f"Added route: {route_key}")
    
    def register_component_handler(self, component_name: str, handler: Callable):
        """Register a component handler for message processing"""
        self.component_handlers[component_name] = handler
        self.logger.info(f"Registered handler for component: {component_name}")
    
    def route_message(self, message: MCPMessage, route_type: RouteType) -> bool:
        """Route a message to the appropriate component"""
        start_time = time.time()
        
        try:
            # Update metrics
            with self.lock:
                self.metrics.total_messages += 1
                self.metrics.last_route_time = datetime.now()
            
            # Determine target component
            target_component = self._determine_target_component(message, route_type)
            if not target_component:
                self.logger.error(f"Could not determine target component for message: {message.message_id}")
                return False
            
            # Get handler for target component
            handler = self.component_handlers.get(target_component)
            if not handler:
                self.logger.error(f"No handler registered for component: {target_component}")
                return False
            
            # Route message based on priority
            if message.priority == Priority.HIGH:
                self.priority_queue.put((0, time.time(), message, target_component, handler))
            else:
                self.priority_queue.put((1, time.time(), message, target_component, handler))
            
            # Update component load
            with self.lock:
                self.metrics.component_loads[target_component] = self.metrics.component_loads.get(target_component, 0) + 1
            
            # Calculate routing time
            routing_time = time.time() - start_time
            with self.lock:
                self.metrics.total_routing_time += routing_time
                self.metrics.average_routing_time = self.metrics.total_routing_time / self.metrics.total_messages
            
            self.logger.debug(f"Routed message {message.message_id} to {target_component} in {routing_time:.3f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error routing message {message.message_id}: {e}")
            with self.lock:
                self.metrics.failed_routes += 1
            return False
    
    def _determine_target_component(self, message: MCPMessage, route_type: RouteType) -> Optional[str]:
        """Determine the target component for a message"""
        
        if route_type == RouteType.PLANNER_TO_COORDINATOR:
            # Route to appropriate coordinator based on task type
            if message.task_type == TaskType.RETRIEVAL:
                return "document_coordinator"
            elif message.task_type == TaskType.ANALYSIS:
                return "sme_context_coordinator"
            else:
                return "document_coordinator"  # Default
        
        elif route_type == RouteType.COORDINATOR_TO_WORKER:
            # Route to appropriate worker based on payload
            payload = message.payload
            if 'pdf_processing' in payload:
                return "pdf_processor_worker"
            elif 'text_cleaning' in payload:
                return "text_cleaner_worker"
            elif 'embedding_generation' in payload:
                return "embedding_generator_worker"
            elif 'table_extraction' in payload:
                return "table_extractor_worker"
            elif 'live_context' in payload:
                return "live_context_worker"
            else:
                return "pdf_processor_worker"  # Default
        
        elif route_type == RouteType.WORKER_TO_COORDINATOR:
            # Route back to the coordinator that sent the message
            return message.context.get('source_coordinator', 'document_coordinator')
        
        elif route_type == RouteType.COORDINATOR_TO_PLANNER:
            # Route back to the planner
            return "enhanced_planner"
        
        elif route_type == RouteType.BROADCAST:
            # Broadcast to all registered components
            return "broadcast"
        
        return None
    
    def start_routing(self):
        """Start the message routing thread"""
        if self.running:
            self.logger.warning("Message router is already running")
            return
        
        self.running = True
        self.router_thread = threading.Thread(target=self._routing_loop, daemon=True)
        self.router_thread.start()
        self.logger.info("Message router started")
    
    def stop_routing(self):
        """Stop the message routing thread"""
        self.running = False
        if self.router_thread:
            self.router_thread.join(timeout=5.0)
        self.logger.info("Message router stopped")
    
    def _routing_loop(self):
        """Main routing loop"""
        while self.running:
            try:
                # Process priority queue
                if not self.priority_queue.empty():
                    priority, timestamp, message, target_component, handler = self.priority_queue.get_nowait()
                    
                    # Check for timeout
                    if time.time() - timestamp > 30.0:  # 30 second timeout
                        self.logger.warning(f"Message {message.message_id} timed out")
                        with self.lock:
                            self.metrics.failed_routes += 1
                        continue
                    
                    # Process message
                    success = self._process_message(message, target_component, handler)
                    
                    with self.lock:
                        if success:
                            self.metrics.successful_routes += 1
                        else:
                            self.metrics.failed_routes += 1
                
                # Process broadcast queue
                if not self.broadcast_queue.empty():
                    broadcast_message = self.broadcast_queue.get_nowait()
                    self._process_broadcast(broadcast_message)
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
                
            except Exception as e:
                self.logger.error(f"Error in routing loop: {e}")
                time.sleep(0.1)  # Longer sleep on error
    
    def _process_message(self, message: MCPMessage, target_component: str, handler: Callable) -> bool:
        """Process a message with the target component handler"""
        try:
            # Call the handler
            result = handler(message)
            
            # Log success
            self.logger.debug(f"Successfully processed message {message.message_id} with {target_component}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing message {message.message_id} with {target_component}: {e}")
            return False
    
    def _process_broadcast(self, message: MCPMessage):
        """Process a broadcast message to all components"""
        try:
            for component_name, handler in self.component_handlers.items():
                try:
                    handler(message)
                    self.logger.debug(f"Broadcast message {message.message_id} sent to {component_name}")
                except Exception as e:
                    self.logger.error(f"Error broadcasting to {component_name}: {e}")
        except Exception as e:
            self.logger.error(f"Error in broadcast processing: {e}")
    
    def broadcast_message(self, message: MCPMessage):
        """Broadcast a message to all registered components"""
        self.broadcast_queue.put(message)
        self.logger.info(f"Broadcast message {message.message_id} queued")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current routing metrics"""
        with self.lock:
            return {
                "total_messages": self.metrics.total_messages,
                "successful_routes": self.metrics.successful_routes,
                "failed_routes": self.metrics.failed_routes,
                "success_rate": self.metrics.successful_routes / max(self.metrics.total_messages, 1),
                "average_routing_time": self.metrics.average_routing_time,
                "component_loads": dict(self.metrics.component_loads),
                "last_route_time": self.metrics.last_route_time.isoformat() if self.metrics.last_route_time else None,
                "router_status": "running" if self.running else "stopped"
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the message router"""
        return {
            "component": "message_router",
            "status": "healthy" if self.running else "unhealthy",
            "metrics": self.get_metrics(),
            "registered_components": list(self.component_handlers.keys()),
            "active_routes": len(self.routes),
            "queue_sizes": {
                "priority_queue": self.priority_queue.qsize(),
                "broadcast_queue": self.broadcast_queue.qsize()
            },
            "timestamp": datetime.now().isoformat()
        }
