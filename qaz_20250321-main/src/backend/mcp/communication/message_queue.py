"""
MCP Message Queue

Message queuing system for MCP communication.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from collections import deque

from ..protocol.message_schemas import MCPMessageSchema, MessagePriority


class MCPMessageQueue:
    """Message queuing system for MCP communication"""
    
    def __init__(self, max_size: int = 1000):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.max_size = max_size
        self.queues: Dict[str, deque] = {
            "high": deque(),
            "normal": deque(),
            "low": deque()
        }
        self.message_handlers: Dict[str, Callable] = {}
        self.is_running = False
        self.processing_task = None

    async def start(self):
        """Start the message queue processor"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_messages())
        self.logger.info("Message queue processor started")

    async def stop(self):
        """Stop the message queue processor"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Message queue processor stopped")

    def enqueue(self, message: MCPMessageSchema) -> bool:
        """Add message to appropriate queue"""
        if len(self._get_all_messages()) >= self.max_size:
            self.logger.warning("Message queue is full, dropping oldest message")
            self._drop_oldest_message()

        priority_queue = self._get_priority_queue(message.priority)
        priority_queue.append(message)
        
        self.logger.debug(f"Enqueued message {message.message_id} with priority {message.priority}")
        return True

    def dequeue(self, priority: Optional[MessagePriority] = None) -> Optional[MCPMessageSchema]:
        """Remove and return next message from queue"""
        if priority:
            priority_queue = self._get_priority_queue(priority)
            return priority_queue.popleft() if priority_queue else None
        
        # Try high priority first, then normal, then low
        for queue_name in ["high", "normal", "low"]:
            if self.queues[queue_name]:
                return self.queues[queue_name].popleft()
        
        return None

    def peek(self, priority: Optional[MessagePriority] = None) -> Optional[MCPMessageSchema]:
        """View next message without removing it"""
        if priority:
            priority_queue = self._get_priority_queue(priority)
            return priority_queue[0] if priority_queue else None
        
        # Try high priority first, then normal, then low
        for queue_name in ["high", "normal", "low"]:
            if self.queues[queue_name]:
                return self.queues[queue_name][0]
        
        return None

    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered handler for message type: {message_type}")

    def get_queue_size(self, priority: Optional[MessagePriority] = None) -> int:
        """Get current queue size"""
        if priority:
            return len(self._get_priority_queue(priority))
        return len(self._get_all_messages())

    def clear_queue(self, priority: Optional[MessagePriority] = None):
        """Clear messages from queue"""
        if priority:
            self._get_priority_queue(priority).clear()
        else:
            for queue in self.queues.values():
                queue.clear()
        self.logger.info("Message queue cleared")

    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "total_messages": len(self._get_all_messages()),
            "high_priority": len(self.queues["high"]),
            "normal_priority": len(self.queues["normal"]),
            "low_priority": len(self.queues["low"]),
            "max_size": self.max_size,
            "is_running": self.is_running,
            "registered_handlers": list(self.message_handlers.keys())
        }

    async def _process_messages(self):
        """Process messages from the queue"""
        while self.is_running:
            try:
                message = self.dequeue()
                if message:
                    await self._handle_message(message)
                else:
                    # No messages, wait a bit
                    await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                await asyncio.sleep(1)  # Wait before retrying

    async def _handle_message(self, message: MCPMessageSchema):
        """Handle a single message"""
        try:
            message_type = message.message_type.value
            handler = self.message_handlers.get(message_type)
            
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
                self.logger.debug(f"Handled message {message.message_id}")
            else:
                self.logger.warning(f"No handler registered for message type: {message_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling message {message.message_id}: {e}")

    def _get_priority_queue(self, priority: MessagePriority) -> deque:
        """Get queue for specific priority"""
        if priority == MessagePriority.CRITICAL or priority == MessagePriority.HIGH:
            return self.queues["high"]
        elif priority == MessagePriority.NORMAL:
            return self.queues["normal"]
        else:
            return self.queues["low"]

    def _get_all_messages(self) -> List[MCPMessageSchema]:
        """Get all messages from all queues"""
        all_messages = []
        for queue in self.queues.values():
            all_messages.extend(queue)
        return all_messages

    def _drop_oldest_message(self):
        """Drop the oldest message from the lowest priority queue"""
        for queue_name in ["low", "normal", "high"]:
            if self.queues[queue_name]:
                dropped_message = self.queues[queue_name].popleft()
                self.logger.warning(f"Dropped message {dropped_message.message_id} from {queue_name} queue")
                break
