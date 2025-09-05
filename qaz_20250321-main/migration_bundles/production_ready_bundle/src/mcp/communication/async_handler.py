"""
MCP Async Handler

Async communication handling for MCP layers.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from ..protocol.message_schemas import MCPMessageSchema


class AsyncHandler:
    """Async communication handling for MCP layers"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.async_handlers: Dict[str, Callable] = {}
        self.pending_tasks: List[asyncio.Task] = []
        self.is_running = False

    async def register_async_handler(self, message_type: str, handler: Callable):
        """Register an async handler"""
        self.async_handlers[message_type] = handler
        self.logger.info(f"Registered async handler for: {message_type}")

    async def handle_message_async(self, message: MCPMessageSchema) -> bool:
        """Handle message asynchronously"""
        message_type = message.message_type.value
        handler = self.async_handlers.get(message_type)
        
        if handler:
            try:
                task = asyncio.create_task(handler(message))
                self.pending_tasks.append(task)
                self.logger.debug(f"Created async task for message {message.message_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error creating async task for message {message.message_id}: {e}")
                return False
        else:
            self.logger.warning(f"No async handler registered for message type: {message_type}")
            return False

    async def wait_for_completion(self, timeout: Optional[float] = None):
        """Wait for all pending tasks to complete"""
        if not self.pending_tasks:
            return
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.pending_tasks, return_exceptions=True),
                timeout=timeout
            )
            self.logger.info(f"Completed {len(self.pending_tasks)} async tasks")
        except asyncio.TimeoutError:
            self.logger.warning("Timeout waiting for async tasks to complete")
        except Exception as e:
            self.logger.error(f"Error waiting for async tasks: {e}")
        finally:
            self.pending_tasks.clear()

    def get_async_statistics(self) -> Dict[str, Any]:
        """Get async handler statistics"""
        return {
            "pending_tasks": len(self.pending_tasks),
            "registered_handlers": list(self.async_handlers.keys()),
            "is_running": self.is_running
        }
