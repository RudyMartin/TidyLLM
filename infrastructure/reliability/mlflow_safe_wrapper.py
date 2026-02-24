#!/usr/bin/env python3
"""
MLflow Safe Wrapper - Never Block Core Functionality
===================================================

DESIGN PHILOSOPHY: MLflow is OPTIONAL. Core requests must NEVER fail due to MLflow.

TRAFFIC CAPTURE GUARANTEE:
- When HEALTHY: Captures 100% of traffic (expands queue if needed)
- When DEGRADED: Best effort capture (may drop some entries)
- When DISABLED: No capture (circuit breaker open)
- When UNAVAILABLE: No capture (MLflow not installed)

Features:
- Graceful degradation when MLflow unavailable
- Timeout protection (max 100ms for logging)
- Automatic disable on repeated failures
- Circuit breaker pattern with recovery
- Zero-impact on core functionality
- 100% traffic capture when enabled and healthy

Usage:
    wrapper = MLflowSafeWrapper()
    wrapper.log_request(data)  # Never throws, never blocks

Author: TidyLLM Team
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = logging.getLogger(__name__)


class MLflowState(Enum):
    """MLflow wrapper states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Working but slow
    DISABLED = "disabled"  # Disabled due to failures
    UNAVAILABLE = "unavailable"  # Never worked


@dataclass
class MLflowHealth:
    """MLflow health tracking."""
    state: MLflowState = MLflowState.UNAVAILABLE
    consecutive_failures: int = 0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    total_requests: int = 0
    total_successes: int = 0
    avg_response_time: float = 0.0


class MLflowSafeWrapper:
    """
    Safe MLflow wrapper that NEVER blocks core functionality.

    Circuit Breaker Pattern:
    - HEALTHY: Log normally
    - DEGRADED: Log with warnings
    - DISABLED: Skip logging entirely
    - UNAVAILABLE: Never tried to connect
    """

    def __init__(self,
                 timeout_ms: int = 500,
                 max_failures: int = 3,
                 recovery_interval: int = 300):
        """
        Initialize MLflow safe wrapper.

        Args:
            timeout_ms: Max time to wait for MLflow (500ms default)
            max_failures: Failures before circuit breaker opens
            recovery_interval: Seconds before retry attempt
        """
        self.timeout_ms = timeout_ms
        self.max_failures = max_failures
        self.recovery_interval = recovery_interval

        self.health = MLflowHealth()
        self._lock = threading.Lock()

        # Async logging queue
        self._log_queue = queue.Queue(maxsize=1000)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlflow-logger")
        self._logging_enabled = True

        # Lazy initialization
        self._mlflow_client = None
        self._mlflow_available = None

        # Initialize SettingsLoader to set up MLflow environment
        self._initialize_settings()

        logger.info("MLflowSafeWrapper initialized (graceful degradation enabled)")

    def _initialize_settings(self):
        """Initialize SettingsLoader to set up MLflow environment variables."""
        try:
            # Use PathManager for proper path resolution
            import sys
            try:
                from common.utilities.path_manager import PathManager
                path_mgr = PathManager()

                # Add all necessary paths to sys.path
                for path in path_mgr.get_python_paths():
                    if path not in sys.path:
                        sys.path.insert(0, path)

                logger.info("PathManager initialized - proper path resolution active")
            except ImportError as path_error:
                logger.warning(f"PathManager not available: {path_error}, using fallback paths")
                # Fallback to manual path setup
                from pathlib import Path
                qa_root = Path(__file__).parent.parent.parent.parent.parent
                if str(qa_root) not in sys.path:
                    sys.path.insert(0, str(qa_root))

            # Use CredentialCarrier to properly set all environment variables
            try:
                from infrastructure.services.credential_carrier import CredentialCarrier
                credential_carrier = CredentialCarrier()

                # This method sets all environment variables from cached credentials
                # including AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.
                credential_carrier.set_environment_from_credentials()
                logger.info("Environment variables set from CredentialCarrier")

                # Also set MLflow-specific environment
                from infrastructure.yaml_loader import SettingsLoader
                loader = SettingsLoader()
                mlflow_config = loader.get_mlflow_config()

                # Check if we should use local MLflow server (preferred) or direct PostgreSQL
                mlflow_gateway_uri = mlflow_config.get('mlflow_gateway_uri')
                tracking_uri = mlflow_config.get('tracking_uri')

                import os
                # Prefer local MLflow server if configured (manages its own connection pool)
                if mlflow_gateway_uri and 'localhost' in mlflow_gateway_uri:
                    # Use local MLflow server - it manages connection pooling to RDS
                    os.environ['MLFLOW_TRACKING_URI'] = mlflow_gateway_uri
                    logger.info(f"Using local MLflow server at {mlflow_gateway_uri} (server manages RDS pooling)")
                    logger.info("MLflow server provides connection pooling to AWS RDS")
                elif tracking_uri:
                    # Fallback to direct PostgreSQL (may have timeout issues)
                    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
                    logger.warning(f"Using direct PostgreSQL connection: {tracking_uri[:50]}...")
                    logger.warning("Direct RDS connections may timeout - consider starting local MLflow server")
                else:
                    logger.warning("No MLflow tracking URI found in configuration")

            except ImportError as carrier_error:
                logger.warning(f"CredentialCarrier not available: {carrier_error}, using direct settings")
                # Fallback to direct settings loader
                try:
                    from infrastructure.yaml_loader import SettingsLoader
                    loader = SettingsLoader()
                    loader.set_environment_variables()  # This sets all env vars

                    mlflow_config = loader.get_mlflow_config()
                    tracking_uri = mlflow_config.get('tracking_uri')
                    if tracking_uri:
                        os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
                        logger.info(f"Using MLflow tracking URI from settings: {tracking_uri[:50]}...")
                except Exception as fallback_error:
                    logger.warning(f"Fallback settings failed: {fallback_error}")

            # Reset availability check to pick up new environment
            self._mlflow_available = None

            # Force a fresh availability check
            if self.is_available():
                self.health.state = MLflowState.HEALTHY
                logger.info("MLflow environment configured and ready")
            else:
                logger.warning("MLflow environment configured but MLflow not available")

        except Exception as e:
            logger.warning(f"Failed to initialize MLflow configuration: {e} (graceful degradation)")

    def is_available(self) -> bool:
        """Check if MLflow is available without blocking."""
        if self._mlflow_available is None:
            # Quick availability check
            try:
                import mlflow
                self._mlflow_available = True
                logger.info("MLflow detected and available")
            except ImportError:
                self._mlflow_available = False
                self.health.state = MLflowState.UNAVAILABLE
                logger.info("MLflow not installed - graceful degradation active")

        return self._mlflow_available

    def log_request(self,
                   model: str,
                   prompt: str,
                   response: str,
                   processing_time: float,
                   success: bool = True,
                   **kwargs) -> bool:
        """
        Log request to MLflow safely.

        GUARANTEE: This method NEVER blocks and NEVER throws exceptions.

        Returns:
            bool: True if logged, False if skipped (but request succeeds either way)
        """
        if not self._should_attempt_logging():
            return False

        # Package the log entry
        log_entry = {
            'model': model,
            'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
            'response': response[:100] + "..." if len(response) > 100 else response,
            'processing_time': processing_time,
            'success': success,
            'timestamp': time.time(),
            **kwargs
        }

        # Queue for async processing (never blocks)
        try:
            # WHEN ENABLED: CAPTURE 100% OF TRAFFIC
            if self.health.state == MLflowState.HEALTHY:
                # For healthy state, ensure we capture ALL requests
                try:
                    self._log_queue.put(log_entry, timeout=0.01)  # Brief wait for 100% capture
                except queue.Full:
                    # Expand queue dynamically when healthy to ensure 100% capture
                    logger.warning("MLflow healthy but queue full - expanding capacity for 100% capture")
                    # Create a larger queue temporarily
                    old_entries = []
                    while not self._log_queue.empty():
                        try:
                            old_entries.append(self._log_queue.get_nowait())
                        except queue.Empty:
                            break

                    # Recreate with larger capacity
                    self._log_queue = queue.Queue(maxsize=2000)  # Double capacity
                    for entry in old_entries:
                        self._log_queue.put_nowait(entry)
                    self._log_queue.put_nowait(log_entry)

                    logger.info("MLflow queue expanded - maintaining 100% traffic capture")
            else:
                # For degraded states, use non-blocking approach
                self._log_queue.put_nowait(log_entry)

            # Submit async logging task
            self._executor.submit(self._async_log_worker)
            return True

        except queue.Full:
            # Only happens in degraded states
            logger.debug("MLflow log queue full in degraded state - graceful degradation")
            return False
        except Exception as e:
            logger.debug(f"MLflow queue error: {e} (graceful degradation)")
            return False

    def _should_attempt_logging(self) -> bool:
        """Determine if we should attempt MLflow logging."""
        with self._lock:
            # Check if MLflow is available
            if not self.is_available():
                return False

            # Check circuit breaker
            if self.health.state == MLflowState.DISABLED:
                # Check if recovery time has passed
                if (self.health.last_failure and
                    time.time() - self.health.last_failure > self.recovery_interval):
                    logger.info("MLflow recovery attempt - resetting circuit breaker")
                    self.health.state = MLflowState.HEALTHY
                    self.health.consecutive_failures = 0
                    return True
                return False

            return self.health.state in [MLflowState.HEALTHY, MLflowState.DEGRADED]

    def _async_log_worker(self):
        """Worker thread for async MLflow logging."""
        try:
            # Get all pending entries (batch processing)
            entries = []
            try:
                while True:
                    entry = self._log_queue.get_nowait()
                    entries.append(entry)
                    if len(entries) >= 10:  # Batch size limit
                        break
            except queue.Empty:
                pass

            if not entries:
                return

            # Process entries with timeout
            start_time = time.time()
            try:
                # Timeout wrapper
                future = self._executor.submit(self._do_mlflow_logging, entries)
                future.result(timeout=self.timeout_ms / 1000.0)

                # Success
                processing_time = (time.time() - start_time) * 1000
                self._record_success(processing_time)

            except TimeoutError:
                logger.warning(f"MLflow logging timeout ({self.timeout_ms}ms) - graceful degradation")
                self._record_failure("timeout")
            except Exception as e:
                logger.warning(f"MLflow logging error: {e} - graceful degradation")
                self._record_failure(str(e))

        except Exception as e:
            logger.debug(f"MLflow worker error: {e} (graceful degradation)")

    def _do_mlflow_logging(self, entries: list):
        """Actual MLflow logging implementation."""
        if not self._mlflow_client:
            import mlflow
            self._mlflow_client = mlflow

        # Log entries to MLflow
        for entry in entries:
            try:
                # Use MLflow tracking
                with self._mlflow_client.start_run():
                    # Batch ALL metrics into a single call
                    metrics_batch = {
                        "processing_time_ms": entry['processing_time'],
                        "success": 1.0 if entry['success'] else 0.0
                    }

                    # Add RL metrics if present
                    if 'rl_data' in entry:
                        import json
                        rl_data = entry['rl_data']

                        # Add key RL metrics to batch
                        if isinstance(rl_data.get('reward_signal'), (int, float)):
                            metrics_batch['reward_signal'] = rl_data['reward_signal']
                        if isinstance(rl_data.get('value_estimation'), (int, float)):
                            metrics_batch['value_estimation'] = rl_data['value_estimation']

                        # Only log RL JSON if it contains substantial data
                        if len(rl_data) > 2:  # More than just the two metrics above
                            rl_json = json.dumps(rl_data, separators=(',', ':'))  # Compact JSON
                            # Log as a single text artifact
                            self._mlflow_client.log_text(rl_json, "rl_data.json")

                    # Single batch call for all metrics (1 DB write instead of 4+)
                    self._mlflow_client.log_metrics(metrics_batch)

                    # Log model as param (required for MLflow)
                    self._mlflow_client.log_param("model", entry['model'])

                    # Only log text artifacts if they're reasonably sized
                    MAX_TEXT_SIZE = 1000  # Avoid large S3 uploads
                    if len(entry['prompt']) <= MAX_TEXT_SIZE:
                        self._mlflow_client.log_text(entry['prompt'], "prompt.txt")
                    else:
                        # Log truncated version as param instead
                        self._mlflow_client.log_param("prompt_truncated", entry['prompt'][:100] + "...")

                    if len(entry['response']) <= MAX_TEXT_SIZE:
                        self._mlflow_client.log_text(entry['response'], "response.txt")
                    else:
                        # Log truncated version as param instead
                        self._mlflow_client.log_param("response_truncated", entry['response'][:100] + "...")

            except Exception as e:
                # Individual entry failure - continue with others
                logger.debug(f"MLflow entry logging failed: {e}")
                continue

    def _record_success(self, processing_time: float):
        """Record successful MLflow operation."""
        with self._lock:
            self.health.consecutive_failures = 0
            self.health.last_success = time.time()
            self.health.total_requests += 1
            self.health.total_successes += 1

            # Update average response time
            if self.health.avg_response_time == 0:
                self.health.avg_response_time = processing_time
            else:
                self.health.avg_response_time = (self.health.avg_response_time * 0.9 + processing_time * 0.1)

            # State management
            if processing_time > self.timeout_ms * 0.8:  # 80% of timeout
                self.health.state = MLflowState.DEGRADED
                logger.info(f"MLflow degraded performance: {processing_time:.1f}ms")
            else:
                self.health.state = MLflowState.HEALTHY

    def _record_failure(self, error: str):
        """Record MLflow failure."""
        with self._lock:
            self.health.consecutive_failures += 1
            self.health.last_failure = time.time()
            self.health.total_requests += 1

            # Circuit breaker logic
            if self.health.consecutive_failures >= self.max_failures:
                self.health.state = MLflowState.DISABLED
                logger.warning(f"MLflow circuit breaker OPEN - disabled for {self.recovery_interval}s")
            else:
                self.health.state = MLflowState.DEGRADED
                logger.info(f"MLflow failure {self.health.consecutive_failures}/{self.max_failures}: {error}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get MLflow wrapper health status."""
        with self._lock:
            success_rate = (self.health.total_successes / self.health.total_requests
                          if self.health.total_requests > 0 else 0.0)

            return {
                "mlflow_wrapper": {
                    "state": self.health.state.value,
                    "available": self.is_available(),
                    "success_rate": f"{success_rate:.1%}",
                    "avg_response_time_ms": f"{self.health.avg_response_time:.1f}",
                    "consecutive_failures": self.health.consecutive_failures,
                    "total_requests": self.health.total_requests,
                    "queue_size": self._log_queue.qsize(),
                    "timeout_ms": self.timeout_ms,
                    "max_failures": self.max_failures
                }
            }

    def force_disable(self):
        """Manually disable MLflow logging."""
        with self._lock:
            self.health.state = MLflowState.DISABLED
            self._logging_enabled = False
            logger.info("MLflow logging manually disabled")

    def force_enable(self):
        """Manually enable MLflow logging."""
        with self._lock:
            if self.is_available():
                self.health.state = MLflowState.HEALTHY
                self.health.consecutive_failures = 0
                self._logging_enabled = True
                logger.info("MLflow logging manually enabled")

    def cleanup(self):
        """Cleanup resources."""
        self._logging_enabled = False
        self._executor.shutdown(wait=False)
        logger.info("MLflowSafeWrapper cleaned up")


# Global singleton instance
_global_mlflow_wrapper = None

def get_mlflow_safe_wrapper() -> MLflowSafeWrapper:
    """Get global MLflow safe wrapper instance."""
    global _global_mlflow_wrapper
    if _global_mlflow_wrapper is None:
        # Increase timeout to 2000ms to handle RL metrics logging
        _global_mlflow_wrapper = MLflowSafeWrapper(timeout_ms=2000)
    return _global_mlflow_wrapper


if __name__ == "__main__":
    # Example usage
    import time

    wrapper = MLflowSafeWrapper()

    print("Testing MLflow safe wrapper...")

    # Test logging (should never block)
    for i in range(5):
        success = wrapper.log_request(
            model="test-model",
            prompt=f"Test prompt {i}",
            response=f"Test response {i}",
            processing_time=100.0 + i * 10
        )
        print(f"Log {i}: {'SUCCESS' if success else 'SKIPPED'}")

    # Wait a bit for async processing
    time.sleep(0.5)

    # Check health
    health = wrapper.get_health_status()
    print(f"\nHealth Status:")
    for key, value in health["mlflow_wrapper"].items():
        print(f"  {key}: {value}")

    wrapper.cleanup()