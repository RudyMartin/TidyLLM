"""
################################################################################
# *** IMPORTANT: READ docs/2025-09-08/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md ***
# *** BEFORE PLANNING ANY CHANGES TO THIS FILE ***
################################################################################

Base Gateway Interface for TidyLLM
===================================

Common interface that all gateways must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime
from enum import Enum
import logging

# CRITICAL: Import polars for DataFrame processing at all gateway stages
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

if TYPE_CHECKING:
    from . import AIProcessingGateway, CorporateLLMGateway, WorkflowOptimizerGateway

logger = logging.getLogger(__name__)


class GatewayStatus(Enum):
    """Gateway processing status."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


class GatewayState(Enum):
    """Gateway operational state for infrastructure control."""
    ENABLED = "enabled"           # Normal operation
    DISABLED = "disabled"         # Completely disabled
    PAUSED = "paused"            # Temporarily paused
    MAINTENANCE = "maintenance"   # Under maintenance
    CIRCUIT_OPEN = "circuit_open" # Circuit breaker open
    DEGRADED = "degraded"        # Running with limited functionality


class ControlSignal(Enum):
    """Control signals that can be sent to gateways."""
    ENABLE = "enable"
    DISABLE = "disable"
    PAUSE = "pause"
    RESUME = "resume"
    ENTER_MAINTENANCE = "enter_maintenance"
    EXIT_MAINTENANCE = "exit_maintenance"
    RESET_CIRCUIT = "reset_circuit"
    FORCE_DEGRADED = "force_degraded"


@dataclass
class GatewayDependencies:
    """
    Defines gateway dependency requirements.
    
    Dependency Chain:
    CorporateLLMGateway (base) → AIProcessingGateway → WorkflowOptimizerGateway → ContextGateway
    """
    requires_ai_processing: bool = False    # Needs AI/ML model capabilities
    requires_corporate_llm: bool = False    # Needs corporate LLM access control
    requires_workflow_optimizer: bool = False  # Needs workflow optimization
    requires_context: bool = False  # Needs context/knowledge access
    
    def get_required_services(self) -> List[str]:
        """Return list of required service names."""
        services = []
        if self.requires_ai_processing: services.append("ai_processing")
        if self.requires_corporate_llm: services.append("corporate_llm")
        if self.requires_workflow_optimizer: services.append("workflow_optimizer")
        if self.requires_context: services.append("context")
        return services


@dataclass
class GatewayResponse:
    """
    Standard response from gateway operations.
    
    Attributes:
        status: Processing status
        data: Main response data
        metadata: Additional metadata (timing, tokens, etc.)
        errors: List of any errors encountered
        gateway_name: Name of the gateway that processed
        timestamp: When processing occurred
    """
    status: GatewayStatus
    data: Any
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    gateway_name: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def is_success(self) -> bool:
        """Check if operation succeeded."""
        return self.status == GatewayStatus.SUCCESS
    
    @property
    def is_partial(self) -> bool:
        """Check if operation partially succeeded."""
        return self.status == GatewayStatus.PARTIAL
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0
    
    @property
    def success(self) -> bool:
        """Legacy property for backward compatibility."""
        return self.is_success
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "data": self.data,
            "metadata": self.metadata,
            "errors": self.errors,
            "gateway_name": self.gateway_name,
            "timestamp": self.timestamp.isoformat()
        }


class BaseGateway(ABC):
    """
    Abstract base class for all TidyLLM gateways.
    
    All gateways must implement this interface to ensure
    consistent behavior across different processing engines.
    """
    
    def __init__(self, **config):
        """
        Initialize gateway with configuration.
        
        Args:
            **config: Gateway-specific configuration
        """
        self.config = config
        self.name = self.__class__.__name__
        
        # CONTROL: Gateway operational state
        self._state = GatewayState.ENABLED
        self._maintenance_reason = None
        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = config.get('circuit_breaker_threshold', 5)
        
        # INTEGRATION: UnifiedSessionManager support
        self.session_manager = None
        
        # Initialize dependency configuration
        self.dependencies = self._get_default_dependencies()
        self._resolve_dependencies()
    
    def set_session_manager(self, session_manager):
        """
        Inject UnifiedSessionManager for consistent session handling.
        
        Args:
            session_manager: UnifiedSessionManager instance
        """
        self.session_manager = session_manager
        logger.info(f"UnifiedSessionManager set for {self.name}")
    
    def get_s3_client(self):
        """Get S3 client through UnifiedSessionManager if available."""
        if self.session_manager:
            return self.session_manager.get_s3_client()
        else:
            # NO FALLBACK - UnifiedSessionManager is required
            raise RuntimeError("BaseGateway: UnifiedSessionManager is required for S3 access")
    
    def get_postgres_connection(self):
        """Get PostgreSQL connection through UnifiedSessionManager if available."""
        if self.session_manager:
            return self.session_manager.get_postgres_connection()
        else:
            # Fallback would need to be implemented by subclass
            raise NotImplementedError("No session manager available and no fallback implemented")
    
    def create_stage_dataframe(self, data: Dict[str, Any], stage_name: str) -> Optional['pl.DataFrame']:
        """
        Create polars DataFrame for gateway stage processing.
        
        Args:
            data: Dictionary of data to convert to DataFrame
            stage_name: Name of the gateway stage
            
        Returns:
            polars DataFrame or None if polars not available
        """
        if not POLARS_AVAILABLE:
            logger.warning(f"Polars not available for {stage_name} DataFrame processing")
            return None
            
        try:
            # Add metadata columns for audit trail
            df_data = {
                "stage": [stage_name],
                "gateway": [self.name],
                "timestamp": [datetime.now().isoformat()],
                "request_id": [data.get("request_id", "unknown")]
            }
            
            # Add all data fields as columns
            for key, value in data.items():
                if isinstance(value, (str, int, float, bool)):
                    df_data[key] = [value]
                elif isinstance(value, (list, tuple)):
                    df_data[key] = [str(value)]  # Convert complex types to string
                else:
                    df_data[key] = [str(value)]
            
            return pl.DataFrame(df_data)
            
        except Exception as e:
            logger.error(f"Failed to create DataFrame for {stage_name}: {e}")
            return None
    
    def persist_stage_data(self, df: 'pl.DataFrame', stage_name: str, request_id: str) -> bool:
        """
        Persist polars DataFrame for this gateway stage.
        
        Args:
            df: polars DataFrame to persist
            stage_name: Name of the gateway stage
            request_id: Unique request identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not POLARS_AVAILABLE or df is None:
            return False
            
        try:
            if self.session_manager:
                # Persist to S3 as Parquet (polars native format)
                s3_client = self.get_s3_client()
                s3_key = f"gateway_data/{stage_name}/{request_id}/stage_data.parquet"
                
                # Convert DataFrame to bytes
                parquet_bytes = df.write_parquet()
                
                # Upload to S3
                s3_client.put_object(
                    Bucket="nsc-mvp1",  # Default bucket
                    Key=s3_key,
                    Body=parquet_bytes,
                    ContentType="application/parquet"
                )
                
                logger.info(f"Persisted {stage_name} DataFrame to s3://nsc-mvp1/{s3_key}")
                return True
                
            else:
                logger.warning(f"No session manager - cannot persist {stage_name} DataFrame")
                return False
                
        except Exception as e:
            logger.error(f"S3 persistence failed for {stage_name} DataFrame: {e}")
            
            # CRITICAL: Local backup when S3 is down - NO DATA LOSS
            try:
                import os
                from pathlib import Path
                
                # Create local backup directory
                backup_dir = Path("gateway_data_backup") / stage_name / request_id
                backup_dir.mkdir(parents=True, exist_ok=True)
                
                # Save DataFrame locally as Parquet
                local_file = backup_dir / "stage_data.parquet"
                df.write_parquet(local_file)
                
                # Save metadata about the failure
                metadata_file = backup_dir / "backup_metadata.json"
                import json
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "stage": stage_name,
                    "request_id": request_id,
                    "s3_error": str(e),
                    "backup_location": str(local_file),
                    "status": "s3_failed_local_backup"
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.warning(f"S3 DOWN - DataFrame backed up locally: {local_file}")
                logger.info(f"Local backup metadata: {metadata_file}")
                
                return True  # Data is safe in local backup
                
            except Exception as backup_error:
                logger.error(f"CRITICAL: Both S3 and local backup failed for {stage_name}: {backup_error}")
                return False
    
    def sync_local_backups_to_s3(self) -> Dict[str, Any]:
        """
        Sync local backup files to S3 when connection is restored.
        
        Returns:
            Dictionary with sync results
        """
        if not POLARS_AVAILABLE:
            return {"status": "polars_unavailable"}
            
        from pathlib import Path
        import json
        
        backup_root = Path("gateway_data_backup")
        if not backup_root.exists():
            return {"status": "no_backups", "message": "No local backup directory found"}
        
        sync_results = {
            "status": "completed",
            "synced_files": [],
            "failed_files": [],
            "total_files": 0
        }
        
        try:
            # Find all backup metadata files
            metadata_files = list(backup_root.rglob("backup_metadata.json"))
            sync_results["total_files"] = len(metadata_files)
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if this is a pending backup
                    if metadata.get("status") == "s3_failed_local_backup":
                        parquet_file = Path(metadata["backup_location"])
                        
                        if parquet_file.exists():
                            # Try to upload to S3
                            if self.session_manager:
                                s3_client = self.get_s3_client()
                                s3_key = f"gateway_data/{metadata['stage']}/{metadata['request_id']}/stage_data.parquet"
                                
                                with open(parquet_file, 'rb') as f:
                                    s3_client.put_object(
                                        Bucket="nsc-mvp1",
                                        Key=s3_key,
                                        Body=f.read(),
                                        ContentType="application/parquet"
                                    )
                                
                                # Update metadata to mark as synced
                                metadata["status"] = "synced_to_s3"
                                metadata["sync_timestamp"] = datetime.now().isoformat()
                                metadata["s3_location"] = f"s3://nsc-mvp1/{s3_key}"
                                
                                with open(metadata_file, 'w') as f:
                                    json.dump(metadata, f, indent=2)
                                
                                sync_results["synced_files"].append({
                                    "local_file": str(parquet_file),
                                    "s3_location": f"s3://nsc-mvp1/{s3_key}",
                                    "stage": metadata['stage'],
                                    "request_id": metadata['request_id']
                                })
                                
                                logger.info(f"Synced backup to S3: {s3_key}")
                                
                except Exception as file_error:
                    sync_results["failed_files"].append({
                        "file": str(metadata_file),
                        "error": str(file_error)
                    })
                    logger.error(f"Failed to sync backup file {metadata_file}: {file_error}")
            
            logger.info(f"Backup sync completed: {len(sync_results['synced_files'])} files synced, {len(sync_results['failed_files'])} failed")
            
        except Exception as e:
            sync_results["status"] = "error"
            sync_results["error"] = str(e)
            logger.error(f"Backup sync process failed: {e}")
        
        return sync_results
        
        # Validate after dependencies are resolved
        self._validate_config()
    
    @abstractmethod
    def _get_default_dependencies(self) -> GatewayDependencies:
        """
        Get the default dependency configuration for this gateway.
        
        Each gateway must define its dependencies:
        - AIProcessingGateway requires CorporateLLMGateway
        - WorkflowOptimizerGateway requires AIProcessingGateway + CorporateLLMGateway
        - CorporateLLMGateway has no dependencies
        - ContextGateway depends on all other gateways (final orchestrator)
        """
        pass
    
    def _resolve_dependencies(self):
        """
        Resolve and auto-enable dependent gateways based on configuration.
        
        This implements the dependency chain:
        - If AIProcessingGateway is used → enable CorporateLLMGateway
        - If WorkflowOptimizerGateway is used → enable AIProcessingGateway + CorporateLLMGateway
        """
        logger.info(f"Resolving dependencies for {self.name}")
        
        # Log original dependencies
        logger.debug(f"Original dependencies: {self.dependencies.get_required_services()}")
        
        # Apply dependency rules (set by subclasses in _get_default_dependencies)
        if hasattr(self, '_dependency_rules_applied'):
            return  # Avoid infinite recursion
        
        self._dependency_rules_applied = True
        
        # Log resolved dependencies  
        logger.info(f"Resolved dependencies: {self.dependencies.get_required_services()}")
    
    def get_required_services(self) -> List[str]:
        """
        Get list of service names that this gateway depends on.
        
        Returns:
            List of service names that are required by this gateway
        """
        return self.dependencies.get_required_services()
    
    def get_required_gateways(self) -> List[str]:
        """Legacy method for backward compatibility."""
        return self.get_required_services()
    
    @abstractmethod
    async def process(self, 
                     input_data: Any,
                     **kwargs) -> GatewayResponse:
        """
        Main processing method for the gateway.
        
        Args:
            input_data: Data to process (text, dict, etc.)
            **kwargs: Additional processing parameters
            
        Returns:
            GatewayResponse with results
        """
        pass
    
    @abstractmethod
    def process_sync(self, 
                    input_data: Any,
                    **kwargs) -> GatewayResponse:
        """
        Synchronous processing method.
        
        Args:
            input_data: Data to process
            **kwargs: Additional processing parameters
            
        Returns:
            GatewayResponse with results
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate gateway configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get gateway capabilities and features.
        
        Returns:
            Dictionary describing what this gateway can do
            
        Example:
            {
                "models": ["claude-3", "gpt-4"],
                "max_tokens": 4096,
                "supports_streaming": True,
                "supports_async": True,
                "supports_batch": False
            }
        """
        pass
    
    def _validate_config(self):
        """Internal config validation."""
        try:
            self.validate_config()
        except Exception as e:
            raise ValueError(f"Invalid configuration for {self.name}: {e}")
    
    def send_control_signal(self, signal: ControlSignal, reason: str = None) -> bool:
        """
        Send control signal to change gateway state.
        
        Args:
            signal: Control signal to send
            reason: Optional reason for state change
            
        Returns:
            True if signal was processed successfully
        """
        try:
            # CORPORATE SAFETY: Check if control signals are enabled
            if not self._is_control_signals_enabled():
                logger.warning(f"{self.name}: Control signals disabled in settings")
                return False
            
            # CORPORATE SAFETY: Check for protected interfaces
            if self._is_protected_interface(signal):
                logger.warning(f"{self.name}: Attempt to disable protected interface blocked")
                return False
            
            # CORPORATE SAFETY: Require reason for shutdown operations
            if signal in [ControlSignal.DISABLE, ControlSignal.ENTER_MAINTENANCE] and not reason:
                if self.config.get('corporate_control', {}).get('require_reason_for_shutdown', True):
                    logger.error(f"{self.name}: Shutdown operation requires reason")
                    return False
            
            old_state = self._state
            
            if signal == ControlSignal.ENABLE:
                self._state = GatewayState.ENABLED
                self._maintenance_reason = None
                
            elif signal == ControlSignal.DISABLE:
                self._state = GatewayState.DISABLED
                
            elif signal == ControlSignal.PAUSE:
                if self._state == GatewayState.ENABLED:
                    self._state = GatewayState.PAUSED
                    
            elif signal == ControlSignal.RESUME:
                if self._state == GatewayState.PAUSED:
                    self._state = GatewayState.ENABLED
                    
            elif signal == ControlSignal.ENTER_MAINTENANCE:
                self._state = GatewayState.MAINTENANCE
                self._maintenance_reason = reason or "Scheduled maintenance"
                
            elif signal == ControlSignal.EXIT_MAINTENANCE:
                if self._state == GatewayState.MAINTENANCE:
                    self._state = GatewayState.ENABLED
                    self._maintenance_reason = None
                    
            elif signal == ControlSignal.RESET_CIRCUIT:
                self._circuit_breaker_failures = 0
                if self._state == GatewayState.CIRCUIT_OPEN:
                    self._state = GatewayState.ENABLED
                    
            elif signal == ControlSignal.FORCE_DEGRADED:
                self._state = GatewayState.DEGRADED
            
            logger.info(f"{self.name}: State changed from {old_state.value} to {self._state.value} (signal: {signal.value})")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Failed to process control signal {signal.value}: {e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get current gateway state and control information."""
        return {
            "state": self._state.value,
            "maintenance_reason": self._maintenance_reason,
            "circuit_breaker_failures": self._circuit_breaker_failures,
            "circuit_breaker_threshold": self._circuit_breaker_threshold,
            "is_operational": self._state in [GatewayState.ENABLED, GatewayState.DEGRADED]
        }
    
    def _check_circuit_breaker(self, error_occurred: bool = False) -> bool:
        """
        Check circuit breaker status and update if needed.
        
        Args:
            error_occurred: Whether an error just occurred
            
        Returns:
            True if circuit is closed (operational), False if open
        """
        if error_occurred:
            self._circuit_breaker_failures += 1
            if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
                self._state = GatewayState.CIRCUIT_OPEN
                logger.warning(f"{self.name}: Circuit breaker opened ({self._circuit_breaker_failures} failures)")
                return False
        
        return self._state != GatewayState.CIRCUIT_OPEN
    
    def _can_process(self) -> bool:
        """Check if gateway can process requests based on current state."""
        return self._state in [GatewayState.ENABLED, GatewayState.DEGRADED]
    
    def _is_control_signals_enabled(self) -> bool:
        """Check if control signals are enabled in corporate settings."""
        return self.config.get('features', {}).get('security', {}).get('corporate_control', {}).get('enable_control_signals', True)
    
    def _is_protected_interface(self, signal: ControlSignal) -> bool:
        """Check if this gateway is a protected interface that cannot be disabled."""
        if signal not in [ControlSignal.DISABLE, ControlSignal.FORCE_DEGRADED]:
            return False
            
        protected_interfaces = self.config.get('features', {}).get('security', {}).get('corporate_control', {}).get('protected_interfaces', [])
        gateway_type = self.name.lower().replace('gateway', '')
        
        return gateway_type in protected_interfaces or self.name.lower() in protected_interfaces
    
    def emergency_override(self, master_key: str) -> bool:
        """Emergency override to restore gateway functionality."""
        try:
            expected_key = self.config.get('features', {}).get('security', {}).get('corporate_control', {}).get('master_override_key', 'CORPORATE_MASTER_2025')
            
            if master_key != expected_key:
                logger.error(f"{self.name}: Invalid master override key")
                return False
            
            # Force enable the gateway
            self._state = GatewayState.ENABLED
            self._maintenance_reason = None
            self._circuit_breaker_failures = 0
            
            logger.critical(f"{self.name}: EMERGENCY OVERRIDE ACTIVATED - Gateway force enabled")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Emergency override failed: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Check gateway health and availability.
        
        Returns:
            Health status dictionary
        """
        return {
            "gateway": self.name,
            "status": "healthy" if self._can_process() else self._state.value,
            "state": self.get_state(),
            "timestamp": datetime.now().isoformat(),
            "capabilities": self.get_capabilities()
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}({self.config})"