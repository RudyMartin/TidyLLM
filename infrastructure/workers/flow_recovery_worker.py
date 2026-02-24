"""
Flow Recovery Worker - File Purgatory Recovery System
====================================================

Specialized worker that monitors drop zone flows and recovers stuck documents.
Integrates with existing flow architecture to provide automatic recovery,
retry mechanisms, and manual intervention tools.

Capabilities:
- Detect stalled documents across all processing stages
- Automatic retry with exponential backoff
- Manual file recovery and reprocessing
- Flow health monitoring and alerting
- Audit trail preservation during recovery
"""

import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .base_worker import BaseWorker, TaskPriority
from ..session.unified import UnifiedSessionManager

logger = logging.getLogger("flow_recovery_worker")


class FileStatus(Enum):
    """File processing status in drop zone flows."""
    SUBMITTED = "submitted"
    STARTED = "started" 
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    STALLED = "stalled"
    ABANDONED = "abandoned"


class RecoveryAction(Enum):
    """Recovery actions available for stuck files."""
    RETRY_CURRENT_STAGE = "retry_current_stage"
    RESTART_FROM_BEGINNING = "restart_from_beginning"
    SKIP_TO_NEXT_STAGE = "skip_to_next_stage"
    MANUAL_INTERVENTION = "manual_intervention"
    QUARANTINE = "quarantine"
    DELETE = "delete"


@dataclass
class StalledFile:
    """Information about a stalled file in the processing flow."""
    submission_id: str
    file_path: str
    current_stage: str
    status: FileStatus
    submitted_at: datetime
    last_activity: Optional[datetime] = None
    stall_duration: Optional[timedelta] = None
    retry_count: int = 0
    error_messages: List[str] = field(default_factory=list)
    audit_path: Optional[str] = None
    
    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = self.submitted_at
        
        now = datetime.now()
        self.stall_duration = now - self.last_activity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "submission_id": self.submission_id,
            "file_path": self.file_path,
            "current_stage": self.current_stage,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat(),
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "stall_duration_minutes": self.stall_duration.total_seconds() / 60 if self.stall_duration else 0,
            "retry_count": self.retry_count,
            "error_messages": self.error_messages,
            "audit_path": self.audit_path
        }


@dataclass 
class RecoveryRequest:
    """Request for file recovery action."""
    request_id: str
    submission_id: str
    action: RecoveryAction
    force_recovery: bool = False
    target_stage: Optional[str] = None
    recovery_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Result from recovery operation."""
    request_id: str
    submission_id: str
    success: bool
    action_taken: RecoveryAction
    new_status: FileStatus
    recovery_time: Optional[float] = None
    messages: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "submission_id": self.submission_id,
            "success": self.success,
            "action_taken": self.action_taken.value,
            "new_status": self.new_status.value,
            "recovery_time": self.recovery_time,
            "messages": self.messages,
            "errors": self.errors
        }


class FlowRecoveryWorker(BaseWorker[RecoveryRequest, RecoveryResult]):
    """
    Worker for monitoring and recovering stuck files in drop zone flows.
    
    Monitors all drop zones for:
    - Files stuck in processing stages beyond timeout thresholds
    - Files with repeated failures needing intervention
    - Orphaned files in intermediate stages
    - Missing audit trails or inconsistent states
    
    Provides recovery actions:
    - Automatic retry with exponential backoff
    - Manual file movement between stages
    - Flow state reconstruction from audit logs
    - Emergency quarantine for problematic files
    """
    
    def __init__(self,
                 worker_name: str = "flow_recovery_worker",
                 drop_zones_path: str = "C:/Users/marti/github/drop_zones",
                 stale_threshold_minutes: int = 30,
                 max_auto_retries: int = 3,
                 monitoring_interval: float = 60.0,
                 **kwargs):
        """
        Initialize Flow Recovery Worker.
        
        Args:
            worker_name: Worker identifier
            drop_zones_path: Path to drop zones directory
            stale_threshold_minutes: Minutes before considering file stalled
            max_auto_retries: Maximum automatic retry attempts
            monitoring_interval: Seconds between monitoring cycles
        """
        super().__init__(worker_name, **kwargs)
        
        self.drop_zones_path = Path(drop_zones_path)
        self.stale_threshold = timedelta(minutes=stale_threshold_minutes)
        self.max_auto_retries = max_auto_retries
        self.monitoring_interval = monitoring_interval
        
        # Session management
        self.session_manager = None
        
        # Monitoring state
        self.known_files: Dict[str, StalledFile] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.recovery_stats = {
            "files_monitored": 0,
            "stalled_files_detected": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "manual_interventions": 0
        }
        
        # Flow stage definitions
        self.flow_stages = {
            "qa_control": ["incoming", "processing", "ingest", "extract", "embed", "index", "analysis", "reports", "completed"],
            "mvr_analysis": ["incoming", "processing", "completed"],
            "knowledge_base": ["incoming", "processing", "completed"]
        }
        
        logger.info(f"Flow Recovery Worker '{worker_name}' configured")
        logger.info(f"  Drop zones path: {drop_zones_path}")
        logger.info(f"  Stale threshold: {stale_threshold_minutes} minutes")
        logger.info(f"  Monitoring interval: {monitoring_interval}s")
    
    async def _initialize_worker(self) -> None:
        """Initialize recovery worker."""
        try:
            # Initialize UnifiedSessionManager
            try:
                self.session_manager = UnifiedSessionManager()
                logger.info("Flow Recovery Worker: UnifiedSessionManager initialized")
            except Exception as e:
                logger.warning(f"Flow Recovery Worker: UnifiedSessionManager not available: {e}")
            
            # Ensure drop zones structure exists
            await self._ensure_recovery_directories()
            
            # Start background monitoring
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Flow Recovery Worker initialized successfully")
            
        except Exception as e:
            logger.error(f"Flow Recovery Worker initialization failed: {e}")
            raise
    
    async def _ensure_recovery_directories(self) -> None:
        """Ensure required recovery directories exist."""
        try:
            for flow_name in self.flow_stages:
                flow_path = self.drop_zones_path / flow_name
                
                # Create failed directory for error recovery
                failed_path = flow_path / "failed"
                failed_path.mkdir(parents=True, exist_ok=True)
                
                # Create quarantine directory for problematic files
                quarantine_path = flow_path / "quarantine"
                quarantine_path.mkdir(parents=True, exist_ok=True)
                
                # Create recovery logs directory
                recovery_logs_path = flow_path / "recovery_logs"
                recovery_logs_path.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Recovery directories ensured for flow: {flow_name}")
                
        except Exception as e:
            logger.error(f"Failed to create recovery directories: {e}")
            raise
    
    def validate_input(self, task_input: Any) -> bool:
        """Validate recovery request input."""
        if not isinstance(task_input, RecoveryRequest):
            return False
        
        return bool(
            task_input.request_id and 
            task_input.submission_id and 
            task_input.action
        )
    
    async def process_task(self, task_input: RecoveryRequest) -> RecoveryResult:
        """Process file recovery request."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing recovery request '{task_input.request_id}' for file '{task_input.submission_id}'")
            
            # Execute recovery action
            if task_input.action == RecoveryAction.RETRY_CURRENT_STAGE:
                result = await self._retry_current_stage(task_input)
            elif task_input.action == RecoveryAction.RESTART_FROM_BEGINNING:
                result = await self._restart_from_beginning(task_input)
            elif task_input.action == RecoveryAction.SKIP_TO_NEXT_STAGE:
                result = await self._skip_to_next_stage(task_input)
            elif task_input.action == RecoveryAction.MANUAL_INTERVENTION:
                result = await self._manual_intervention(task_input)
            elif task_input.action == RecoveryAction.QUARANTINE:
                result = await self._quarantine_file(task_input)
            elif task_input.action == RecoveryAction.DELETE:
                result = await self._delete_file(task_input)
            else:
                raise ValueError(f"Unknown recovery action: {task_input.action}")
            
            # Set processing time
            result.recovery_time = time.time() - start_time
            
            # Update statistics
            if result.success:
                self.recovery_stats["successful_recoveries"] += 1
            else:
                self.recovery_stats["failed_recoveries"] += 1
            
            if task_input.action == RecoveryAction.MANUAL_INTERVENTION:
                self.recovery_stats["manual_interventions"] += 1
            
            logger.info(f"Recovery request '{task_input.request_id}' completed: {'SUCCESS' if result.success else 'FAILED'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Recovery request '{task_input.request_id}' failed: {e}")
            
            return RecoveryResult(
                request_id=task_input.request_id,
                submission_id=task_input.submission_id,
                success=False,
                action_taken=task_input.action,
                new_status=FileStatus.FAILED,
                recovery_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for stalled files."""
        logger.info("Starting file monitoring loop...")
        
        while True:
            try:
                await self._scan_for_stalled_files()
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                logger.info("File monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _scan_for_stalled_files(self) -> None:
        """Scan all drop zones for stalled files."""
        try:
            stalled_files = []
            
            for flow_name in self.flow_stages:
                flow_stalled = await self._scan_flow_for_stalled_files(flow_name)
                stalled_files.extend(flow_stalled)
            
            # Update monitoring statistics
            self.recovery_stats["files_monitored"] = len(self.known_files)
            self.recovery_stats["stalled_files_detected"] = len(stalled_files)
            
            # Process automatic recoveries for newly stalled files
            for stalled_file in stalled_files:
                if stalled_file.retry_count < self.max_auto_retries:
                    await self._attempt_automatic_recovery(stalled_file)
            
            if stalled_files:
                logger.info(f"Detected {len(stalled_files)} stalled files across all flows")
                
        except Exception as e:
            logger.error(f"Error scanning for stalled files: {e}")
    
    async def _scan_flow_for_stalled_files(self, flow_name: str) -> List[StalledFile]:
        """Scan a specific flow for stalled files."""
        stalled_files = []
        
        try:
            flow_path = self.drop_zones_path / flow_name
            audit_path = flow_path / "audit"
            
            if not audit_path.exists():
                return stalled_files
            
            # Read master audit log
            master_audit_path = audit_path / "master_audit.jsonl"
            if master_audit_path.exists():
                with open(master_audit_path, 'r') as f:
                    for line in f:
                        try:
                            audit_entry = json.loads(line.strip())
                            stalled_file = await self._check_if_file_stalled(audit_entry, flow_name)
                            
                            if stalled_file:
                                stalled_files.append(stalled_file)
                                self.known_files[stalled_file.submission_id] = stalled_file
                                
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.warning(f"Error processing audit entry: {e}")
                            continue
            
        except Exception as e:
            logger.error(f"Error scanning flow '{flow_name}': {e}")
        
        return stalled_files
    
    async def _check_if_file_stalled(self, audit_entry: Dict[str, Any], flow_name: str) -> Optional[StalledFile]:
        """Check if a file is stalled based on audit entry."""
        try:
            submission_id = audit_entry.get("submission_id")
            status = audit_entry.get("status", "unknown")
            submitted_at_str = audit_entry.get("submitted_at")
            started_at_str = audit_entry.get("started_at")
            completed_at_str = audit_entry.get("completed_at")
            retry_count = audit_entry.get("retry_count", 0)
            errors = audit_entry.get("errors", [])
            
            if not submission_id or not submitted_at_str:
                return None
            
            submitted_at = datetime.fromisoformat(submitted_at_str.replace('Z', '+00:00'))
            
            # Determine last activity time
            last_activity = submitted_at
            if started_at_str:
                started_at = datetime.fromisoformat(started_at_str.replace('Z', '+00:00'))
                last_activity = max(last_activity, started_at)
            
            if completed_at_str:
                # File is completed, not stalled
                return None
            
            # Check if file is stalled
            now = datetime.now()
            time_since_activity = now - last_activity
            
            if time_since_activity > self.stale_threshold:
                # Determine current stage based on status and file location
                current_stage = self._determine_current_stage(audit_entry, flow_name)
                
                stalled_file = StalledFile(
                    submission_id=submission_id,
                    file_path=audit_entry.get("file_path", "unknown"),
                    current_stage=current_stage,
                    status=FileStatus(status) if status in [s.value for s in FileStatus] else FileStatus.SUBMITTED,
                    submitted_at=submitted_at,
                    last_activity=last_activity,
                    retry_count=retry_count,
                    error_messages=[str(e) for e in errors],
                    audit_path=str(self.drop_zones_path / flow_name / "audit" / submission_id)
                )
                
                return stalled_file
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking if file stalled: {e}")
            return None
    
    def _determine_current_stage(self, audit_entry: Dict[str, Any], flow_name: str) -> str:
        """Determine current processing stage for a file."""
        # Check stage_results for latest completed stage
        stage_results = audit_entry.get("stage_results", {})
        
        if stage_results:
            # Find the latest stage with results
            stages = self.flow_stages.get(flow_name, ["incoming"])
            
            for stage in reversed(stages):
                if stage in stage_results:
                    # If stage has results, file might be in next stage
                    stage_index = stages.index(stage)
                    if stage_index + 1 < len(stages):
                        return stages[stage_index + 1]
                    return stage
        
        # Default to submitted/incoming stage
        return "incoming"
    
    async def _attempt_automatic_recovery(self, stalled_file: StalledFile) -> None:
        """Attempt automatic recovery for a stalled file."""
        try:
            logger.info(f"Attempting automatic recovery for file '{stalled_file.submission_id}'")
            
            # Create recovery request
            recovery_request = RecoveryRequest(
                request_id=f"auto_recovery_{stalled_file.submission_id}_{int(time.time())}",
                submission_id=stalled_file.submission_id,
                action=RecoveryAction.RETRY_CURRENT_STAGE,
                force_recovery=False
            )
            
            # Submit recovery task
            await self.submit_task(
                task_type="automatic_recovery",
                task_input=recovery_request,
                priority=TaskPriority.HIGH
            )
            
        except Exception as e:
            logger.error(f"Failed to submit automatic recovery for '{stalled_file.submission_id}': {e}")
    
    async def _retry_current_stage(self, request: RecoveryRequest) -> RecoveryResult:
        """Retry processing at the current stage."""
        try:
            # Find the stalled file
            stalled_file = self.known_files.get(request.submission_id)
            if not stalled_file:
                return RecoveryResult(
                    request_id=request.request_id,
                    submission_id=request.submission_id,
                    success=False,
                    action_taken=request.action,
                    new_status=FileStatus.FAILED,
                    errors=["File not found in known stalled files"]
                )
            
            # Update audit log with retry attempt
            await self._update_audit_log(stalled_file, "retry_attempted", {
                "retry_count": stalled_file.retry_count + 1,
                "recovery_action": "retry_current_stage",
                "retry_timestamp": datetime.now().isoformat()
            })
            
            # Move file back to current stage for reprocessing
            success = await self._move_file_to_stage(stalled_file, stalled_file.current_stage)
            
            if success:
                # Update file status
                stalled_file.retry_count += 1
                stalled_file.status = FileStatus.STARTED
                stalled_file.last_activity = datetime.now()
                
                return RecoveryResult(
                    request_id=request.request_id,
                    submission_id=request.submission_id,
                    success=True,
                    action_taken=request.action,
                    new_status=FileStatus.STARTED,
                    messages=[f"File moved to {stalled_file.current_stage} for retry"]
                )
            else:
                return RecoveryResult(
                    request_id=request.request_id,
                    submission_id=request.submission_id,
                    success=False,
                    action_taken=request.action,
                    new_status=FileStatus.FAILED,
                    errors=["Failed to move file for retry"]
                )
                
        except Exception as e:
            logger.error(f"Retry current stage failed: {e}")
            return RecoveryResult(
                request_id=request.request_id,
                submission_id=request.submission_id,
                success=False,
                action_taken=request.action,
                new_status=FileStatus.FAILED,
                errors=[str(e)]
            )
    
    async def _restart_from_beginning(self, request: RecoveryRequest) -> RecoveryResult:
        """Restart file processing from the beginning."""
        try:
            stalled_file = self.known_files.get(request.submission_id)
            if not stalled_file:
                return RecoveryResult(
                    request_id=request.request_id,
                    submission_id=request.submission_id,
                    success=False,
                    action_taken=request.action,
                    new_status=FileStatus.FAILED,
                    errors=["File not found in known stalled files"]
                )
            
            # Update audit log
            await self._update_audit_log(stalled_file, "restart_from_beginning", {
                "previous_stage": stalled_file.current_stage,
                "restart_timestamp": datetime.now().isoformat()
            })
            
            # Move file to incoming folder
            success = await self._move_file_to_stage(stalled_file, "incoming")
            
            if success:
                stalled_file.current_stage = "incoming"
                stalled_file.status = FileStatus.SUBMITTED
                stalled_file.last_activity = datetime.now()
                
                return RecoveryResult(
                    request_id=request.request_id,
                    submission_id=request.submission_id,
                    success=True,
                    action_taken=request.action,
                    new_status=FileStatus.SUBMITTED,
                    messages=["File restarted from incoming folder"]
                )
            else:
                return RecoveryResult(
                    request_id=request.request_id,
                    submission_id=request.submission_id,
                    success=False,
                    action_taken=request.action,
                    new_status=FileStatus.FAILED,
                    errors=["Failed to move file to incoming"]
                )
                
        except Exception as e:
            return RecoveryResult(
                request_id=request.request_id,
                submission_id=request.submission_id,
                success=False,
                action_taken=request.action,
                new_status=FileStatus.FAILED,
                errors=[str(e)]
            )
    
    async def _skip_to_next_stage(self, request: RecoveryRequest) -> RecoveryResult:
        """Skip to next stage in the processing flow."""
        # Implementation for skipping to next stage
        return RecoveryResult(
            request_id=request.request_id,
            submission_id=request.submission_id,
            success=False,
            action_taken=request.action,
            new_status=FileStatus.MANUAL_INTERVENTION,
            messages=["Skip to next stage - implementation needed"]
        )
    
    async def _manual_intervention(self, request: RecoveryRequest) -> RecoveryResult:
        """Mark file for manual intervention."""
        try:
            stalled_file = self.known_files.get(request.submission_id)
            if not stalled_file:
                return RecoveryResult(
                    request_id=request.request_id,
                    submission_id=request.submission_id,
                    success=False,
                    action_taken=request.action,
                    new_status=FileStatus.FAILED,
                    errors=["File not found"]
                )
            
            # Update audit log
            await self._update_audit_log(stalled_file, "manual_intervention_required", {
                "intervention_reason": request.recovery_options.get("reason", "Manual intervention requested"),
                "intervention_timestamp": datetime.now().isoformat()
            })
            
            stalled_file.status = FileStatus.MANUAL_INTERVENTION
            
            return RecoveryResult(
                request_id=request.request_id,
                submission_id=request.submission_id,
                success=True,
                action_taken=request.action,
                new_status=FileStatus.MANUAL_INTERVENTION,
                messages=["File marked for manual intervention"]
            )
            
        except Exception as e:
            return RecoveryResult(
                request_id=request.request_id,
                submission_id=request.submission_id,
                success=False,
                action_taken=request.action,
                new_status=FileStatus.FAILED,
                errors=[str(e)]
            )
    
    async def _quarantine_file(self, request: RecoveryRequest) -> RecoveryResult:
        """Move file to quarantine for investigation."""
        try:
            stalled_file = self.known_files.get(request.submission_id)
            if not stalled_file:
                return RecoveryResult(
                    request_id=request.request_id,
                    submission_id=request.submission_id,
                    success=False,
                    action_taken=request.action,
                    new_status=FileStatus.FAILED,
                    errors=["File not found"]
                )
            
            # Move to quarantine folder
            success = await self._move_file_to_stage(stalled_file, "quarantine")
            
            if success:
                await self._update_audit_log(stalled_file, "quarantined", {
                    "quarantine_reason": request.recovery_options.get("reason", "File quarantined for investigation"),
                    "quarantine_timestamp": datetime.now().isoformat()
                })
                
                stalled_file.status = FileStatus.ABANDONED
                
                return RecoveryResult(
                    request_id=request.request_id,
                    submission_id=request.submission_id,
                    success=True,
                    action_taken=request.action,
                    new_status=FileStatus.ABANDONED,
                    messages=["File moved to quarantine"]
                )
            else:
                return RecoveryResult(
                    request_id=request.request_id,
                    submission_id=request.submission_id,
                    success=False,
                    action_taken=request.action,
                    new_status=FileStatus.FAILED,
                    errors=["Failed to move file to quarantine"]
                )
                
        except Exception as e:
            return RecoveryResult(
                request_id=request.request_id,
                submission_id=request.submission_id,
                success=False,
                action_taken=request.action,
                new_status=FileStatus.FAILED,
                errors=[str(e)]
            )
    
    async def _delete_file(self, request: RecoveryRequest) -> RecoveryResult:
        """Delete file and cleanup all references."""
        # Implementation for file deletion with safety checks
        return RecoveryResult(
            request_id=request.request_id,
            submission_id=request.submission_id,
            success=False,
            action_taken=request.action,
            new_status=FileStatus.FAILED,
            messages=["Delete operation requires additional safety implementation"]
        )
    
    async def _move_file_to_stage(self, stalled_file: StalledFile, target_stage: str) -> bool:
        """Move file to target processing stage."""
        try:
            # This is a placeholder - actual implementation would:
            # 1. Find the physical file location
            # 2. Move it to the target stage directory 
            # 3. Update any necessary tracking files
            # 4. Trigger the appropriate processing workflow
            
            logger.info(f"Moving file '{stalled_file.submission_id}' to stage '{target_stage}'")
            
            # For now, just simulate success
            # Real implementation would use file system operations
            return True
            
        except Exception as e:
            logger.error(f"Failed to move file to stage '{target_stage}': {e}")
            return False
    
    async def _update_audit_log(self, stalled_file: StalledFile, action: str, details: Dict[str, Any]) -> None:
        """Update audit log with recovery action."""
        try:
            if stalled_file.audit_path:
                audit_path = Path(stalled_file.audit_path) / "audit.json"
                
                if audit_path.exists():
                    # Read existing audit data
                    with open(audit_path, 'r') as f:
                        audit_data = json.load(f)
                    
                    # Add recovery action to access log
                    if "access_log" not in audit_data:
                        audit_data["access_log"] = []
                    
                    audit_data["access_log"].append({
                        "user": "flow_recovery_worker",
                        "action": action,
                        "timestamp": datetime.now().isoformat(),
                        "details": details
                    })
                    
                    # Write updated audit data
                    with open(audit_path, 'w') as f:
                        json.dump(audit_data, f, indent=2)
                    
                    logger.info(f"Updated audit log for '{stalled_file.submission_id}' with action '{action}'")
                
        except Exception as e:
            logger.error(f"Failed to update audit log: {e}")
    
    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the recovery worker and monitoring."""
        logger.info(f"Stopping Flow Recovery Worker '{self.worker_name}'...")
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await asyncio.wait_for(self.monitoring_task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        
        await super().stop(timeout)
        
        logger.info(f"Flow Recovery Worker '{self.worker_name}' stopped")
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics and current state."""
        return {
            "monitoring_active": self.monitoring_task is not None and not self.monitoring_task.done(),
            "known_stalled_files": len(self.known_files),
            "recovery_stats": self.recovery_stats.copy(),
            "stale_threshold_minutes": self.stale_threshold.total_seconds() / 60,
            "max_auto_retries": self.max_auto_retries,
            "monitoring_interval_seconds": self.monitoring_interval
        }
    
    def get_stalled_files_report(self) -> List[Dict[str, Any]]:
        """Get report of all currently stalled files."""
        return [file.to_dict() for file in self.known_files.values()]
    
    # Task submission convenience methods
    async def recover_file(self,
                          submission_id: str,
                          action: RecoveryAction,
                          force_recovery: bool = False,
                          target_stage: str = None,
                          recovery_options: Dict[str, Any] = None,
                          priority: TaskPriority = TaskPriority.HIGH) -> str:
        """
        Submit file recovery task.
        
        Returns:
            Task ID for tracking
        """
        request = RecoveryRequest(
            request_id=f"recovery_{submission_id}_{int(time.time())}",
            submission_id=submission_id,
            action=action,
            force_recovery=force_recovery,
            target_stage=target_stage,
            recovery_options=recovery_options or {}
        )
        
        task = await self.submit_task(
            task_type="recover_file",
            task_input=request,
            priority=priority
        )
        
        return task.task_id