#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Gateway - Foundation for all TidyLLM Gateway implementations

Provides the core enterprise features that all gateways inherit:
- Security and authentication
- Audit logging and compliance
- Rate limiting and quota management
- Health monitoring and alerting
- Policy enforcement
- Multi-tenant support
"""

from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
import uuid
import hashlib
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class GatewayStatus(Enum):
    """Gateway operational status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"


class AccessLevel(Enum):
    """User access levels"""
    READ_ONLY = "read_only"
    STANDARD = "standard"
    ELEVATED = "elevated"
    ADMIN = "admin"


@dataclass
class GatewayConfig:
    """Base configuration for all gateways"""
    
    # Connection settings
    base_url: str
    timeout: int = 30
    max_retries: int = 3
    
    # Security settings
    auth_token: Optional[str] = None
    auth_method: str = "bearer"  # bearer, basic, oauth, kerberos
    tls_verify: bool = True
    
    # Enterprise features
    enable_audit_logging: bool = True
    enable_rate_limiting: bool = True
    enable_quota_management: bool = True
    enable_health_monitoring: bool = True
    
    # Tenant settings
    tenant_id: Optional[str] = None
    department: Optional[str] = None
    cost_center: Optional[str] = None
    
    # Policy settings
    policy_file: Optional[str] = None
    compliance_mode: str = "standard"  # strict, standard, permissive
    
    # Performance settings
    connection_pool_size: int = 10
    keep_alive: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "auth_method": self.auth_method,
            "tls_verify": self.tls_verify,
            "tenant_id": self.tenant_id,
            "department": self.department,
            "compliance_mode": self.compliance_mode
        }


@dataclass  
class GatewayResponse:
    """Standardized response from any gateway"""
    
    # Response data
    success: bool
    data: Any
    error: Optional[str] = None
    
    # Metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None
    
    # Enterprise tracking
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    audit_trail: List[str] = field(default_factory=list)
    
    # Performance metrics
    tokens_consumed: Optional[int] = None
    cost_usd: Optional[float] = None
    rate_limit_remaining: Optional[int] = None
    
    # Compliance
    data_classification: str = "internal"
    retention_policy: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "processing_time_ms": self.processing_time_ms,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "tokens_consumed": self.tokens_consumed,
            "cost_usd": self.cost_usd,
            "data_classification": self.data_classification
        }


@dataclass
class AuditRecord:
    """Audit record for compliance and security"""
    
    # Request identification
    request_id: str
    timestamp: datetime
    gateway_type: str
    
    # User context
    user_id: str
    tenant_id: Optional[str]
    department: Optional[str]
    
    # Request details
    endpoint: str
    method: str
    parameters: Dict[str, Any]
    
    # Response details
    success: bool
    response_size_bytes: Optional[int]
    processing_time_ms: float
    
    # Security
    client_ip: Optional[str]
    user_agent: Optional[str]
    auth_method: str
    
    # Compliance
    data_classification: str
    audit_reason: Optional[str]
    
    # Cost tracking
    tokens_consumed: Optional[int] = 0
    cost_usd: Optional[float] = 0.0
    
    def to_json(self) -> str:
        """Convert audit record to JSON for storage"""
        data = {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "gateway_type": self.gateway_type,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "department": self.department,
            "endpoint": self.endpoint,
            "method": self.method,
            "parameters": self._sanitize_parameters(self.parameters),
            "success": self.success,
            "response_size_bytes": self.response_size_bytes,
            "processing_time_ms": self.processing_time_ms,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "auth_method": self.auth_method,
            "data_classification": self.data_classification,
            "audit_reason": self.audit_reason,
            "tokens_consumed": self.tokens_consumed,
            "cost_usd": self.cost_usd
        }
        return json.dumps(data)
    
    def _sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from parameters for audit log"""
        sanitized = {}
        for key, value in params.items():
            if any(sensitive in key.lower() for sensitive in ['password', 'token', 'key', 'secret']):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = str(value)[:200]  # Truncate long values
        return sanitized


class BaseGateway(ABC):
    """
    Abstract base class for all TidyLLM gateways.
    
    Provides enterprise features:
    - Authentication and authorization
    - Audit logging and compliance
    - Rate limiting and quota management
    - Health monitoring
    - Policy enforcement
    - Multi-tenant support
    """
    
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.gateway_type = self.__class__.__name__
        self.status = GatewayStatus.HEALTHY
        self.request_history: List[AuditRecord] = []
        
        # Initialize enterprise features
        self._init_security()
        self._init_audit_logging()
        self._init_rate_limiting()
        self._init_health_monitoring()
        
        logger.info(f"🚀 {self.gateway_type} initialized")
        logger.info(f"   Base URL: {config.base_url}")
        logger.info(f"   Tenant: {config.tenant_id}")
        logger.info(f"   Compliance: {config.compliance_mode}")
    
    def _init_security(self):
        """Initialize security components"""
        # This would integrate with corporate security systems
        self.security_enabled = True
        logger.info("🔒 Security manager initialized")
    
    def _init_audit_logging(self):
        """Initialize audit logging"""
        if self.config.enable_audit_logging:
            self.audit_logger = logging.getLogger(f"{self.gateway_type}.audit")
            logger.info("📝 Audit logging enabled")
    
    def _init_rate_limiting(self):
        """Initialize rate limiting"""
        if self.config.enable_rate_limiting:
            self.rate_limits = {
                "requests_per_minute": 1000,
                "requests_per_hour": 10000,
                "requests_per_day": 50000
            }
            logger.info("🚦 Rate limiting enabled")
    
    def _init_health_monitoring(self):
        """Initialize health monitoring"""
        if self.config.enable_health_monitoring:
            self.health_metrics = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_response_time_ms": 0.0,
                "last_health_check": datetime.utcnow()
            }
            logger.info("📊 Health monitoring enabled")
    
    @abstractmethod
    def _execute_request(self, endpoint: str, data: Dict[str, Any], **kwargs) -> Any:
        """Execute the actual request - implemented by specific gateways"""
        pass
    
    def execute(
        self, 
        endpoint: str, 
        data: Dict[str, Any],
        user_id: str,
        audit_reason: Optional[str] = None,
        **kwargs
    ) -> GatewayResponse:
        """
        Execute a request through the gateway with full enterprise features.
        
        Args:
            endpoint: Service endpoint to call
            data: Request data
            user_id: User making the request (for audit)
            audit_reason: Reason for request (for compliance)
            **kwargs: Additional parameters
            
        Returns:
            GatewayResponse with standardized format
        """
        
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            # Pre-request validation
            self._validate_request(endpoint, data, user_id)
            self._check_rate_limits(user_id)
            self._enforce_policies(endpoint, data, user_id)
            
            # Execute request
            logger.info(f"🔄 Executing {self.gateway_type} request: {endpoint}")
            result = self._execute_request(endpoint, data, **kwargs)
            
            # Calculate metrics
            end_time = datetime.utcnow()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Create response
            response = GatewayResponse(
                success=True,
                data=result,
                request_id=request_id,
                timestamp=end_time,
                processing_time_ms=processing_time_ms,
                tenant_id=self.config.tenant_id,
                user_id=user_id
            )
            
            # Post-request processing
            self._update_metrics(True, processing_time_ms)
            self._log_audit_record(request_id, endpoint, data, user_id, response, audit_reason)
            
            logger.info(f"✅ {self.gateway_type} request completed in {processing_time_ms:.1f}ms")
            return response
            
        except Exception as e:
            # Error handling
            end_time = datetime.utcnow() 
            processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            error_response = GatewayResponse(
                success=False,
                data=None,
                error=str(e),
                request_id=request_id,
                timestamp=end_time,
                processing_time_ms=processing_time_ms,
                tenant_id=self.config.tenant_id,
                user_id=user_id
            )
            
            self._update_metrics(False, processing_time_ms)
            self._log_audit_record(request_id, endpoint, data, user_id, error_response, audit_reason)
            
            logger.error(f"❌ {self.gateway_type} request failed: {e}")
            return error_response
    
    def _validate_request(self, endpoint: str, data: Dict[str, Any], user_id: str):
        """Validate request before execution"""
        if not endpoint:
            raise ValueError("Endpoint cannot be empty")
        if not user_id:
            raise ValueError("User ID required for audit trail")
        if not data:
            raise ValueError("Request data cannot be empty")
    
    def _check_rate_limits(self, user_id: str):
        """Check if user has exceeded rate limits"""
        if not self.config.enable_rate_limiting:
            return
        
        # Simple rate limiting logic (would be more sophisticated in production)
        recent_requests = [
            r for r in self.request_history 
            if r.user_id == user_id and 
            r.timestamp > datetime.utcnow() - timedelta(minutes=1)
        ]
        
        if len(recent_requests) > self.rate_limits["requests_per_minute"]:
            raise ValueError(f"Rate limit exceeded for user {user_id}")
    
    def _enforce_policies(self, endpoint: str, data: Dict[str, Any], user_id: str):
        """Enforce corporate policies"""
        # Policy enforcement would be implemented here
        # Example: Check if user has permission for this endpoint
        # Example: Check if data classification allows this operation
        pass
    
    def _update_metrics(self, success: bool, processing_time_ms: float):
        """Update health metrics"""
        if not self.config.enable_health_monitoring:
            return
        
        self.health_metrics["total_requests"] += 1
        if success:
            self.health_metrics["successful_requests"] += 1
        else:
            self.health_metrics["failed_requests"] += 1
        
        # Update average response time
        total = self.health_metrics["total_requests"]
        current_avg = self.health_metrics["avg_response_time_ms"]
        self.health_metrics["avg_response_time_ms"] = (
            (current_avg * (total - 1) + processing_time_ms) / total
        )
    
    def _log_audit_record(
        self, 
        request_id: str, 
        endpoint: str, 
        data: Dict[str, Any], 
        user_id: str,
        response: GatewayResponse,
        audit_reason: Optional[str]
    ):
        """Log request for audit and compliance"""
        if not self.config.enable_audit_logging:
            return
        
        audit_record = AuditRecord(
            request_id=request_id,
            timestamp=response.timestamp,
            gateway_type=self.gateway_type,
            user_id=user_id,
            tenant_id=self.config.tenant_id,
            department=self.config.department,
            endpoint=endpoint,
            method="POST",  # Most gateway calls are POST
            parameters=data,
            success=response.success,
            response_size_bytes=len(str(response.data)) if response.data else 0,
            processing_time_ms=response.processing_time_ms or 0,
            client_ip=None,  # Would be populated in real implementation
            user_agent=None,  # Would be populated in real implementation  
            auth_method=self.config.auth_method,
            data_classification=response.data_classification,
            audit_reason=audit_reason,
            tokens_consumed=response.tokens_consumed or 0,
            cost_usd=response.cost_usd or 0.0
        )
        
        # Store audit record
        self.request_history.append(audit_record)
        
        # Log to audit system
        self.audit_logger.info(audit_record.to_json())
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status and metrics"""
        return {
            "gateway_type": self.gateway_type,
            "status": self.status.value,
            "config": self.config.to_dict(),
            "metrics": self.health_metrics,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_records = [
            r for r in self.request_history 
            if r.timestamp > cutoff_time
        ]
        
        total_requests = len(recent_records)
        successful_requests = sum(1 for r in recent_records if r.success)
        failed_requests = total_requests - successful_requests
        total_cost = sum(r.cost_usd or 0 for r in recent_records)
        
        return {
            "time_period_hours": hours,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            "total_cost_usd": total_cost,
            "average_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "unique_users": len(set(r.user_id for r in recent_records)),
            "most_active_endpoint": self._get_most_active_endpoint(recent_records)
        }
    
    def _get_most_active_endpoint(self, records: List[AuditRecord]) -> str:
        """Find the most frequently used endpoint"""
        if not records:
            return "none"
        
        endpoint_counts = {}
        for record in records:
            endpoint_counts[record.endpoint] = endpoint_counts.get(record.endpoint, 0) + 1
        
        return max(endpoint_counts.items(), key=lambda x: x[1])[0]