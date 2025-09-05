#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rate Limiter and Quota Manager

Enterprise rate limiting and quota management for gateway access control.
Provides flexible rate limiting strategies and budget controls.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


class RateLimitStrategy(Enum):
    """Rate limiting strategy types"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class QuotaPeriod(Enum):
    """Quota time periods"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests: int                    # Number of requests
    period: QuotaPeriod             # Time period
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_allowance: int = 0        # Additional burst capacity
    cooldown_seconds: int = 0       # Cooldown after limit hit


@dataclass
class QuotaConfig:
    """Quota configuration for resource limits"""
    
    # Request limits
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    
    # Token/usage limits
    tokens_per_hour: Optional[int] = None
    tokens_per_day: Optional[int] = None
    
    # Cost limits
    cost_per_hour_usd: Optional[float] = None
    cost_per_day_usd: Optional[float] = None
    cost_per_month_usd: Optional[float] = None
    
    # Per-user limits
    user_requests_per_minute: Optional[int] = None
    user_requests_per_hour: Optional[int] = None
    
    # Department/tenant limits
    tenant_requests_per_hour: Optional[int] = None
    tenant_cost_per_day_usd: Optional[float] = None


@dataclass
class UsageMetrics:
    """Usage tracking metrics"""
    requests: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    first_request: Optional[datetime] = None
    last_request: Optional[datetime] = None
    
    def add_usage(self, tokens: int = 0, cost_usd: float = 0.0):
        """Add usage to metrics"""
        now = datetime.utcnow()
        
        self.requests += 1
        self.tokens += tokens
        self.cost_usd += cost_usd
        
        if self.first_request is None:
            self.first_request = now
        self.last_request = now


class RateLimiter:
    """
    Enterprise rate limiter with multiple strategies
    
    Features:
    - Multiple rate limiting algorithms
    - User and tenant-level controls
    - Burst capacity management
    - Cooldown periods
    - Real-time metrics
    - Graceful degradation
    """
    
    def __init__(self, quota_config: QuotaConfig):
        self.config = quota_config
        
        # User tracking
        self.user_windows: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self.user_tokens: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.user_cooldowns: Dict[str, datetime] = {}
        
        # Tenant tracking
        self.tenant_windows: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self.tenant_tokens: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Global tracking
        self.global_windows: Dict[str, deque] = defaultdict(deque)
        self.global_tokens: Dict[str, int] = defaultdict(int)
        
        # Usage tracking
        self.usage_by_user: Dict[str, Dict[str, UsageMetrics]] = defaultdict(lambda: defaultdict(UsageMetrics))
        self.usage_by_tenant: Dict[str, Dict[str, UsageMetrics]] = defaultdict(lambda: defaultdict(UsageMetrics))
        
        self.logger = logging.getLogger(f"{__name__}.RateLimiter")
        self.logger.info("🚦 Rate Limiter initialized")
    
    def check_rate_limit(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        endpoint: str = "default",
        tokens_requested: int = 0,
        cost_usd: float = 0.0
    ) -> Dict[str, Any]:
        """
        Check if request is within rate limits
        
        Args:
            user_id: User making the request
            tenant_id: Tenant/department ID
            endpoint: Specific endpoint being accessed
            tokens_requested: Number of tokens to be consumed
            cost_usd: Estimated cost of request
            
        Returns:
            Rate limit check result
        """
        
        now = datetime.utcnow()
        
        try:
            # Check user cooldown
            if self._is_user_in_cooldown(user_id, now):
                return self._create_rate_limit_response(False, "User in cooldown period")
            
            # Check user limits
            user_check = self._check_user_limits(user_id, endpoint, tokens_requested, cost_usd, now)
            if not user_check["allowed"]:
                return self._create_rate_limit_response(False, user_check["reason"])
            
            # Check tenant limits if applicable
            if tenant_id:
                tenant_check = self._check_tenant_limits(tenant_id, endpoint, tokens_requested, cost_usd, now)
                if not tenant_check["allowed"]:
                    return self._create_rate_limit_response(False, tenant_check["reason"])
            
            # Check global limits
            global_check = self._check_global_limits(endpoint, tokens_requested, cost_usd, now)
            if not global_check["allowed"]:
                return self._create_rate_limit_response(False, global_check["reason"])
            
            # All checks passed - record usage
            self._record_usage(user_id, tenant_id, endpoint, tokens_requested, cost_usd, now)
            
            # Get remaining quota info
            remaining_info = self._get_remaining_quotas(user_id, tenant_id, endpoint, now)
            
            return self._create_rate_limit_response(
                True, 
                "Request allowed",
                remaining=remaining_info
            )
            
        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
            # Fail open - allow request but log error
            return self._create_rate_limit_response(True, "Rate limit check failed - allowing request")
    
    def _is_user_in_cooldown(self, user_id: str, now: datetime) -> bool:
        """Check if user is in cooldown period"""
        if user_id not in self.user_cooldowns:
            return False
        
        return now < self.user_cooldowns[user_id]
    
    def _check_user_limits(
        self, 
        user_id: str, 
        endpoint: str, 
        tokens_requested: int, 
        cost_usd: float, 
        now: datetime
    ) -> Dict[str, Any]:
        """Check user-specific rate limits"""
        
        # Check per-minute limits
        if self.config.user_requests_per_minute:
            if not self._check_sliding_window_limit(
                self.user_windows[user_id]["minute"],
                self.config.user_requests_per_minute,
                60,  # 60 seconds
                now
            ):
                return {"allowed": False, "reason": "User minute limit exceeded"}
        
        # Check per-hour limits
        if self.config.user_requests_per_hour:
            if not self._check_sliding_window_limit(
                self.user_windows[user_id]["hour"],
                self.config.user_requests_per_hour,
                3600,  # 3600 seconds
                now
            ):
                return {"allowed": False, "reason": "User hour limit exceeded"}
        
        return {"allowed": True}
    
    def _check_tenant_limits(
        self, 
        tenant_id: str, 
        endpoint: str, 
        tokens_requested: int, 
        cost_usd: float, 
        now: datetime
    ) -> Dict[str, Any]:
        """Check tenant-specific rate limits"""
        
        # Check tenant request limits
        if self.config.tenant_requests_per_hour:
            if not self._check_sliding_window_limit(
                self.tenant_windows[tenant_id]["hour"],
                self.config.tenant_requests_per_hour,
                3600,
                now
            ):
                return {"allowed": False, "reason": "Tenant hour limit exceeded"}
        
        # Check tenant cost limits
        if self.config.tenant_cost_per_day_usd:
            daily_usage = self.usage_by_tenant[tenant_id]["day"]
            
            # Reset daily usage if needed
            if self._should_reset_period(daily_usage.first_request, now, "day"):
                self.usage_by_tenant[tenant_id]["day"] = UsageMetrics()
                daily_usage = self.usage_by_tenant[tenant_id]["day"]
            
            if daily_usage.cost_usd + cost_usd > self.config.tenant_cost_per_day_usd:
                return {"allowed": False, "reason": "Tenant daily cost limit exceeded"}
        
        return {"allowed": True}
    
    def _check_global_limits(
        self, 
        endpoint: str, 
        tokens_requested: int, 
        cost_usd: float, 
        now: datetime
    ) -> Dict[str, Any]:
        """Check global rate limits"""
        
        # Check global request limits
        if self.config.requests_per_minute:
            if not self._check_sliding_window_limit(
                self.global_windows["minute"],
                self.config.requests_per_minute,
                60,
                now
            ):
                return {"allowed": False, "reason": "Global minute limit exceeded"}
        
        if self.config.requests_per_hour:
            if not self._check_sliding_window_limit(
                self.global_windows["hour"],
                self.config.requests_per_hour,
                3600,
                now
            ):
                return {"allowed": False, "reason": "Global hour limit exceeded"}
        
        if self.config.requests_per_day:
            if not self._check_sliding_window_limit(
                self.global_windows["day"],
                self.config.requests_per_day,
                86400,  # 24 hours
                now
            ):
                return {"allowed": False, "reason": "Global day limit exceeded"}
        
        return {"allowed": True}
    
    def _check_sliding_window_limit(
        self, 
        window: deque, 
        limit: int, 
        window_seconds: int, 
        now: datetime
    ) -> bool:
        """Check sliding window rate limit"""
        
        # Remove old entries outside the window
        cutoff_time = now - timedelta(seconds=window_seconds)
        while window and window[0] <= cutoff_time:
            window.popleft()
        
        # Check if adding this request would exceed limit
        return len(window) < limit
    
    def _record_usage(
        self, 
        user_id: str, 
        tenant_id: Optional[str], 
        endpoint: str, 
        tokens_used: int, 
        cost_usd: float, 
        now: datetime
    ):
        """Record usage across all tracking systems"""
        
        # Record in sliding windows
        self.user_windows[user_id]["minute"].append(now)
        self.user_windows[user_id]["hour"].append(now)
        
        if tenant_id:
            self.tenant_windows[tenant_id]["hour"].append(now)
        
        self.global_windows["minute"].append(now)
        self.global_windows["hour"].append(now)
        self.global_windows["day"].append(now)
        
        # Record in usage metrics
        self.usage_by_user[user_id]["hour"].add_usage(tokens_used, cost_usd)
        self.usage_by_user[user_id]["day"].add_usage(tokens_used, cost_usd)
        
        if tenant_id:
            self.usage_by_tenant[tenant_id]["hour"].add_usage(tokens_used, cost_usd)
            self.usage_by_tenant[tenant_id]["day"].add_usage(tokens_used, cost_usd)
    
    def _should_reset_period(self, first_request: Optional[datetime], now: datetime, period: str) -> bool:
        """Check if usage period should be reset"""
        if first_request is None:
            return False
        
        if period == "hour":
            return now - first_request > timedelta(hours=1)
        elif period == "day":
            return now - first_request > timedelta(days=1)
        elif period == "month":
            return now - first_request > timedelta(days=30)
        
        return False
    
    def _get_remaining_quotas(
        self, 
        user_id: str, 
        tenant_id: Optional[str], 
        endpoint: str, 
        now: datetime
    ) -> Dict[str, Any]:
        """Get remaining quota information"""
        
        remaining = {}
        
        # User quotas
        if self.config.user_requests_per_minute:
            user_minute_count = len(self.user_windows[user_id]["minute"])
            remaining["user_requests_per_minute"] = max(0, self.config.user_requests_per_minute - user_minute_count)
        
        if self.config.user_requests_per_hour:
            user_hour_count = len(self.user_windows[user_id]["hour"])
            remaining["user_requests_per_hour"] = max(0, self.config.user_requests_per_hour - user_hour_count)
        
        # Global quotas
        if self.config.requests_per_minute:
            global_minute_count = len(self.global_windows["minute"])
            remaining["global_requests_per_minute"] = max(0, self.config.requests_per_minute - global_minute_count)
        
        return remaining
    
    def _create_rate_limit_response(
        self, 
        allowed: bool, 
        reason: str, 
        remaining: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create standardized rate limit response"""
        
        response = {
            "allowed": allowed,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if remaining:
            response["remaining"] = remaining
        
        return response
    
    def set_user_cooldown(self, user_id: str, cooldown_seconds: int):
        """Set cooldown period for user after limit violation"""
        self.user_cooldowns[user_id] = datetime.utcnow() + timedelta(seconds=cooldown_seconds)
        self.logger.warning(f"User {user_id} in cooldown for {cooldown_seconds} seconds")
    
    def get_user_usage_summary(self, user_id: str) -> Dict[str, Any]:
        """Get usage summary for user"""
        
        now = datetime.utcnow()
        
        # Get current window counts
        minute_requests = len(self.user_windows[user_id]["minute"])
        hour_requests = len(self.user_windows[user_id]["hour"])
        
        # Get usage metrics
        hour_usage = self.usage_by_user[user_id]["hour"]
        day_usage = self.usage_by_user[user_id]["day"]
        
        return {
            "user_id": user_id,
            "current_usage": {
                "requests_last_minute": minute_requests,
                "requests_last_hour": hour_requests,
                "tokens_last_hour": hour_usage.tokens,
                "cost_last_hour_usd": hour_usage.cost_usd,
                "tokens_last_day": day_usage.tokens,
                "cost_last_day_usd": day_usage.cost_usd
            },
            "limits": {
                "requests_per_minute": self.config.user_requests_per_minute,
                "requests_per_hour": self.config.user_requests_per_hour
            },
            "cooldown_until": self.user_cooldowns.get(user_id),
            "last_request": hour_usage.last_request,
            "timestamp": now.isoformat()
        }


class QuotaManager:
    """
    Enterprise quota manager for resource allocation
    
    Features:
    - Budget management and tracking
    - Department/team quotas
    - Automatic quota allocation
    - Usage forecasting
    - Alert system for quota exhaustion
    """
    
    def __init__(self, quota_configs: Dict[str, QuotaConfig]):
        self.quota_configs = quota_configs  # Quota configs by tenant/department
        self.quota_usage: Dict[str, Dict[str, UsageMetrics]] = defaultdict(lambda: defaultdict(UsageMetrics))
        self.quota_alerts: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(f"{__name__}.QuotaManager")
        self.logger.info("📊 Quota Manager initialized")
    
    def allocate_quota(
        self, 
        tenant_id: str, 
        quota_config: QuotaConfig,
        effective_date: Optional[datetime] = None
    ):
        """Allocate new quota for tenant"""
        
        effective_date = effective_date or datetime.utcnow()
        
        self.quota_configs[tenant_id] = quota_config
        
        # Reset usage if needed
        if tenant_id not in self.quota_usage:
            self.quota_usage[tenant_id] = defaultdict(UsageMetrics)
        
        self.logger.info(f"Quota allocated for tenant {tenant_id}")
    
    def check_quota_available(
        self, 
        tenant_id: str, 
        tokens_requested: int = 0, 
        cost_usd: float = 0.0
    ) -> Dict[str, Any]:
        """Check if quota is available for request"""
        
        if tenant_id not in self.quota_configs:
            return {"available": False, "reason": "No quota configured"}
        
        config = self.quota_configs[tenant_id]
        now = datetime.utcnow()
        
        # Check daily limits
        if config.cost_per_day_usd:
            daily_usage = self.quota_usage[tenant_id]["day"]
            
            # Reset if new day
            if self._should_reset_daily_quota(daily_usage.first_request, now):
                self.quota_usage[tenant_id]["day"] = UsageMetrics()
                daily_usage = self.quota_usage[tenant_id]["day"]
            
            if daily_usage.cost_usd + cost_usd > config.cost_per_day_usd:
                return {
                    "available": False, 
                    "reason": "Daily cost quota exceeded",
                    "limit": config.cost_per_day_usd,
                    "used": daily_usage.cost_usd,
                    "requested": cost_usd
                }
        
        # Check monthly limits
        if config.cost_per_month_usd:
            monthly_usage = self.quota_usage[tenant_id]["month"]
            
            # Reset if new month
            if self._should_reset_monthly_quota(monthly_usage.first_request, now):
                self.quota_usage[tenant_id]["month"] = UsageMetrics()
                monthly_usage = self.quota_usage[tenant_id]["month"]
            
            if monthly_usage.cost_usd + cost_usd > config.cost_per_month_usd:
                return {
                    "available": False, 
                    "reason": "Monthly cost quota exceeded",
                    "limit": config.cost_per_month_usd,
                    "used": monthly_usage.cost_usd,
                    "requested": cost_usd
                }
        
        return {"available": True}
    
    def record_quota_usage(
        self, 
        tenant_id: str, 
        tokens_used: int = 0, 
        cost_usd: float = 0.0
    ):
        """Record quota usage"""
        
        if tenant_id not in self.quota_configs:
            self.logger.warning(f"Recording usage for unconfigured tenant: {tenant_id}")
            return
        
        # Record in all relevant periods
        self.quota_usage[tenant_id]["day"].add_usage(tokens_used, cost_usd)
        self.quota_usage[tenant_id]["month"].add_usage(tokens_used, cost_usd)
        
        # Check for alerts
        self._check_quota_alerts(tenant_id)
    
    def _should_reset_daily_quota(self, first_request: Optional[datetime], now: datetime) -> bool:
        """Check if daily quota should be reset"""
        if first_request is None:
            return False
        
        return now.date() > first_request.date()
    
    def _should_reset_monthly_quota(self, first_request: Optional[datetime], now: datetime) -> bool:
        """Check if monthly quota should be reset"""
        if first_request is None:
            return False
        
        return (now.year, now.month) > (first_request.year, first_request.month)
    
    def _check_quota_alerts(self, tenant_id: str):
        """Check for quota alert conditions"""
        
        config = self.quota_configs[tenant_id]
        
        # Check daily quota alerts
        if config.cost_per_day_usd:
            daily_usage = self.quota_usage[tenant_id]["day"]
            usage_pct = daily_usage.cost_usd / config.cost_per_day_usd * 100
            
            if usage_pct > 90:  # 90% threshold
                self._create_quota_alert(tenant_id, "daily", usage_pct, "cost")
        
        # Check monthly quota alerts
        if config.cost_per_month_usd:
            monthly_usage = self.quota_usage[tenant_id]["month"]
            usage_pct = monthly_usage.cost_usd / config.cost_per_month_usd * 100
            
            if usage_pct > 80:  # 80% threshold for monthly
                self._create_quota_alert(tenant_id, "monthly", usage_pct, "cost")
    
    def _create_quota_alert(self, tenant_id: str, period: str, usage_pct: float, resource_type: str):
        """Create quota alert"""
        
        alert = {
            "tenant_id": tenant_id,
            "period": period,
            "resource_type": resource_type,
            "usage_percentage": usage_pct,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "high" if usage_pct > 95 else "medium"
        }
        
        self.quota_alerts.append(alert)
        
        self.logger.warning(f"Quota alert: {tenant_id} {period} {resource_type} usage at {usage_pct:.1f}%")
    
    def get_quota_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive quota summary"""
        
        if tenant_id not in self.quota_configs:
            return {"error": "No quota configured"}
        
        config = self.quota_configs[tenant_id]
        daily_usage = self.quota_usage[tenant_id]["day"]
        monthly_usage = self.quota_usage[tenant_id]["month"]
        
        summary = {
            "tenant_id": tenant_id,
            "quotas": {
                "daily_cost_limit": config.cost_per_day_usd,
                "monthly_cost_limit": config.cost_per_month_usd,
            },
            "usage": {
                "daily": {
                    "requests": daily_usage.requests,
                    "tokens": daily_usage.tokens,
                    "cost_usd": daily_usage.cost_usd,
                    "usage_pct": (daily_usage.cost_usd / config.cost_per_day_usd * 100) if config.cost_per_day_usd else 0
                },
                "monthly": {
                    "requests": monthly_usage.requests,
                    "tokens": monthly_usage.tokens,
                    "cost_usd": monthly_usage.cost_usd,
                    "usage_pct": (monthly_usage.cost_usd / config.cost_per_month_usd * 100) if config.cost_per_month_usd else 0
                }
            },
            "alerts": [alert for alert in self.quota_alerts if alert["tenant_id"] == tenant_id],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return summary