#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enterprise Spend Tracking System

Comprehensive cost tracking, budget management, and financial reporting
for LLM usage across enterprise organizations. Provides granular cost
attribution, forecasting, and budget controls.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json


class BudgetPeriod(Enum):
    """Budget time periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class CostCategory(Enum):
    """Cost categorization"""
    COMPUTE = "compute"              # LLM inference costs
    STORAGE = "storage"              # Embedding/vector storage
    NETWORK = "network"              # Data transfer costs
    MANAGEMENT = "management"        # Gateway infrastructure
    COMPLIANCE = "compliance"        # Audit and compliance overhead


class AlertType(Enum):
    """Budget alert types"""
    THRESHOLD_WARNING = "threshold_warning"      # 80% of budget
    THRESHOLD_CRITICAL = "threshold_critical"    # 95% of budget
    BUDGET_EXCEEDED = "budget_exceeded"          # Over budget
    UNUSUAL_SPENDING = "unusual_spending"        # Anomaly detection
    FORECAST_OVERRUN = "forecast_overrun"        # Projected overrun


@dataclass
class SpendRecord:
    """Individual spend record"""
    record_id: str
    timestamp: datetime
    user_id: str
    department: str
    tenant_id: str
    
    # Request details
    model: str
    provider: str
    endpoint: str
    
    # Usage metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Cost breakdown
    compute_cost_usd: float = 0.0
    storage_cost_usd: float = 0.0
    network_cost_usd: float = 0.0
    management_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    
    # Metadata
    request_id: str = ""
    audit_reason: str = ""
    cost_category: CostCategory = CostCategory.COMPUTE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "department": self.department,
            "tenant_id": self.tenant_id,
            "model": self.model,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "compute_cost_usd": self.compute_cost_usd,
            "storage_cost_usd": self.storage_cost_usd,
            "network_cost_usd": self.network_cost_usd,
            "management_cost_usd": self.management_cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "request_id": self.request_id,
            "audit_reason": self.audit_reason,
            "cost_category": self.cost_category.value
        }


@dataclass
class BudgetConfig:
    """Budget configuration"""
    budget_id: str
    name: str
    
    # Budget scope
    tenant_id: Optional[str] = None
    department: Optional[str] = None
    user_id: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    
    # Budget limits
    limit_usd: float = 0.0
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    
    # Alert thresholds
    warning_threshold: float = 0.8      # 80%
    critical_threshold: float = 0.95    # 95%
    
    # Budget controls
    hard_limit: bool = False            # Reject requests when exceeded
    rollover_unused: bool = True        # Carry over unused budget
    
    # Validity
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    active: bool = True


@dataclass
class BudgetStatus:
    """Current budget status"""
    budget_id: str
    name: str
    
    # Current period
    period_start: datetime
    period_end: datetime
    
    # Budget tracking
    limit_usd: float
    spent_usd: float
    remaining_usd: float
    utilization_pct: float
    
    # Projections
    projected_spend_usd: float
    projected_overrun_usd: float
    days_until_exhaustion: Optional[int]
    
    # Status indicators
    over_warning_threshold: bool
    over_critical_threshold: bool
    budget_exceeded: bool
    
    # Recent activity
    spend_last_24h_usd: float
    spend_trend: str  # "increasing", "stable", "decreasing"


@dataclass
class SpendAlert:
    """Budget alert"""
    alert_id: str
    alert_type: AlertType
    severity: str  # "low", "medium", "high", "critical"
    
    # Alert context
    budget_id: str
    tenant_id: str
    department: Optional[str] = None
    
    # Alert details
    message: str
    threshold_value: float
    current_value: float
    
    # Timing
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolved: bool = False
    
    # Actions
    notification_sent: bool = False
    action_taken: str = ""


class EnterpriseSpendTracker:
    """
    Enterprise Spend Tracking System
    
    Features:
    - Granular cost tracking by user, department, model, provider
    - Multi-level budget management with hierarchical controls
    - Real-time spend monitoring and alerting
    - Cost forecasting and trend analysis
    - Automated budget enforcement
    - Comprehensive financial reporting
    - Chargeback and cost allocation
    """
    
    def __init__(self, storage_backend: Optional[str] = None):
        # Spend records storage
        self.spend_records: List[SpendRecord] = []
        self.spend_by_user: Dict[str, List[SpendRecord]] = defaultdict(list)
        self.spend_by_department: Dict[str, List[SpendRecord]] = defaultdict(list)
        self.spend_by_tenant: Dict[str, List[SpendRecord]] = defaultdict(list)
        
        # Budget management
        self.budgets: Dict[str, BudgetConfig] = {}
        self.budget_status: Dict[str, BudgetStatus] = {}
        
        # Alerting
        self.alerts: List[SpendAlert] = []
        self.active_alerts: Dict[str, SpendAlert] = {}
        
        # Analytics
        self.daily_totals: Dict[str, float] = {}  # date -> total_cost
        self.model_costs: Dict[str, float] = defaultdict(float)
        self.provider_costs: Dict[str, float] = defaultdict(float)
        
        # Forecasting data (rolling 30 days)
        self.spend_history: deque = deque(maxlen=30)
        
        self.logger = logging.getLogger(f"{__name__}.SpendTracker")
        self.logger.info("💰 Enterprise Spend Tracker initialized")
    
    def record_spend(
        self,
        user_id: str,
        department: str,
        tenant_id: str,
        model: str,
        provider: str,
        endpoint: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        compute_cost_usd: float = 0.0,
        request_id: str = "",
        audit_reason: str = ""
    ) -> str:
        """Record a spend transaction"""
        
        # Create spend record
        record_id = f"spend_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{user_id[:8]}"
        
        # Calculate additional costs (management overhead, etc.)
        management_cost = compute_cost_usd * 0.05  # 5% management fee
        total_cost = compute_cost_usd + management_cost
        
        spend_record = SpendRecord(
            record_id=record_id,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            department=department,
            tenant_id=tenant_id,
            model=model,
            provider=provider,
            endpoint=endpoint,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            compute_cost_usd=compute_cost_usd,
            management_cost_usd=management_cost,
            total_cost_usd=total_cost,
            request_id=request_id,
            audit_reason=audit_reason,
            cost_category=CostCategory.COMPUTE
        )
        
        # Store record
        self.spend_records.append(spend_record)
        self.spend_by_user[user_id].append(spend_record)
        self.spend_by_department[department].append(spend_record)
        self.spend_by_tenant[tenant_id].append(spend_record)
        
        # Update analytics
        today = datetime.utcnow().date().isoformat()
        self.daily_totals[today] = self.daily_totals.get(today, 0.0) + total_cost
        self.model_costs[model] += total_cost
        self.provider_costs[provider] += total_cost
        
        # Update spend history for forecasting
        self.spend_history.append({
            "date": datetime.utcnow().date(),
            "cost": total_cost
        })
        
        # Check budgets and trigger alerts
        self._check_budgets_for_spend(spend_record)
        
        self.logger.debug(f"Recorded spend: {record_id} - ${total_cost:.4f}")
        
        return record_id
    
    def create_budget(
        self,
        name: str,
        limit_usd: float,
        period: BudgetPeriod = BudgetPeriod.MONTHLY,
        tenant_id: Optional[str] = None,
        department: Optional[str] = None,
        user_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        hard_limit: bool = False,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95
    ) -> str:
        """Create a new budget"""
        
        budget_id = f"budget_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        budget = BudgetConfig(
            budget_id=budget_id,
            name=name,
            tenant_id=tenant_id,
            department=department,
            user_id=user_id,
            model=model,
            provider=provider,
            limit_usd=limit_usd,
            period=period,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            hard_limit=hard_limit
        )
        
        self.budgets[budget_id] = budget
        
        # Initialize budget status
        self._initialize_budget_status(budget)
        
        self.logger.info(f"Created budget: {name} (${limit_usd:.2f} {period.value})")
        
        return budget_id
    
    def _initialize_budget_status(self, budget: BudgetConfig):
        """Initialize budget status tracking"""
        
        period_start, period_end = self._get_budget_period_dates(budget.period)
        
        # Calculate current spend for this budget scope
        spent_usd = self._calculate_budget_spend(budget, period_start, period_end)
        remaining_usd = max(0.0, budget.limit_usd - spent_usd)
        utilization_pct = (spent_usd / budget.limit_usd * 100) if budget.limit_usd > 0 else 0.0
        
        # Forecasting
        projected_spend = self._forecast_spend(budget, period_start, period_end)
        projected_overrun = max(0.0, projected_spend - budget.limit_usd)
        
        # Days until exhaustion
        days_until_exhaustion = None
        if remaining_usd > 0:
            daily_rate = self._calculate_daily_spend_rate(budget)
            if daily_rate > 0:
                days_until_exhaustion = int(remaining_usd / daily_rate)
        
        status = BudgetStatus(
            budget_id=budget.budget_id,
            name=budget.name,
            period_start=period_start,
            period_end=period_end,
            limit_usd=budget.limit_usd,
            spent_usd=spent_usd,
            remaining_usd=remaining_usd,
            utilization_pct=utilization_pct,
            projected_spend_usd=projected_spend,
            projected_overrun_usd=projected_overrun,
            days_until_exhaustion=days_until_exhaustion,
            over_warning_threshold=utilization_pct > (budget.warning_threshold * 100),
            over_critical_threshold=utilization_pct > (budget.critical_threshold * 100),
            budget_exceeded=spent_usd > budget.limit_usd,
            spend_last_24h_usd=self._calculate_recent_spend(budget, hours=24),
            spend_trend=self._calculate_spend_trend(budget)
        )
        
        self.budget_status[budget.budget_id] = status
    
    def _get_budget_period_dates(self, period: BudgetPeriod) -> Tuple[datetime, datetime]:
        """Get start and end dates for budget period"""
        
        now = datetime.utcnow()
        
        if period == BudgetPeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == BudgetPeriod.WEEKLY:
            days_since_monday = now.weekday()
            start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(weeks=1)
        elif period == BudgetPeriod.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)
        elif period == BudgetPeriod.QUARTERLY:
            quarter = ((now.month - 1) // 3) + 1
            start_month = ((quarter - 1) * 3) + 1
            start = now.replace(month=start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_month = start_month + 3
            if end_month > 12:
                end = start.replace(year=now.year + 1, month=end_month - 12)
            else:
                end = start.replace(month=end_month)
        elif period == BudgetPeriod.ANNUAL:
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(year=now.year + 1)
        else:
            raise ValueError(f"Unknown budget period: {period}")
        
        return start, end
    
    def _calculate_budget_spend(
        self, 
        budget: BudgetConfig, 
        period_start: datetime, 
        period_end: datetime
    ) -> float:
        """Calculate spend for budget scope in period"""
        
        total_spend = 0.0
        
        for record in self.spend_records:
            # Check time period
            if not (period_start <= record.timestamp < period_end):
                continue
            
            # Check budget scope filters
            if budget.tenant_id and record.tenant_id != budget.tenant_id:
                continue
            if budget.department and record.department != budget.department:
                continue
            if budget.user_id and record.user_id != budget.user_id:
                continue
            if budget.model and record.model != budget.model:
                continue
            if budget.provider and record.provider != budget.provider:
                continue
            
            total_spend += record.total_cost_usd
        
        return total_spend
    
    def _forecast_spend(
        self, 
        budget: BudgetConfig, 
        period_start: datetime, 
        period_end: datetime
    ) -> float:
        """Forecast spend for rest of budget period"""
        
        # Calculate current spend rate
        now = datetime.utcnow()
        elapsed_days = (now - period_start).days + 1
        total_period_days = (period_end - period_start).days
        
        current_spend = self._calculate_budget_spend(budget, period_start, now)
        
        if elapsed_days == 0:
            return current_spend
        
        # Simple linear projection based on current rate
        daily_rate = current_spend / elapsed_days
        remaining_days = total_period_days - elapsed_days
        
        return current_spend + (daily_rate * remaining_days)
    
    def _calculate_daily_spend_rate(self, budget: BudgetConfig) -> float:
        """Calculate daily spend rate for budget scope"""
        
        # Look at last 7 days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        total_spend = self._calculate_budget_spend(budget, start_date, end_date)
        
        return total_spend / 7.0
    
    def _calculate_recent_spend(self, budget: BudgetConfig, hours: int = 24) -> float:
        """Calculate spend in recent hours"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours)
        
        return self._calculate_budget_spend(budget, start_date, end_date)
    
    def _calculate_spend_trend(self, budget: BudgetConfig) -> str:
        """Calculate spend trend direction"""
        
        now = datetime.utcnow()
        
        # Compare last 24 hours to previous 24 hours
        recent_spend = self._calculate_recent_spend(budget, hours=24)
        
        previous_end = now - timedelta(hours=24)
        previous_start = previous_end - timedelta(hours=24)
        previous_spend = self._calculate_budget_spend(budget, previous_start, previous_end)
        
        if recent_spend > previous_spend * 1.1:  # 10% increase
            return "increasing"
        elif recent_spend < previous_spend * 0.9:  # 10% decrease
            return "decreasing"
        else:
            return "stable"
    
    def _check_budgets_for_spend(self, spend_record: SpendRecord):
        """Check all applicable budgets for alerts"""
        
        for budget_id, budget in self.budgets.items():
            if not budget.active:
                continue
            
            # Check if spend record matches budget scope
            if not self._spend_matches_budget_scope(spend_record, budget):
                continue
            
            # Update budget status
            self._initialize_budget_status(budget)
            status = self.budget_status[budget_id]
            
            # Check for alert conditions
            self._check_budget_alerts(budget, status)
    
    def _spend_matches_budget_scope(self, record: SpendRecord, budget: BudgetConfig) -> bool:
        """Check if spend record matches budget scope"""
        
        if budget.tenant_id and record.tenant_id != budget.tenant_id:
            return False
        if budget.department and record.department != budget.department:
            return False
        if budget.user_id and record.user_id != budget.user_id:
            return False
        if budget.model and record.model != budget.model:
            return False
        if budget.provider and record.provider != budget.provider:
            return False
        
        return True
    
    def _check_budget_alerts(self, budget: BudgetConfig, status: BudgetStatus):
        """Check budget for alert conditions"""
        
        alerts_to_create = []
        
        # Budget exceeded
        if status.budget_exceeded and not self._alert_exists(budget.budget_id, AlertType.BUDGET_EXCEEDED):
            alerts_to_create.append({
                "type": AlertType.BUDGET_EXCEEDED,
                "severity": "critical",
                "message": f"Budget '{budget.name}' exceeded: ${status.spent_usd:.2f} / ${status.limit_usd:.2f}",
                "threshold": status.limit_usd,
                "current": status.spent_usd
            })
        
        # Critical threshold
        elif status.over_critical_threshold and not self._alert_exists(budget.budget_id, AlertType.THRESHOLD_CRITICAL):
            alerts_to_create.append({
                "type": AlertType.THRESHOLD_CRITICAL,
                "severity": "high",
                "message": f"Budget '{budget.name}' at {status.utilization_pct:.1f}% (critical threshold)",
                "threshold": budget.critical_threshold * 100,
                "current": status.utilization_pct
            })
        
        # Warning threshold  
        elif status.over_warning_threshold and not self._alert_exists(budget.budget_id, AlertType.THRESHOLD_WARNING):
            alerts_to_create.append({
                "type": AlertType.THRESHOLD_WARNING,
                "severity": "medium",
                "message": f"Budget '{budget.name}' at {status.utilization_pct:.1f}% (warning threshold)",
                "threshold": budget.warning_threshold * 100,
                "current": status.utilization_pct
            })
        
        # Forecast overrun
        if status.projected_overrun_usd > 0 and not self._alert_exists(budget.budget_id, AlertType.FORECAST_OVERRUN):
            alerts_to_create.append({
                "type": AlertType.FORECAST_OVERRUN,
                "severity": "medium",
                "message": f"Budget '{budget.name}' projected to exceed by ${status.projected_overrun_usd:.2f}",
                "threshold": status.limit_usd,
                "current": status.projected_spend_usd
            })
        
        # Create alerts
        for alert_data in alerts_to_create:
            self._create_alert(budget, alert_data)
    
    def _alert_exists(self, budget_id: str, alert_type: AlertType) -> bool:
        """Check if alert already exists for budget"""
        
        alert_key = f"{budget_id}_{alert_type.value}"
        return alert_key in self.active_alerts and not self.active_alerts[alert_key].resolved
    
    def _create_alert(self, budget: BudgetConfig, alert_data: Dict[str, Any]):
        """Create budget alert"""
        
        alert_id = f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{budget.budget_id}"
        
        alert = SpendAlert(
            alert_id=alert_id,
            alert_type=alert_data["type"],
            severity=alert_data["severity"],
            budget_id=budget.budget_id,
            tenant_id=budget.tenant_id or "unknown",
            department=budget.department,
            message=alert_data["message"],
            threshold_value=alert_data["threshold"],
            current_value=alert_data["current"]
        )
        
        self.alerts.append(alert)
        alert_key = f"{budget.budget_id}_{alert_data['type'].value}"
        self.active_alerts[alert_key] = alert
        
        self.logger.warning(f"Budget alert created: {alert.message}")
    
    def check_budget_approval(
        self,
        user_id: str,
        department: str,
        tenant_id: str,
        model: str,
        provider: str,
        estimated_cost_usd: float
    ) -> Dict[str, Any]:
        """Check if request is approved by budget constraints"""
        
        # Find applicable budgets
        applicable_budgets = []
        for budget in self.budgets.values():
            if not budget.active:
                continue
            
            # Check scope match
            scope_match = True
            if budget.tenant_id and budget.tenant_id != tenant_id:
                scope_match = False
            if budget.department and budget.department != department:
                scope_match = False
            if budget.user_id and budget.user_id != user_id:
                scope_match = False
            if budget.model and budget.model != model:
                scope_match = False
            if budget.provider and budget.provider != provider:
                scope_match = False
            
            if scope_match:
                applicable_budgets.append(budget)
        
        # Check each applicable budget
        for budget in applicable_budgets:
            status = self.budget_status.get(budget.budget_id)
            if not status:
                continue
            
            # Check if adding this cost would exceed budget
            if budget.hard_limit and (status.spent_usd + estimated_cost_usd) > budget.limit_usd:
                return {
                    "approved": False,
                    "reason": f"Hard budget limit exceeded: {budget.name}",
                    "budget_id": budget.budget_id,
                    "limit_usd": budget.limit_usd,
                    "current_spend_usd": status.spent_usd,
                    "estimated_cost_usd": estimated_cost_usd
                }
        
        return {"approved": True}
    
    def get_spend_summary(
        self,
        tenant_id: Optional[str] = None,
        department: Optional[str] = None,
        user_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive spend summary"""
        
        # Filter records
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        filtered_records = []
        for record in self.spend_records:
            if record.timestamp < start_date:
                continue
            
            if tenant_id and record.tenant_id != tenant_id:
                continue
            if department and record.department != department:
                continue
            if user_id and record.user_id != user_id:
                continue
            
            filtered_records.append(record)
        
        # Calculate summary statistics
        total_cost = sum(r.total_cost_usd for r in filtered_records)
        total_tokens = sum(r.total_tokens for r in filtered_records)
        total_requests = len(filtered_records)
        
        # Cost by category
        compute_cost = sum(r.compute_cost_usd for r in filtered_records)
        management_cost = sum(r.management_cost_usd for r in filtered_records)
        
        # Top models
        model_costs = defaultdict(float)
        for record in filtered_records:
            model_costs[record.model] += record.total_cost_usd
        
        top_models = sorted(model_costs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Top users
        user_costs = defaultdict(float)
        for record in filtered_records:
            user_costs[record.user_id] += record.total_cost_usd
        
        top_users = sorted(user_costs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "summary": {
                "period_days": days,
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "total_requests": total_requests,
                "avg_cost_per_request": total_cost / total_requests if total_requests else 0,
                "avg_cost_per_1k_tokens": (total_cost / total_tokens * 1000) if total_tokens else 0
            },
            "cost_breakdown": {
                "compute_cost_usd": compute_cost,
                "management_cost_usd": management_cost,
                "compute_percentage": (compute_cost / total_cost * 100) if total_cost else 0,
                "management_percentage": (management_cost / total_cost * 100) if total_cost else 0
            },
            "top_models": [{"model": model, "cost_usd": cost} for model, cost in top_models],
            "top_users": [{"user_id": user, "cost_usd": cost} for user, cost in top_users],
            "filters_applied": {
                "tenant_id": tenant_id,
                "department": department, 
                "user_id": user_id
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_budget_dashboard(self) -> Dict[str, Any]:
        """Get budget dashboard data"""
        
        # Update all budget statuses
        for budget in self.budgets.values():
            if budget.active:
                self._initialize_budget_status(budget)
        
        # Get active alerts
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        # Budget summaries
        budget_summaries = []
        for budget_id, status in self.budget_status.items():
            budget_summaries.append({
                "budget_id": budget_id,
                "name": status.name,
                "limit_usd": status.limit_usd,
                "spent_usd": status.spent_usd,
                "utilization_pct": status.utilization_pct,
                "status": "exceeded" if status.budget_exceeded else
                        "critical" if status.over_critical_threshold else
                        "warning" if status.over_warning_threshold else "healthy",
                "days_until_exhaustion": status.days_until_exhaustion,
                "spend_trend": status.spend_trend
            })
        
        # Overall metrics
        total_budgets = len(self.budgets)
        healthy_budgets = sum(1 for s in self.budget_status.values() 
                             if not s.over_warning_threshold)
        warning_budgets = sum(1 for s in self.budget_status.values() 
                             if s.over_warning_threshold and not s.over_critical_threshold)
        critical_budgets = sum(1 for s in self.budget_status.values() 
                              if s.over_critical_threshold and not s.budget_exceeded)
        exceeded_budgets = sum(1 for s in self.budget_status.values() 
                              if s.budget_exceeded)
        
        return {
            "overview": {
                "total_budgets": total_budgets,
                "healthy_budgets": healthy_budgets,
                "warning_budgets": warning_budgets,
                "critical_budgets": critical_budgets,
                "exceeded_budgets": exceeded_budgets,
                "active_alerts": len(active_alerts)
            },
            "budgets": budget_summaries,
            "recent_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "type": alert.alert_type.value,
                    "severity": alert.severity,
                    "message": alert.message,
                    "triggered_at": alert.triggered_at.isoformat()
                }
                for alert in sorted(active_alerts, key=lambda x: x.triggered_at, reverse=True)[:10]
            ],
            "timestamp": datetime.utcnow().isoformat()
        }


# Export main classes
__all__ = [
    'EnterpriseSpendTracker',
    'SpendRecord',
    'BudgetConfig',
    'BudgetStatus', 
    'SpendAlert',
    'BudgetPeriod',
    'CostCategory',
    'AlertType'
]