"""
Base Flow Agreement - Foundation for all workflow contracts
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json


@dataclass
class FlowAgreementConfig:
    """Configuration for a Flow Agreement."""
    agreement_id: str
    agreement_type: str
    created_by: str
    valid_until: Optional[datetime] = None
    max_files_per_day: Optional[int] = None
    max_cost_per_month: Optional[float] = None
    approved_gateways: List[str] = None
    audit_requirements: List[str] = None
    auto_optimizations: List[str] = None
    
    def __post_init__(self):
        if self.valid_until is None:
            # Default to 1 year validity
            self.valid_until = datetime.now() + timedelta(days=365)
        if self.approved_gateways is None:
            self.approved_gateways = ["dspy"]  # Default to DSPy
        if self.audit_requirements is None:
            self.audit_requirements = []
        if self.auto_optimizations is None:
            self.auto_optimizations = []


class BaseFlowAgreement(ABC):
    """
    Base class for all Flow Agreements.
    
    Flow Agreements are pre-negotiated workflow contracts that make it easy
    for users to get started with TidyLLM without having to configure everything
    from scratch.
    """
    
    def __init__(self, config: FlowAgreementConfig):
        self.config = config
        self.activated = False
        self.activation_time = None
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate that this agreement is still valid and can be used."""
        pass
    
    @abstractmethod
    def get_gateway_config(self) -> Dict[str, Any]:
        """Get the gateway configuration for this agreement."""
        pass
    
    @abstractmethod
    def get_drop_zone_config(self) -> Dict[str, Any]:
        """Get the drop zone configuration for this agreement."""
        pass
    
    def activate(self) -> Dict[str, Any]:
        """
        Activate this Flow Agreement and return ready-to-use setup.
        
        Returns:
            Dict containing gateway, drop_zone, and other configurations
        """
        if not self.validate():
            raise ValueError(f"Flow Agreement {self.config.agreement_id} is not valid")
        
        # Import gateways dynamically to avoid circular imports
        from ..gateways import get_gateway
        
        # Get the primary gateway for this agreement
        primary_gateway_name = self.config.approved_gateways[0]
        gateway = get_gateway(primary_gateway_name, **self.get_gateway_config())
        
        # Create drop zone configuration
        drop_zone_config = self.get_drop_zone_config()
        
        # Mark as activated
        self.activated = True
        self.activation_time = datetime.now()
        
        return {
            'agreement_id': self.config.agreement_id,
            'gateway': gateway,
            'drop_zone_config': drop_zone_config,
            'welcome_message': self.get_welcome_message(),
            'quick_start_guide': self.get_quick_start_guide()
        }
    
    def get_welcome_message(self) -> str:
        """Get personalized welcome message for this agreement."""
        return f"Welcome! Your {self.config.agreement_type} Flow Agreement is active."
    
    def get_quick_start_guide(self) -> List[str]:
        """Get quick start steps for this agreement."""
        return [
            f"Your {self.config.approved_gateways[0]} gateway is ready",
            "Drop your files in the configured drop zone",
            "Processing will begin automatically"
        ]
    
    def is_valid(self) -> bool:
        """Check if agreement is currently valid."""
        if self.config.valid_until and datetime.now() > self.config.valid_until:
            return False
        return True
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this agreement."""
        return {
            'agreement_id': self.config.agreement_id,
            'activated': self.activated,
            'activation_time': self.activation_time.isoformat() if self.activation_time else None,
            'valid_until': self.config.valid_until.isoformat() if self.config.valid_until else None,
            'approved_gateways': self.config.approved_gateways
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agreement to dictionary for serialization."""
        return {
            'config': {
                'agreement_id': self.config.agreement_id,
                'agreement_type': self.config.agreement_type,
                'created_by': self.config.created_by,
                'valid_until': self.config.valid_until.isoformat() if self.config.valid_until else None,
                'approved_gateways': self.config.approved_gateways,
                'audit_requirements': self.config.audit_requirements,
                'auto_optimizations': self.config.auto_optimizations
            },
            'status': {
                'activated': self.activated,
                'activation_time': self.activation_time.isoformat() if self.activation_time else None
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create agreement from dictionary."""
        config_data = data['config']
        config = FlowAgreementConfig(
            agreement_id=config_data['agreement_id'],
            agreement_type=config_data['agreement_type'],
            created_by=config_data['created_by'],
            valid_until=datetime.fromisoformat(config_data['valid_until']) if config_data.get('valid_until') else None,
            approved_gateways=config_data.get('approved_gateways', ['dspy']),
            audit_requirements=config_data.get('audit_requirements', []),
            auto_optimizations=config_data.get('auto_optimizations', [])
        )
        
        agreement = cls(config)
        
        # Restore status
        status_data = data.get('status', {})
        agreement.activated = status_data.get('activated', False)
        if status_data.get('activation_time'):
            agreement.activation_time = datetime.fromisoformat(status_data['activation_time'])
        
        return agreement