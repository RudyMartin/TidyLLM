"""
Temporal Resolver Implementation
===============================

Compliance-owned temporal conflict resolution using newest-wins strategy.
Handles date-based precedence for SOP guidance conflicts.

Part of tidyllm-compliance: Automated compliance with complete transparency
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class TemporalConflict:
    """Structure for temporal conflict information"""
    query: str
    conflicting_dates: List[str]
    authoritative_date: str
    deprecated_dates: List[str]
    resolution_confidence: str

class TemporalResolver:
    """Compliance temporal conflict resolution using newest-wins strategy"""
    
    def __init__(self):
        self.resolution_strategy = "newest_wins"
        
    def resolve_temporal_conflicts(self, conflicts: List[Dict[str, Any]]) -> List[TemporalConflict]:
        """
        Resolve temporal conflicts using compliance-approved newest-wins strategy
        
        Args:
            conflicts: List of conflict data with dates involved
            
        Returns:
            List of TemporalConflict objects with resolution decisions
        """
        resolved_conflicts = []
        
        for conflict in conflicts:
            query = conflict.get('query', '')
            dates_involved = conflict.get('dates_involved', [])
            
            if len(dates_involved) > 1:
                # Apply temporal resolution
                temporal_conflict = self._apply_newest_wins_resolution(query, dates_involved)
                resolved_conflicts.append(temporal_conflict)
            else:
                # No conflict - single date
                resolved_conflicts.append(TemporalConflict(
                    query=query,
                    conflicting_dates=dates_involved,
                    authoritative_date=dates_involved[0] if dates_involved else '',
                    deprecated_dates=[],
                    resolution_confidence='HIGH'
                ))
        
        return resolved_conflicts
    
    def _apply_newest_wins_resolution(self, query: str, dates_involved: List[str]) -> TemporalConflict:
        """Apply newest-wins temporal resolution strategy"""
        
        # Sort dates to find newest
        sorted_dates = sorted(dates_involved, reverse=True)  # Newest first
        authoritative_date = sorted_dates[0]
        deprecated_dates = sorted_dates[1:]
        
        # Determine confidence based on date spread
        if self._calculate_date_spread(dates_involved) <= 7:  # Within a week
            confidence = 'HIGH'
        elif self._calculate_date_spread(dates_involved) <= 30:  # Within a month
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
            
        return TemporalConflict(
            query=query,
            conflicting_dates=dates_involved,
            authoritative_date=authoritative_date,
            deprecated_dates=deprecated_dates,
            resolution_confidence=confidence
        )
    
    def _calculate_date_spread(self, dates: List[str]) -> int:
        """Calculate the spread between earliest and latest dates in days"""
        try:
            date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
            earliest = min(date_objects)
            latest = max(date_objects)
            return (latest - earliest).days
        except Exception:
            return 0  # Default to 0 if date parsing fails
    
    def validate_temporal_resolution(self, resolved_conflicts: List[TemporalConflict]) -> Dict[str, Any]:
        """
        Validate temporal resolution decisions for compliance
        
        Args:
            resolved_conflicts: List of resolved temporal conflicts
            
        Returns:
            Validation report for compliance purposes
        """
        total_conflicts = len(resolved_conflicts)
        high_confidence = len([c for c in resolved_conflicts if c.resolution_confidence == 'HIGH'])
        low_confidence = len([c for c in resolved_conflicts if c.resolution_confidence == 'LOW'])
        
        return {
            'validation_type': 'Temporal Resolution Validation',
            'timestamp': datetime.now().isoformat(),
            'resolution_metrics': {
                'total_conflicts_resolved': total_conflicts,
                'high_confidence_resolutions': high_confidence,
                'medium_confidence_resolutions': total_conflicts - high_confidence - low_confidence,
                'low_confidence_resolutions': low_confidence,
                'resolution_success_rate': (high_confidence / total_conflicts * 100) if total_conflicts > 0 else 0
            },
            'compliance_status': 'PASS' if low_confidence == 0 else 'REVIEW_REQUIRED',
            'recommendation': 'Review low-confidence resolutions manually' if low_confidence > 0 else 'All resolutions meet compliance standards'
        }