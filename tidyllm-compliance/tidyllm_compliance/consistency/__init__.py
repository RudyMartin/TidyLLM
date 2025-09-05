"""
Argument Consistency Analysis Module

Monitor argument consistency for determining review scope based on:
- Logical structure evaluation
- Internal contradiction detection  
- Scope factor analysis
- Priority and risk assessment
"""

from .analysis import ConsistencyAnalyzer

__all__ = ["ConsistencyAnalyzer"]