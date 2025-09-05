"""
TidyLLM-{Domain} - {Brief Description}

{Longer description of what this application does and its business value.}

Architecture:
- Built on tidyllm utility layer (core ML algorithms)
- Focuses on {domain}-specific business problems
- Provides business-friendly analysis and insights
- Includes production-ready interfaces

Usage:
    from tidyllm_{domain} import {MainClass}
    
    analyzer = {MainClass}()
    results = analyzer.analyze(data)
    insights = analyzer.generate_business_insights(results)
"""

# Core API exports - keep this clean and business-focused
from .core.analyzer import {MainClass}
from .analysis.business_intelligence import {BusinessAnalyzer}

# Version information
__version__ = "0.1.0"
__author__ = "TidyLLM Ecosystem"

# Public API
__all__ = [
    "{MainClass}",
    "{BusinessAnalyzer}",
]

# Quick access to common functionality
def analyze(data, **kwargs):
    """
    Quick analysis function for common use cases.
    
    Args:
        data: Input data to analyze
        **kwargs: Additional parameters
    
    Returns:
        Analysis results with business insights
    """
    analyzer = {MainClass}()
    return analyzer.analyze(data, **kwargs)

def generate_business_report(results, format="markdown"):
    """
    Generate business-friendly report from analysis results.
    
    Args:
        results: Analysis results from analyze()
        format: Output format ("markdown", "html", "json")
    
    Returns:
        Formatted business report
    """
    business_analyzer = {BusinessAnalyzer}()
    return business_analyzer.generate_report(results, format)