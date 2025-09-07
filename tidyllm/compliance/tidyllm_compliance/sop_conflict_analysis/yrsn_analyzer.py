"""
YRSN Noise Analyzer Implementation
==================================

Yes/Relevant/Specific/No-fluff compliance validation method.
Quantifies signal-to-noise ratio in SOP guidance content.

Part of tidyllm-compliance: Automated compliance with complete transparency
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class NoiseScore:
    """YRSN noise analysis results"""
    noise_percentage: float
    quality_assessment: str
    actionable_content_ratio: float
    specific_guidance_found: int
    noise_indicators_found: int

class YRSNNoiseAnalyzer:
    """Compliance validation using YRSN (Yes/Relevant/Specific/No-fluff) analysis"""
    
    def __init__(self):
        self.actionable_indicators = [
            'use', 'should use', 'must use', 'required', 'official',
            'pattern is', 'recommended', 'standard', 'implement',
            'configure', 'set to', 'enable', 'disable'
        ]
        
        self.noise_indicators = [
            'may be', 'could be', 'might', 'unclear', 'depends on',
            'various', 'multiple', 'different approaches', 'consider',
            'potentially', 'possibly', 'generally', 'typically'
        ]
    
    def analyze_guidance_quality(self, content_sections: List[str], query: str) -> NoiseScore:
        """
        Calculate YRSN (Yes/Relevant/Specific/No-fluff) noise metric for compliance validation
        
        Args:
            content_sections: List of guidance content sections
            query: Original query being analyzed
            
        Returns:
            NoiseScore with compliance validation metrics
        """
        if not content_sections:
            return NoiseScore(
                noise_percentage=100.0,
                quality_assessment='COMPLIANCE FAILURE - No content found',
                actionable_content_ratio=0.0,
                specific_guidance_found=0,
                noise_indicators_found=0
            )
        
        total_chars = sum(len(section) for section in content_sections)
        actionable_chars = 0
        specific_guidance_found = 0
        noise_indicators_found = 0
        
        # Analyze each section for compliance signal vs noise
        for section in content_sections:
            section_lower = section.lower()
            
            # Count actionable compliance content (specific guidance)
            for indicator in self.actionable_indicators:
                if indicator in section_lower:
                    actionable_chars += len(indicator) * 3  # Weight actionable content higher
                    specific_guidance_found += 1
            
            # Penalize vague language (compliance noise indicators)
            for noise in self.noise_indicators:
                if noise in section_lower:
                    actionable_chars = max(0, actionable_chars - len(noise))
                    noise_indicators_found += 1
        
        # Calculate compliance metrics
        actionable_ratio = actionable_chars / total_chars if total_chars > 0 else 0
        noise_percentage = max(0, 100 - (actionable_ratio * 100))
        
        # Determine compliance quality assessment
        if noise_percentage >= 90:
            quality = f"CRITICAL COMPLIANCE FAILURE ({noise_percentage:.1f}% noise) - No actionable guidance"
        elif noise_percentage >= 70:
            quality = f"HIGH COMPLIANCE RISK ({noise_percentage:.1f}% noise) - Minimal actionable content"
        elif noise_percentage >= 50:
            quality = f"MODERATE COMPLIANCE RISK ({noise_percentage:.1f}% noise) - Some actionable content"
        elif noise_percentage >= 30:
            quality = f"ACCEPTABLE COMPLIANCE ({noise_percentage:.1f}% noise) - Good actionable content"
        else:
            quality = f"EXCELLENT COMPLIANCE ({noise_percentage:.1f}% noise) - High actionable content"
        
        return NoiseScore(
            noise_percentage=noise_percentage,
            quality_assessment=quality,
            actionable_content_ratio=actionable_ratio,
            specific_guidance_found=specific_guidance_found,
            noise_indicators_found=noise_indicators_found
        )
    
    def validate_sop_response(self, response_content: str, query: str) -> Dict[str, Any]:
        """
        Validate SOP response quality for compliance purposes
        
        Args:
            response_content: The SOP response content to validate
            query: Original query
            
        Returns:
            Compliance validation result with YRSN metrics
        """
        content_sections = [response_content] if response_content else []
        noise_score = self.analyze_guidance_quality(content_sections, query)
        
        return {
            'compliance_status': 'PASS' if noise_score.noise_percentage < 50 else 'FAIL',
            'yrsn_metrics': {
                'noise_percentage': noise_score.noise_percentage,
                'quality_assessment': noise_score.quality_assessment,
                'actionable_ratio': noise_score.actionable_content_ratio,
                'specific_guidance_count': noise_score.specific_guidance_found,
                'noise_indicators_count': noise_score.noise_indicators_found
            },
            'validation_timestamp': self._get_timestamp(),
            'query_analyzed': query
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for compliance audit trail"""
        from datetime import datetime
        return datetime.now().isoformat()