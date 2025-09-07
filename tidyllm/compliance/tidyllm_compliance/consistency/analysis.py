"""
Argument Consistency Analysis Implementation

Monitor argument consistency for determining review scope based on:
- Logical structure and flow
- Internal contradiction detection
- Materiality and risk factor analysis
- Review scope and priority determination

Part of the tidyllm-verse: Educational ML with complete transparency
"""

import re
from typing import Dict, List, Tuple, Any

class ConsistencyAnalyzer:
    """Monitor argument consistency for determining review scope."""
    
    def __init__(self):
        self.consistency_framework = self._initialize_consistency_framework()
        
    def _initialize_consistency_framework(self) -> Dict[str, Any]:
        """Initialize argument consistency analysis framework."""
        return {
            'logical_structure_patterns': {
                'premise_indicators': r'(?:because|since|given that|assuming|based on)',
                'conclusion_indicators': r'(?:therefore|thus|hence|consequently|as a result)',
                'evidence_indicators': r'(?:evidence shows|data indicates|research demonstrates)',
                'qualification_indicators': r'(?:however|although|despite|nevertheless|on the other hand)'
            },
            
            'scope_determination_criteria': {
                'materiality_indicators': r'(?:material|significant|substantial|major impact)',
                'risk_level_indicators': r'(?:high risk|critical|urgent|immediate attention)',
                'regulatory_indicators': r'(?:regulatory|compliance|legal requirement|mandatory)',
                'financial_impact_indicators': r'(?:\$[\d,]+|[\d]+%|material impact|financial exposure)'
            },
            
            'consistency_checks': {
                'internal_contradictions': [
                    (r'(?:always|never|all|none)', r'(?:sometimes|may|might|could)'),
                    (r'(?:increase|grow|expand)', r'(?:decrease|shrink|reduce)'),
                    (r'(?:approve|accept|endorse)', r'(?:reject|deny|oppose)')
                ],
                
                'temporal_consistency': [
                    r'(?:before|after|during|while)[\s\w]*\d{4}',
                    r'(?:previous|current|future|next)[\s\w]*(?:year|quarter|month)'
                ],
                
                'quantitative_consistency': [
                    r'\$?[\d,]+\.?\d*[kmb]?',  # Monetary amounts
                    r'\d+\.?\d*%',  # Percentages
                    r'[\d,]+\s*(?:units|items|cases|samples)'  # Quantities
                ]
            }
        }
    
    def analyze_document(self, document_text: str) -> Dict[str, Any]:
        """Analyze document for argument consistency and determine review scope."""
        analysis_results = {
            'consistency_score': 0.0,
            'logical_structure_score': 0.0,
            'review_scope_recommendation': 'unknown',
            'identified_issues': [],
            'scope_factors': {},
            'priority_level': 'medium'
        }
        
        text_lower = document_text.lower()
        
        # Analyze logical structure
        structure_score = self._analyze_logical_structure(text_lower)
        analysis_results['logical_structure_score'] = structure_score
        
        # Check for internal contradictions
        contradictions = self._detect_contradictions(text_lower)
        analysis_results['identified_issues'].extend(contradictions)
        
        # Analyze scope determination factors
        scope_factors = self._analyze_scope_factors(text_lower)
        analysis_results['scope_factors'] = scope_factors
        
        # Determine review scope
        scope_recommendation = self._determine_review_scope(scope_factors, contradictions, structure_score)
        analysis_results['review_scope_recommendation'] = scope_recommendation['recommendation']
        analysis_results['priority_level'] = scope_recommendation['priority']
        
        # Calculate overall consistency score
        contradiction_penalty = min(len(contradictions) * 0.1, 0.5)  # Max 50% penalty
        analysis_results['consistency_score'] = max(0, structure_score - contradiction_penalty)
        
        return analysis_results
    
    def _analyze_logical_structure(self, text: str) -> float:
        """Analyze the logical structure of arguments."""
        structure_elements = 0
        total_expected = len(self.consistency_framework['logical_structure_patterns'])
        
        for element_type, pattern in self.consistency_framework['logical_structure_patterns'].items():
            if re.search(pattern, text, re.IGNORECASE):
                structure_elements += 1
        
        return structure_elements / total_expected
    
    def _detect_contradictions(self, text: str) -> List[str]:
        """Detect potential internal contradictions."""
        contradictions = []
        
        for positive_pattern, negative_pattern in self.consistency_framework['consistency_checks']['internal_contradictions']:
            positive_matches = re.findall(positive_pattern, text, re.IGNORECASE)
            negative_matches = re.findall(negative_pattern, text, re.IGNORECASE)
            
            if positive_matches and negative_matches:
                contradictions.append(f"Potential contradiction: {positive_matches[0]} vs {negative_matches[0]}")
        
        return contradictions
    
    def _analyze_scope_factors(self, text: str) -> Dict[str, Any]:
        """Analyze factors that determine review scope."""
        scope_factors = {}
        
        for factor_type, pattern in self.consistency_framework['scope_determination_criteria'].items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            scope_factors[factor_type] = {
                'found': len(matches) > 0,
                'count': len(matches),
                'examples': matches[:3] if matches else []  # First 3 examples
            }
        
        return scope_factors
    
    def _determine_review_scope(self, scope_factors: Dict, contradictions: List, structure_score: float) -> Dict[str, str]:
        """Determine recommended review scope based on analysis."""
        
        # High priority triggers
        high_priority_triggers = 0
        if scope_factors.get('materiality_indicators', {}).get('found', False):
            high_priority_triggers += 1
        if scope_factors.get('risk_level_indicators', {}).get('found', False):
            high_priority_triggers += 1
        if scope_factors.get('regulatory_indicators', {}).get('found', False):
            high_priority_triggers += 1
        if len(contradictions) > 2:
            high_priority_triggers += 1
        if structure_score < 0.3:
            high_priority_triggers += 1
        
        # Determine scope and priority
        if high_priority_triggers >= 3:
            return {
                'recommendation': 'full_review_required',
                'priority': 'critical'
            }
        elif high_priority_triggers >= 2:
            return {
                'recommendation': 'comprehensive_review',
                'priority': 'high'
            }
        elif high_priority_triggers >= 1 or len(contradictions) > 0:
            return {
                'recommendation': 'focused_review',
                'priority': 'medium'
            }
        else:
            return {
                'recommendation': 'standard_review',
                'priority': 'low'
            }
    
    def get_analysis_framework_summary(self) -> Dict[str, Any]:
        """Get summary of analysis framework used."""
        return {
            'logical_structure_patterns': list(self.consistency_framework['logical_structure_patterns'].keys()),
            'scope_criteria': list(self.consistency_framework['scope_determination_criteria'].keys()),
            'consistency_check_types': list(self.consistency_framework['consistency_checks'].keys()),
            'contradiction_pattern_count': len(self.consistency_framework['consistency_checks']['internal_contradictions']),
            'review_scope_options': ['standard_review', 'focused_review', 'comprehensive_review', 'full_review_required'],
            'priority_levels': ['low', 'medium', 'high', 'critical']
        }