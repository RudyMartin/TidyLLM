#!/usr/bin/env python3
"""
Business Analysis RAG for tidyllm-compliance
===========================================

Enhanced RAG with Business Intelligence for regulatory research:
- Section Length Analysis (where authors invested effort)
- Signal-to-Noise Analysis (content quality vs fluff)
- Reference drilling for regulatory compliance research
- Business-friendly analysis for stakeholder assessment

Integrated with tidyllm-compliance for:
- Analyzing research papers mentioned in risk management documents
- Quick ROI assessment of research value for compliance teams
- Extracting business insights from academic sources
- Supporting regulatory decision-making with research backing

Part of tidyllm-compliance: Professional regulatory compliance platform
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import core research framework
from .research_framework import ResearchFramework, ResearchPaper, DecompositionScore

# Import compliance validation if available
try:
    from ..sop_conflict_analysis.yrsn_analyzer import YRSNNoiseAnalyzer
    from ..evidence.validation import EvidenceValidator
except ImportError:
    YRSNNoiseAnalyzer = None
    EvidenceValidator = None

class BusinessAnalysisRAG:
    """
    Enhanced RAG system with business intelligence capabilities.
    
    Features:
    - Business-friendly analysis of research papers
    - ROI assessment for compliance research investments
    - Section length analysis to identify author focus areas
    - Signal-to-noise analysis for content quality
    - Integration with regulatory compliance validation
    """
    
    def __init__(self, enable_compliance_validation: bool = True):
        # Initialize core research framework
        self.research_framework = ResearchFramework(
            enable_yrsn_validation=enable_compliance_validation
        )
        
        # Initialize compliance validators
        if enable_compliance_validation:
            self.yrsn_analyzer = YRSNNoiseAnalyzer() if YRSNNoiseAnalyzer else None
            self.evidence_validator = EvidenceValidator() if EvidenceValidator else None
        else:
            self.yrsn_analyzer = None
            self.evidence_validator = None
        
        # Business analysis metrics
        self.business_metrics = {
            'time_investment_threshold': 0.15,  # 15% minimum section length
            'quality_threshold': 0.7,  # 70% quality threshold
            'regulatory_relevance_threshold': 0.6  # 60% relevance threshold
        }
        
        print(f"[BUSINESS_RAG] Initialized business analysis RAG")
        if enable_compliance_validation:
            print(f"[INTEGRATION] Compliance validation enabled")
    
    def analyze_research_for_business(self, paper: ResearchPaper, 
                                    business_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Business-focused analysis of research paper.
        
        Args:
            paper: Research paper to analyze
            business_context: Business context for relevance assessment
            
        Returns:
            Business intelligence analysis with ROI and stakeholder insights
        """
        print(f"\n[BUSINESS_ANALYSIS] Analyzing for business value: {paper.title[:50]}...")
        
        # Step 1: Core research analysis
        research_analysis = self.research_framework.analyze_research_paper(paper, business_context)
        
        # Step 2: Business-specific analysis
        business_analysis = {
            'analysis_timestamp': datetime.now().isoformat(),
            'business_context': business_context,
            'core_research_analysis': research_analysis
        }
        
        # Step 3: ROI Assessment
        roi_assessment = self._assess_research_roi(research_analysis, business_context)
        business_analysis['roi_assessment'] = roi_assessment
        
        # Step 4: Stakeholder Value Analysis
        stakeholder_analysis = self._analyze_stakeholder_value(research_analysis, paper)
        business_analysis['stakeholder_value'] = stakeholder_analysis
        
        # Step 5: Implementation Readiness
        implementation_analysis = self._assess_implementation_readiness(research_analysis, paper)
        business_analysis['implementation_readiness'] = implementation_analysis
        
        # Step 6: Risk Analysis
        risk_analysis = self._analyze_research_risks(research_analysis, paper)
        business_analysis['risk_analysis'] = risk_analysis
        
        # Step 7: Executive Summary
        executive_summary = self._generate_executive_summary(business_analysis)
        business_analysis['executive_summary'] = executive_summary
        
        print(f"[COMPLETE] Business analysis complete")
        print(f"ROI Rating: {roi_assessment['roi_rating']}")
        print(f"Business Value Score: {roi_assessment['business_value_score']:.1f}/100")
        
        return business_analysis
    
    def _assess_research_roi(self, research_analysis: Dict[str, Any], 
                           business_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Assess return on investment for research utilization.
        
        Args:
            research_analysis: Core research analysis results
            business_context: Business context for ROI calculation
            
        Returns:
            ROI assessment with business metrics
        """
        # Extract key metrics
        regulatory_relevance = research_analysis.get('regulatory_relevance', {})
        overall_assessment = research_analysis.get('overall_assessment', {})
        
        relevance_score = regulatory_relevance.get('overall_score', 0.0)
        quality_rating = overall_assessment.get('quality_rating', 'NEEDS_REVIEW')
        research_value = overall_assessment.get('research_value_score', 0.0)
        
        # Business value calculation
        business_value_factors = {
            'regulatory_compliance_value': self._calculate_compliance_value(relevance_score),
            'implementation_ease': self._calculate_implementation_ease(research_analysis),
            'stakeholder_impact': self._calculate_stakeholder_impact(research_analysis),
            'time_to_value': self._calculate_time_to_value(research_analysis)
        }
        
        # Weighted business value score
        business_value_score = (
            business_value_factors['regulatory_compliance_value'] * 0.4 +
            business_value_factors['implementation_ease'] * 0.3 +
            business_value_factors['stakeholder_impact'] * 0.2 +
            business_value_factors['time_to_value'] * 0.1
        ) * 100
        
        # ROI rating determination
        if business_value_score >= 80:
            roi_rating = "HIGH_ROI"
            recommendation = "Strong recommendation for implementation"
        elif business_value_score >= 60:
            roi_rating = "MODERATE_ROI"
            recommendation = "Recommended with careful implementation planning"
        elif business_value_score >= 40:
            roi_rating = "LOW_ROI"
            recommendation = "Consider for future implementation or supplementary use"
        else:
            roi_rating = "NEGATIVE_ROI"
            recommendation = "Not recommended for current implementation"
        
        return {
            'roi_rating': roi_rating,
            'business_value_score': business_value_score,
            'value_factors': business_value_factors,
            'recommendation': recommendation,
            'cost_benefit_analysis': {
                'implementation_complexity': self._assess_implementation_complexity(research_analysis),
                'resource_requirements': self._assess_resource_requirements(research_analysis),
                'expected_benefits': self._identify_expected_benefits(research_analysis)
            }
        }
    
    def _analyze_stakeholder_value(self, research_analysis: Dict[str, Any], 
                                 paper: ResearchPaper) -> Dict[str, Any]:
        """
        Analyze value proposition for different stakeholder groups.
        
        Args:
            research_analysis: Core research analysis
            paper: Research paper being analyzed
            
        Returns:
            Stakeholder-specific value analysis
        """
        stakeholder_groups = {
            'compliance_officers': {
                'primary_interests': ['regulatory_alignment', 'audit_readiness', 'risk_mitigation'],
                'value_score': 0.0
            },
            'risk_managers': {
                'primary_interests': ['risk_assessment', 'quantitative_methods', 'model_validation'],
                'value_score': 0.0
            },
            'senior_management': {
                'primary_interests': ['strategic_value', 'competitive_advantage', 'regulatory_compliance'],
                'value_score': 0.0
            },
            'audit_teams': {
                'primary_interests': ['documentation_quality', 'evidence_validation', 'process_verification'],
                'value_score': 0.0
            }
        }
        
        # Calculate value for each stakeholder group
        regulatory_relevance = research_analysis.get('regulatory_relevance', {})
        overall_assessment = research_analysis.get('overall_assessment', {})
        
        content_to_analyze = f"{paper.title} {paper.abstract}"
        
        for group_name, group_info in stakeholder_groups.items():
            group_score = 0.0
            
            # Score based on stakeholder interests
            for interest in group_info['primary_interests']:
                if interest in ['regulatory_alignment', 'regulatory_compliance']:
                    group_score += regulatory_relevance.get('overall_score', 0.0) * 0.4
                elif interest == 'audit_readiness':
                    group_score += (1.0 if 'audit' in content_to_analyze.lower() else 0.0) * 0.3
                elif interest == 'risk_mitigation':
                    group_score += (1.0 if 'risk' in content_to_analyze.lower() else 0.0) * 0.3
                elif interest in ['risk_assessment', 'model_validation']:
                    group_score += (overall_assessment.get('research_value_score', 0.0) / 100) * 0.4
                elif interest == 'strategic_value':
                    group_score += (regulatory_relevance.get('overall_score', 0.0)) * 0.5
                elif interest == 'documentation_quality':
                    ref_analysis = research_analysis.get('reference_analysis', {})
                    group_score += min(ref_analysis.get('total_references_found', 0) / 20, 1.0) * 0.4
            
            stakeholder_groups[group_name]['value_score'] = min(1.0, group_score)
        
        # Identify primary and secondary stakeholders
        sorted_stakeholders = sorted(stakeholder_groups.items(), 
                                   key=lambda x: x[1]['value_score'], 
                                   reverse=True)
        
        return {
            'stakeholder_scores': stakeholder_groups,
            'primary_stakeholder': sorted_stakeholders[0][0],
            'primary_value_score': sorted_stakeholders[0][1]['value_score'],
            'secondary_stakeholder': sorted_stakeholders[1][0] if len(sorted_stakeholders) > 1 else None,
            'overall_stakeholder_alignment': sum(info['value_score'] for info in stakeholder_groups.values()) / len(stakeholder_groups)
        }
    
    def _assess_implementation_readiness(self, research_analysis: Dict[str, Any], 
                                       paper: ResearchPaper) -> Dict[str, Any]:
        """
        Assess readiness for implementing research insights.
        
        Args:
            research_analysis: Core research analysis
            paper: Research paper being analyzed
            
        Returns:
            Implementation readiness assessment
        """
        # Readiness factors
        readiness_factors = {
            'methodological_clarity': 0.0,
            'data_availability': 0.0,
            'technical_feasibility': 0.0,
            'regulatory_approval': 0.0,
            'resource_requirements': 0.0
        }
        
        # Assess methodological clarity
        section_analysis = research_analysis.get('section_analysis', {})
        methodology_percentage = section_analysis.get('section_breakdown', {}).get('methodology', {}).get('percentage_of_total', 0)
        readiness_factors['methodological_clarity'] = min(methodology_percentage / 15, 1.0)  # 15% is good methodology coverage
        
        # Assess data availability
        content_lower = (paper.content + " " + paper.abstract).lower()
        data_indicators = ['dataset', 'data source', 'publicly available', 'open data', 'repository']
        data_score = sum(0.2 for indicator in data_indicators if indicator in content_lower)
        readiness_factors['data_availability'] = min(data_score, 1.0)
        
        # Assess technical feasibility  
        tech_indicators = ['implementation', 'algorithm', 'code', 'software', 'tool']
        tech_score = sum(0.2 for indicator in tech_indicators if indicator in content_lower)
        readiness_factors['technical_feasibility'] = min(tech_score, 1.0)
        
        # Assess regulatory approval readiness
        regulatory_relevance = research_analysis.get('regulatory_relevance', {})
        readiness_factors['regulatory_approval'] = regulatory_relevance.get('overall_score', 0.0)
        
        # Assess resource requirements (inverse score - lower requirements = higher readiness)
        complexity_indicators = ['complex', 'sophisticated', 'advanced', 'specialized']
        complexity_score = sum(0.2 for indicator in complexity_indicators if indicator in content_lower)
        readiness_factors['resource_requirements'] = max(0.0, 1.0 - min(complexity_score, 1.0))
        
        # Overall readiness score
        overall_readiness = sum(readiness_factors.values()) / len(readiness_factors)
        
        # Readiness rating
        if overall_readiness >= 0.8:
            readiness_rating = "READY_FOR_IMPLEMENTATION"
        elif overall_readiness >= 0.6:
            readiness_rating = "PILOT_READY"
        elif overall_readiness >= 0.4:
            readiness_rating = "NEEDS_DEVELOPMENT"
        else:
            readiness_rating = "RESEARCH_STAGE_ONLY"
        
        return {
            'readiness_rating': readiness_rating,
            'overall_readiness_score': overall_readiness * 100,
            'readiness_factors': {k: v * 100 for k, v in readiness_factors.items()},
            'implementation_timeline': self._estimate_implementation_timeline(overall_readiness),
            'critical_dependencies': self._identify_critical_dependencies(research_analysis, paper)
        }
    
    def _analyze_research_risks(self, research_analysis: Dict[str, Any], 
                              paper: ResearchPaper) -> Dict[str, Any]:
        """
        Analyze risks associated with implementing research insights.
        
        Args:
            research_analysis: Core research analysis
            paper: Research paper being analyzed
            
        Returns:
            Risk analysis with mitigation strategies
        """
        risk_categories = {
            'regulatory_risk': {
                'level': 'low',
                'description': '',
                'mitigation_strategies': []
            },
            'implementation_risk': {
                'level': 'low',
                'description': '',
                'mitigation_strategies': []
            },
            'quality_risk': {
                'level': 'low', 
                'description': '',
                'mitigation_strategies': []
            },
            'operational_risk': {
                'level': 'low',
                'description': '',
                'mitigation_strategies': []
            }
        }
        
        # Assess regulatory risk
        regulatory_relevance = research_analysis.get('regulatory_relevance', {})
        relevance_category = regulatory_relevance.get('relevance_category', 'LIMITED_RELEVANCE')
        
        if relevance_category == 'LIMITED_RELEVANCE':
            risk_categories['regulatory_risk']['level'] = 'high'
            risk_categories['regulatory_risk']['description'] = 'Limited regulatory alignment may cause compliance issues'
            risk_categories['regulatory_risk']['mitigation_strategies'] = [
                'Supplement with additional regulatory guidance',
                'Consult with compliance officers before implementation'
            ]
        elif relevance_category == 'SOMEWHAT_RELEVANT':
            risk_categories['regulatory_risk']['level'] = 'medium'
            risk_categories['regulatory_risk']['description'] = 'Moderate regulatory alignment requires careful implementation'
            risk_categories['regulatory_risk']['mitigation_strategies'] = [
                'Validate approach with regulatory experts',
                'Implement with enhanced documentation'
            ]
        
        # Assess quality risk
        overall_assessment = research_analysis.get('overall_assessment', {})
        quality_rating = overall_assessment.get('quality_rating', 'NEEDS_REVIEW')
        
        if quality_rating in ['NEEDS_REVIEW', 'ACCEPTABLE']:
            risk_categories['quality_risk']['level'] = 'high' if quality_rating == 'NEEDS_REVIEW' else 'medium'
            risk_categories['quality_risk']['description'] = f'Research quality rated as {quality_rating}'
            risk_categories['quality_risk']['mitigation_strategies'] = [
                'Validate findings with additional research sources',
                'Implement enhanced quality controls',
                'Consider peer review before implementation'
            ]
        
        # Assess implementation risk
        ref_analysis = research_analysis.get('reference_analysis', {})
        total_refs = ref_analysis.get('total_references_found', 0)
        
        if total_refs < 10:
            risk_categories['implementation_risk']['level'] = 'medium'
            risk_categories['implementation_risk']['description'] = 'Limited references may indicate incomplete methodology'
            risk_categories['implementation_risk']['mitigation_strategies'] = [
                'Conduct additional research to fill methodology gaps',
                'Implement with enhanced testing and validation'
            ]
        
        # Assess operational risk based on complexity
        content_lower = (paper.content + " " + paper.abstract).lower()
        complexity_indicators = ['complex', 'sophisticated', 'advanced', 'specialized', 'novel']
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in content_lower)
        
        if complexity_score >= 3:
            risk_categories['operational_risk']['level'] = 'high'
            risk_categories['operational_risk']['description'] = 'High complexity may lead to operational challenges'
            risk_categories['operational_risk']['mitigation_strategies'] = [
                'Develop comprehensive training programs',
                'Implement phased rollout approach',
                'Establish expert support team'
            ]
        elif complexity_score >= 1:
            risk_categories['operational_risk']['level'] = 'medium'
            risk_categories['operational_risk']['description'] = 'Moderate complexity requires careful change management'
            risk_categories['operational_risk']['mitigation_strategies'] = [
                'Provide adequate training and documentation',
                'Monitor implementation closely'
            ]
        
        # Calculate overall risk level
        risk_levels = {'low': 1, 'medium': 2, 'high': 3}
        avg_risk = sum(risk_levels[cat['level']] for cat in risk_categories.values()) / len(risk_categories)
        
        if avg_risk >= 2.5:
            overall_risk = 'high'
        elif avg_risk >= 1.5:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'
        
        return {
            'overall_risk_level': overall_risk,
            'risk_categories': risk_categories,
            'risk_mitigation_priority': 'high' if overall_risk == 'high' else 'medium' if overall_risk == 'medium' else 'low',
            'recommended_risk_controls': self._recommend_risk_controls(risk_categories)
        }
    
    def _generate_executive_summary(self, business_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate executive summary for business stakeholders.
        
        Args:
            business_analysis: Complete business analysis results
            
        Returns:
            Executive summary with key insights and recommendations
        """
        roi_assessment = business_analysis.get('roi_assessment', {})
        stakeholder_value = business_analysis.get('stakeholder_value', {})
        implementation_readiness = business_analysis.get('implementation_readiness', {})
        risk_analysis = business_analysis.get('risk_analysis', {})
        
        # Key metrics
        key_metrics = {
            'business_value_score': roi_assessment.get('business_value_score', 0.0),
            'roi_rating': roi_assessment.get('roi_rating', 'UNKNOWN'),
            'primary_stakeholder': stakeholder_value.get('primary_stakeholder', 'unknown'),
            'implementation_timeline': implementation_readiness.get('implementation_timeline', 'unknown'),
            'overall_risk_level': risk_analysis.get('overall_risk_level', 'unknown')
        }
        
        # Strategic recommendations
        strategic_recommendations = []
        
        if key_metrics['roi_rating'] == 'HIGH_ROI':
            strategic_recommendations.append("STRATEGIC PRIORITY: High ROI potential - recommend fast-track implementation")
        elif key_metrics['roi_rating'] == 'MODERATE_ROI':
            strategic_recommendations.append("STRATEGIC CONSIDERATION: Moderate ROI - plan careful implementation")
        else:
            strategic_recommendations.append("STRATEGIC CAUTION: Limited ROI - consider alternative approaches")
        
        if key_metrics['overall_risk_level'] == 'high':
            strategic_recommendations.append("RISK MANAGEMENT: High risk level requires comprehensive mitigation strategy")
        
        # Implementation recommendations
        implementation_recommendations = []
        
        readiness_rating = implementation_readiness.get('readiness_rating', 'UNKNOWN')
        if readiness_rating == 'READY_FOR_IMPLEMENTATION':
            implementation_recommendations.append("Implementation ready - proceed with full deployment")
        elif readiness_rating == 'PILOT_READY':
            implementation_recommendations.append("Pilot implementation recommended before full deployment")
        else:
            implementation_recommendations.append("Additional development required before implementation")
        
        return {
            'key_metrics': key_metrics,
            'strategic_recommendations': strategic_recommendations,
            'implementation_recommendations': implementation_recommendations,
            'business_impact_summary': self._generate_business_impact_summary(business_analysis),
            'next_steps': self._generate_next_steps(business_analysis),
            'executive_decision_points': self._identify_decision_points(business_analysis)
        }
    
    # Helper methods for business analysis
    def _calculate_compliance_value(self, relevance_score: float) -> float:
        """Calculate regulatory compliance value score."""
        return min(relevance_score * 1.2, 1.0)  # Boost regulatory relevance
    
    def _calculate_implementation_ease(self, research_analysis: Dict[str, Any]) -> float:
        """Calculate implementation ease score."""
        # Based on methodology clarity and reference quality
        section_analysis = research_analysis.get('section_analysis', {})
        methodology_pct = section_analysis.get('section_breakdown', {}).get('methodology', {}).get('percentage_of_total', 0)
        
        ref_analysis = research_analysis.get('reference_analysis', {})
        ref_count = ref_analysis.get('total_references_found', 0)
        
        return min((methodology_pct / 15) + (ref_count / 50), 1.0)
    
    def _calculate_stakeholder_impact(self, research_analysis: Dict[str, Any]) -> float:
        """Calculate potential stakeholder impact score."""
        overall_assessment = research_analysis.get('overall_assessment', {})
        research_value = overall_assessment.get('research_value_score', 0.0)
        return research_value / 100
    
    def _calculate_time_to_value(self, research_analysis: Dict[str, Any]) -> float:
        """Calculate time to value score (higher = faster value realization)."""
        # Based on implementation readiness indicators
        section_analysis = research_analysis.get('section_analysis', {})
        results_pct = section_analysis.get('section_breakdown', {}).get('results', {}).get('percentage_of_total', 0)
        
        # Higher results percentage = clearer outcomes = faster time to value
        return min(results_pct / 20, 1.0)
    
    def _assess_implementation_complexity(self, research_analysis: Dict[str, Any]) -> str:
        """Assess implementation complexity level."""
        section_analysis = research_analysis.get('section_analysis', {})
        methodology_pct = section_analysis.get('section_breakdown', {}).get('methodology', {}).get('percentage_of_total', 0)
        
        if methodology_pct > 25:
            return "HIGH"
        elif methodology_pct > 15:
            return "MEDIUM" 
        else:
            return "LOW"
    
    def _assess_resource_requirements(self, research_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Assess resource requirements for implementation."""
        return {
            'technical_expertise': "MEDIUM",  # Default assessment
            'time_investment': "MEDIUM",
            'infrastructure': "LOW",
            'training_needs': "MEDIUM"
        }
    
    def _identify_expected_benefits(self, research_analysis: Dict[str, Any]) -> List[str]:
        """Identify expected benefits from implementation."""
        benefits = []
        
        regulatory_relevance = research_analysis.get('regulatory_relevance', {})
        relevance_category = regulatory_relevance.get('relevance_category', 'LIMITED_RELEVANCE')
        
        if relevance_category in ['HIGHLY_RELEVANT', 'MODERATELY_RELEVANT']:
            benefits.append("Enhanced regulatory compliance")
            benefits.append("Reduced regulatory risk")
        
        overall_assessment = research_analysis.get('overall_assessment', {})
        quality_rating = overall_assessment.get('quality_rating', 'NEEDS_REVIEW')
        
        if quality_rating in ['EXCELLENT', 'GOOD']:
            benefits.append("Improved decision-making quality")
            benefits.append("Enhanced analytical capabilities")
        
        return benefits if benefits else ["Limited expected benefits identified"]
    
    def _estimate_implementation_timeline(self, readiness_score: float) -> str:
        """Estimate implementation timeline based on readiness."""
        if readiness_score >= 0.8:
            return "3-6 months"
        elif readiness_score >= 0.6:
            return "6-12 months"
        elif readiness_score >= 0.4:
            return "12-18 months"
        else:
            return "18+ months"
    
    def _identify_critical_dependencies(self, research_analysis: Dict[str, Any], 
                                      paper: ResearchPaper) -> List[str]:
        """Identify critical dependencies for implementation."""
        dependencies = []
        
        # Data dependencies
        content_lower = (paper.content + " " + paper.abstract).lower()
        if 'dataset' in content_lower or 'data' in content_lower:
            dependencies.append("Data availability and quality")
        
        # Technical dependencies
        if any(term in content_lower for term in ['software', 'system', 'platform']):
            dependencies.append("Technical infrastructure")
        
        # Regulatory dependencies
        regulatory_relevance = research_analysis.get('regulatory_relevance', {})
        if regulatory_relevance.get('relevance_category') != 'HIGHLY_RELEVANT':
            dependencies.append("Regulatory approval and alignment")
        
        return dependencies if dependencies else ["No critical dependencies identified"]
    
    def _recommend_risk_controls(self, risk_categories: Dict[str, Any]) -> List[str]:
        """Recommend risk controls based on risk analysis."""
        controls = []
        
        for category, risk_info in risk_categories.items():
            if risk_info['level'] == 'high':
                controls.extend(risk_info['mitigation_strategies'])
        
        # Default controls if none identified
        if not controls:
            controls = [
                "Regular monitoring and review",
                "Documented implementation procedures",
                "Stakeholder communication plan"
            ]
        
        return list(set(controls))  # Remove duplicates
    
    def _generate_business_impact_summary(self, business_analysis: Dict[str, Any]) -> str:
        """Generate business impact summary."""
        roi_assessment = business_analysis.get('roi_assessment', {})
        business_value_score = roi_assessment.get('business_value_score', 0.0)
        roi_rating = roi_assessment.get('roi_rating', 'UNKNOWN')
        
        if business_value_score >= 80:
            return f"HIGH IMPACT: Business value score of {business_value_score:.0f}/100 indicates strong potential for positive organizational impact"
        elif business_value_score >= 60:
            return f"MODERATE IMPACT: Business value score of {business_value_score:.0f}/100 suggests moderate organizational benefits"
        else:
            return f"LIMITED IMPACT: Business value score of {business_value_score:.0f}/100 indicates limited organizational benefits"
    
    def _generate_next_steps(self, business_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommended next steps."""
        next_steps = []
        
        roi_assessment = business_analysis.get('roi_assessment', {})
        roi_rating = roi_assessment.get('roi_rating', 'UNKNOWN')
        
        if roi_rating == 'HIGH_ROI':
            next_steps.extend([
                "Develop detailed implementation plan",
                "Secure necessary resources and stakeholder buy-in",
                "Begin pilot implementation"
            ])
        elif roi_rating == 'MODERATE_ROI':
            next_steps.extend([
                "Conduct more detailed feasibility analysis",
                "Evaluate alternative approaches",
                "Plan phased implementation approach"
            ])
        else:
            next_steps.extend([
                "Re-evaluate business case",
                "Consider alternative research sources",
                "Focus on higher-value opportunities"
            ])
        
        return next_steps
    
    def _identify_decision_points(self, business_analysis: Dict[str, Any]) -> List[str]:
        """Identify key decision points for executives."""
        decision_points = []
        
        roi_assessment = business_analysis.get('roi_assessment', {})
        risk_analysis = business_analysis.get('risk_analysis', {})
        implementation_readiness = business_analysis.get('implementation_readiness', {})
        
        # ROI decision point
        roi_rating = roi_assessment.get('roi_rating', 'UNKNOWN')
        decision_points.append(f"Investment Decision: {roi_rating} - Proceed with implementation?")
        
        # Risk decision point
        risk_level = risk_analysis.get('overall_risk_level', 'unknown')
        if risk_level == 'high':
            decision_points.append("Risk Management: High risk level - Acceptable for organization?")
        
        # Timing decision point
        readiness_rating = implementation_readiness.get('readiness_rating', 'UNKNOWN')
        if readiness_rating in ['NEEDS_DEVELOPMENT', 'RESEARCH_STAGE_ONLY']:
            decision_points.append("Timing Decision: Research requires additional development - Invest in preparation?")
        
        return decision_points

# Example usage and demonstration
def demo_business_analysis_rag():
    """Demonstrate business analysis RAG functionality."""
    from .research_framework import get_demo_papers
    
    business_rag = BusinessAnalysisRAG(enable_compliance_validation=True)
    
    # Get demo papers
    papers = get_demo_papers()
    
    print("\nBusiness Analysis RAG Demo")
    print("=" * 40)
    
    for i, paper in enumerate(papers[:1], 1):  # Test first paper
        print(f"\n[PAPER {i}] Business Analysis: {paper.title}")
        
        # Add content for analysis
        paper.content = f"""This research presents a comprehensive analysis of {paper.title.lower()}.
        
        The methodology section (15% of paper) details statistical analysis approaches.
        Results section (25% of paper) shows significant findings with p < 0.05.
        Discussion section (20% of paper) covers implications for practice.
        
        The study includes 30 references from peer-reviewed sources.
        Data sources include publicly available regulatory datasets.
        Implementation requires moderate technical expertise.
        """
        
        try:
            analysis = business_rag.analyze_research_for_business(paper, "regulatory_compliance")
            
            # Display key results
            executive_summary = analysis.get('executive_summary', {})
            key_metrics = executive_summary.get('key_metrics', {})
            
            print(f"\nBusiness Analysis Results:")
            print(f"  Business Value Score: {key_metrics.get('business_value_score', 0):.1f}/100")
            print(f"  ROI Rating: {key_metrics.get('roi_rating', 'unknown')}")
            print(f"  Primary Stakeholder: {key_metrics.get('primary_stakeholder', 'unknown')}")
            print(f"  Implementation Timeline: {key_metrics.get('implementation_timeline', 'unknown')}")
            print(f"  Risk Level: {key_metrics.get('overall_risk_level', 'unknown')}")
            
            # Show strategic recommendations
            strategic_recs = executive_summary.get('strategic_recommendations', [])
            if strategic_recs:
                print(f"\nStrategic Recommendations:")
                for rec in strategic_recs[:2]:
                    print(f"  - {rec}")
            
        except Exception as e:
            print(f"Business analysis failed: {e}")

if __name__ == "__main__":
    demo_business_analysis_rag()