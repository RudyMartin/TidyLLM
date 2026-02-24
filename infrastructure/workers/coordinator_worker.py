# Coordinator Worker - Multi-Template Result Synthesis
# 🔄 RESULTS SYNTHESIS WORKER - Intelligent Multi-Template Coordination
#
# This worker coordinates and synthesizes results from multiple template processing:
# - Collects results from multiple PromptWorkers
# - Resolves conflicts between different analytical perspectives
# - Synthesizes comprehensive final reports
# - Manages quality assurance across multiple outputs
#
# Dependencies:
# - BaseWorker for task queue management
# - Template results from PromptWorkers
# - AI-Assisted Manager for coordination instructions

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .base_worker import BaseWorker, TaskInput, TaskResult

logger = logging.getLogger(__name__)

class SynthesisStrategy(Enum):
    CONSENSUS_BUILDING = "consensus_building"
    PRIORITY_WEIGHTED = "priority_weighted"
    CONFLICT_RESOLUTION = "conflict_resolution"
    COMPREHENSIVE_MERGE = "comprehensive_merge"

@dataclass
class TemplateResult:
    template_name: str
    result_content: str
    confidence_score: float
    processing_time: int
    quality_metrics: Dict[str, float]
    recommendations: List[str]
    risk_indicators: List[str]

@dataclass
class CoordinatorTask:
    document_path: str
    template_results: List[TemplateResult]
    synthesis_strategy: SynthesisStrategy
    priority_weights: Optional[Dict[str, float]] = None
    quality_requirements: Optional[Dict[str, float]] = None

@dataclass
class CoordinatorResult:
    synthesized_report: str
    consensus_areas: List[str]
    conflict_resolutions: List[Dict[str, Any]]
    final_recommendations: List[str]
    quality_assessment: Dict[str, float]
    confidence_score: float

class CoordinatorWorker(BaseWorker[CoordinatorTask, CoordinatorResult]):
    """
    Coordinator Worker for synthesizing multi-template analysis results.
    
    This worker provides:
    1. Result Collection - Gathers outputs from multiple PromptWorkers
    2. Conflict Resolution - Identifies and resolves analytical conflicts
    3. Synthesis - Creates comprehensive unified reports
    4. Quality Assurance - Ensures output quality across templates
    """
    
    def __init__(self, synthesis_prompts_path: str = "C:/Users/marti/github/prompts/synthesis"):
        super().__init__()
        self.synthesis_prompts_path = Path(synthesis_prompts_path)
        self.conflict_resolution_strategies = {
            SynthesisStrategy.CONSENSUS_BUILDING: self._consensus_building_synthesis,
            SynthesisStrategy.PRIORITY_WEIGHTED: self._priority_weighted_synthesis,
            SynthesisStrategy.CONFLICT_RESOLUTION: self._conflict_resolution_synthesis,
            SynthesisStrategy.COMPREHENSIVE_MERGE: self._comprehensive_merge_synthesis
        }
        
        # Quality thresholds for synthesis validation
        self.quality_thresholds = {
            "consistency": 0.85,
            "completeness": 0.90,
            "accuracy": 0.88,
            "actionability": 0.80
        }
    
    async def process_task(self, task: CoordinatorTask) -> CoordinatorResult:
        """
        Main coordination processing for multi-template result synthesis.
        """
        try:
            logger.info(f"Starting coordination for {task.document_path} with {len(task.template_results)} results")
            
            # Step 1: Validate input results
            validated_results = await self._validate_template_results(task.template_results)
            
            # Step 2: Analyze conflicts and consensus areas
            consensus_analysis = await self._analyze_consensus_and_conflicts(validated_results)
            
            # Step 3: Apply synthesis strategy
            synthesis_method = self.conflict_resolution_strategies[task.synthesis_strategy]
            synthesized_content = await synthesis_method(validated_results, task, consensus_analysis)
            
            # Step 4: Generate final recommendations
            final_recommendations = await self._generate_final_recommendations(
                synthesized_content, validated_results, consensus_analysis
            )
            
            # Step 5: Quality assessment
            quality_assessment = await self._assess_synthesis_quality(
                synthesized_content, validated_results, final_recommendations
            )
            
            # Step 6: Calculate overall confidence
            confidence_score = self._calculate_confidence_score(validated_results, consensus_analysis)
            
            return CoordinatorResult(
                synthesized_report=synthesized_content,
                consensus_areas=consensus_analysis["consensus_areas"],
                conflict_resolutions=consensus_analysis["resolved_conflicts"],
                final_recommendations=final_recommendations,
                quality_assessment=quality_assessment,
                confidence_score=confidence_score,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Coordinator processing failed: {e}")
            return CoordinatorResult(
                synthesized_report="",
                consensus_areas=[],
                conflict_resolutions=[],
                final_recommendations=[],
                quality_assessment={},
                confidence_score=0.0,
                success=False,
                error=str(e)
            )
    
    async def _validate_template_results(self, results: List[TemplateResult]) -> List[TemplateResult]:
        """
        Validate and filter template results for synthesis quality.
        """
        validated = []
        
        for result in results:
            # Basic validation checks
            if not result.result_content or len(result.result_content.strip()) < 100:
                logger.warning(f"Skipping {result.template_name} - insufficient content")
                continue
            
            if result.confidence_score < 0.5:
                logger.warning(f"Low confidence result from {result.template_name}: {result.confidence_score}")
                # Still include but flag for careful handling
            
            # Quality threshold checks
            quality_ok = True
            for metric, threshold in self.quality_thresholds.items():
                if result.quality_metrics.get(metric, 1.0) < threshold * 0.7:  # 70% of threshold for inclusion
                    logger.warning(f"{result.template_name} failed {metric} quality check")
                    quality_ok = False
                    break
            
            if quality_ok or len(results) <= 2:  # Always include if we have few results
                validated.append(result)
        
        logger.info(f"Validated {len(validated)} of {len(results)} template results")
        return validated
    
    async def _analyze_consensus_and_conflicts(self, results: List[TemplateResult]) -> Dict[str, Any]:
        """
        Analyze areas of consensus and conflict between template results.
        """
        consensus_areas = []
        conflicts = []
        
        # Extract key themes and recommendations from each result
        all_themes = {}
        all_recommendations = {}
        all_risks = {}
        
        for result in results:
            # Extract themes (simplified - could use NLP here)
            themes = self._extract_themes(result.result_content)
            for theme in themes:
                if theme not in all_themes:
                    all_themes[theme] = []
                all_themes[theme].append(result.template_name)
            
            # Track recommendations
            for rec in result.recommendations:
                rec_key = rec.lower()[:50]  # First 50 chars as key
                if rec_key not in all_recommendations:
                    all_recommendations[rec_key] = []
                all_recommendations[rec_key].append((result.template_name, rec))
            
            # Track risk indicators
            for risk in result.risk_indicators:
                risk_key = risk.lower()[:50]
                if risk_key not in all_risks:
                    all_risks[risk_key] = []
                all_risks[risk_key].append((result.template_name, risk))
        
        # Identify consensus areas (mentioned by multiple templates)
        for theme, templates in all_themes.items():
            if len(templates) >= max(2, len(results) // 2):
                consensus_areas.append({
                    "theme": theme,
                    "supporting_templates": templates,
                    "consensus_strength": len(templates) / len(results)
                })
        
        # Identify conflicts (contradictory recommendations/assessments)
        resolved_conflicts = []
        for rec_key, rec_instances in all_recommendations.items():
            if len(rec_instances) > 1:
                # Check for contradictions
                recommendations_text = [r[1] for r in rec_instances]
                if self._detect_contradiction(recommendations_text):
                    conflicts.append({
                        "type": "recommendation_conflict",
                        "key": rec_key,
                        "conflicting_recommendations": rec_instances,
                        "resolution_strategy": "priority_weighted"
                    })
        
        # Resolve conflicts
        for conflict in conflicts:
            resolution = await self._resolve_conflict(conflict, results)
            resolved_conflicts.append(resolution)
        
        return {
            "consensus_areas": consensus_areas,
            "conflicts": conflicts,
            "resolved_conflicts": resolved_conflicts,
            "theme_analysis": all_themes,
            "recommendation_analysis": all_recommendations,
            "risk_analysis": all_risks
        }
    
    def _extract_themes(self, content: str) -> List[str]:
        """
        Extract key themes from result content (simplified implementation).
        """
        # Simplified theme extraction - in production would use NLP
        themes = []
        content_lower = content.lower()
        
        theme_keywords = {
            "risk_assessment": ["risk", "threat", "vulnerability", "exposure"],
            "financial_analysis": ["financial", "revenue", "profit", "cost", "budget"],
            "compliance": ["compliance", "regulation", "policy", "standard"],
            "quality_control": ["quality", "accuracy", "completeness", "validation"],
            "recommendations": ["recommend", "suggest", "propose", "should"],
            "data_analysis": ["data", "analysis", "findings", "results"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def _detect_contradiction(self, recommendations: List[str]) -> bool:
        """
        Detect contradictory recommendations (simplified implementation).
        """
        # Simplified contradiction detection
        contradiction_pairs = [
            (["approve", "accept"], ["reject", "deny", "decline"]),
            (["increase", "expand"], ["decrease", "reduce", "limit"]),
            (["continue", "proceed"], ["stop", "halt", "discontinue"]),
            (["low risk"], ["high risk"]),
            (["compliant"], ["non-compliant", "violation"])
        ]
        
        rec_text = " ".join(recommendations).lower()
        
        for positive_terms, negative_terms in contradiction_pairs:
            has_positive = any(term in rec_text for term in positive_terms)
            has_negative = any(term in rec_text for term in negative_terms)
            
            if has_positive and has_negative:
                return True
        
        return False
    
    async def _resolve_conflict(self, conflict: Dict[str, Any], results: List[TemplateResult]) -> Dict[str, Any]:
        """
        Resolve a specific conflict between template results.
        """
        resolution = {
            "conflict_type": conflict["type"],
            "conflict_key": conflict["key"],
            "resolution_method": conflict["resolution_strategy"],
            "resolved_recommendation": "",
            "confidence": 0.0,
            "rationale": ""
        }
        
        if conflict["resolution_strategy"] == "priority_weighted":
            # Weight by template confidence and quality
            weighted_recommendations = []
            
            for template_name, recommendation in conflict["conflicting_recommendations"]:
                # Find the template result for weighting
                template_result = next((r for r in results if r.template_name == template_name), None)
                if template_result:
                    weight = (
                        template_result.confidence_score * 0.4 +
                        template_result.quality_metrics.get("accuracy", 0.8) * 0.3 +
                        template_result.quality_metrics.get("completeness", 0.8) * 0.3
                    )
                    weighted_recommendations.append((recommendation, weight, template_name))
            
            # Select highest weighted recommendation
            if weighted_recommendations:
                weighted_recommendations.sort(key=lambda x: x[1], reverse=True)
                best_rec, best_weight, best_template = weighted_recommendations[0]
                
                resolution.update({
                    "resolved_recommendation": best_rec,
                    "confidence": best_weight,
                    "rationale": f"Selected recommendation from {best_template} based on highest quality/confidence weighting ({best_weight:.2f})"
                })
        
        return resolution
    
    async def _consensus_building_synthesis(
        self, 
        results: List[TemplateResult], 
        task: CoordinatorTask, 
        consensus_analysis: Dict[str, Any]
    ) -> str:
        """
        Build synthesis focusing on consensus areas and common findings.
        """
        synthesis_sections = []
        
        # Executive Summary based on consensus
        consensus_themes = [area["theme"] for area in consensus_analysis["consensus_areas"]]
        synthesis_sections.append(f"""
# Comprehensive Analysis Report - Consensus-Based Synthesis

## Executive Summary
Based on analysis using {len(results)} analytical frameworks, the following consensus has emerged:

**Primary Themes Identified:**
{chr(10).join(f"- {theme.replace('_', ' ').title()}" for theme in consensus_themes)}

**Consensus Strength:** {len(consensus_analysis['consensus_areas'])} areas of agreement across multiple analytical perspectives.
""")
        
        # Detailed consensus areas
        synthesis_sections.append("## Areas of Analytical Consensus")
        for area in consensus_analysis["consensus_areas"]:
            strength_pct = int(area["consensus_strength"] * 100)
            synthesis_sections.append(f"""
### {area['theme'].replace('_', ' ').title()}
- **Consensus Strength:** {strength_pct}% ({len(area['supporting_templates'])} of {len(results)} frameworks agree)
- **Supporting Analysis:** {', '.join(area['supporting_templates'])}
""")
        
        # Synthesized recommendations from consensus
        synthesis_sections.append(self._build_consensus_recommendations(results, consensus_analysis))
        
        return "\n".join(synthesis_sections)
    
    async def _priority_weighted_synthesis(
        self, 
        results: List[TemplateResult], 
        task: CoordinatorTask, 
        consensus_analysis: Dict[str, Any]
    ) -> str:
        """
        Build synthesis using priority weighting based on template importance and quality.
        """
        # Calculate weights for each template
        template_weights = {}
        if task.priority_weights:
            template_weights = task.priority_weights
        else:
            # Default weighting based on confidence and quality
            for result in results:
                weight = (
                    result.confidence_score * 0.4 +
                    result.quality_metrics.get("accuracy", 0.8) * 0.3 +
                    result.quality_metrics.get("completeness", 0.8) * 0.3
                )
                template_weights[result.template_name] = weight
        
        # Sort results by priority weight
        sorted_results = sorted(results, key=lambda r: template_weights.get(r.template_name, 0.5), reverse=True)
        
        synthesis_sections = []
        synthesis_sections.append(f"""
# Comprehensive Analysis Report - Priority-Weighted Synthesis

## Executive Summary
Analysis performed using {len(results)} frameworks with priority weighting based on relevance and quality metrics.

**Framework Priority Ranking:**
{chr(10).join(f"{i+1}. {r.template_name.replace('_', ' ').title()} (Weight: {template_weights.get(r.template_name, 0.5):.2f})" 
             for i, r in enumerate(sorted_results))}
""")
        
        # Lead with highest priority analysis
        primary_result = sorted_results[0]
        synthesis_sections.append(f"""
## Primary Analysis ({primary_result.template_name.replace('_', ' ').title()})
**Priority Weight:** {template_weights.get(primary_result.template_name, 0.5):.2f}

{self._extract_key_findings(primary_result.result_content)}
""")
        
        # Supporting analyses
        if len(sorted_results) > 1:
            synthesis_sections.append("## Supporting Analysis")
            for result in sorted_results[1:]:
                weight = template_weights.get(result.template_name, 0.5)
                synthesis_sections.append(f"""
### {result.template_name.replace('_', ' ').title()} (Weight: {weight:.2f})
{self._extract_key_findings(result.result_content, max_length=500)}
""")
        
        # Weighted recommendations
        synthesis_sections.append(self._build_weighted_recommendations(sorted_results, template_weights))
        
        return "\n".join(synthesis_sections)
    
    async def _conflict_resolution_synthesis(
        self, 
        results: List[TemplateResult], 
        task: CoordinatorTask, 
        consensus_analysis: Dict[str, Any]
    ) -> str:
        """
        Build synthesis focusing on resolving conflicts between analyses.
        """
        synthesis_sections = []
        synthesis_sections.append(f"""
# Comprehensive Analysis Report - Conflict Resolution Synthesis

## Executive Summary
Analysis identified and resolved {len(consensus_analysis['conflicts'])} areas of conflicting assessments across {len(results)} analytical frameworks.
""")
        
        # Document conflicts and resolutions
        if consensus_analysis['resolved_conflicts']:
            synthesis_sections.append("## Conflict Resolution Summary")
            for resolution in consensus_analysis['resolved_conflicts']:
                synthesis_sections.append(f"""
### {resolution['conflict_type'].replace('_', ' ').title()}
- **Conflict Area:** {resolution['conflict_key'][:100]}...
- **Resolution Method:** {resolution['resolution_method']}
- **Final Resolution:** {resolution['resolved_recommendation']}
- **Confidence Level:** {resolution['confidence']:.2f}
- **Rationale:** {resolution['rationale']}
""")
        
        # Consensus areas (uncontested findings)
        synthesis_sections.append("## Uncontested Findings")
        for area in consensus_analysis["consensus_areas"]:
            synthesis_sections.append(f"- **{area['theme'].replace('_', ' ').title()}**: Consistent across {len(area['supporting_templates'])} frameworks")
        
        # Final synthesized assessment
        synthesis_sections.append(self._build_conflict_resolved_recommendations(consensus_analysis))
        
        return "\n".join(synthesis_sections)
    
    async def _comprehensive_merge_synthesis(
        self, 
        results: List[TemplateResult], 
        task: CoordinatorTask, 
        consensus_analysis: Dict[str, Any]
    ) -> str:
        """
        Build comprehensive synthesis merging all analytical perspectives.
        """
        synthesis_sections = []
        synthesis_sections.append(f"""
# Comprehensive Multi-Framework Analysis Report

## Executive Summary
This report synthesizes findings from {len(results)} complementary analytical frameworks to provide a comprehensive assessment.

**Analytical Frameworks Applied:**
{chr(10).join(f"- {r.template_name.replace('_', ' ').title()}" for r in results)}
""")
        
        # Comprehensive findings by category
        all_categories = set()
        for result in results:
            categories = self._categorize_findings(result.result_content)
            all_categories.update(categories.keys())
        
        for category in sorted(all_categories):
            synthesis_sections.append(f"## {category.replace('_', ' ').title()} Analysis")
            
            category_findings = []
            for result in results:
                findings = self._categorize_findings(result.result_content)
                if category in findings:
                    category_findings.append(f"**{result.template_name.replace('_', ' ').title()}:** {findings[category]}")
            
            synthesis_sections.extend(category_findings)
        
        # Comprehensive recommendations
        synthesis_sections.append(self._build_comprehensive_recommendations(results))
        
        return "\n".join(synthesis_sections)
    
    def _extract_key_findings(self, content: str, max_length: int = 1000) -> str:
        """
        Extract key findings from analysis content.
        """
        # Simplified extraction - look for key sections
        lines = content.split('\n')
        key_sections = []
        
        current_section = []
        capturing = False
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['key findings', 'summary', 'conclusions', 'recommendations']):
                capturing = True
                current_section = [line]
            elif line.startswith('#') and capturing:
                # New section, stop capturing
                break
            elif capturing:
                current_section.append(line)
                if len('\n'.join(current_section)) > max_length:
                    break
        
        if current_section:
            return '\n'.join(current_section)[:max_length]
        else:
            # Fallback - return first portion
            return content[:max_length]
    
    def _categorize_findings(self, content: str) -> Dict[str, str]:
        """
        Categorize findings from content into standard categories.
        """
        categories = {
            "risk_assessment": "",
            "financial_impact": "",
            "compliance_status": "",
            "quality_indicators": "",
            "recommendations": ""
        }
        
        content_lower = content.lower()
        lines = content.split('\n')
        
        current_category = None
        current_content = []
        
        for line in lines:
            line_lower = line.lower()
            
            # Detect category headers
            if 'risk' in line_lower:
                current_category = 'risk_assessment'
            elif 'financial' in line_lower or 'cost' in line_lower:
                current_category = 'financial_impact'
            elif 'compliance' in line_lower or 'regulatory' in line_lower:
                current_category = 'compliance_status'
            elif 'quality' in line_lower:
                current_category = 'quality_indicators'
            elif 'recommend' in line_lower:
                current_category = 'recommendations'
            elif line.startswith('#'):
                # Save previous category
                if current_category and current_content:
                    categories[current_category] = '\n'.join(current_content)
                current_content = []
                current_category = None
            
            if current_category:
                current_content.append(line)
        
        # Save final category
        if current_category and current_content:
            categories[current_category] = '\n'.join(current_content)
        
        return {k: v for k, v in categories.items() if v}
    
    def _build_consensus_recommendations(self, results: List[TemplateResult], consensus_analysis: Dict[str, Any]) -> str:
        """
        Build recommendations based on consensus findings.
        """
        recommendations_section = ["## Consensus-Based Recommendations"]
        
        # Aggregate recommendations by frequency
        rec_frequency = {}
        for result in results:
            for rec in result.recommendations:
                rec_key = rec.lower()[:100]
                if rec_key not in rec_frequency:
                    rec_frequency[rec_key] = []
                rec_frequency[rec_key].append((result.template_name, rec))
        
        # Sort by frequency and include only consensus recommendations
        consensus_recs = [(key, instances) for key, instances in rec_frequency.items() 
                         if len(instances) >= max(2, len(results) // 2)]
        consensus_recs.sort(key=lambda x: len(x[1]), reverse=True)
        
        for i, (rec_key, instances) in enumerate(consensus_recs[:10]):  # Top 10
            templates = [instance[0] for instance in instances]
            recommendation = instances[0][1]  # Use first instance as template
            
            recommendations_section.append(f"""
### Recommendation {i+1}
- **Priority:** High (Supported by {len(instances)} frameworks)
- **Supporting Analysis:** {', '.join(templates)}
- **Recommendation:** {recommendation}
""")
        
        return '\n'.join(recommendations_section)
    
    def _build_weighted_recommendations(self, sorted_results: List[TemplateResult], weights: Dict[str, float]) -> str:
        """
        Build recommendations using priority weighting.
        """
        recommendations_section = ["## Priority-Weighted Recommendations"]
        
        # Weight recommendations by template priority
        weighted_recs = []
        for result in sorted_results:
            weight = weights.get(result.template_name, 0.5)
            for rec in result.recommendations:
                weighted_recs.append((rec, weight, result.template_name))
        
        # Sort by weight
        weighted_recs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (rec, weight, template) in enumerate(weighted_recs[:10]):  # Top 10
            recommendations_section.append(f"""
### Recommendation {i+1}
- **Priority Weight:** {weight:.2f}
- **Source Framework:** {template.replace('_', ' ').title()}
- **Recommendation:** {rec}
""")
        
        return '\n'.join(recommendations_section)
    
    def _build_conflict_resolved_recommendations(self, consensus_analysis: Dict[str, Any]) -> str:
        """
        Build recommendations from resolved conflicts.
        """
        recommendations_section = ["## Final Recommendations (Post-Conflict Resolution)"]
        
        for i, resolution in enumerate(consensus_analysis['resolved_conflicts']):
            if resolution['resolved_recommendation']:
                recommendations_section.append(f"""
### Recommendation {i+1}
- **Resolution Confidence:** {resolution['confidence']:.2f}
- **Recommendation:** {resolution['resolved_recommendation']}
- **Rationale:** {resolution['rationale']}
""")
        
        # Add consensus recommendations
        recommendations_section.append("### Additional Consensus Recommendations")
        for area in consensus_analysis['consensus_areas']:
            recommendations_section.append(f"- Continue focus on {area['theme'].replace('_', ' ').lower()} (consensus strength: {area['consensus_strength']:.1%})")
        
        return '\n'.join(recommendations_section)
    
    def _build_comprehensive_recommendations(self, results: List[TemplateResult]) -> str:
        """
        Build comprehensive recommendations from all frameworks.
        """
        recommendations_section = ["## Comprehensive Recommendations"]
        
        # Categorize all recommendations
        rec_categories = {}
        for result in results:
            for rec in result.recommendations:
                category = self._categorize_recommendation(rec)
                if category not in rec_categories:
                    rec_categories[category] = []
                rec_categories[category].append((rec, result.template_name))
        
        for category, recs in rec_categories.items():
            recommendations_section.append(f"### {category.replace('_', ' ').title()} Recommendations")
            for rec, template in recs:
                recommendations_section.append(f"- **{template.replace('_', ' ').title()}:** {rec}")
        
        return '\n'.join(recommendations_section)
    
    def _categorize_recommendation(self, recommendation: str) -> str:
        """
        Categorize a recommendation into standard types.
        """
        rec_lower = recommendation.lower()
        
        if any(word in rec_lower for word in ['immediate', 'urgent', 'asap']):
            return 'immediate_actions'
        elif any(word in rec_lower for word in ['risk', 'mitigate', 'address']):
            return 'risk_mitigation'
        elif any(word in rec_lower for word in ['improve', 'enhance', 'optimize']):
            return 'process_improvements'
        elif any(word in rec_lower for word in ['monitor', 'track', 'review']):
            return 'ongoing_monitoring'
        elif any(word in rec_lower for word in ['investigate', 'analyze', 'research']):
            return 'further_investigation'
        else:
            return 'general_recommendations'
    
    async def _generate_final_recommendations(
        self, 
        synthesized_content: str, 
        results: List[TemplateResult], 
        consensus_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Generate final actionable recommendations from synthesis.
        """
        final_recommendations = []
        
        # Extract top recommendations from synthesis
        if consensus_analysis['consensus_areas']:
            final_recommendations.append(
                f"Focus on {len(consensus_analysis['consensus_areas'])} consensus areas identified across multiple analytical frameworks"
            )
        
        if consensus_analysis['resolved_conflicts']:
            final_recommendations.append(
                f"Implement {len(consensus_analysis['resolved_conflicts'])} conflict-resolved recommendations with high confidence"
            )
        
        # Add highest priority recommendations from each template
        for result in results:
            if result.recommendations:
                final_recommendations.append(
                    f"{result.template_name.replace('_', ' ').title()}: {result.recommendations[0]}"
                )
        
        # Add quality-based recommendations
        avg_quality = sum(r.quality_metrics.get('accuracy', 0.8) for r in results) / len(results)
        if avg_quality < 0.85:
            final_recommendations.append("Recommend additional quality validation due to moderate confidence levels")
        
        return final_recommendations[:10]  # Limit to top 10
    
    async def _assess_synthesis_quality(
        self, 
        synthesized_content: str, 
        results: List[TemplateResult], 
        recommendations: List[str]
    ) -> Dict[str, float]:
        """
        Assess the quality of the synthesis output.
        """
        quality_metrics = {}
        
        # Completeness assessment
        content_length = len(synthesized_content)
        expected_min_length = len(results) * 1000  # Expect ~1000 chars per template
        completeness = min(1.0, content_length / expected_min_length)
        quality_metrics['completeness'] = completeness
        
        # Consistency assessment (simplified)
        consistency = 0.85  # Default - would need NLP for proper assessment
        quality_metrics['consistency'] = consistency
        
        # Actionability assessment
        actionable_recs = len([r for r in recommendations if any(word in r.lower() for word in ['should', 'must', 'recommend', 'implement'])])
        actionability = min(1.0, actionable_recs / max(1, len(recommendations)))
        quality_metrics['actionability'] = actionability
        
        # Overall synthesis quality
        overall_quality = (completeness * 0.3 + consistency * 0.4 + actionability * 0.3)
        quality_metrics['overall_quality'] = overall_quality
        
        return quality_metrics
    
    def _calculate_confidence_score(self, results: List[TemplateResult], consensus_analysis: Dict[str, Any]) -> float:
        """
        Calculate overall confidence in the synthesis.
        """
        if not results:
            return 0.0
        
        # Average template confidence
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        
        # Consensus strength bonus
        consensus_bonus = len(consensus_analysis['consensus_areas']) / max(1, len(results)) * 0.1
        
        # Conflict penalty
        conflict_penalty = len(consensus_analysis['conflicts']) / max(1, len(results)) * 0.05
        
        # Quality factor
        avg_quality = sum(r.quality_metrics.get('accuracy', 0.8) for r in results) / len(results)
        
        final_confidence = min(1.0, avg_confidence + consensus_bonus - conflict_penalty) * avg_quality
        
        return final_confidence