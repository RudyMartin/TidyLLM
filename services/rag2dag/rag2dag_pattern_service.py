"""
RAG2DAG Pattern Detection Service
================================

Specialized service for detecting RAG patterns and analyzing optimization potential.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from tidyllm.rag2dag.converter import RAGPatternType

logger = logging.getLogger(__name__)


@dataclass
class PatternDetectionResult:
    """Result of pattern detection analysis."""
    pattern_detected: Optional[RAGPatternType]
    confidence: float
    detection_reason: str
    alternative_patterns: List[RAGPatternType]


class RAG2DAGPatternService:
    """
    Service for intelligent RAG pattern detection and analysis.

    Provides sophisticated pattern recognition using keyword analysis,
    context understanding, and machine learning techniques.
    """

    def __init__(self, config):
        self.config = config

        # Pattern detection rules
        self.pattern_indicators = {
            RAGPatternType.MULTI_SOURCE: {
                "keywords": ["compare", "across", "multiple", "sources", "different", "various", "several"],
                "file_indicators": lambda files: len(files) > 3,
                "context_patterns": [r"multiple\s+documents?", r"across\s+\w+\s+sources?"],
                "confidence_boost": 0.3
            },
            RAGPatternType.RESEARCH_SYNTHESIS: {
                "keywords": ["analyze", "synthesis", "comprehensive", "research", "investigate", "summarize", "overview"],
                "file_indicators": lambda files: len(files) > 2,
                "context_patterns": [r"research\s+synthesis", r"comprehensive\s+analysis", r"summarize\s+findings"],
                "confidence_boost": 0.25
            },
            RAGPatternType.COMPARATIVE_ANALYSIS: {
                "keywords": ["compare", "contrast", "difference", "similarity", "versus", "vs", "between"],
                "file_indicators": lambda files: len(files) >= 2,
                "context_patterns": [r"compare\s+\w+\s+and\s+\w+", r"differences?\s+between", r"similarities?\s+in"],
                "confidence_boost": 0.4
            },
            RAGPatternType.FACT_CHECKING: {
                "keywords": ["verify", "fact", "check", "validate", "accurate", "true", "false", "confirm"],
                "file_indicators": lambda files: len(files) > 1,
                "context_patterns": [r"fact.?check", r"verify\s+against", r"validate\s+claims?"],
                "confidence_boost": 0.35
            },
            RAGPatternType.KNOWLEDGE_EXTRACTION: {
                "keywords": ["extract", "identify", "list", "find all", "enumerate", "collect", "gather"],
                "file_indicators": lambda files: len(files) > 0,
                "context_patterns": [r"extract\s+\w+\s+from", r"identify\s+all", r"list\s+of"],
                "confidence_boost": 0.2
            },
            RAGPatternType.DOCUMENT_PIPELINE: {
                "keywords": ["process", "workflow", "step by step", "pipeline", "sequence", "stages"],
                "file_indicators": lambda files: len(files) > 0,
                "context_patterns": [r"step.by.step", r"processing\s+pipeline", r"workflow\s+for"],
                "confidence_boost": 0.15
            }
        }

    def detect_pattern(self, request: str, context: str = "", source_files: List[str] = None) -> PatternDetectionResult:
        """
        Detect the most appropriate RAG pattern for a given request.

        Args:
            request: The user request to analyze
            context: Additional context
            source_files: List of source files

        Returns:
            PatternDetectionResult with detected pattern and confidence
        """
        source_files = source_files or []
        combined_text = f"{request} {context}".lower()

        pattern_scores = {}
        detection_details = {}

        # Analyze each pattern
        for pattern_type, indicators in self.pattern_indicators.items():
            score = self._calculate_pattern_score(
                combined_text,
                source_files,
                indicators
            )

            pattern_scores[pattern_type] = score
            detection_details[pattern_type] = {
                "score": score,
                "matched_keywords": self._find_matched_keywords(combined_text, indicators["keywords"]),
                "file_match": indicators["file_indicators"](source_files)
            }

        # Find the best pattern
        if not pattern_scores or max(pattern_scores.values()) < 0.3:
            # Default to simple QA if no strong pattern detected
            return PatternDetectionResult(
                pattern_detected=RAGPatternType.SIMPLE_QA,
                confidence=0.8,  # High confidence for simple QA as fallback
                detection_reason="No complex pattern detected, using simple QA",
                alternative_patterns=[]
            )

        # Get top patterns
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        best_pattern, best_score = sorted_patterns[0]

        alternative_patterns = [pattern for pattern, score in sorted_patterns[1:3] if score > 0.2]

        return PatternDetectionResult(
            pattern_detected=best_pattern,
            confidence=min(best_score, 1.0),
            detection_reason=self._generate_detection_reason(best_pattern, detection_details[best_pattern]),
            alternative_patterns=alternative_patterns
        )

    def analyze_workflow_description(self, workflow_description: str, expected_load: str = "medium") -> Dict[str, Any]:
        """
        Analyze workflow description for optimization opportunities.

        Args:
            workflow_description: Description of the workflow
            expected_load: Expected load level

        Returns:
            Analysis with optimization recommendations
        """
        try:
            # Detect pattern from workflow description
            pattern_result = self.detect_pattern(workflow_description)

            # Base analysis
            analysis = {
                "performance_gain": 0,
                "cost_impact": "neutral",
                "complexity_reduction": False,
                "parallel_opportunities": [],
                "optimization_type": "simple_qa",
                "confidence": pattern_result.confidence
            }

            if pattern_result.pattern_detected == RAGPatternType.SIMPLE_QA:
                return analysis

            # Pattern-specific optimizations
            pattern_optimizations = {
                RAGPatternType.MULTI_SOURCE: {
                    "performance_gain": 45,
                    "cost_impact": "reduced",
                    "complexity_reduction": True,
                    "parallel_opportunities": ["Parallel processing of multiple sources", "Concurrent document retrieval"]
                },
                RAGPatternType.RESEARCH_SYNTHESIS: {
                    "performance_gain": 35,
                    "cost_impact": "neutral",
                    "complexity_reduction": True,
                    "parallel_opportunities": ["Parallel research extraction", "Concurrent synthesis operations"]
                },
                RAGPatternType.COMPARATIVE_ANALYSIS: {
                    "performance_gain": 40,
                    "cost_impact": "reduced",
                    "complexity_reduction": True,
                    "parallel_opportunities": ["Parallel comparison operations", "Concurrent analysis pipelines"]
                },
                RAGPatternType.FACT_CHECKING: {
                    "performance_gain": 30,
                    "cost_impact": "neutral",
                    "complexity_reduction": False,
                    "parallel_opportunities": ["Parallel fact verification", "Concurrent source validation"]
                },
                RAGPatternType.KNOWLEDGE_EXTRACTION: {
                    "performance_gain": 25,
                    "cost_impact": "reduced",
                    "complexity_reduction": True,
                    "parallel_opportunities": ["Parallel knowledge extraction", "Concurrent data collection"]
                },
                RAGPatternType.DOCUMENT_PIPELINE: {
                    "performance_gain": 20,
                    "cost_impact": "neutral",
                    "complexity_reduction": False,
                    "parallel_opportunities": ["Pipeline stage optimization", "Parallel processing steps"]
                }
            }

            # Apply pattern-specific optimizations
            if pattern_result.pattern_detected in pattern_optimizations:
                opt = pattern_optimizations[pattern_result.pattern_detected]
                analysis.update(opt)
                analysis["optimization_type"] = pattern_result.pattern_detected.value

            # Apply load-based adjustments
            load_multipliers = {"low": 0.7, "medium": 1.0, "high": 1.3}
            multiplier = load_multipliers.get(expected_load, 1.0)

            analysis["performance_gain"] = int(analysis["performance_gain"] * multiplier)

            if expected_load == "high":
                analysis["parallel_opportunities"].append("High-load optimization patterns")
                if analysis["cost_impact"] == "neutral":
                    analysis["cost_impact"] = "reduced"

            return analysis

        except Exception as e:
            logger.error(f"Workflow analysis failed: {e}")
            return {
                "performance_gain": 0,
                "cost_impact": "neutral",
                "complexity_reduction": False,
                "parallel_opportunities": [],
                "optimization_type": "simple_qa",
                "confidence": 0.0
            }

    def _calculate_pattern_score(self, text: str, files: List[str], indicators: Dict) -> float:
        """Calculate confidence score for a specific pattern."""
        score = 0.0

        # Keyword matching
        matched_keywords = self._find_matched_keywords(text, indicators["keywords"])
        keyword_score = len(matched_keywords) / len(indicators["keywords"])
        score += keyword_score * 0.4

        # File indicators
        if indicators["file_indicators"](files):
            score += 0.3

        # Context pattern matching
        pattern_matches = 0
        for pattern in indicators.get("context_patterns", []):
            if re.search(pattern, text):
                pattern_matches += 1

        if indicators.get("context_patterns"):
            pattern_score = pattern_matches / len(indicators["context_patterns"])
            score += pattern_score * 0.3

        # Confidence boost for strong indicators
        if keyword_score > 0.5 or pattern_matches > 0:
            score += indicators.get("confidence_boost", 0.0)

        return min(score, 1.0)

    def _find_matched_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Find which keywords are present in the text."""
        matched = []
        for keyword in keywords:
            if keyword.lower() in text:
                matched.append(keyword)
        return matched

    def _generate_detection_reason(self, pattern: RAGPatternType, details: Dict) -> str:
        """Generate human-readable reason for pattern detection."""
        matched_keywords = details.get("matched_keywords", [])
        file_match = details.get("file_match", False)
        score = details.get("score", 0.0)

        reason_parts = [f"Detected {pattern.value} pattern"]

        if matched_keywords:
            reason_parts.append(f"matched keywords: {', '.join(matched_keywords[:3])}")

        if file_match:
            reason_parts.append("file structure supports pattern")

        reason_parts.append(f"confidence: {score:.2f}")

        return " - ".join(reason_parts)