"""
MCP Quality Checker

Quality assessment for MCP results.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime


class QualityChecker:
    """Quality assessment for MCP results"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.quality_metrics: Dict[str, callable] = {}
        self.quality_history: List[Dict[str, Any]] = []

    def register_quality_metric(self, metric_name: str, metric_func: callable):
        """Register a quality metric"""
        self.quality_metrics[metric_name] = metric_func
        self.logger.info(f"Registered quality metric: {metric_name}")

    def assess_quality(self, result: Dict[str, Any], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Assess quality of result using specified metrics"""
        if not metrics:
            metrics = list(self.quality_metrics.keys())

        quality_assessment = {
            "overall_score": 0.0,
            "metrics": {},
            "timestamp": datetime.now().isoformat()
        }

        total_score = 0.0
        metric_count = 0

        for metric_name in metrics:
            metric_func = self.quality_metrics.get(metric_name)
            if metric_func:
                try:
                    score = metric_func(result)
                    quality_assessment["metrics"][metric_name] = score
                    total_score += score
                    metric_count += 1
                except Exception as e:
                    self.logger.error(f"Error calculating metric '{metric_name}': {e}")
                    quality_assessment["metrics"][metric_name] = 0.0

        if metric_count > 0:
            quality_assessment["overall_score"] = total_score / metric_count

        self.quality_history.append(quality_assessment)
        self.logger.debug(f"Assessed quality with {len(metrics)} metrics, overall score: {quality_assessment['overall_score']}")

        return quality_assessment

    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get quality assessment statistics"""
        if not self.quality_history:
            return {"total_assessments": 0}

        total_assessments = len(self.quality_history)
        avg_overall_score = sum(a["overall_score"] for a in self.quality_history) / total_assessments

        metric_scores = {}
        for assessment in self.quality_history:
            for metric_name, score in assessment["metrics"].items():
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                metric_scores[metric_name].append(score)

        avg_metric_scores = {}
        for metric_name, scores in metric_scores.items():
            avg_metric_scores[metric_name] = sum(scores) / len(scores)

        return {
            "total_assessments": total_assessments,
            "average_overall_score": avg_overall_score,
            "average_metric_scores": avg_metric_scores,
            "registered_metrics": list(self.quality_metrics.keys())
        }
