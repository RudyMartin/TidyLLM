"""
Analysis Worker - Content Analysis and Reasoning

Specialized worker for content analysis, reasoning, and data processing
operations including sentiment analysis, key point extraction, and compliance checking.
"""

from typing import List, Dict, Any, Optional
from ..worker import Worker
import json
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class AnalysisWorker(Worker):
    """Specialized worker for content analysis and reasoning"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__("analyzer", "content analysis and reasoning", model_config)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def _create_execution_prompt(self, task: Dict[str, Any], context: MCPContext) -> str:
        """Create specialized execution prompt for analysis tasks"""
        
        task_description = task.get("task", "")
        input_data = task.get("input_data", {})
        constraints = task.get("constraints", {})
        
        # Extract analysis parameters
        content = input_data.get("content", "")
        analysis_type = input_data.get("analysis_type", "general")
        
        prompt = f"""
        As a content analysis specialist, execute this analysis task:
        
        TASK: {task_description}
        
        ANALYSIS TYPE: {analysis_type}
        
        CONTENT TO ANALYZE: {content[:2000]}...
        
        ANALYSIS CONSTRAINTS: {json.dumps(constraints, indent=2)}
        
        ANALYSIS INSTRUCTIONS:
        1. Perform {analysis_type} analysis on the provided content
        2. Identify key insights, patterns, and findings
        3. Provide structured analysis results
        4. Include confidence levels and reasoning
        5. Highlight important observations and implications
        6. Consider context and constraints in the analysis
        
        Provide a comprehensive, well-structured analysis that addresses the specific analysis type.
        """
        
        return prompt
    
    def _format_result(self, result: str, task: Dict[str, Any], execution_duration: float) -> Dict[str, Any]:
        """Format analysis result with specialized metadata"""
        
        # Extract analysis-specific metadata
        analysis_metadata = self._extract_analysis_metadata(result, task)
        
        base_result = super()._format_result(result, task, execution_duration)
        
        # Add analysis-specific fields
        base_result.update({
            "analysis_metadata": {
                "analysis_type": analysis_metadata.get("analysis_type", "general"),
                "sentiment_score": analysis_metadata.get("sentiment_score", 0.0),
                "key_points_count": analysis_metadata.get("key_points_count", 0),
                "compliance_status": analysis_metadata.get("compliance_status", "unknown"),
                "confidence_level": analysis_metadata.get("confidence_level", "medium"),
                "analysis_scope": analysis_metadata.get("analysis_scope", "general")
            },
            "analysis_parameters": {
                "content_length": len(task.get("input_data", {}).get("content", "")),
                "analysis_type": task.get("input_data", {}).get("analysis_type", "general"),
                "constraints_count": len(task.get("constraints", {}))
            }
        })
        
        return base_result
    
    def _extract_analysis_metadata(self, result: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract analysis-specific metadata from result"""
        metadata = {
            "analysis_type": "general",
            "sentiment_score": 0.0,
            "key_points_count": 0,
            "compliance_status": "unknown",
            "confidence_level": "medium",
            "analysis_scope": "general"
        }
        
        # Get analysis type from task
        analysis_type = task.get("input_data", {}).get("analysis_type", "general")
        metadata["analysis_type"] = analysis_type
        
        result_lower = result.lower()
        
        # Extract sentiment score if sentiment analysis
        if analysis_type == "sentiment":
            sentiment_score = self._extract_sentiment_score(result)
            metadata["sentiment_score"] = sentiment_score
        
        # Count key points
        key_points = self._extract_key_points_count(result)
        metadata["key_points_count"] = key_points
        
        # Check compliance status
        if analysis_type == "compliance":
            compliance_status = self._extract_compliance_status(result)
            metadata["compliance_status"] = compliance_status
        
        # Determine confidence level
        confidence_level = self._extract_confidence_level(result)
        metadata["confidence_level"] = confidence_level
        
        # Determine analysis scope
        analysis_scope = self._extract_analysis_scope(result, analysis_type)
        metadata["analysis_scope"] = analysis_scope
        
        return metadata
    
    def _extract_sentiment_score(self, result: str) -> float:
        """Extract sentiment score from analysis result"""
        try:
            # Look for sentiment indicators
            result_lower = result.lower()
            
            if "positive" in result_lower and "negative" not in result_lower:
                return 0.8
            elif "negative" in result_lower and "positive" not in result_lower:
                return -0.8
            elif "neutral" in result_lower:
                return 0.0
            elif "mixed" in result_lower:
                return 0.1
            
            # Look for numerical scores
            score_pattern = r"sentiment.*?([-]?\d+\.?\d*)"
            match = re.search(score_pattern, result_lower)
            if match:
                score = float(match.group(1))
                return max(-1.0, min(1.0, score))  # Clamp between -1 and 1
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error extracting sentiment score: {e}")
            return 0.0
    
    def _extract_key_points_count(self, result: str) -> int:
        """Extract number of key points from analysis result"""
        try:
            # Count numbered points or bullet points
            numbered_points = len(re.findall(r'\d+\.', result))
            bullet_points = len(re.findall(r'[-*•]\s', result))
            
            return max(numbered_points, bullet_points)
            
        except Exception as e:
            self.logger.error(f"Error extracting key points count: {e}")
            return 0
    
    def _extract_compliance_status(self, result: str) -> str:
        """Extract compliance status from analysis result"""
        result_lower = result.lower()
        
        if "compliant" in result_lower and "non-compliant" not in result_lower:
            return "compliant"
        elif "non-compliant" in result_lower or "violation" in result_lower:
            return "non-compliant"
        elif "partial" in result_lower or "mixed" in result_lower:
            return "partial"
        else:
            return "unknown"
    
    def _extract_confidence_level(self, result: str) -> str:
        """Extract confidence level from analysis result"""
        result_lower = result.lower()
        
        if any(word in result_lower for word in ["high confidence", "very confident", "certain"]):
            return "high"
        elif any(word in result_lower for word in ["medium confidence", "moderate", "likely"]):
            return "medium"
        elif any(word in result_lower for word in ["low confidence", "uncertain", "unclear"]):
            return "low"
        else:
            return "medium"
    
    def _extract_analysis_scope(self, result: str, analysis_type: str) -> str:
        """Extract analysis scope from result"""
        result_lower = result.lower()
        
        if analysis_type == "sentiment":
            return "sentiment"
        elif analysis_type == "compliance":
            return "compliance"
        elif analysis_type == "key_points":
            return "content_extraction"
        elif "financial" in result_lower or "money" in result_lower:
            return "financial"
        elif "technical" in result_lower or "code" in result_lower:
            return "technical"
        elif "legal" in result_lower or "regulatory" in result_lower:
            return "legal"
        else:
            return "general"
    
    def _get_specialization_capabilities(self) -> list:
        """Get analysis-specific capabilities"""
        return [
            "Content analysis and reasoning",
            "Sentiment analysis",
            "Key point extraction",
            "Compliance checking",
            "Data processing and pattern recognition",
            "Text classification and categorization",
            "Fact verification and validation",
            "Trend analysis and insights",
            "Quality assessment",
            "Risk analysis and evaluation",
            "Comparative analysis",
            "Statistical analysis",
            "Contextual understanding",
            "Bias detection and analysis"
        ]
    
    def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Perform sentiment analysis on content"""
        try:
            sentiment_prompt = f"""
            Analyze the sentiment of the following content:
            
            Content: {content}
            
            Provide a JSON response with:
            - overall_sentiment: positive/negative/neutral
            - sentiment_score: -1.0 to 1.0 (negative to positive)
            - confidence: 0-1 score
            - key_phrases: list of sentiment-indicating phrases
            - reasoning: brief explanation of the sentiment analysis
            """
            
            response = self.llm_manager.generate_response(sentiment_prompt)
            return self._parse_json_response(response)
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.5,
                "key_phrases": [],
                "reasoning": f"Analysis failed: {e}"
            }
    
    def extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content"""
        try:
            key_points_prompt = f"""
            Extract the key points from the following content:
            
            Content: {content}
            
            Provide a numbered list of the most important points.
            Focus on:
            1. Main ideas and concepts
            2. Important facts and figures
            3. Key insights and conclusions
            4. Critical information
            
            Format as a numbered list.
            """
            
            response = self.llm_manager.generate_response(key_points_prompt)
            return self._parse_list_response(response)
            
        except Exception as e:
            self.logger.error(f"Key points extraction error: {e}")
            return [f"Extraction failed: {e}"]
    
    def check_compliance(self, content: str, rules: List[str]) -> Dict[str, Any]:
        """Check content against compliance rules"""
        try:
            compliance_prompt = f"""
            Check the following content against these compliance rules:
            
            Content: {content}
            
            Rules: {json.dumps(rules, indent=2)}
            
            Provide a JSON response with:
            - compliant: true/false
            - violations: list of rule violations found
            - recommendations: list of improvement suggestions
            - risk_level: low/medium/high
            - compliance_score: 0-100 percentage
            """
            
            response = self.llm_manager.generate_response(compliance_prompt)
            return self._parse_json_response(response)
            
        except Exception as e:
            self.logger.error(f"Compliance check error: {e}")
            return {
                "compliant": False,
                "violations": [f"Analysis failed: {e}"],
                "recommendations": ["Review content manually"],
                "risk_level": "high",
                "compliance_score": 0
            }
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            return json.loads(json_str)
            
        except Exception as e:
            self.logger.error(f"JSON parsing error: {e}")
            return {"error": f"Failed to parse JSON: {e}"}
    
    def _parse_list_response(self, response: str) -> List[str]:
        """Parse list response from LLM"""
        try:
            # Extract numbered or bulleted items
            lines = response.split('\n')
            items = []
            
            for line in lines:
                line = line.strip()
                # Remove numbering or bullets
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^[-*•]\s*', '', line)
                
                if line and len(line) > 5:  # Minimum meaningful length
                    items.append(line)
            
            return items
            
        except Exception as e:
            self.logger.error(f"List parsing error: {e}")
            return [f"Parsing failed: {e}"]
    
    def get_analysis_analytics(self) -> Dict[str, Any]:
        """Get analysis analytics and performance metrics"""
        metrics = self.get_performance_metrics()
        
        # Add analysis-specific analytics
        analysis_analytics = {
            "total_analyses": metrics["total_tasks"],
            "average_analysis_duration": metrics["average_duration"],
            "analysis_success_rate": metrics["success_rate"],
            "analysis_types": {
                "sentiment": 0,
                "key_points": 0,
                "compliance": 0,
                "general": 0
            },
            "confidence_levels": {
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
        
        # Analyze task history for analysis patterns
        for task_record in self.task_history:
            task = task_record["task"]
            result = task_record["result"]
            
            # Count analysis types
            analysis_type = result.get("analysis_metadata", {}).get("analysis_type", "general")
            analysis_analytics["analysis_types"][analysis_type] = analysis_analytics["analysis_types"].get(analysis_type, 0) + 1
            
            # Count confidence levels
            confidence = result.get("analysis_metadata", {}).get("confidence_level", "medium")
            analysis_analytics["confidence_levels"][confidence] = analysis_analytics["confidence_levels"].get(confidence, 0) + 1
        
        return analysis_analytics
