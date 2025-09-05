"""
Writer Worker - Report Generation and Writing

Specialized worker for report generation, content creation, and writing tasks
including document creation, summarization, and content formatting.
"""

from typing import List, Dict, Any, Optional
from ..worker import Worker
import json
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class WriterWorker(Worker):
    """Specialized worker for report generation and writing"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__("writer", "report generation and writing", model_config)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def _create_execution_prompt(self, task: Dict[str, Any], context: MCPContext) -> str:
        """Create specialized execution prompt for writing tasks"""
        
        task_description = task.get("task", "")
        input_data = task.get("input_data", {})
        constraints = task.get("constraints", {})
        
        # Extract writing parameters
        content_data = input_data.get("content_data", {})
        output_format = input_data.get("output_format", "text")
        style_guide = input_data.get("style_guide", {})
        
        prompt = f"""
        As a professional writer and content creator, execute this writing task:
        
        TASK: {task_description}
        
        OUTPUT FORMAT: {output_format}
        
        CONTENT DATA: {json.dumps(content_data, indent=2)}
        
        STYLE GUIDE: {json.dumps(style_guide, indent=2)}
        
        WRITING CONSTRAINTS: {json.dumps(constraints, indent=2)}
        
        WRITING INSTRUCTIONS:
        1. Create high-quality, well-structured content
        2. Follow the specified output format and style guide
        3. Organize information logically and coherently
        4. Use appropriate tone and language for the target audience
        5. Ensure clarity, accuracy, and professionalism
        6. Include relevant details and supporting information
        7. Maintain consistency throughout the content
        
        Provide a comprehensive, well-written response that meets all requirements.
        """
        
        return prompt
    
    def _format_result(self, result: str, task: Dict[str, Any], execution_duration: float) -> Dict[str, Any]:
        """Format writing result with specialized metadata"""
        
        # Extract writing-specific metadata
        writing_metadata = self._extract_writing_metadata(result, task)
        
        base_result = super()._format_result(result, task, execution_duration)
        
        # Add writing-specific fields
        base_result.update({
            "writing_metadata": {
                "output_format": writing_metadata.get("output_format", "text"),
                "word_count": writing_metadata.get("word_count", 0),
                "section_count": writing_metadata.get("section_count", 0),
                "writing_style": writing_metadata.get("writing_style", "professional"),
                "content_type": writing_metadata.get("content_type", "general"),
                "readability_score": writing_metadata.get("readability_score", 0.8)
            },
            "writing_parameters": {
                "output_format": task.get("input_data", {}).get("output_format", "text"),
                "style_guide_applied": bool(task.get("input_data", {}).get("style_guide", {})),
                "content_data_keys": list(task.get("input_data", {}).get("content_data", {}).keys())
            }
        })
        
        return base_result
    
    def _extract_writing_metadata(self, result: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract writing-specific metadata from result"""
        metadata = {
            "output_format": "text",
            "word_count": 0,
            "section_count": 0,
            "writing_style": "professional",
            "content_type": "general",
            "readability_score": 0.8
        }
        
        # Get output format from task
        output_format = task.get("input_data", {}).get("output_format", "text")
        metadata["output_format"] = output_format
        
        # Count words
        words = result.split()
        metadata["word_count"] = len(words)
        
        # Count sections (headers, paragraphs, etc.)
        section_count = self._count_sections(result)
        metadata["section_count"] = section_count
        
        # Determine writing style
        writing_style = self._determine_writing_style(result)
        metadata["writing_style"] = writing_style
        
        # Determine content type
        content_type = self._determine_content_type(result, output_format)
        metadata["content_type"] = content_type
        
        # Calculate readability score
        readability_score = self._calculate_readability_score(result)
        metadata["readability_score"] = readability_score
        
        return metadata
    
    def _count_sections(self, result: str) -> int:
        """Count sections in the written content"""
        try:
            # Count headers, numbered sections, and major breaks
            header_patterns = [
                r'^#+\s',  # Markdown headers
                r'^\d+\.\s',  # Numbered sections
                r'^[A-Z][A-Z\s]+$',  # ALL CAPS headers
                r'^\*\*[^*]+\*\*$',  # Bold headers
            ]
            
            lines = result.split('\n')
            section_count = 0
            
            for line in lines:
                line = line.strip()
                if any(re.match(pattern, line) for pattern in header_patterns):
                    section_count += 1
                elif line and len(line) > 50:  # Long paragraphs as sections
                    section_count += 1
            
            return max(1, section_count)  # At least 1 section
            
        except Exception as e:
            self.logger.error(f"Error counting sections: {e}")
            return 1
    
    def _determine_writing_style(self, result: str) -> str:
        """Determine the writing style used"""
        result_lower = result.lower()
        
        # Check for formal indicators
        formal_indicators = ["therefore", "furthermore", "consequently", "moreover", "thus"]
        if any(indicator in result_lower for indicator in formal_indicators):
            return "formal"
        
        # Check for technical indicators
        technical_indicators = ["algorithm", "implementation", "architecture", "protocol", "framework"]
        if any(indicator in result_lower for indicator in technical_indicators):
            return "technical"
        
        # Check for casual indicators
        casual_indicators = ["you know", "basically", "actually", "like", "well"]
        if any(indicator in result_lower for indicator in casual_indicators):
            return "casual"
        
        # Check for academic indicators
        academic_indicators = ["research", "study", "analysis", "findings", "conclusion"]
        if any(indicator in result_lower for indicator in academic_indicators):
            return "academic"
        
        return "professional"
    
    def _determine_content_type(self, result: str, output_format: str) -> str:
        """Determine the type of content written"""
        result_lower = result.lower()
        
        if output_format == "report":
            return "report"
        elif output_format == "summary":
            return "summary"
        elif output_format == "executive_brief":
            return "executive_brief"
        elif "report" in result_lower or "analysis" in result_lower:
            return "report"
        elif "summary" in result_lower or "overview" in result_lower:
            return "summary"
        elif "recommendation" in result_lower or "suggestion" in result_lower:
            return "recommendation"
        elif "instruction" in result_lower or "guide" in result_lower:
            return "instruction"
        else:
            return "general"
    
    def _calculate_readability_score(self, result: str) -> float:
        """Calculate a simple readability score"""
        try:
            sentences = re.split(r'[.!?]+', result)
            words = result.split()
            
            if not sentences or not words:
                return 0.8
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Simple readability formula (lower is more readable)
            readability = (avg_sentence_length * 0.39) + (avg_word_length * 11.8) - 15.59
            
            # Convert to 0-1 scale (higher is more readable)
            if readability <= 0:
                return 1.0
            elif readability >= 100:
                return 0.0
            else:
                return max(0.0, min(1.0, 1.0 - (readability / 100)))
                
        except Exception as e:
            self.logger.error(f"Error calculating readability: {e}")
            return 0.8
    
    def _get_specialization_capabilities(self) -> list:
        """Get writing-specific capabilities"""
        return [
            "Report generation and writing",
            "Content creation and composition",
            "Document writing and formatting",
            "Executive summaries and briefs",
            "Technical documentation",
            "Content summarization",
            "Copywriting and marketing content",
            "Academic writing",
            "Business writing",
            "Creative writing",
            "Content editing and revision",
            "Style guide compliance",
            "Multi-format content creation",
            "Content optimization",
            "Writing quality assessment"
        ]
    
    def generate_report(self, content_data: Dict[str, Any], style_guide: Dict[str, Any]) -> str:
        """Generate comprehensive report"""
        try:
            report_prompt = f"""
            Generate a comprehensive report based on the following data:
            
            CONTENT DATA: {json.dumps(content_data, indent=2)}
            
            STYLE GUIDE: {json.dumps(style_guide, indent=2)}
            
            REPORT STRUCTURE:
            1. Executive Summary
            2. Introduction and Background
            3. Key Findings and Analysis
            4. Detailed Analysis
            5. Recommendations
            6. Conclusion
            
            Ensure the report is:
            - Well-structured and professional
            - Comprehensive and detailed
            - Clear and easy to understand
            - Actionable and insightful
            """
            
            return self.llm_manager.generate_response(report_prompt)
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return f"Report generation failed: {e}"
    
    def generate_summary(self, content_data: Dict[str, Any]) -> str:
        """Generate concise summary"""
        try:
            summary_prompt = f"""
            Create a concise summary of the following data:
            
            CONTENT DATA: {json.dumps(content_data, indent=2)}
            
            SUMMARY REQUIREMENTS:
            - Focus on the most important points
            - Provide key insights and findings
            - Keep it concise and clear
            - Highlight critical information
            - Maintain professional tone
            """
            
            return self.llm_manager.generate_response(summary_prompt)
            
        except Exception as e:
            self.logger.error(f"Summary generation error: {e}")
            return f"Summary generation failed: {e}"
    
    def generate_executive_brief(self, content_data: Dict[str, Any]) -> str:
        """Generate executive brief"""
        try:
            brief_prompt = f"""
            Create an executive brief based on the following data:
            
            CONTENT DATA: {json.dumps(content_data, indent=2)}
            
            EXECUTIVE BRIEF REQUIREMENTS:
            - High-level overview for executives
            - Key insights and implications
            - Strategic recommendations
            - Risk assessment and opportunities
            - Clear action items
            - Professional and concise format
            """
            
            return self.llm_manager.generate_response(brief_prompt)
            
        except Exception as e:
            self.logger.error(f"Executive brief generation error: {e}")
            return f"Executive brief generation failed: {e}"
    
    def format_content(self, content: str, format_type: str, style_guide: Dict[str, Any]) -> str:
        """Format content according to specified format and style"""
        try:
            format_prompt = f"""
            Format the following content according to the specified format and style:
            
            CONTENT: {content}
            
            FORMAT TYPE: {format_type}
            
            STYLE GUIDE: {json.dumps(style_guide, indent=2)}
            
            FORMATTING INSTRUCTIONS:
            1. Apply the specified format type
            2. Follow the style guide requirements
            3. Maintain content integrity
            4. Ensure proper structure and organization
            5. Apply consistent formatting throughout
            """
            
            return self.llm_manager.generate_response(format_prompt)
            
        except Exception as e:
            self.logger.error(f"Content formatting error: {e}")
            return f"Content formatting failed: {e}"
    
    def get_writing_analytics(self) -> Dict[str, Any]:
        """Get writing analytics and performance metrics"""
        metrics = self.get_performance_metrics()
        
        # Add writing-specific analytics
        writing_analytics = {
            "total_writing_tasks": metrics["total_tasks"],
            "average_writing_duration": metrics["average_duration"],
            "writing_success_rate": metrics["success_rate"],
            "output_formats": {
                "report": 0,
                "summary": 0,
                "executive_brief": 0,
                "text": 0
            },
            "writing_styles": {
                "professional": 0,
                "formal": 0,
                "technical": 0,
                "casual": 0,
                "academic": 0
            },
            "content_types": {
                "report": 0,
                "summary": 0,
                "recommendation": 0,
                "instruction": 0,
                "general": 0
            },
            "average_word_count": 0,
            "average_readability_score": 0.0
        }
        
        # Analyze task history for writing patterns
        total_words = 0
        total_readability = 0.0
        valid_entries = 0
        
        for task_record in self.task_history:
            task = task_record["task"]
            result = task_record["result"]
            
            # Count output formats
            output_format = result.get("writing_metadata", {}).get("output_format", "text")
            writing_analytics["output_formats"][output_format] = writing_analytics["output_formats"].get(output_format, 0) + 1
            
            # Count writing styles
            writing_style = result.get("writing_metadata", {}).get("writing_style", "professional")
            writing_analytics["writing_styles"][writing_style] = writing_analytics["writing_styles"].get(writing_style, 0) + 1
            
            # Count content types
            content_type = result.get("writing_metadata", {}).get("content_type", "general")
            writing_analytics["content_types"][content_type] = writing_analytics["content_types"].get(content_type, 0) + 1
            
            # Accumulate word count and readability
            word_count = result.get("writing_metadata", {}).get("word_count", 0)
            readability_score = result.get("writing_metadata", {}).get("readability_score", 0.8)
            
            if word_count > 0:
                total_words += word_count
                total_readability += readability_score
                valid_entries += 1
        
        # Calculate averages
        if valid_entries > 0:
            writing_analytics["average_word_count"] = total_words / valid_entries
            writing_analytics["average_readability_score"] = total_readability / valid_entries
        
        return writing_analytics
