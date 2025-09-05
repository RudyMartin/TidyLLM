#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-Enhanced Agents

This module provides LLM-enhanced agents for document processing,
all using the centralized LLM Gateway for all LLM interactions.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json

from .llm_gateway import LLMGateway, LLMResponse

logger = logging.getLogger(__name__)


class LLMEnhancedDocumentClassifier:
    """LLM-powered document classification and analysis"""
    
    def __init__(self, llm_gateway: LLMGateway):
        self.llm_gateway = llm_gateway
        self.classification_prompts = self._load_classification_prompts()
    
    def _load_classification_prompts(self) -> Dict[str, str]:
        """Load classification prompt templates"""
        return {
            'classification': """
Analyze this document and classify it into one of these categories based on the content and structure:

- validation_report: Compliance reviews, audits, validation studies, QA reports, model validation reports
- standards: Guidelines, frameworks, policies, best practices, procedures, regulatory standards
- independent_review: External reviews, third-party assessments, peer reviews, consultant reports
- whitepaper: Research papers, methodology documents, technical analysis, academic papers, professional analysis
- test_results: Performance data, benchmark results, testing outcomes, evaluation reports, model performance
- template: Reusable frameworks, checklists, forms, standardized documents, validation templates

Look for these indicators:
- validation_report: Contains validation results, compliance findings, audit outcomes
- standards: Contains guidelines, frameworks, best practices, regulatory requirements
- independent_review: External perspective, third-party analysis, consultant insights
- whitepaper: Professional analysis, research findings, industry insights, methodology discussion
- test_results: Performance metrics, testing data, evaluation results, benchmarks
- template: Structured format, reusable components, standardized procedures

Document content: {content}

CRITICAL: You must respond with ONLY valid JSON. No explanations, no markdown formatting, no additional text.

{{
    "classification": "category_name",
    "confidence": 85,
    "reasoning": "Brief explanation of classification",
    "key_themes": ["theme1", "theme2", "theme3"],
    "document_purpose": "Brief description of document purpose"
}}
""",
            'metadata_extraction': """
Extract key metadata from this document:

Document: {content}

CRITICAL: You must respond with ONLY valid JSON. No explanations, no markdown formatting, no additional text.

{{
    "title": "Document title",
    "authors": ["Author 1", "Author 2"],
    "date": "YYYY-MM-DD or approximate date",
    "version": "Version number if available",
    "organization": "Organization or institution",
    "document_type": "Specific document type",
    "key_topics": ["topic1", "topic2", "topic3"],
    "summary": "Brief 2-3 sentence summary"
}}
"""
        }
    
    def classify_with_llm(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document using LLM with enhanced content understanding"""
        content = document.get('content', '')[:3000]  # Limit content length
        
        # First, try rule-based classification as backup
        rule_based_classification = self._rule_based_classification(content, document.get('filename', ''))
        
        prompt = self.classification_prompts['classification'].format(content=content)
        
        try:
            response = self.llm_gateway.call_llm(
                agent_name="document_classifier",
                task_type="classification",
                prompt=prompt,
                model_preference="gpt-4"
            )
            
            # Parse JSON response
            classification_result = self._parse_json_response(response.content)
            
            # Add document info
            classification_result['document_filename'] = document.get('filename', 'Unknown')
            classification_result['document_size'] = len(content)
            
            # If LLM classification failed, use rule-based as fallback
            if 'error' in classification_result or classification_result.get('classification') == 'unknown':
                logger.info(f"LLM classification failed, using rule-based fallback for: {document.get('filename', 'Unknown')}")
                classification_result.update(rule_based_classification)
            
            logger.info(f"Classified document: {document.get('filename', 'Unknown')} -> {classification_result.get('classification', 'Unknown')}")
            
            return classification_result
            
        except Exception as e:
            logger.error(f"Failed to classify document: {e}")
            # Use rule-based classification as fallback
            fallback_result = rule_based_classification.copy()
            fallback_result['reasoning'] = f'Classification failed: {str(e)}'
            fallback_result['document_filename'] = document.get('filename', 'Unknown')
            return fallback_result
    
    def _rule_based_classification(self, content: str, filename: str) -> Dict[str, Any]:
        """Rule-based document classification as fallback"""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        # Classification rules based on content and filename patterns
        classification_rules = [
            {
                'category': 'whitepaper',
                'indicators': [
                    'whitepaper', 'research paper', 'analysis', 'study', 'professional',
                    'industry insights', 'best practices', 'methodology', 'framework',
                    'regulatory', 'compliance', 'risk management', 'model validation'
                ],
                'filename_patterns': ['wp', 'whitepaper', 'analysis', 'study']
            },
            {
                'category': 'validation_report',
                'indicators': [
                    'validation report', 'validation results', 'compliance review',
                    'audit report', 'validation findings', 'model validation',
                    'validation assessment', 'validation outcomes'
                ],
                'filename_patterns': ['validation', 'report', 'audit', 'compliance']
            },
            {
                'category': 'standards',
                'indicators': [
                    'standards', 'guidelines', 'framework', 'policy', 'procedures',
                    'best practices', 'regulatory requirements', 'compliance standards',
                    'governance', 'controls', 'requirements'
                ],
                'filename_patterns': ['standards', 'guidelines', 'framework', 'policy']
            },
            {
                'category': 'template',
                'indicators': [
                    'template', 'checklist', 'form', 'standardized', 'reusable',
                    'framework template', 'validation template', 'assessment template'
                ],
                'filename_patterns': ['template', 'checklist', 'form']
            },
            {
                'category': 'independent_review',
                'indicators': [
                    'independent review', 'third party', 'external review',
                    'consultant report', 'peer review', 'independent assessment',
                    'external audit', 'third-party analysis'
                ],
                'filename_patterns': ['independent', 'external', 'third-party', 'consultant']
            },
            {
                'category': 'test_results',
                'indicators': [
                    'test results', 'performance data', 'benchmark', 'testing outcomes',
                    'evaluation results', 'performance metrics', 'test data',
                    'benchmarking results', 'performance analysis'
                ],
                'filename_patterns': ['test', 'performance', 'benchmark', 'results']
            }
        ]
        
        # Score each category
        category_scores = {}
        for rule in classification_rules:
            score = 0
            category = rule['category']
            
            # Check content indicators
            for indicator in rule['indicators']:
                if indicator in content_lower:
                    score += 2  # Content matches are weighted higher
            
            # Check filename patterns
            for pattern in rule['filename_patterns']:
                if pattern in filename_lower:
                    score += 1
            
            category_scores[category] = score
        
        # Find the best classification
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            best_score = category_scores[best_category]
            
            if best_score > 0:
                confidence = min(85, best_score * 15)  # Scale score to confidence
                return {
                    'classification': best_category,
                    'confidence': confidence,
                    'reasoning': f'Rule-based classification based on {best_score} indicators found',
                    'key_themes': self._extract_key_themes(content),
                    'document_purpose': f'Document appears to be a {best_category.replace("_", " ")}'
                }
        
        # Default fallback
        return {
            'classification': 'unknown',
            'confidence': 0,
            'reasoning': 'No clear classification indicators found',
            'key_themes': self._extract_key_themes(content),
            'document_purpose': 'Unknown document type'
        }
    
    def _extract_key_themes(self, content: str) -> List[str]:
        """Extract key themes from document content"""
        content_lower = content.lower()
        themes = []
        
        # Common themes in model validation documents
        theme_keywords = {
            'model validation': ['model validation', 'validation', 'model risk'],
            'regulatory compliance': ['regulatory', 'compliance', 'sr11-7', 'regulation'],
            'risk management': ['risk management', 'model risk', 'risk assessment'],
            'governance': ['governance', 'controls', 'policies', 'procedures'],
            'performance testing': ['performance', 'testing', 'benchmark', 'metrics'],
            'documentation': ['documentation', 'documentation standards', 'reporting'],
            'independent review': ['independent', 'external', 'third party', 'audit'],
            'best practices': ['best practices', 'standards', 'guidelines', 'framework']
        }
        
        for theme, keywords in theme_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    themes.append(theme)
                    break
        
        return themes[:5]  # Return top 5 themes
    
    def extract_metadata_with_llm(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata using LLM"""
        content = document.get('content', '')[:2000]  # Limit content length
        
        prompt = self.classification_prompts['metadata_extraction'].format(content=content)
        
        try:
            response = self.llm_gateway.call_llm(
                agent_name="document_classifier",
                task_type="metadata_extraction",
                prompt=prompt,
                model_preference="gpt-3.5-turbo"
            )
            
            metadata = self._parse_json_response(response.content)
            metadata['document_filename'] = document.get('filename', 'Unknown')
            
            logger.info(f"Extracted metadata for: {document.get('filename', 'Unknown')}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return {
                'title': 'Unknown',
                'authors': [],
                'date': 'Unknown',
                'version': 'Unknown',
                'organization': 'Unknown',
                'document_type': 'Unknown',
                'key_topics': [],
                'summary': 'Metadata extraction failed',
                'document_filename': document.get('filename', 'Unknown')
            }
    
    def _parse_json_response(self, response_content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with enhanced error handling and multiple parsing strategies"""
        try:
            # Clean the response content
            content = response_content.strip()
            
            # Method 1: Try direct JSON parsing if content starts with {
            if content.startswith('{') and content.endswith('}'):
                return json.loads(content)
            
            # Method 2: Extract JSON from markdown code blocks
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # Method 3: Find first complete JSON object with improved brace counting
            start_idx = content.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                in_string = False
                escape_next = False
                
                for i, char in enumerate(content[start_idx:], start_idx):
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                
                if brace_count == 0:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
            
            # Method 4: Try to extract JSON from common LLM response patterns
            patterns = [
                r'(?:Here\'s the JSON:|Response:|JSON:|Output:)\s*(\{.*?\})',
                r'(\{[^{}]*"[^"]*"[^{}]*\})',  # Simple JSON with one level
                r'(\{[^{}]*"[^"]*"[^{}]*"[^"]*"[^{}]*\})'  # JSON with multiple fields
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except:
                        continue
            
            # Method 5: Try to extract key-value pairs manually
            logger.warning("Attempting manual parsing as fallback")
            return self._manual_json_parse(content)
                
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {'error': f'Failed to parse response: {str(e)}', 'raw_response': response_content[:500]}
    
    def _manual_json_parse(self, content: str) -> Dict[str, Any]:
        """Manual parsing as fallback for malformed JSON"""
        result = {}
        
        # Extract common fields with regex
        patterns = {
            'classification': r'"classification":\s*"([^"]*)"',
            'confidence': r'"confidence":\s*(\d+)',
            'reasoning': r'"reasoning":\s*"([^"]*)"',
            'title': r'"title":\s*"([^"]*)"',
            'organization': r'"organization":\s*"([^"]*)"'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1)
                if key == 'confidence':
                    result[key] = int(value)
                else:
                    result[key] = value
        
        # Extract arrays
        array_patterns = {
            'key_themes': r'"key_themes":\s*\[(.*?)\]',
            'authors': r'"authors":\s*\[(.*?)\]',
            'key_topics': r'"key_topics":\s*\[(.*?)\]'
        }
        
        for key, pattern in array_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                items_str = match.group(1)
                items = [item.strip(' "') for item in items_str.split(',') if item.strip()]
                result[key] = items
        
        return result if result else {'error': 'Could not parse any fields from response'}


class LLMEnhancedStandardsLibrarian:
    """LLM-powered standards extraction and application"""
    
    def __init__(self, llm_gateway: LLMGateway):
        self.llm_gateway = llm_gateway
        self.standards_prompts = self._load_standards_prompts()
    
    def _load_standards_prompts(self) -> Dict[str, str]:
        """Load standards extraction prompt templates"""
        return {
            'standards_extraction': """
Extract standards, best practices, and methodologies from this document:

Document: {content}

CRITICAL: You must respond with ONLY valid JSON. No explanations, no markdown formatting, no additional text.
{{
    "best_practices": [
        {{
            "practice": "Description of best practice",
            "context": "When/where to apply",
            "benefits": "Why this is beneficial"
        }}
    ],
    "methodologies": [
        {{
            "method": "Methodology name/description",
            "steps": ["step1", "step2", "step3"],
            "applicability": "When to use this methodology"
        }}
    ],
    "compliance_requirements": [
        {{
            "requirement": "Compliance requirement",
            "standard": "Related standard/framework",
            "implementation": "How to implement"
        }}
    ],
    "risk_factors": [
        {{
            "risk": "Risk description",
            "mitigation": "How to mitigate",
            "severity": "High/Medium/Low"
        }}
    ],
    "success_metrics": [
        {{
            "metric": "Metric name",
            "description": "What it measures",
            "target": "Target value or range"
        }}
    ],
    "related_standards": ["standard1", "standard2"],
    "implementation_guidelines": "Overall implementation guidance"
}}
""",
            'standards_application': """
Apply existing standards to this document:

Document: {content}

Existing Standards: {existing_standards}

Analyze the document against existing standards and provide:
{{
    "compliance_score": 85,
    "standards_applied": ["standard1", "standard2"],
    "gaps_identified": [
        {{
            "gap": "Description of gap",
            "standard": "Related standard",
            "recommendation": "How to address"
        }}
    ],
    "improvements_suggested": [
        {{
            "area": "Area for improvement",
            "suggestion": "Specific suggestion",
            "priority": "High/Medium/Low"
        }}
    ],
    "overall_assessment": "Summary of compliance and recommendations"
}}
"""
        }
    
    def extract_standards_with_llm(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standards and best practices using LLM"""
        content = document.get('content', '')[:4000]  # Limit content length
        
        prompt = self.standards_prompts['standards_extraction'].format(content=content)
        
        try:
            response = self.llm_gateway.call_llm(
                agent_name="standards_librarian",
                task_type="standards_extraction",
                prompt=prompt,
                model_preference="claude-3-sonnet"
            )
            
            standards = self._parse_json_response(response.content)
            standards['document_filename'] = document.get('filename', 'Unknown')
            standards['extraction_timestamp'] = self._get_timestamp()
            
            logger.info(f"Extracted standards from: {document.get('filename', 'Unknown')}")
            
            return standards
            
        except Exception as e:
            logger.error(f"Failed to extract standards: {e}")
            return {
                'best_practices': [],
                'methodologies': [],
                'compliance_requirements': [],
                'risk_factors': [],
                'success_metrics': [],
                'related_standards': [],
                'implementation_guidelines': 'Standards extraction failed',
                'document_filename': document.get('filename', 'Unknown'),
                'error': str(e)
            }
    
    def apply_standards_with_llm(self, document: Dict[str, Any], existing_standards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply existing standards to document using LLM"""
        content = document.get('content', '')[:3000]
        standards_text = json.dumps(existing_standards, indent=2)
        
        prompt = self.standards_prompts['standards_application'].format(
            content=content,
            existing_standards=standards_text
        )
        
        try:
            response = self.llm_gateway.call_llm(
                agent_name="standards_librarian",
                task_type="standards_application",
                prompt=prompt,
                model_preference="gpt-4"
            )
            
            application_result = self._parse_json_response(response.content)
            application_result['document_filename'] = document.get('filename', 'Unknown')
            application_result['application_timestamp'] = self._get_timestamp()
            
            logger.info(f"Applied standards to: {document.get('filename', 'Unknown')}")
            
            return application_result
            
        except Exception as e:
            logger.error(f"Failed to apply standards: {e}")
            return {
                'compliance_score': 0,
                'standards_applied': [],
                'gaps_identified': [],
                'improvements_suggested': [],
                'overall_assessment': f'Standards application failed: {str(e)}',
                'document_filename': document.get('filename', 'Unknown'),
                'error': str(e)
            }
    
    def _parse_json_response(self, response_content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {'error': f'Failed to parse response: {str(e)}'}
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


class LLMEnhancedDigestGenerator:
    """LLM-powered document summarization and digest generation"""
    
    def __init__(self, llm_gateway: LLMGateway):
        self.llm_gateway = llm_gateway
        self.digest_prompts = self._load_digest_prompts()
    
    def _load_digest_prompts(self) -> Dict[str, str]:
        """Load digest generation prompt templates"""
        return {
            'executive_summary': """
Create an executive summary for this document:

Document: {content}

Generate a concise executive summary (2-3 paragraphs) that includes:
- Key findings and insights
- Business impact and implications
- Main recommendations
- Critical observations

Target audience: Senior management and decision makers
""",
            'technical_summary': """
Create a technical summary for this document:

Document: {content}

Generate a technical summary that includes:
- Methodology overview
- Technical approach and implementation
- Key technical findings
- Performance metrics (if applicable)
- Technical recommendations

Target audience: Technical teams and implementation specialists
""",
            'key_findings': """
Extract key findings from this document:

Document: {content}

CRITICAL: You must respond with ONLY valid JSON. No explanations, no markdown formatting, no additional text.
{{
    "main_findings": [
        {{
            "finding": "Description of finding",
            "significance": "Why this is important",
            "evidence": "Supporting evidence or data"
        }}
    ],
    "action_items": [
        {{
            "action": "Specific action to take",
            "priority": "High/Medium/Low",
            "owner": "Who should take action",
            "timeline": "When to complete"
        }}
    ],
    "risks_identified": [
        {{
            "risk": "Risk description",
            "impact": "Potential impact",
            "mitigation": "How to address"
        }}
    ],
    "opportunities": [
        {{
            "opportunity": "Opportunity description",
            "benefit": "Potential benefit",
            "next_steps": "How to pursue"
        }}
    ]
}}
""",
            'comprehensive_digest': """
Create a comprehensive digest for this document:

Document: {content}

Generate a comprehensive digest that includes:

## Executive Summary
[2-3 paragraph executive summary]

## Key Findings
[Bullet points of main findings]

## Technical Overview
[Technical methodology and approach]

## Recommendations
[Specific actionable recommendations]

## Risk Assessment
[Identified risks and mitigation strategies]

## Business Impact
[Impact on business operations and strategy]

## Next Steps
[Immediate and long-term action items]

Target audience: Quant teams and review committees
"""
        }
    
    def generate_executive_summary(self, document: Dict[str, Any]) -> str:
        """Generate executive summary using LLM"""
        content = document.get('content', '')[:3000]
        
        prompt = self.digest_prompts['executive_summary'].format(content=content)
        
        try:
            response = self.llm_gateway.call_llm(
                agent_name="digest_generator",
                task_type="executive_summary",
                prompt=prompt,
                model_preference="gpt-4"
            )
            
            logger.info(f"Generated executive summary for: {document.get('filename', 'Unknown')}")
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            return f"Failed to generate executive summary: {str(e)}"
    
    def generate_technical_summary(self, document: Dict[str, Any]) -> str:
        """Generate technical summary using LLM"""
        content = document.get('content', '')[:3000]
        
        prompt = self.digest_prompts['technical_summary'].format(content=content)
        
        try:
            response = self.llm_gateway.call_llm(
                agent_name="digest_generator",
                task_type="technical_summary",
                prompt=prompt,
                model_preference="gpt-4"
            )
            
            logger.info(f"Generated technical summary for: {document.get('filename', 'Unknown')}")
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate technical summary: {e}")
            return f"Failed to generate technical summary: {str(e)}"
    
    def extract_key_findings(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key findings using LLM"""
        content = document.get('content', '')[:3000]
        
        prompt = self.digest_prompts['key_findings'].format(content=content)
        
        try:
            response = self.llm_gateway.call_llm(
                agent_name="digest_generator",
                task_type="key_findings",
                prompt=prompt,
                model_preference="gpt-4"
            )
            
            findings = self._parse_json_response(response.content)
            findings['document_filename'] = document.get('filename', 'Unknown')
            findings['extraction_timestamp'] = self._get_timestamp()
            
            logger.info(f"Extracted key findings from: {document.get('filename', 'Unknown')}")
            
            return findings
            
        except Exception as e:
            logger.error(f"Failed to extract key findings: {e}")
            return {
                'main_findings': [],
                'action_items': [],
                'risks_identified': [],
                'opportunities': [],
                'document_filename': document.get('filename', 'Unknown'),
                'error': str(e)
            }
    
    def generate_comprehensive_digest(self, document: Dict[str, Any]) -> str:
        """Generate comprehensive digest using LLM"""
        content = document.get('content', '')[:4000]
        
        prompt = self.digest_prompts['comprehensive_digest'].format(content=content)
        
        try:
            response = self.llm_gateway.call_llm(
                agent_name="digest_generator",
                task_type="comprehensive_digest",
                prompt=prompt,
                model_preference="gpt-4"
            )
            
            logger.info(f"Generated comprehensive digest for: {document.get('filename', 'Unknown')}")
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive digest: {e}")
            return f"Failed to generate comprehensive digest: {str(e)}"
    
    def generate_digest_with_llm(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete digest package using LLM"""
        try:
            # Generate all digest components
            executive_summary = self.generate_executive_summary(document)
            technical_summary = self.generate_technical_summary(document)
            key_findings = self.extract_key_findings(document)
            comprehensive_digest = self.generate_comprehensive_digest(document)
            
            digest_package = {
                'document_filename': document.get('filename', 'Unknown'),
                'generation_timestamp': self._get_timestamp(),
                'executive_summary': executive_summary,
                'technical_summary': technical_summary,
                'key_findings': key_findings,
                'comprehensive_digest': comprehensive_digest,
                'metadata': {
                    'document_size': len(document.get('content', '')),
                    'digest_version': '1.0',
                    'generated_by': 'LLM Enhanced Digest Generator'
                }
            }
            
            logger.info(f"Generated complete digest package for: {document.get('filename', 'Unknown')}")
            return digest_package
            
        except Exception as e:
            logger.error(f"Failed to generate digest package: {e}")
            return {
                'document_filename': document.get('filename', 'Unknown'),
                'error': f'Failed to generate digest package: {str(e)}',
                'generation_timestamp': self._get_timestamp()
            }
    
    def _parse_json_response(self, response_content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {'error': f'Failed to parse response: {str(e)}'}
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
