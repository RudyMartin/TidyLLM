"""
Business Document Template Processing

Specialized processing for common business document types with pattern-based
metadata extraction and document type classification.

Part of the tidyllm-verse: Educational ML with complete transparency
"""

import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from ..classification import DocumentClassifier
from ..extraction import TextExtractor, MetadataExtractor

@dataclass
class DocumentTemplate:
    """Template definition for a business document type."""
    name: str
    keywords: List[str]
    required_patterns: List[str]
    optional_patterns: List[str]
    confidence_boost: float

class BusinessDocumentProcessor:
    """Process business documents using specialized templates and patterns."""
    
    def __init__(self):
        """Initialize with common business document templates."""
        self.templates = self._initialize_business_templates()
        self.text_extractor = TextExtractor()
        self.metadata_extractor = MetadataExtractor()
        
        # Initialize classifier with business categories
        business_categories = list(self.templates.keys())
        self.classifier = DocumentClassifier(business_categories)
        
    def _initialize_business_templates(self) -> Dict[str, DocumentTemplate]:
        """Initialize business document templates."""
        return {
            'invoice': DocumentTemplate(
                name='invoice',
                keywords=['invoice', 'bill', 'payment', 'due', 'total', 'amount'],
                required_patterns=['invoice_number', 'total_amount'],
                optional_patterns=['invoice_date', 'due_date', 'email_address', 'phone_number'],
                confidence_boost=0.2
            ),
            
            'contract': DocumentTemplate(
                name='contract',
                keywords=['agreement', 'contract', 'terms', 'conditions', 'party', 'parties'],
                required_patterns=['contract_number', 'invoice_date'],  # Using invoice_date as generic date
                optional_patterns=['total_amount', 'due_date', 'email_address', 'phone_number'],
                confidence_boost=0.3
            ),
            
            'purchase_order': DocumentTemplate(
                name='purchase_order',
                keywords=['purchase', 'order', 'buy', 'vendor', 'supplier'],
                required_patterns=['purchase_order'],
                optional_patterns=['total_amount', 'invoice_date', 'email_address'],
                confidence_boost=0.25
            ),
            
            'financial_statement': DocumentTemplate(
                name='financial_statement',
                keywords=['statement', 'balance', 'account', 'financial', 'bank'],
                required_patterns=['account_number'],
                optional_patterns=['invoice_date', 'total_amount'],
                confidence_boost=0.2
            ),
            
            'legal_document': DocumentTemplate(
                name='legal_document',
                keywords=['legal', 'court', 'lawsuit', 'attorney', 'law', 'clause'],
                required_patterns=['invoice_date'],  # Using generic date
                optional_patterns=['due_date', 'reference_number', 'email_address'],
                confidence_boost=0.15
            ),
            
            'report': DocumentTemplate(
                name='report',
                keywords=['report', 'analysis', 'summary', 'findings', 'results'],
                required_patterns=[],
                optional_patterns=['invoice_date', 'reference_number', 'email_address'],
                confidence_boost=0.1
            )
        }
    
    def process_document(self, document_path: str, max_pages: int = 5) -> Dict[str, Any]:
        """
        Process a business document with specialized template matching.
        
        Args:
            document_path: Path to document file
            max_pages: Maximum pages to process
            
        Returns:
            Dictionary with enhanced processing results
        """
        # Extract text
        document_text, extraction_meta = self.text_extractor.extract_text(document_path, max_pages)
        
        if not document_text:
            return {
                'document_type': 'unknown',
                'confidence': 0.0,
                'error': extraction_meta.get('error', 'No text extracted'),
                'extracted_fields': {}
            }
        
        # Classify using templates
        document_type, confidence = self._classify_with_templates(document_text)
        
        # Extract metadata using relevant patterns
        template = self.templates.get(document_type)
        relevant_fields = []
        if template:
            relevant_fields = template.required_patterns + template.optional_patterns
        
        extracted_fields = self.metadata_extractor.extract_metadata(document_text, relevant_fields)
        
        # Get metadata with confidence
        detailed_metadata = self.metadata_extractor.extract_with_confidence(document_text, relevant_fields)
        
        return {
            'document_type': document_type,
            'confidence': confidence,
            'template_used': template.name if template else None,
            'text_length': len(document_text),
            'extracted_fields': extracted_fields,
            'detailed_metadata': detailed_metadata,
            'extraction_metadata': extraction_meta
        }
    
    def _classify_with_templates(self, text: str) -> Tuple[str, float]:
        """Classify document using business templates."""
        text_lower = text.lower()
        template_scores = {}
        
        for template_name, template in self.templates.items():
            score = 0.0
            
            # Check for keywords
            keyword_matches = sum(1 for keyword in template.keywords if keyword in text_lower)
            if template.keywords:
                keyword_score = keyword_matches / len(template.keywords)
                score += keyword_score * 0.4
            
            # Check for required patterns
            required_found = 0
            for pattern_name in template.required_patterns:
                if self._check_pattern_exists(text_lower, pattern_name):
                    required_found += 1
            
            if template.required_patterns:
                required_score = required_found / len(template.required_patterns)
                score += required_score * 0.4
            else:
                score += 0.4  # No required patterns means this section scores full
            
            # Check for optional patterns
            optional_found = 0
            for pattern_name in template.optional_patterns:
                if self._check_pattern_exists(text_lower, pattern_name):
                    optional_found += 1
            
            if template.optional_patterns:
                optional_score = optional_found / len(template.optional_patterns)
                score += optional_score * 0.2
            else:
                score += 0.2  # No optional patterns means this section scores full
            
            # Apply template-specific confidence boost
            score += template.confidence_boost
            
            template_scores[template_name] = min(score, 1.0)  # Cap at 1.0
        
        if template_scores:
            best_template = max(template_scores, key=template_scores.get)
            confidence = template_scores[best_template]
            return best_template, confidence
        else:
            return 'unknown', 0.0
    
    def _check_pattern_exists(self, text: str, pattern_name: str) -> bool:
        """Check if a metadata pattern exists in the text."""
        # Use the metadata extractor to check if pattern would find matches
        result = self.metadata_extractor.extract_metadata(text, [pattern_name])
        return pattern_name in result and result[pattern_name]
    
    def get_template_summary(self) -> Dict[str, Any]:
        """Get summary of available business templates."""
        summary = {
            'total_templates': len(self.templates),
            'template_details': {}
        }
        
        for template_name, template in self.templates.items():
            summary['template_details'][template_name] = {
                'keywords_count': len(template.keywords),
                'required_patterns_count': len(template.required_patterns),
                'optional_patterns_count': len(template.optional_patterns),
                'confidence_boost': template.confidence_boost,
                'keywords': template.keywords
            }
        
        return summary
    
    def add_custom_template(self, template: DocumentTemplate):
        """Add a custom business document template."""
        self.templates[template.name] = template
        
        # Update classifier categories
        business_categories = list(self.templates.keys())
        self.classifier = DocumentClassifier(business_categories)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return self.text_extractor.get_supported_formats()