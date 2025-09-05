"""
Metadata Extraction Implementation

Extract structured metadata from business documents using pattern matching and NLP.
Supports common business document fields like amounts, dates, IDs, contacts, etc.

Part of the tidyllm-verse: Educational ML with complete transparency
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime

class MetadataExtractor:
    """Extract structured metadata from document text using pattern matching."""
    
    def __init__(self):
        """Initialize metadata extractor with common business patterns."""
        self.patterns = self._initialize_extraction_patterns()
    
    def _initialize_extraction_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common business document extraction patterns."""
        return {
            # Financial patterns
            'invoice_number': {
                'pattern': r'(?:invoice|inv)[\s#:]*([a-z0-9-]+)',
                'description': 'Invoice identification number',
                'validation': r'^[A-Z0-9-]{3,20}$'
            },
            
            'total_amount': {
                'pattern': r'(?:total|amount due)[\s:$]*(\$?[\d,]+\.?\d{0,2})',
                'description': 'Total monetary amount',
                'validation': r'^\$?[\d,]+\.?\d{0,2}$'
            },
            
            'account_number': {
                'pattern': r'(?:account|acct)[\s#:]*(\d{8,16})',
                'description': 'Account number',
                'validation': r'^\d{8,16}$'
            },
            
            # Date patterns  
            'invoice_date': {
                'pattern': r'(?:invoice date|date)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})',
                'description': 'Invoice or document date',
                'validation': r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$|^\w+\s+\d{1,2},?\s+\d{4}$'
            },
            
            'due_date': {
                'pattern': r'(?:due date|payment due)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})',
                'description': 'Payment due date',
                'validation': r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$|^\w+\s+\d{1,2},?\s+\d{4}$'
            },
            
            # Contact information
            'email_address': {
                'pattern': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                'description': 'Email addresses',
                'validation': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            },
            
            'phone_number': {
                'pattern': r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
                'description': 'Phone numbers',
                'validation': r'^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$'
            },
            
            # Contract patterns
            'contract_number': {
                'pattern': r'(?:contract|agreement)[\s#:]*([a-z0-9-]+)',
                'description': 'Contract identification number',
                'validation': r'^[A-Z0-9-]{3,20}$'
            },
            
            'contract_value': {
                'pattern': r'(?:contract value|total value)[\s:$]*(\$?[\d,]+\.?\d{0,2})',
                'description': 'Total contract value',
                'validation': r'^\$?[\d,]+\.?\d{0,2}$'
            },
            
            # Purchase order patterns
            'purchase_order': {
                'pattern': r'(?:purchase order|po)[\s#:]*([a-z0-9-]+)',
                'description': 'Purchase order number',
                'validation': r'^[A-Z0-9-]{3,20}$'
            },
            
            # Tax and legal
            'tax_id': {
                'pattern': r'(?:tax id|ein|ssn)[\s#:]*(\d{2}-?\d{7}|\d{3}-?\d{2}-?\d{4})',
                'description': 'Tax identification number',
                'validation': r'^\d{2}-?\d{7}$|^\d{3}-?\d{2}-?\d{4}$'
            },
            
            # Project and reference
            'project_code': {
                'pattern': r'(?:project|proj)[\s#:]*([a-z0-9-]{3,15})',
                'description': 'Project identification code',
                'validation': r'^[A-Z0-9-]{3,15}$'
            },
            
            'reference_number': {
                'pattern': r'(?:reference|ref)[\s#:]*([a-z0-9-]+)',
                'description': 'Reference number',
                'validation': r'^[A-Z0-9-]{3,20}$'
            }
        }
    
    def extract_metadata(self, document_text: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract metadata from document text.
        
        Args:
            document_text: Text content to analyze
            fields: Specific fields to extract (if None, extracts all available)
            
        Returns:
            Dictionary with extracted metadata fields and values
        """
        if fields is None:
            fields = list(self.patterns.keys())
        
        extracted_metadata = {}
        text_lower = document_text.lower()
        
        for field_name in fields:
            if field_name not in self.patterns:
                continue
                
            pattern_info = self.patterns[field_name]
            pattern = pattern_info['pattern']
            validation_pattern = pattern_info.get('validation')
            
            # Find matches
            matches = re.findall(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                # Take the first match and validate if validation pattern exists
                value = matches[0].strip()
                
                if validation_pattern:
                    if re.match(validation_pattern, value, re.IGNORECASE):
                        extracted_metadata[field_name] = value
                    # If validation fails, try other matches
                    else:
                        for match in matches[1:]:
                            test_value = match.strip()
                            if re.match(validation_pattern, test_value, re.IGNORECASE):
                                extracted_metadata[field_name] = test_value
                                break
                else:
                    extracted_metadata[field_name] = value
        
        return extracted_metadata
    
    def extract_with_confidence(self, document_text: str, fields: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Extract metadata with confidence scoring and additional information.
        
        Args:
            document_text: Text content to analyze
            fields: Specific fields to extract (if None, extracts all available)
            
        Returns:
            Dictionary with field names as keys and dicts containing value, confidence, and metadata
        """
        if fields is None:
            fields = list(self.patterns.keys())
        
        results = {}
        text_lower = document_text.lower()
        
        for field_name in fields:
            if field_name not in self.patterns:
                continue
                
            pattern_info = self.patterns[field_name]
            pattern = pattern_info['pattern']
            validation_pattern = pattern_info.get('validation')
            
            # Find all matches
            matches = re.findall(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                # Calculate confidence based on number of matches and validation
                confidence = min(0.5 + (len(matches) * 0.1), 1.0)  # Base 50% + 10% per match, max 100%
                
                # Take first valid match
                best_value = matches[0].strip()
                is_valid = True
                
                if validation_pattern:
                    is_valid = bool(re.match(validation_pattern, best_value, re.IGNORECASE))
                    if not is_valid:
                        # Try other matches
                        for match in matches[1:]:
                            test_value = match.strip()
                            if re.match(validation_pattern, test_value, re.IGNORECASE):
                                best_value = test_value
                                is_valid = True
                                break
                    
                    # Adjust confidence based on validation
                    if is_valid:
                        confidence = min(confidence + 0.2, 1.0)  # Boost for valid data
                    else:
                        confidence = max(confidence - 0.3, 0.1)  # Penalty for invalid data
                
                results[field_name] = {
                    'value': best_value,
                    'confidence': confidence,
                    'match_count': len(matches),
                    'is_valid': is_valid,
                    'description': pattern_info['description'],
                    'all_matches': matches[:5]  # First 5 matches for reference
                }
        
        return results
    
    def validate_extracted_data(self, metadata: Dict[str, str]) -> Dict[str, bool]:
        """
        Validate extracted metadata against pattern validation rules.
        
        Args:
            metadata: Dictionary of extracted metadata to validate
            
        Returns:
            Dictionary with field names as keys and validation results as booleans
        """
        validation_results = {}
        
        for field_name, value in metadata.items():
            if field_name in self.patterns:
                validation_pattern = self.patterns[field_name].get('validation')
                if validation_pattern:
                    validation_results[field_name] = bool(re.match(validation_pattern, value, re.IGNORECASE))
                else:
                    validation_results[field_name] = True  # No validation rule means valid
            else:
                validation_results[field_name] = False  # Unknown pattern
        
        return validation_results
    
    def get_available_fields(self) -> Dict[str, str]:
        """Get list of all available extraction fields with descriptions."""
        return {field_name: info['description'] for field_name, info in self.patterns.items()}
    
    def add_custom_pattern(self, field_name: str, pattern: str, description: str, validation: Optional[str] = None):
        """
        Add a custom extraction pattern.
        
        Args:
            field_name: Name of the field to extract
            pattern: Regular expression pattern for extraction
            description: Human-readable description of the field
            validation: Optional validation pattern for extracted values
        """
        self.patterns[field_name] = {
            'pattern': pattern,
            'description': description,
            'validation': validation
        }