"""
Label Normalization Utilities for VectorQA Sage

This module provides robust label normalization functionality for QA datasets,
ensuring consistent label formats across different data sources and user inputs.

The normalizer handles various input formats including emojis, different text cases,
and edge cases like None values and empty strings, providing a standardized output format.

TODO - Add support for custom label mappings
TODO - Add label validation and quality checks
TODO - Add label statistics and reporting
TODO - Add batch processing capabilities
"""

from typing import Union, Optional

def normalize_label(label: Union[str, None]) -> str:
    """
    Normalize a label to a standard format.
    
    Args:
        label: The input label to normalize (can be string, None, or other types)
        
    Returns:
        Normalized label string: 'Correct', 'Missing Info', 'Inconsistent', or 'Other'
        
    Examples:
        >>> normalize_label('correct')
        'Correct'
        >>> normalize_label('✔️')
        'Correct'
        >>> normalize_label(None)
        'Other'
        >>> normalize_label('')
        'Other'
    """
    # Handle None and empty values
    if label is None:
        return 'Other'
    
    # Convert to string and handle empty strings
    label_str = str(label).strip().lower()
    if not label_str:
        return 'Other'
    
    if label_str in ['correct', '✔️', 'true', 'yes']:
        return 'Correct'
    elif label_str in ['missing info', 'incomplete', 'missing']:
        return 'Missing Info'
    elif label_str in ['inconsistent', 'wrong', 'no', 'false']:
        return 'Inconsistent'
    else:
        return 'Other'
