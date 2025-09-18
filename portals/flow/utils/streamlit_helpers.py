"""
Streamlit Helper Utilities for TidyLLM Flow Portal
=================================================

Utility functions to handle common Streamlit issues and provide
consistent behavior across the portal interface.
"""

from typing import Tuple, Union, Optional


def ensure_numeric_type_consistency(
    field_type: str,
    min_val: Union[int, float],
    max_val: Union[int, float],
    default_value: Optional[Union[int, float]]
) -> Tuple[Union[int, float], Union[int, float], Union[int, float]]:
    """
    Ensure numeric type consistency for Streamlit number_input.

    Streamlit requires all numeric parameters (min_value, max_value, value) to be
    of the same type (all int or all float) to avoid StreamlitMixedNumericTypesError.

    This utility function ensures type consistency by converting all values to the
    appropriate type based on the field_type specification.

    Args:
        field_type: 'integer' or 'number' (float)
        min_val: minimum value (any numeric type)
        max_val: maximum value (any numeric type)
        default_value: default value (any numeric type or None)

    Returns:
        tuple: (min_val, max_val, default_val) all of consistent type

    Example:
        >>> min_val, max_val, default_val = ensure_numeric_type_consistency(
        ...     'integer', 0.0, 100.0, 50.0
        ... )
        >>> # Returns: (0, 100, 50) - all integers

        >>> min_val, max_val, default_val = ensure_numeric_type_consistency(
        ...     'number', 0, 100, None
        ... )
        >>> # Returns: (0.0, 100.0, 0.0) - all floats
    """
    if field_type == 'integer':
        min_val = int(min_val)
        max_val = int(max_val)
        default_val = int(default_value) if default_value is not None else min_val
    else:  # number (float)
        min_val = float(min_val)
        max_val = float(max_val)
        default_val = float(default_value) if default_value is not None else float(min_val)

    return min_val, max_val, default_val


def safe_get_rag_systems(rag_status: Union[dict, list]) -> list:
    """
    Safely extract available RAG systems from status data.

    Handles both dict and list formats of rag_status to prevent
    'list' object has no attribute 'items' errors.

    Args:
        rag_status: RAG status data (dict or list)

    Returns:
        list: Available RAG system names
    """
    if isinstance(rag_status, dict):
        return [k for k, v in rag_status.items() if v]
    elif isinstance(rag_status, list):
        return [f"rag_system_{i}" for i, v in enumerate(rag_status) if v]
    else:
        return ["No RAG systems available"]


def format_status_icon(is_available: bool, style: str = "professional") -> str:
    """
    Format status icons for consistent UI appearance.

    Args:
        is_available: Whether the item is available/healthy
        style: Icon style ('professional', 'emoji', 'text')

    Returns:
        str: Formatted status indicator
    """
    if style == "professional":
        return "✓" if is_available else "○"
    elif style == "emoji":
        return "✅" if is_available else "❌"
    elif style == "text":
        return "Available" if is_available else "Unavailable"
    else:
        return "✓" if is_available else "○"


def safe_dict_iteration(data: Union[dict, list], default_key_prefix: str = "item") -> list:
    """
    Safely iterate over data that might be dict or list.

    Args:
        data: Data to iterate (dict or list)
        default_key_prefix: Prefix for generated keys if data is list

    Returns:
        list: List of (key, value) tuples
    """
    if isinstance(data, dict):
        return list(data.items())
    elif isinstance(data, list):
        return [(f"{default_key_prefix}_{i}", value) for i, value in enumerate(data)]
    else:
        return []