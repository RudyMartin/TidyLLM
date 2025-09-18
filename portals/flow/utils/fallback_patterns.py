"""
Fallback Pattern Utilities for TidyLLM Flow Portal
=================================================

Utilities for handling common fallback patterns, error recovery,
and graceful degradation used throughout the portal.
"""

from typing import Union, List, Dict, Any, Optional, Callable
from pathlib import Path
import json


def safe_file_access(
    primary_path: str,
    fallback_paths: List[str],
    default_content: Any = None
) -> tuple[bool, Any]:
    """
    Safely access files with fallback paths.

    Args:
        primary_path: Primary file path to try
        fallback_paths: List of fallback paths to try
        default_content: Default content if all paths fail

    Returns:
        tuple: (success, content)
    """
    all_paths = [primary_path] + fallback_paths

    for path in all_paths:
        try:
            file_path = Path(path)
            if file_path.exists():
                if path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return True, json.load(f)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return True, f.read()
        except Exception:
            continue

    return False, default_content


def safe_json_parse(
    data: Union[str, dict],
    default: dict = None
) -> dict:
    """
    Safely parse JSON data with fallback.

    Args:
        data: JSON string or dict
        default: Default value if parsing fails

    Returns:
        Parsed dict or default
    """
    if default is None:
        default = {}

    if isinstance(data, dict):
        return data

    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return default

    return default


def safe_criteria_access(
    criteria: Union[Dict, List, str],
    key: str,
    default: Any = None
) -> Any:
    """
    Safely access criteria data regardless of format.

    Args:
        criteria: Criteria data (dict, list, or string)
        key: Key to access
        default: Default value if key not found

    Returns:
        Value or default
    """
    if isinstance(criteria, dict):
        return criteria.get(key, default)
    elif isinstance(criteria, str):
        try:
            parsed = json.loads(criteria)
            return parsed.get(key, default) if isinstance(parsed, dict) else default
        except json.JSONDecodeError:
            return default
    else:
        return default


def get_default_use_cases() -> List[str]:
    """
    Get default use cases as fallback.

    Returns:
        List of default use cases
    """
    return [
        "Document Analysis",
        "Process Automation",
        "Quality Control",
        "Compliance Checking",
        "Data Extraction"
    ]


def safe_template_field_access(
    template_fields: Union[Dict, List],
    operation: str = "items"
) -> Any:
    """
    Safely access template fields regardless of format.

    Args:
        template_fields: Template field data
        operation: Operation to perform ('items', 'count', 'keys')

    Returns:
        Result based on operation
    """
    if isinstance(template_fields, dict):
        if operation == "items":
            return list(template_fields.items())
        elif operation == "count":
            return len(template_fields)
        elif operation == "keys":
            return list(template_fields.keys())
        else:
            return template_fields
    elif isinstance(template_fields, list):
        if operation == "items":
            return [(f"field_{i}", field) for i, field in enumerate(template_fields)]
        elif operation == "count":
            return len(template_fields)
        elif operation == "keys":
            return [f"field_{i}" for i in range(len(template_fields))]
        else:
            return template_fields
    else:
        if operation in ["items", "keys"]:
            return []
        elif operation == "count":
            return 0
        else:
            return {}


def safe_workflow_selection(
    workflows: Union[Dict, List],
    selection_criteria: Callable[[Dict], bool] = None
) -> Dict[str, Dict]:
    """
    Safely select workflows with optional criteria.

    Args:
        workflows: Workflow data
        selection_criteria: Function to filter workflows

    Returns:
        Dict of selected workflows
    """
    if selection_criteria is None:
        selection_criteria = lambda w: True

    if isinstance(workflows, dict):
        return {
            wid: workflow for wid, workflow in workflows.items()
            if selection_criteria(workflow)
        }
    elif isinstance(workflows, list):
        return {
            f"workflow_{i}": workflow for i, workflow in enumerate(workflows)
            if selection_criteria(workflow)
        }
    else:
        return {}


def graceful_error_message(
    error: Exception,
    context: str = "operation",
    user_friendly: bool = True
) -> str:
    """
    Generate graceful error messages for users.

    Args:
        error: The exception that occurred
        context: Context where error occurred
        user_friendly: Whether to show user-friendly message

    Returns:
        Formatted error message
    """
    if user_friendly:
        error_messages = {
            "FileNotFoundError": f"Required configuration not found for {context}",
            "JSONDecodeError": f"Configuration format issue in {context}",
            "KeyError": f"Missing required information in {context}",
            "TypeError": f"Data format issue in {context}",
            "AttributeError": f"Service temporarily unavailable for {context}",
        }

        error_type = type(error).__name__
        return error_messages.get(error_type, f"{context.title()} temporarily unavailable")
    else:
        return f"{context} error: {str(error)}"


def safe_directory_fallback(
    primary_dir: str,
    fallback_dirs: List[str],
    required_files: List[str] = None
) -> Optional[str]:
    """
    Find first available directory with required files.

    Args:
        primary_dir: Primary directory to check
        fallback_dirs: List of fallback directories
        required_files: Files that must exist in directory

    Returns:
        Path to first valid directory or None
    """
    if required_files is None:
        required_files = []

    all_dirs = [primary_dir] + fallback_dirs

    for directory in all_dirs:
        dir_path = Path(directory)
        if dir_path.exists() and dir_path.is_dir():
            # Check if all required files exist
            if all((dir_path / file).exists() for file in required_files):
                return str(dir_path)

    return None


def create_safe_default(
    data_type: str,
    context: str = "default"
) -> Any:
    """
    Create safe default values based on data type.

    Args:
        data_type: Type of default needed
        context: Context for the default

    Returns:
        Safe default value
    """
    defaults = {
        "workflows": [],
        "rag_status": {},
        "template_fields": {},
        "criteria": {},
        "use_cases": get_default_use_cases(),
        "context_info": {},
        "metrics": {"total": 0, "available": 0},
        "status": {"healthy": False, "message": f"{context} checking..."}
    }

    return defaults.get(data_type, {})


def retry_with_fallback(
    primary_func: Callable,
    fallback_func: Callable,
    max_retries: int = 2,
    context: str = "operation"
) -> tuple[bool, Any]:
    """
    Retry operation with fallback function.

    Args:
        primary_func: Primary function to try
        fallback_func: Fallback function if primary fails
        max_retries: Maximum retry attempts
        context: Context for error messages

    Returns:
        tuple: (success, result)
    """
    for attempt in range(max_retries + 1):
        try:
            result = primary_func()
            return True, result
        except Exception as e:
            if attempt == max_retries:
                # Try fallback
                try:
                    result = fallback_func()
                    return True, result
                except Exception:
                    return False, graceful_error_message(e, context)

    return False, f"{context} failed after {max_retries} attempts"