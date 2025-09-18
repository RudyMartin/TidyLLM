"""
Data Handling Utilities for TidyLLM Flow Portal
===============================================

Utilities for handling inconsistent data structures (dict vs list)
and defensive programming patterns commonly used throughout the portal.
"""

from typing import Union, List, Dict, Any, Tuple, Optional


def safe_workflow_access(workflows: Union[Dict, List], operation: str = "count") -> Any:
    """
    Safely access workflow data regardless of dict/list format.

    Args:
        workflows: Workflow data (dict or list)
        operation: Type of operation ('count', 'types', 'active', 'values')

    Returns:
        Appropriate result based on operation
    """
    if isinstance(workflows, dict):
        if operation == "count":
            return len(workflows)
        elif operation == "types":
            return list(set(w.get("workflow_type", "unknown") for w in workflows.values()))
        elif operation == "active":
            return [w for w in workflows.values() if w.get('status') != 'disabled']
        elif operation == "values":
            return list(workflows.values())
        else:
            return workflows
    elif isinstance(workflows, list):
        if operation == "count":
            return len(workflows)
        elif operation == "types":
            return list(set(w.get("workflow_type", "unknown") for w in workflows))
        elif operation == "active":
            return [w for w in workflows if w.get('status') != 'disabled']
        elif operation == "values":
            return workflows
        else:
            return workflows
    else:
        # Fallback for unknown types
        if operation == "count":
            return 0
        elif operation in ["types", "active", "values"]:
            return []
        else:
            return {}


def safe_status_counts(status_data: Union[Dict, List]) -> Tuple[int, int]:
    """
    Safely count available vs total items from status data.

    Args:
        status_data: Status data (dict of bool or list of bool)

    Returns:
        tuple: (available_count, total_count)
    """
    if isinstance(status_data, dict):
        available = sum(1 for available in status_data.values() if available)
        total = len(status_data)
    elif isinstance(status_data, list):
        available = sum(1 for item in status_data if item)
        total = len(status_data)
    else:
        available, total = 0, 0

    return available, total


def safe_get_available_items(
    status_data: Union[Dict, List],
    name_prefix: str = "item"
) -> List[str]:
    """
    Safely get list of available item names from status data.

    Args:
        status_data: Status data (dict of bool or list of bool)
        name_prefix: Prefix for generated names if data is list

    Returns:
        List of available item names
    """
    if isinstance(status_data, dict):
        return [k for k, v in status_data.items() if v]
    elif isinstance(status_data, list):
        return [f"{name_prefix}_{i}" for i, v in enumerate(status_data) if v]
    else:
        return ["No items available"]


def safe_check_availability(
    status_data: Union[Dict, List],
    item_name: str,
    default: bool = False
) -> bool:
    """
    Safely check if specific item is available in status data.

    Args:
        status_data: Status data (dict or list)
        item_name: Name or index of item to check
        default: Default value if item not found

    Returns:
        bool: Whether item is available
    """
    if isinstance(status_data, dict):
        return status_data.get(item_name, default)
    elif isinstance(status_data, list):
        try:
            # Try to extract index from item_name like "rag_system_0"
            if "_" in item_name:
                index = int(item_name.split("_")[-1])
                return status_data[index] if 0 <= index < len(status_data) else default
            else:
                # Assume all items available for list format
                return True if status_data else default
        except (ValueError, IndexError):
            return default
    else:
        return default


def safe_iterate_with_status(
    status_data: Union[Dict, List],
    name_prefix: str = "item"
) -> List[Tuple[str, bool]]:
    """
    Safely iterate over status data returning (name, status) pairs.

    Args:
        status_data: Status data (dict or list)
        name_prefix: Prefix for generated names if data is list

    Returns:
        List of (name, status) tuples
    """
    if isinstance(status_data, dict):
        return list(status_data.items())
    elif isinstance(status_data, list):
        return [(f"{name_prefix}_{i}", status) for i, status in enumerate(status_data)]
    else:
        return []


def safe_any_check(data: Union[Dict, List]) -> bool:
    """
    Safely check if any items in data structure are truthy.

    Args:
        data: Data to check (dict or list)

    Returns:
        bool: Whether any items are truthy
    """
    if isinstance(data, dict):
        return any(data.values())
    elif isinstance(data, list):
        return any(data)
    else:
        return False


def normalize_workflow_types(workflows: Union[Dict, List]) -> Dict[str, List[Dict]]:
    """
    Normalize workflows by type regardless of input format.

    Args:
        workflows: Workflow data (dict or list)

    Returns:
        Dict mapping workflow_type to list of workflows
    """
    if isinstance(workflows, dict):
        workflow_list = list(workflows.values())
    elif isinstance(workflows, list):
        workflow_list = workflows
    else:
        workflow_list = []

    # Group by type
    by_type = {}
    for workflow in workflow_list:
        wf_type = workflow.get("workflow_type", "unknown")
        if wf_type not in by_type:
            by_type[wf_type] = []
        by_type[wf_type].append(workflow)

    return by_type


def safe_context_check(context_info: Union[Dict, List]) -> bool:
    """
    Safely check if context info has content.

    Args:
        context_info: Context data (dict or list)

    Returns:
        bool: Whether context has meaningful content
    """
    if isinstance(context_info, dict):
        return any(context_info.values())
    elif isinstance(context_info, list):
        return any(context_info)
    else:
        return False