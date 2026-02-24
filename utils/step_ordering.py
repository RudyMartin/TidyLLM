"""
Step Ordering Utilities
=======================

Reusable utilities for managing step/template execution order in workflows.

Key Concepts:
- step_id: Permanent identifier (never changes)
- step_index: 0-based array position (changes on reorder)
- step_number: 1-based display number (changes on reorder)
"""

from typing import List, Dict, Any


def reorder_steps_by_position(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reorder steps based on desired_position field.

    WHAT CHANGES:
    - Array index (position in list)
    - step_index (0-based position)
    - step_number (1-based display like "1.0", "2.0")

    WHAT STAYS THE SAME:
    - step_id (e.g., 'metadata_extraction' - permanent identifier)
    - template_id (linked template)
    - All other step properties

    Args:
        steps: List of step dictionaries with 'desired_position' field

    Returns:
        Reordered list with updated indices

    Example:
        >>> steps = [
        ...     {"step_id": "extract", "desired_position": 2},
        ...     {"step_id": "analyze", "desired_position": 1}
        ... ]
        >>> reordered = reorder_steps_by_position(steps)
        >>> reordered[0]["step_id"]
        'analyze'
    """
    # Create a copy to avoid modifying original
    steps_copy = [s.copy() for s in steps]

    # Sort steps by desired_position (this shuffles the array indices)
    reordered = sorted(steps_copy, key=lambda s: s.get('desired_position', 999))

    # Update position-related fields for each step
    for new_index, step in enumerate(reordered):
        # Remove temporary UI field
        if 'desired_position' in step:
            del step['desired_position']

        # Update position fields (but NOT step_id)
        step['step_index'] = new_index  # 0-based array position
        step['step_number'] = f"{new_index + 1}.0"  # 1-based display number

        # step_id UNCHANGED - it's the permanent identifier

    return reordered


def check_for_order_changes(steps: List[Dict[str, Any]]) -> bool:
    """
    Check if any steps have different desired positions.

    Args:
        steps: List of step dictionaries

    Returns:
        True if any positions changed, False otherwise
    """
    for i, step in enumerate(steps):
        if step.get('desired_position', i+1) != i+1:
            return True
    return False


def renumber_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Update step_index and step_number based on current array position.
    Useful after adding or removing steps.

    Args:
        steps: List of step dictionaries

    Returns:
        List with updated numbering
    """
    for i, step in enumerate(steps):
        step['step_index'] = i
        step['step_number'] = f"{i + 1}.0"
    return steps


def get_step_by_id(steps: List[Dict[str, Any]], step_id: str) -> Dict[str, Any]:
    """
    Find a step by its permanent ID.

    Args:
        steps: List of step dictionaries
        step_id: The permanent step identifier

    Returns:
        The step dictionary or None if not found
    """
    for step in steps:
        if step.get('step_id') == step_id:
            return step
    return None


def get_step_by_position(steps: List[Dict[str, Any]], position: int) -> Dict[str, Any]:
    """
    Get step at a specific position (1-based).

    Args:
        steps: List of step dictionaries
        position: 1-based position number

    Returns:
        The step dictionary or None if position invalid
    """
    index = position - 1
    if 0 <= index < len(steps):
        return steps[index]
    return None


def swap_steps(steps: List[Dict[str, Any]], pos1: int, pos2: int) -> List[Dict[str, Any]]:
    """
    Swap two steps by position (1-based).

    Args:
        steps: List of step dictionaries
        pos1: First position (1-based)
        pos2: Second position (1-based)

    Returns:
        List with swapped steps and updated numbering
    """
    idx1, idx2 = pos1 - 1, pos2 - 1

    if 0 <= idx1 < len(steps) and 0 <= idx2 < len(steps):
        steps[idx1], steps[idx2] = steps[idx2], steps[idx1]
        return renumber_steps(steps)

    return steps


def move_step_to_position(steps: List[Dict[str, Any]], from_pos: int, to_pos: int) -> List[Dict[str, Any]]:
    """
    Move a step from one position to another (1-based).

    Args:
        steps: List of step dictionaries
        from_pos: Current position (1-based)
        to_pos: Target position (1-based)

    Returns:
        List with moved step and updated numbering
    """
    from_idx = from_pos - 1
    to_idx = to_pos - 1

    if 0 <= from_idx < len(steps) and 0 <= to_idx < len(steps):
        step = steps.pop(from_idx)
        steps.insert(to_idx, step)
        return renumber_steps(steps)

    return steps


def validate_step_order(steps: List[Dict[str, Any]]) -> List[str]:
    """
    Validate that step ordering is consistent.

    Args:
        steps: List of step dictionaries

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check for duplicate step_ids
    step_ids = [s.get('step_id') for s in steps if s.get('step_id')]
    if len(step_ids) != len(set(step_ids)):
        errors.append("Duplicate step_id values found")

    # Check step_index consistency
    for i, step in enumerate(steps):
        if step.get('step_index') != i:
            errors.append(f"Step at position {i} has incorrect step_index: {step.get('step_index')}")

    # Check step_number format
    for i, step in enumerate(steps):
        expected = f"{i + 1}.0"
        if step.get('step_number') != expected:
            errors.append(f"Step at position {i} has incorrect step_number: {step.get('step_number')}")

    return errors


# Export all utilities
__all__ = [
    'reorder_steps_by_position',
    'check_for_order_changes',
    'renumber_steps',
    'get_step_by_id',
    'get_step_by_position',
    'swap_steps',
    'move_step_to_position',
    'validate_step_order'
]