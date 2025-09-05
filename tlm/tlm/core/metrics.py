from typing import List, Optional

# Type definitions for consistency
Scalar = float
Vector = List[Scalar]
Matrix = List[Vector]

__all__ = ['confusion_matrix','accuracy']

def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: Optional[int] = None) -> List[List[int]]:
    """Compute confusion matrix."""
    K = (max(max(y_true), max(y_pred)) + 1) if num_classes is None else num_classes
    cm = [[0 for _ in range(K)] for _ in range(K)]
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt)][int(yp)] += 1
    return cm

def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Calculate classification accuracy."""
    n = len(y_true)
    return sum(1 for a,b in zip(y_true, y_pred) if a==b) / n
