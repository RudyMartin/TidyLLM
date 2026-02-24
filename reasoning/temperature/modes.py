"""
Reasoning modes based on temperature.
"""

from enum import Enum


class ReasoningMode(Enum):
    """Reasoning modes controlled by temperature parameter.

    Temperature determines which reasoning approach to use:
    - SYMBOLIC: T ≈ 0 (certifiable, rule-based, deterministic)
    - HYBRID: 0 < T < 0.5 (mixed symbolic + analogical, weighted blend)
    - ANALOGICAL: T ≥ 0.5 (case-based, similarity-driven, exploratory)

    Based on Pedro Domingos's Tensor Logic framework that unifies
    symbolic and analogical reasoning on a temperature continuum.
    """
    # T ≈ 0: Pure symbolic reasoning - uses rules, logic, formal verification
    SYMBOLIC = "symbolic"

    # 0 < T < 0.5: Hybrid mode - combines symbolic and analogical with weights
    HYBRID = "hybrid"

    # T ≥ 0.5: Pure analogical reasoning - uses case similarity, embeddings
    ANALOGICAL = "analogical"

    def __str__(self):
        """Return string value for easy printing."""
        return self.value
