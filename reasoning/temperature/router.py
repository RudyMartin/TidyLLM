"""
Temperature-based routing between reasoning modes.
"""

from .modes import ReasoningMode


class TemperatureRouter:
    """Routes queries to appropriate reasoning mode based on temperature.

    Temperature ranges:
    - T ≤ 0.05: Pure symbolic (certifiable)
    - 0.05 < T < 0.5: Hybrid (symbolic + analogical)
    - T ≥ 0.5: Pure analogical (case-based)
    """

    def __init__(self, symbolic_threshold=0.05, hybrid_threshold=0.5):
        """Initialize router with temperature thresholds.

        Args:
            symbolic_threshold: Below this = pure symbolic (default: 0.05)
            hybrid_threshold: Above this = pure analogical (default: 0.5)
        """
        # Store threshold values that define mode boundaries
        self.symbolic_threshold = symbolic_threshold  # T ≤ this → SYMBOLIC
        self.hybrid_threshold = hybrid_threshold      # T ≥ this → ANALOGICAL

    def get_mode(self, temperature):
        """Determine reasoning mode from temperature.

        Args:
            temperature: Float >= 0

        Returns:
            ReasoningMode enum value
        """
        # Map temperature to one of three reasoning modes
        if temperature <= self.symbolic_threshold:
            # Very low T → Symbolic reasoning (certifiable, rule-based)
            return ReasoningMode.SYMBOLIC

        elif temperature < self.hybrid_threshold:
            # Middle range T → Hybrid (weighted mix of symbolic + analogical)
            return ReasoningMode.HYBRID

        else:
            # High T → Analogical reasoning (case-based, similarity)
            return ReasoningMode.ANALOGICAL

    def get_weights(self, temperature):
        """Get mixing weights for hybrid mode.

        Computes how much weight to give to symbolic vs analogical reasoning.
        Weights always sum to 1.0 for proper averaging.

        Args:
            temperature: Float in [0, 1]

        Returns:
            Dict with 'symbolic' and 'analogical' weights (sum to 1.0)
        """
        if temperature <= self.symbolic_threshold:
            # Pure symbolic: 100% symbolic, 0% analogical
            return {'symbolic': 1.0, 'analogical': 0.0}

        elif temperature >= self.hybrid_threshold:
            # Pure analogical: 0% symbolic, 100% analogical
            return {'symbolic': 0.0, 'analogical': 1.0}

        else:
            # Hybrid mode: linear interpolation between thresholds
            # Normalize temperature to 0-1 range within hybrid zone
            t_normalized = (temperature - self.symbolic_threshold) / \
                          (self.hybrid_threshold - self.symbolic_threshold)

            # As temperature increases, symbolic weight decreases, analogical increases
            return {
                'symbolic': 1.0 - t_normalized,      # Decreases from 1.0 to 0.0
                'analogical': t_normalized           # Increases from 0.0 to 1.0
            }
