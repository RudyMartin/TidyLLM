"""
TensorLogicService - Main orchestration for temperature-controlled reasoning.
"""

from .temperature.router import TemperatureRouter
from .temperature.modes import ReasoningMode


class TensorLogicService:
    """Temperature-controlled reasoning service.

    Orchestrates symbolic and analogical reasoning based on temperature:
    - T=0.0: Pure symbolic (certifiable, rule-based)
    - T=0.1-0.4: Hybrid (symbolic + analogical)
    - T≥0.5: Pure analogical (case-based)

    Example:
        >>> service = TensorLogicService()
        >>> result = service.infer("Is data validation required?", temperature=0.0)
        >>> print(result['certifiable'])  # True for symbolic reasoning
        True
    """

    def __init__(self,
                 rules=None,
                 case_base=None,
                 embedding_method='lsa',
                 symbolic_threshold=0.05,
                 hybrid_threshold=0.5):
        """Initialize service with configuration.

        Args:
            rules: Symbolic rules (list of dicts or Rule objects)
            case_base: Analogical cases (list of strings)
            embedding_method: Method for embeddings ('tfidf', 'lsa', 'word_avg')
            symbolic_threshold: Temperature threshold for symbolic mode
            hybrid_threshold: Temperature threshold for analogical mode
        """
        # Create temperature router to map T values to reasoning modes
        self.router = TemperatureRouter(symbolic_threshold, hybrid_threshold)

        # Store reasoning knowledge bases
        self.rules = rules or []              # Symbolic rules (formal logic)
        self.case_base = case_base or []      # Analogical cases (examples)
        self.embedding_method = embedding_method  # How to embed text for similarity

        # Placeholder for reasoning engines (to be implemented with actual adapters)
        # These will be populated when integrating with compliance-qa adapters
        self.symbolic_engine = None      # For rule-based inference
        self.analogical_engine = None    # For case-based retrieval
        self.yrsn_scorer = None          # For trustworthiness scoring

    def infer(self, query, context=None, temperature=0.0, score_trustworthiness=False):
        """Run inference with temperature-controlled reasoning.

        Args:
            query: Query string
            context: Optional context dict
            temperature: Float >= 0 controlling reasoning mode
            score_trustworthiness: Whether to compute YRSN trustworthiness

        Returns:
            Dict with:
            - answer: Inferred answer
            - confidence: Confidence score (0-1)
            - reasoning_mode: ReasoningMode enum value
            - certifiable: Boolean (True only for pure symbolic)
            - trustworthiness: YRSN trust score (if requested)
            - evidence: List of evidence items
            - components: Dict of component scores

        Example:
            >>> service = TensorLogicService()
            >>> result = service.infer("Query?", temperature=0.0)
            >>> result['reasoning_mode']
            'symbolic'
        """
        # Route query to appropriate reasoning mode based on temperature
        mode = self.router.get_mode(temperature)

        # Execute reasoning strategy for determined mode
        if mode == ReasoningMode.SYMBOLIC:
            # T ≈ 0: Use symbolic reasoning (rules, logic)
            result = self._infer_symbolic(query, context)
            certifiable = True  # Symbolic results are certifiable (provable)

        elif mode == ReasoningMode.ANALOGICAL:
            # T ≥ 0.5: Use analogical reasoning (case similarity)
            result = self._infer_analogical(query, context, temperature)
            certifiable = False  # Analogical results are probabilistic

        else:  # HYBRID
            # 0 < T < 0.5: Mix symbolic and analogical with weights
            result = self._infer_hybrid(query, context, temperature)
            certifiable = False  # Hybrid results are not fully certifiable

        # Compute YRSN trustworthiness score if requested and scorer available
        if score_trustworthiness and self.yrsn_scorer:
            trust_score = self.yrsn_scorer.score(query, str(result['answer']))
        else:
            trust_score = None

        # Return unified result format across all modes
        return {
            'answer': result.get('answer'),                  # Inferred answer
            'confidence': result.get('confidence', 0.0),     # 0-1 confidence
            'reasoning_mode': mode.value,                    # Which mode was used
            'certifiable': certifiable,                      # Is result provable?
            'trustworthiness': trust_score,                  # YRSN score (optional)
            'evidence': result.get('evidence', []),          # Supporting evidence
            'components': result.get('components', {})       # Component details
        }

    def _infer_symbolic(self, query, context):
        """Pure symbolic reasoning using rules.

        Uses formal logic, rules, and deduction. Results are certifiable
        (can be proven true given the rule base).

        Args:
            query: Query string
            context: Optional context

        Returns:
            Dict with answer, confidence, evidence
        """
        # If symbolic engine is configured, use it
        if self.symbolic_engine:
            return self.symbolic_engine.infer(query, context)

        # Placeholder implementation (returns mock symbolic answer)
        # In production, this would use actual rule engine from compliance-qa
        return {
            'answer': f"Symbolic answer for: {query}",
            'confidence': 1.0,  # Symbolic answers have perfect confidence if proven
            'evidence': ['Rule-based inference'],
            'components': {}
        }

    def _infer_analogical(self, query, context, temperature):
        """Pure analogical reasoning using cases.

        Uses case-based retrieval and similarity matching. Results are
        probabilistic based on similarity to known examples.

        Args:
            query: Query string
            context: Optional context
            temperature: Temperature for similarity weighting

        Returns:
            Dict with answer, confidence, evidence
        """
        # If analogical engine is configured, use it
        if self.analogical_engine:
            return self.analogical_engine.infer(query, context, temperature)

        # Placeholder implementation using tidyllm-sentence if available
        try:
            import sys
            import os
            # Try to import tidyllm_sentence from sibling package
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sys.path.insert(0, os.path.join(parent_dir, 'tidyllm-sentence'))
            import tidyllm_sentence as tls

            # If we have a case base, do actual case retrieval
            if self.case_base:
                # Find top 3 most similar cases using embeddings
                results = tls.case_retrieval(
                    query=query,
                    case_base=self.case_base,
                    method=self.embedding_method,
                    top_k=3
                )
                # Use best matching case as answer
                best_case, best_score = results[0] if results else (None, 0.0)
                return {
                    'answer': best_case or f"No similar cases found for: {query}",
                    'confidence': best_score,  # Similarity score as confidence
                    'evidence': [f"Case-based retrieval ({self.embedding_method})"],
                    'components': {'similar_cases': results}
                }
        except Exception as e:
            # Fall through to placeholder if import fails
            pass

        # Fallback placeholder (when no case_base or tidyllm-sentence unavailable)
        return {
            'answer': f"Analogical answer for: {query}",
            'confidence': 0.7,  # Moderate confidence for placeholder
            'evidence': ['Case-based retrieval'],
            'components': {}
        }

    def _infer_hybrid(self, query, context, temperature):
        """Hybrid reasoning combining symbolic and analogical.

        Runs both symbolic and analogical reasoning, then combines results
        using temperature-weighted averaging. Allows smooth transition
        between pure symbolic (T→0) and pure analogical (T→0.5).

        Args:
            query: Query string
            context: Optional context
            temperature: Temperature for weighting (determines blend ratio)

        Returns:
            Dict with answer, confidence, evidence
        """
        # Get mixing weights based on temperature
        # E.g., T=0.2 might give {symbolic: 0.6, analogical: 0.4}
        weights = self.router.get_weights(temperature)

        # Run both reasoning modes independently
        sym_result = self._infer_symbolic(query, context)
        ana_result = self._infer_analogical(query, context, temperature)

        # Combine confidence scores using weighted average
        # Higher temperature → more weight on analogical confidence
        combined_confidence = (
            weights['symbolic'] * sym_result['confidence'] +
            weights['analogical'] * ana_result['confidence']
        )

        # Return combined result with both components
        return {
            'answer': f"Hybrid: {sym_result['answer']} + {ana_result['answer']}",
            'confidence': combined_confidence,
            'evidence': sym_result['evidence'] + ana_result['evidence'],  # Merge evidence
            'components': {
                'symbolic': sym_result,      # Full symbolic result
                'analogical': ana_result,    # Full analogical result
                'weights': weights           # How they were combined
            }
        }
