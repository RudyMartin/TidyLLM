"""
Convenience factory functions for creating reasoners.
"""

from .service import TensorLogicService


def create_reasoner(rules=None, cases=None, embedding_method='lsa'):
    """Create TensorLogicService with default configuration.

    Convenience factory that creates a reasoner with sensible defaults.
    Simplifies common usage pattern: create service, use with temperature control.

    Args:
        rules: Symbolic rules (optional) - for T≈0 symbolic reasoning
        cases: Analogical cases (optional) - for T≥0.5 analogical reasoning
        embedding_method: Embedding method for analogical reasoning
                         ('lsa', 'tfidf', or 'word_avg')

    Returns:
        TensorLogicService instance ready for inference

    Examples:
        >>> # Create basic reasoner (symbolic mode)
        >>> reasoner = create_reasoner()
        >>> result = reasoner.infer("Query?", temperature=0.0)
        >>> print(result['reasoning_mode'])
        symbolic

        >>> # With cases for analogical reasoning
        >>> cases = ["Case 1", "Case 2", "Case 3"]
        >>> reasoner = create_reasoner(cases=cases)
        >>> result = reasoner.infer("Query?", temperature=0.7)
        >>> print(result['reasoning_mode'])
        analogical
    """
    # Create and return TensorLogicService with provided knowledge bases
    # Uses default temperature thresholds (symbolic: 0.05, hybrid: 0.5)
    return TensorLogicService(
        rules=rules,                    # Symbolic knowledge (rules, logic)
        case_base=cases,                # Analogical knowledge (examples)
        embedding_method=embedding_method  # How to embed text for similarity
    )
