"""
Value Objects - Immutable domain objects with no identity
Following V2 specification for value objects
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import hashlib


@dataclass(frozen=True)
class Embedding:
    """
    Immutable embedding value object
    As specified in V2 architecture docs
    """
    vector: Tuple[float, ...]  # Tuple for immutability
    model_name: str
    dimensions: int

    def __post_init__(self):
        """Validate embedding dimensions"""
        if len(self.vector) != self.dimensions:
            raise ValueError(f"Vector has {len(self.vector)} dimensions, expected {self.dimensions}")

    def similarity_to(self, other: 'Embedding') -> float:
        """Calculate cosine similarity with another embedding"""
        if self.dimensions != other.dimensions:
            raise ValueError("Embeddings must have same dimensions")

        dot_product = sum(a * b for a, b in zip(self.vector, other.vector))
        magnitude_self = sum(a * a for a in self.vector) ** 0.5
        magnitude_other = sum(b * b for b in other.vector) ** 0.5

        if magnitude_self * magnitude_other == 0:
            return 0.0

        return dot_product / (magnitude_self * magnitude_other)


@dataclass(frozen=True)
class AuthorityTier:
    """
    Value object representing compliance authority tier
    Immutable representation of tier with business meaning
    """
    level: int
    name: str
    description: str

    def __post_init__(self):
        """Validate tier level"""
        if self.level not in [1, 2, 3]:
            raise ValueError("Authority tier must be 1 (Regulatory), 2 (SOP), or 3 (Technical)")

    @classmethod
    def regulatory(cls) -> 'AuthorityTier':
        """Factory method for regulatory tier"""
        return cls(1, "Regulatory", "Government and regulatory requirements")

    @classmethod
    def sop(cls) -> 'AuthorityTier':
        """Factory method for SOP tier"""
        return cls(2, "SOP", "Standard Operating Procedures")

    @classmethod
    def technical(cls) -> 'AuthorityTier':
        """Factory method for technical tier"""
        return cls(3, "Technical", "Technical guidelines and best practices")

    def is_higher_than(self, other: 'AuthorityTier') -> bool:
        """Check if this tier has higher authority (lower number)"""
        return self.level < other.level


@dataclass(frozen=True)
class DocumentChecksum:
    """
    Value object for document integrity verification
    Immutable checksum that uniquely identifies content
    """
    algorithm: str
    hash_value: str

    @classmethod
    def from_content(cls, content: str, algorithm: str = "sha256") -> 'DocumentChecksum':
        """Create checksum from document content"""
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(content.encode('utf-8'))
        return cls(algorithm=algorithm, hash_value=hash_obj.hexdigest())

    def verify(self, content: str) -> bool:
        """Verify content matches this checksum"""
        other = DocumentChecksum.from_content(content, self.algorithm)
        return self.hash_value == other.hash_value


@dataclass(frozen=True)
class ComplianceScore:
    """
    Value object for compliance scoring
    Immutable score with business rules
    """
    value: float
    max_value: float = 1.0

    def __post_init__(self):
        """Validate score is within bounds"""
        if not 0 <= self.value <= self.max_value:
            raise ValueError(f"Score must be between 0 and {self.max_value}")

    @property
    def percentage(self) -> float:
        """Get score as percentage"""
        return (self.value / self.max_value) * 100

    @property
    def is_compliant(self) -> bool:
        """Business rule: 80% or higher is compliant"""
        return self.percentage >= 80

    @property
    def is_conditionally_compliant(self) -> bool:
        """Business rule: 50-79% is conditionally compliant"""
        return 50 <= self.percentage < 80

    @property
    def compliance_level(self) -> str:
        """Get compliance level as string"""
        if self.is_compliant:
            return "COMPLIANT"
        elif self.is_conditionally_compliant:
            return "CONDITIONALLY_COMPLIANT"
        else:
            return "NON_COMPLIANT"