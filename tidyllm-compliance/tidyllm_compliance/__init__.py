"""
tidyllm-compliance: Automated compliance monitoring with complete transparency

This package provides specialized compliance monitoring capabilities:
1. Model Risk Development Standards (SR 11-7, OCC guidance)
2. Evidence Validation (authenticity and completeness assessment)  
3. Argument Consistency Analysis (review scope determination)
4. SOP Golden Answers (validated Standard Operating Procedures)
5. SOP Conflict Analysis (YRSN-validated conflict detection and resolution)

Part of the tidyllm-verse: Educational ML tools with complete algorithmic transparency.
"""

from .model_risk import ModelRiskMonitor
from .evidence import EvidenceValidator  
from .consistency import ConsistencyAnalyzer
from .sop_golden_answers import SOPValidator
from .sop_conflict_analysis import SOPConflictReporter, YRSNNoiseAnalyzer, TemporalResolver, ComplianceSOPFallback

__version__ = "0.1.0"
__author__ = "Rudy Martin"

__all__ = [
    "ModelRiskMonitor",
    "EvidenceValidator", 
    "ConsistencyAnalyzer",
    "SOPValidator",
    "SOPConflictReporter",
    "YRSNNoiseAnalyzer",
    "TemporalResolver", 
    "ComplianceSOPFallback"
]

# Package metadata
DESCRIPTION = "Automated compliance monitoring with complete algorithmic transparency"
LICENSE = "CC-BY-4.0"
HOMEPAGE = "https://github.com/tidyllm-verse/tidyllm-compliance"