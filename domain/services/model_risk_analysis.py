"""
Model Risk Management Document Analyzer
======================================

Real model risk management analysis using regulatory documents and industry guidance.
Extracts, classifies, and analyzes regulatory requirements for model validation.

Uses V2 documents module with V1 capabilities for professional analysis.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import re

# Add documents module
sys.path.append(str(Path(__file__).parent))

try:
    from documents import process_document, SimpleDocumentProcessor
    DOCUMENTS_AVAILABLE = True
except ImportError as e:
    DOCUMENTS_AVAILABLE = False
    print(f"ERROR: Documents module not available: {e}")

@dataclass
class ModelRiskAnalysis:
    """Analysis results for model risk management documents."""
    document_path: str
    document_type: str
    regulatory_source: str
    key_requirements: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    governance_requirements: List[str] = field(default_factory=list)
    testing_requirements: List[str] = field(default_factory=list)
    documentation_requirements: List[str] = field(default_factory=list)
    risk_categories: List[str] = field(default_factory=list)
    compliance_score: float = 0.0
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)

class ModelRiskAnalyzer:
    """Analyze regulatory documents for model risk management requirements."""
    
    def __init__(self):
        """Initialize the analyzer."""
        if not DOCUMENTS_AVAILABLE:
            raise ImportError("Documents module required for analysis")
        
        self.processor = SimpleDocumentProcessor()
        
        # Model risk management keywords and patterns
        self.patterns = {
            'regulatory_requirements': [
                r'shall\s+(?:establish|implement|maintain|document)',
                r'must\s+(?:establish|implement|maintain|document)',
                r'required?\s+to\s+(?:establish|implement|maintain)',
                r'institution\s+(?:shall|must|should)',
                r'board\s+(?:oversight|approval|responsibility)',
                r'senior\s+management\s+(?:oversight|responsibility)'
            ],
            'validation_criteria': [
                r'model\s+validation',
                r'back-?testing',
                r'benchmarking',
                r'sensitivity\s+analysis',
                r'stress\s+testing',
                r'performance\s+monitoring',
                r'outcome\s+analysis',
                r'challenger\s+model'
            ],
            'governance_requirements': [
                r'model\s+inventory',
                r'model\s+risk\s+management\s+framework',
                r'three\s+lines\s+of\s+defense',
                r'independent\s+validation',
                r'model\s+approval',
                r'model\s+committee',
                r'governance\s+structure'
            ],
            'testing_requirements': [
                r'conceptual\s+soundness',
                r'ongoing\s+monitoring',
                r'outcomes\s+analysis',
                r'process\s+verification',
                r'implementation\s+testing',
                r'data\s+quality',
                r'assumptions\s+testing'
            ],
            'documentation_requirements': [
                r'model\s+documentation',
                r'validation\s+report',
                r'model\s+development\s+documentation',
                r'user\s+guide',
                r'technical\s+specification',
                r'change\s+management',
                r'version\s+control'
            ]
        }
        
        # Regulatory source identification
        self.regulatory_sources = {
            'fed': ['federal reserve', 'fed', 'frb', 'board of governors'],
            'occ': ['occ', 'office of the comptroller', 'comptroller of the currency'],
            'fdic': ['fdic', 'federal deposit insurance'],
            'basel': ['basel', 'bcbs', 'basel committee'],
            'eba': ['eba', 'european banking authority'],
            'industry': ['acams', 'garp', 'prmia', 'rma'],
            'academic': ['arxiv', 'ssrn', 'journal', 'university']
        }
        
        # Model risk categories
        self.risk_categories = [
            'credit risk', 'market risk', 'operational risk', 'liquidity risk',
            'concentration risk', 'interest rate risk', 'stress testing',
            'capital planning', 'fair value', 'allowance estimation'
        ]
    
    def analyze_document(self, pdf_path: Path) -> ModelRiskAnalysis:
        """Analyze a single document for model risk management content."""
        print(f"Analyzing: {pdf_path.name}")
        
        # Extract document content
        result = process_document(str(pdf_path), max_pages=20, create_chunks=True)
        
        if not result.success or not result.text:
            print(f"  ERROR: Failed to extract text from {pdf_path.name}")
            return ModelRiskAnalysis(
                document_path=str(pdf_path),
                document_type="unknown",
                regulatory_source="unknown"
            )
        
        # Create analysis
        analysis = ModelRiskAnalysis(
            document_path=str(pdf_path),
            document_type=result.classification.get('document_type', 'unknown'),
            regulatory_source=self._identify_regulatory_source(pdf_path.name, result.text)
        )
        
        # Extract requirements and criteria
        analysis.key_requirements = self._extract_patterns(result.text, 'regulatory_requirements')
        analysis.validation_criteria = self._extract_patterns(result.text, 'validation_criteria')
        analysis.governance_requirements = self._extract_patterns(result.text, 'governance_requirements')
        analysis.testing_requirements = self._extract_patterns(result.text, 'testing_requirements')
        analysis.documentation_requirements = self._extract_patterns(result.text, 'documentation_requirements')
        
        # Identify risk categories
        analysis.risk_categories = self._identify_risk_categories(result.text)
        
        # Calculate compliance score
        analysis.compliance_score = self._calculate_compliance_score(analysis)
        
        # Store analysis metadata
        analysis.analysis_metadata = {
            'text_length': len(result.text),
            'chunks_created': len(result.chunks),
            'processing_time': result.processing_time,
            'classification_confidence': result.classification.get('confidence', 0),
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        print(f"  SUCCESS: Found {len(analysis.key_requirements)} requirements, "
              f"score: {analysis.compliance_score:.1%}")
        
        return analysis
    
    def _identify_regulatory_source(self, filename: str, text: str) -> str:
        """Identify the regulatory source of the document."""
        filename_lower = filename.lower()
        text_lower = text[:2000].lower()  # Check first 2000 chars
        
        for source, keywords in self.regulatory_sources.items():
            for keyword in keywords:
                if keyword in filename_lower or keyword in text_lower:
                    return source
        
        return "unknown"
    
    def _extract_patterns(self, text: str, pattern_type: str) -> List[str]:
        """Extract patterns from text based on pattern type."""
        if pattern_type not in self.patterns:
            return []
        
        found_items = []
        text_lower = text.lower()
        
        for pattern in self.patterns[pattern_type]:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Extract surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 100)
                context = text[start:end].strip()
                
                # Clean up the context
                context = re.sub(r'\s+', ' ', context)
                if len(context) > 10 and context not in found_items:
                    found_items.append(context)
        
        return found_items[:10]  # Limit to top 10 matches
    
    def _identify_risk_categories(self, text: str) -> List[str]:
        """Identify model risk categories mentioned in the document."""
        text_lower = text.lower()
        found_categories = []
        
        for category in self.risk_categories:
            if category in text_lower:
                found_categories.append(category)
        
        return found_categories
    
    def _calculate_compliance_score(self, analysis: ModelRiskAnalysis) -> float:
        """Calculate a compliance completeness score based on extracted requirements."""
        score = 0.0
        max_score = 6.0  # 6 categories
        
        # Score based on presence of different requirement types
        if analysis.key_requirements:
            score += 1.0
        if analysis.validation_criteria:
            score += 1.0  
        if analysis.governance_requirements:
            score += 1.0
        if analysis.testing_requirements:
            score += 1.0
        if analysis.documentation_requirements:
            score += 1.0
        if analysis.risk_categories:
            score += 1.0
        
        return score / max_score
    
    def analyze_knowledge_base(self, knowledge_base_path: Path) -> List[ModelRiskAnalysis]:
        """Analyze all documents in the knowledge base."""
        pdfs_path = knowledge_base_path / "pdfs"
        
        if not pdfs_path.exists():
            print(f"ERROR: PDFs directory not found: {pdfs_path}")
            return []
        
        pdf_files = list(pdfs_path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF documents for analysis")
        
        analyses = []
        for pdf_file in pdf_files[:10]:  # Limit to first 10 for this demo
            try:
                analysis = self.analyze_document(pdf_file)
                analyses.append(analysis)
            except Exception as e:
                print(f"ERROR analyzing {pdf_file.name}: {e}")
        
        return analyses
    
    def generate_regulatory_summary(self, analyses: List[ModelRiskAnalysis]) -> Dict[str, Any]:
        """Generate a comprehensive regulatory summary."""
        if not analyses:
            return {"error": "No analyses provided"}
        
        # Aggregate by regulatory source
        source_analysis = {}
        for analysis in analyses:
            source = analysis.regulatory_source
            if source not in source_analysis:
                source_analysis[source] = {
                    'documents': [],
                    'total_requirements': 0,
                    'avg_compliance_score': 0,
                    'key_themes': set()
                }
            
            source_analysis[source]['documents'].append(analysis.document_path)
            source_analysis[source]['total_requirements'] += len(analysis.key_requirements)
            source_analysis[source]['avg_compliance_score'] += analysis.compliance_score
            source_analysis[source]['key_themes'].update(analysis.risk_categories)
        
        # Calculate averages
        for source_data in source_analysis.values():
            doc_count = len(source_data['documents'])
            if doc_count > 0:
                source_data['avg_compliance_score'] /= doc_count
                source_data['key_themes'] = list(source_data['key_themes'])
        
        # Overall statistics
        total_docs = len(analyses)
        avg_score = sum(a.compliance_score for a in analyses) / total_docs if total_docs > 0 else 0
        
        return {
            'total_documents_analyzed': total_docs,
            'overall_avg_compliance_score': avg_score,
            'regulatory_sources': source_analysis,
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'top_risk_categories': self._get_top_risk_categories(analyses),
            'validation_coverage': self._assess_validation_coverage(analyses)
        }
    
    def _get_top_risk_categories(self, analyses: List[ModelRiskAnalysis]) -> List[Dict[str, Any]]:
        """Get top risk categories across all documents."""
        category_counts = {}
        for analysis in analyses:
            for category in analysis.risk_categories:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Sort by frequency
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'category': cat, 'document_count': count, 'frequency': count/len(analyses)}
            for cat, count in sorted_categories[:10]
        ]
    
    def _assess_validation_coverage(self, analyses: List[ModelRiskAnalysis]) -> Dict[str, Any]:
        """Assess how well validation requirements are covered."""
        validation_areas = [
            'conceptual soundness', 'ongoing monitoring', 'outcomes analysis',
            'back-testing', 'benchmarking', 'sensitivity analysis'
        ]
        
        coverage = {}
        for area in validation_areas:
            docs_covering = 0
            for analysis in analyses:
                area_text = ' '.join(analysis.validation_criteria).lower()
                if area.lower() in area_text:
                    docs_covering += 1
            
            coverage[area] = {
                'documents_covering': docs_covering,
                'coverage_percentage': docs_covering / len(analyses) if analyses else 0
            }
        
        return coverage

def main():
    """Main analysis function."""
    print("="*60)
    print("MODEL RISK MANAGEMENT DOCUMENT ANALYZER")
    print("="*60)
    
    if not DOCUMENTS_AVAILABLE:
        print("ERROR: Documents module not available")
        return 1
    
    # Initialize analyzer
    analyzer = ModelRiskAnalyzer()
    
    # Set paths
    knowledge_base_path = Path("C:/Users/marti/AI-Scoring/v2/knowledge_base")
    
    if not knowledge_base_path.exists():
        print(f"ERROR: Knowledge base not found: {knowledge_base_path}")
        return 1
    
    # Analyze documents
    print("\nAnalyzing regulatory documents...")
    analyses = analyzer.analyze_knowledge_base(knowledge_base_path)
    
    if not analyses:
        print("ERROR: No documents analyzed successfully")
        return 1
    
    # Generate summary
    print(f"\nGenerating regulatory summary for {len(analyses)} documents...")
    summary = analyzer.generate_regulatory_summary(analyses)
    
    # Display results
    print("\n" + "="*60)
    print("REGULATORY ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Total Documents: {summary['total_documents_analyzed']}")
    print(f"Overall Compliance Score: {summary['overall_avg_compliance_score']:.1%}")
    
    print(f"\nRegulatory Sources:")
    for source, data in summary['regulatory_sources'].items():
        print(f"  {source.upper()}: {len(data['documents'])} docs, "
              f"score: {data['avg_compliance_score']:.1%}")
    
    print(f"\nTop Risk Categories:")
    for cat_data in summary['top_risk_categories'][:5]:
        print(f"  {cat_data['category'].title()}: {cat_data['document_count']} docs "
              f"({cat_data['frequency']:.1%})")
    
    print(f"\nValidation Coverage:")
    for area, coverage in summary['validation_coverage'].items():
        print(f"  {area.title()}: {coverage['coverage_percentage']:.1%} coverage")
    
    # Save detailed results
    results_file = Path("model_risk_analysis_results.json")
    with open(results_file, 'w') as f:
        # Convert analyses to dicts for JSON serialization
        analyses_data = [
            {
                'document_path': a.document_path,
                'document_type': a.document_type,
                'regulatory_source': a.regulatory_source,
                'key_requirements': a.key_requirements,
                'validation_criteria': a.validation_criteria,
                'governance_requirements': a.governance_requirements,
                'testing_requirements': a.testing_requirements,
                'documentation_requirements': a.documentation_requirements,
                'risk_categories': a.risk_categories,
                'compliance_score': a.compliance_score,
                'analysis_metadata': a.analysis_metadata
            }
            for a in analyses
        ]
        
        json.dump({
            'summary': summary,
            'detailed_analyses': analyses_data
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)