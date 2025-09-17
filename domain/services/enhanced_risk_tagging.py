#!/usr/bin/env python3
"""
Enhanced Risk Tagging System
============================

Extended risk assessment for multiple file types including:
- Code files: .py, .js, .ts, .java, .cpp, .c, .go, .rs
- Documentation: .md, .txt, .rst, .adoc
- Web files: .html, .htm, .xml, .json, .yaml, .yml
- Data files: .csv, .tsv, .json, .xml
- Office files: .docx, .doc, .xlsx, .xls, .pptx, .ppt
- PDF files: .pdf
- Configuration: .ini, .cfg, .conf, .properties, .env
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import mimetypes

@dataclass
class FileRiskAssessment:
    """Risk assessment for a single file."""
    file_path: str
    file_type: str
    mime_type: str
    risk_level: str
    risk_score: float
    compliance_requirements: List[str]
    security_concerns: List[str]
    privacy_concerns: List[str]
    data_classification: str
    needs_audit: bool
    needs_review: bool
    file_size_mb: float
    line_count: int
    assessment_metadata: Dict

class EnhancedRiskTagger:
    """Enhanced risk tagging for multiple file types."""

    # Risk patterns for different file types
    RISK_PATTERNS = {
        'general': [
            r'@risk:\s*(\w+)',           # @risk: HIGH
            r'Risk Level:\s*(\w+)',      # Risk Level: HIGH
            r'RISK:(\w+)',               # RISK:HIGH
            r'"risk_level":\s*"(\w+)"'  # "risk_level": "HIGH"
        ],
        'compliance': [
            r'@compliance:\s*([\w\-,\s]+)',     # @compliance: SR-11-7
            r'Compliance:\s*([\w\-,\s]+)',      # Compliance: SR 11-7
            r'"compliance_required":\s*\[(.*?)\]'  # "compliance_required": ["SR-11-7"]
        ],
        'security': [
            r'password\s*=\s*["\']([^"\']+)["\']',  # password = "secret"
            r'api_key\s*=\s*["\']([^"\']+)["\']',   # api_key = "key123"
            r'secret\s*=\s*["\']([^"\']+)["\']',    # secret = "secret"
            r'token\s*=\s*["\']([^"\']+)["\']',     # token = "token123"
            r'private_key',                          # private_key
            r'public_key',                           # public_key
            r'certificate',                          # certificate
            r'credential',                           # credential
        ],
        'privacy': [
            r'pii\b',                               # PII
            r'personal\s+information',               # personal information
            r'personally\s+identifiable',            # personally identifiable
            r'social\s+security\s+number',           # social security number
            r'ssn\b',                               # SSN
            r'credit\s+card\s+number',               # credit card number
            r'bank\s+account\s+number',              # bank account number
            r'email\s+address',                      # email address
            r'phone\s+number',                       # phone number
            r'date\s+of\s+birth',                    # date of birth
            r'gdpr\b',                              # GDPR
            r'ccpa\b',                              # CCPA
            r'hipaa\b',                             # HIPAA
        ],
        'financial': [
            r'payment\s+processing',                 # payment processing
            r'financial\s+transaction',              # financial transaction
            r'money\s+transfer',                     # money transfer
            r'banking\s+information',                # banking information
            r'account\s+balance',                    # account balance
            r'transaction\s+history',                # transaction history
            r'credit\s+score',                       # credit score
            r'loan\s+application',                   # loan application
        ]
    }

    # File type categories
    FILE_CATEGORIES = {
        'code': ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb', '.swift', '.kt'],
        'documentation': ['.md', '.txt', '.rst', '.adoc', '.tex'],
        'web': ['.html', '.htm', '.xml', '.json', '.yaml', '.yml', '.toml'],
        'data': ['.csv', '.tsv', '.json', '.xml', '.parquet', '.avro'],
        'office': ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.odt', '.ods', '.odp'],
        'pdf': ['.pdf'],
        'config': ['.ini', '.cfg', '.conf', '.properties', '.env', '.config'],
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.tiff'],
        'archive': ['.zip', '.tar', '.gz', '.rar', '.7z']
    }

    # Data classification levels
    DATA_CLASSIFICATION = {
        'PUBLIC': 0.1,
        'INTERNAL': 0.3,
        'CONFIDENTIAL': 0.7,
        'RESTRICTED': 1.0
    }

    def __init__(self):
        """Initialize the enhanced risk tagger."""
        self.assessed_files = {}
        self.total_risk_score = 0.0
        self.file_type_stats = {category: {'count': 0, 'risk_score': 0.0} 
                               for category in self.FILE_CATEGORIES.keys()}

    def get_file_category(self, file_path: Path) -> str:
        """Determine the category of a file based on its extension."""
        suffix = file_path.suffix.lower()
        
        for category, extensions in self.FILE_CATEGORIES.items():
            if suffix in extensions:
                return category
        
        return 'unknown'

    def get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for a file."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or 'application/octet-stream'

    def extract_text_content(self, file_path: Path) -> str:
        """Extract text content from various file types."""
        try:
            if file_path.suffix.lower() in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', 
                                          '.md', '.txt', '.rst', '.adoc', '.html', '.htm', '.xml', 
                                          '.json', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.env']:
                # Text files - read directly
                return file_path.read_text(encoding='utf-8', errors='ignore')
            
            elif file_path.suffix.lower() in ['.csv', '.tsv']:
                # CSV/TSV files - read as text
                return file_path.read_text(encoding='utf-8', errors='ignore')
            
            elif file_path.suffix.lower() == '.pdf':
                # PDF files - would need PyPDF2 or similar
                # For now, return placeholder
                return f"[PDF file: {file_path.name}]"
            
            elif file_path.suffix.lower() in ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt']:
                # Office files - would need python-docx, openpyxl, etc.
                # For now, return placeholder
                return f"[Office file: {file_path.name}]"
            
            else:
                # Binary or unknown files
                return f"[Binary file: {file_path.name}]"
                
        except Exception as e:
            return f"[Error reading file: {e}]"

    def assess_file_risk(self, file_path: Path) -> FileRiskAssessment:
        """Comprehensive risk assessment for a single file."""
        
        # Basic file information
        file_category = self.get_file_category(file_path)
        mime_type = self.get_mime_type(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0
        
        # Extract content
        content = self.extract_text_content(file_path)
        line_count = len(content.split('\n')) if content else 0
        
        # Risk level detection
        risk_level = self._detect_risk_level(content)
        risk_score = self._calculate_risk_score(risk_level, file_category, file_size_mb)
        
        # Compliance requirements
        compliance_requirements = self._extract_compliance_requirements(content)
        
        # Security concerns
        security_concerns = self._detect_security_concerns(content)
        
        # Privacy concerns
        privacy_concerns = self._detect_privacy_concerns(content)
        
        # Data classification
        data_classification = self._classify_data(content, file_category)
        
        # Assessment flags
        needs_audit = risk_level == 'HIGH' or len(security_concerns) > 0
        needs_review = risk_level in ['HIGH', 'MEDIUM'] or len(privacy_concerns) > 0
        
        return FileRiskAssessment(
            file_path=str(file_path),
            file_type=file_category,
            mime_type=mime_type,
            risk_level=risk_level,
            risk_score=risk_score,
            compliance_requirements=compliance_requirements,
            security_concerns=security_concerns,
            privacy_concerns=privacy_concerns,
            data_classification=data_classification,
            needs_audit=needs_audit,
            needs_review=needs_review,
            file_size_mb=file_size_mb,
            line_count=line_count,
            assessment_metadata={
                'assessment_timestamp': '2024-01-01T00:00:00Z',  # Would use datetime.utcnow()
                'assessor_version': '1.0.0',
                'content_preview': content[:200] if content else ''
            }
        )

    def _detect_risk_level(self, content: str) -> str:
        """Detect risk level from content."""
        content_lower = content.lower()
        
        # Check for explicit risk tags
        for pattern in self.RISK_PATTERNS['general']:
            match = re.search(pattern, content_lower, re.IGNORECASE)
            if match:
                risk_level = match.group(1).upper()
                if risk_level in ['HIGH', 'MEDIUM', 'LOW', 'NONE']:
                    return risk_level
        
        # Default risk assessment based on content
        if any(keyword in content_lower for keyword in ['password', 'secret', 'private_key', 'api_key']):
            return 'HIGH'
        elif any(keyword in content_lower for keyword in ['pii', 'personal information', 'gdpr', 'ccpa']):
            return 'MEDIUM'
        else:
            return 'UNTAGGED'

    def _calculate_risk_score(self, risk_level: str, file_category: str, file_size_mb: float) -> float:
        """Calculate risk score based on multiple factors."""
        base_scores = {
            'HIGH': 1.0,
            'MEDIUM': 0.5,
            'LOW': 0.2,
            'NONE': 0.0,
            'UNTAGGED': 0.5
        }
        
        base_score = base_scores.get(risk_level, 0.5)
        
        # Adjust for file category
        category_multipliers = {
            'code': 1.0,
            'config': 1.2,  # Config files are more sensitive
            'data': 1.1,    # Data files may contain sensitive info
            'documentation': 0.8,
            'web': 0.9,
            'office': 1.0,
            'pdf': 1.0,
            'unknown': 1.0
        }
        
        multiplier = category_multipliers.get(file_category, 1.0)
        
        # Adjust for file size (larger files might be more risky)
        size_factor = min(1.2, 1.0 + (file_size_mb / 100))  # Cap at 1.2x for very large files
        
        return min(1.0, base_score * multiplier * size_factor)

    def _extract_compliance_requirements(self, content: str) -> List[str]:
        """Extract compliance requirements from content."""
        requirements = []
        content_lower = content.lower()
        
        for pattern in self.RISK_PATTERNS['compliance']:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                requirements.extend([req.strip() for req in match.split(',')])
        
        # Also look for common compliance frameworks
        compliance_frameworks = [
            'sox', 'pci-dss', 'gdpr', 'ccpa', 'hipaa', 'ferpa', 'glba',
            'basel', 'sr-11-7', 'sr-15-18', 'ccar', 'dfast'
        ]
        
        for framework in compliance_frameworks:
            if framework in content_lower:
                requirements.append(framework.upper())
        
        return list(set(requirements))  # Remove duplicates

    def _detect_security_concerns(self, content: str) -> List[str]:
        """Detect security concerns in content."""
        concerns = []
        content_lower = content.lower()
        
        for pattern in self.RISK_PATTERNS['security']:
            if re.search(pattern, content_lower, re.IGNORECASE):
                concerns.append(f"Potential security issue: {pattern}")
        
        return concerns

    def _detect_privacy_concerns(self, content: str) -> List[str]:
        """Detect privacy concerns in content."""
        concerns = []
        content_lower = content.lower()
        
        for pattern in self.RISK_PATTERNS['privacy']:
            if re.search(pattern, content_lower, re.IGNORECASE):
                concerns.append(f"Privacy concern: {pattern}")
        
        return concerns

    def _classify_data(self, content: str, file_category: str) -> str:
        """Classify data sensitivity level."""
        content_lower = content.lower()
        
        # Check for restricted data
        if any(keyword in content_lower for keyword in ['password', 'secret', 'private_key', 'ssn', 'credit card']):
            return 'RESTRICTED'
        
        # Check for confidential data
        elif any(keyword in content_lower for keyword in ['pii', 'personal information', 'gdpr', 'ccpa']):
            return 'CONFIDENTIAL'
        
        # Check for internal data
        elif any(keyword in content_lower for keyword in ['internal', 'proprietary', 'confidential']):
            return 'INTERNAL'
        
        # Default to public
        else:
            return 'PUBLIC'

    def scan_directory(self, directory: Path) -> Dict:
        """Scan entire directory for risk assessment."""
        print(f"Scanning directory: {directory}")
        
        results = {
            'total_files_scanned': 0,
            'file_type_breakdown': {},
            'risk_level_breakdown': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNTAGGED': 0},
            'data_classification_breakdown': {'PUBLIC': 0, 'INTERNAL': 0, 'CONFIDENTIAL': 0, 'RESTRICTED': 0},
            'total_risk_score': 0.0,
            'files_needing_audit': [],
            'files_needing_review': [],
            'compliance_requirements': set(),
            'security_concerns': [],
            'privacy_concerns': [],
            'detailed_assessments': []
        }
        
        # Scan all supported file types
        supported_extensions = []
        for extensions in self.FILE_CATEGORIES.values():
            supported_extensions.extend(extensions)
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    assessment = self.assess_file_risk(file_path)
                    results['detailed_assessments'].append(assessment)
                    results['total_files_scanned'] += 1
                    
                    # Update breakdowns
                    file_type = assessment.file_type
                    if file_type not in results['file_type_breakdown']:
                        results['file_type_breakdown'][file_type] = 0
                    results['file_type_breakdown'][file_type] += 1
                    
                    results['risk_level_breakdown'][assessment.risk_level] += 1
                    results['data_classification_breakdown'][assessment.data_classification] += 1
                    
                    results['total_risk_score'] += assessment.risk_score
                    
                    if assessment.needs_audit:
                        results['files_needing_audit'].append(assessment.file_path)
                    
                    if assessment.needs_review:
                        results['files_needing_review'].append(assessment.file_path)
                    
                    results['compliance_requirements'].update(assessment.compliance_requirements)
                    results['security_concerns'].extend(assessment.security_concerns)
                    results['privacy_concerns'].extend(assessment.privacy_concerns)
                    
                except Exception as e:
                    print(f"Error assessing {file_path}: {e}")
                    continue
        
        # Convert set to list for JSON serialization
        results['compliance_requirements'] = list(results['compliance_requirements'])
        
        return results

    def generate_regulatory_overview(self, scan_results: Dict) -> Dict:
        """Generate comprehensive regulatory overview."""
        return {
            'executive_summary': {
                'total_files_assessed': scan_results['total_files_scanned'],
                'overall_risk_level': self._calculate_overall_risk_level(scan_results),
                'compliance_readiness': self._assess_compliance_readiness(scan_results),
                'data_protection_status': self._assess_data_protection(scan_results)
            },
            'regulatory_frameworks': {
                'identified_frameworks': scan_results['compliance_requirements'],
                'coverage_assessment': self._assess_framework_coverage(scan_results),
                'gap_analysis': self._identify_compliance_gaps(scan_results)
            },
            'risk_assessment': {
                'file_type_risks': scan_results['file_type_breakdown'],
                'data_classification_risks': scan_results['data_classification_breakdown'],
                'security_concerns': scan_results['security_concerns'],
                'privacy_concerns': scan_results['privacy_concerns']
            },
            'audit_readiness': {
                'files_requiring_audit': len(scan_results['files_needing_audit']),
                'files_requiring_review': len(scan_results['files_needing_review']),
                'audit_priority_files': scan_results['files_needing_audit'][:10]  # Top 10
            },
            'recommendations': self._generate_regulatory_recommendations(scan_results)
        }

    def _calculate_overall_risk_level(self, scan_results: Dict) -> str:
        """Calculate overall risk level."""
        total_files = scan_results['total_files_scanned']
        if total_files == 0:
            return 'UNKNOWN'
        
        high_risk_ratio = scan_results['risk_level_breakdown']['HIGH'] / total_files
        medium_risk_ratio = scan_results['risk_level_breakdown']['MEDIUM'] / total_files
        
        if high_risk_ratio > 0.1:  # More than 10% high risk
            return 'HIGH'
        elif medium_risk_ratio > 0.3:  # More than 30% medium risk
            return 'MEDIUM'
        else:
            return 'LOW'

    def _assess_compliance_readiness(self, scan_results: Dict) -> str:
        """Assess compliance readiness."""
        compliance_count = len(scan_results['compliance_requirements'])
        total_files = scan_results['total_files_scanned']
        
        if compliance_count == 0:
            return 'NO_COMPLIANCE_FRAMEWORKS_IDENTIFIED'
        elif compliance_count >= 3:
            return 'GOOD_COMPLIANCE_COVERAGE'
        else:
            return 'LIMITED_COMPLIANCE_COVERAGE'

    def _assess_data_protection(self, scan_results: Dict) -> str:
        """Assess data protection status."""
        restricted_count = scan_results['data_classification_breakdown']['RESTRICTED']
        confidential_count = scan_results['data_classification_breakdown']['CONFIDENTIAL']
        
        if restricted_count > 0:
            return 'HIGH_SENSITIVITY_DATA_PRESENT'
        elif confidential_count > 0:
            return 'CONFIDENTIAL_DATA_PRESENT'
        else:
            return 'LOW_SENSITIVITY_DATA_ONLY'

    def _assess_framework_coverage(self, scan_results: Dict) -> Dict:
        """Assess coverage of regulatory frameworks."""
        frameworks = scan_results['compliance_requirements']
        coverage = {}
        
        for framework in frameworks:
            framework_lower = framework.lower()
            if 'sox' in framework_lower:
                coverage['SOX'] = 'IDENTIFIED'
            elif 'pci' in framework_lower:
                coverage['PCI-DSS'] = 'IDENTIFIED'
            elif 'gdpr' in framework_lower:
                coverage['GDPR'] = 'IDENTIFIED'
            elif 'ccpa' in framework_lower:
                coverage['CCPA'] = 'IDENTIFIED'
            elif 'hipaa' in framework_lower:
                coverage['HIPAA'] = 'IDENTIFIED'
        
        return coverage

    def _identify_compliance_gaps(self, scan_results: Dict) -> List[str]:
        """Identify compliance gaps."""
        gaps = []
        frameworks = [f.lower() for f in scan_results['compliance_requirements']]
        
        # Check for common missing frameworks
        if 'sox' not in frameworks:
            gaps.append('SOX compliance not identified')
        if 'pci' not in frameworks:
            gaps.append('PCI-DSS compliance not identified')
        if 'gdpr' not in frameworks:
            gaps.append('GDPR compliance not identified')
        
        return gaps

    def _generate_regulatory_recommendations(self, scan_results: Dict) -> List[Dict]:
        """Generate regulatory recommendations."""
        recommendations = []
        
        # High-risk files
        high_risk_count = scan_results['risk_level_breakdown']['HIGH']
        if high_risk_count > 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'RISK_MANAGEMENT',
                'issue': f'{high_risk_count} high-risk files identified',
                'recommendation': 'Implement comprehensive risk mitigation for high-risk files',
                'action_items': [
                    'Review all high-risk files',
                    'Implement additional security controls',
                    'Document risk acceptance decisions'
                ]
            })
        
        # Compliance gaps
        if len(scan_results['compliance_requirements']) == 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'COMPLIANCE',
                'issue': 'No compliance frameworks identified',
                'recommendation': 'Implement compliance framework identification',
                'action_items': [
                    'Tag files with applicable compliance requirements',
                    'Implement compliance monitoring',
                    'Create compliance documentation'
                ]
            })
        
        # Data classification
        restricted_count = scan_results['data_classification_breakdown']['RESTRICTED']
        if restricted_count > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'DATA_PROTECTION',
                'issue': f'{restricted_count} files contain restricted data',
                'recommendation': 'Implement enhanced data protection controls',
                'action_items': [
                    'Encrypt restricted data files',
                    'Implement access controls',
                    'Create data handling procedures'
                ]
            })
        
        return recommendations

def main():
    """Demonstrate enhanced risk tagging."""
    print("ENHANCED RISK TAGGING SYSTEM")
    print("=" * 50)
    
    tagger = EnhancedRiskTagger()
    
    # Test with current directory
    test_dir = Path(".")
    results = tagger.scan_directory(test_dir)
    
    print(f"\nScan Results:")
    print(f"Total files scanned: {results['total_files_scanned']}")
    print(f"File types: {results['file_type_breakdown']}")
    print(f"Risk levels: {results['risk_level_breakdown']}")
    print(f"Data classifications: {results['data_classification_breakdown']}")
    print(f"Total risk score: {results['total_risk_score']:.2f}")
    
    # Generate regulatory overview
    regulatory_overview = tagger.generate_regulatory_overview(results)
    
    print(f"\nRegulatory Overview:")
    print(f"Overall risk level: {regulatory_overview['executive_summary']['overall_risk_level']}")
    print(f"Compliance readiness: {regulatory_overview['executive_summary']['compliance_readiness']}")
    print(f"Data protection status: {regulatory_overview['executive_summary']['data_protection_status']}")
    
    # Save results
    with open('enhanced_risk_assessment.json', 'w') as f:
        json.dump({
            'scan_results': results,
            'regulatory_overview': regulatory_overview
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: enhanced_risk_assessment.json")

if __name__ == "__main__":
    main()
