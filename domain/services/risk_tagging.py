"""
Simple Risk Tagging System
===========================

Embed risk ratings in code and docs for automatic compliance assessment.
"""

import re
from pathlib import Path
from typing import Dict, Optional

class RiskTagger:
    """Extract and assess risk from any file."""

    # Simple patterns to find risk tags
    RISK_PATTERNS = [
        r'@risk:\s*(\w+)',           # @risk: HIGH
        r'Risk Level:\s*(\w+)',      # Risk Level: HIGH
        r'RISK:(\w+)',               # RISK:HIGH
        r'"risk_level":\s*"(\w+)"'  # "risk_level": "HIGH"
    ]

    COMPLIANCE_PATTERNS = [
        r'@compliance:\s*([\w\-,\s]+)',     # @compliance: SR-11-7
        r'Compliance:\s*([\w\-,\s]+)',      # Compliance: SR 11-7
        r'"compliance_required":\s*\[(.*?)\]'  # "compliance_required": ["SR-11-7"]
    ]

    def tag_file(self, file_path: Path) -> Dict:
        """Extract risk tags from any file."""

        try:
            content = file_path.read_text()

            # Find risk level
            risk_level = None
            for pattern in self.RISK_PATTERNS:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    risk_level = match.group(1).upper()
                    break

            # Find compliance requirements
            compliance = []
            for pattern in self.COMPLIANCE_PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    compliance.extend([c.strip() for c in match.split(',')])

            # Calculate risk score
            risk_score = {
                'HIGH': 1.0,
                'MEDIUM': 0.5,
                'LOW': 0.2,
                'NONE': 0.0
            }.get(risk_level, 0.5)  # Default to medium if not tagged

            return {
                'file': str(file_path),
                'risk_level': risk_level or 'UNTAGGED',
                'risk_score': risk_score,
                'compliance': list(set(compliance)),  # Unique compliance items
                'needs_audit': risk_level == 'HIGH',
                'needs_review': risk_level in ['HIGH', 'MEDIUM']
            }

        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'risk_level': 'UNKNOWN',
                'risk_score': 0.5
            }

    def scan_directory(self, directory: Path) -> Dict:
        """Scan entire directory for risk tags."""

        results = {
            'high_risk_files': [],
            'medium_risk_files': [],
            'low_risk_files': [],
            'untagged_files': [],
            'total_risk_score': 0.0,
            'compliance_requirements': set(),
            'total_files_scanned': 0
        }

        # Scan Python files
        for py_file in directory.rglob('*.py'):
            results['total_files_scanned'] += 1
            tag_result = self.tag_file(py_file)

            # Categorize by risk
            if tag_result['risk_level'] == 'HIGH':
                results['high_risk_files'].append(tag_result)
            elif tag_result['risk_level'] == 'MEDIUM':
                results['medium_risk_files'].append(tag_result)
            elif tag_result['risk_level'] == 'LOW':
                results['low_risk_files'].append(tag_result)
            else:
                results['untagged_files'].append(tag_result)

            # Accumulate scores and compliance
            results['total_risk_score'] += tag_result.get('risk_score', 0)
            results['compliance_requirements'].update(tag_result.get('compliance', []))

        # Scan Markdown files
        for md_file in directory.rglob('*.md'):
            tag_result = self.tag_file(md_file)

            if tag_result['risk_level'] == 'HIGH':
                results['high_risk_files'].append(tag_result)
            elif tag_result['risk_level'] == 'MEDIUM':
                results['medium_risk_files'].append(tag_result)

        results['compliance_requirements'] = list(results['compliance_requirements'])

        return results

    def generate_compliance_report(self, directory: Path) -> str:
        """Generate simple compliance report."""

        results = self.scan_directory(directory)

        report = f"""
COMPLIANCE RISK REPORT
======================

HIGH RISK FILES: {len(results['high_risk_files'])}
{'='*50}
"""
        for item in results['high_risk_files']:
            report += f"  - {Path(item['file']).name}\n"
            if item.get('compliance'):
                report += f"    Compliance: {', '.join(item['compliance'])}\n"

        report += f"""

MEDIUM RISK FILES: {len(results['medium_risk_files'])}
{'='*50}
"""
        for item in results['medium_risk_files']:
            report += f"  - {Path(item['file']).name}\n"

        report += f"""

COMPLIANCE REQUIREMENTS FOUND:
{'='*50}
"""
        for req in results['compliance_requirements']:
            report += f"  - {req}\n"

        report += f"""

OVERALL RISK SCORE: {results['total_risk_score']:.1f}
UNTAGGED FILES: {len(results['untagged_files'])}

RECOMMENDATION: {'Tag all files with risk levels' if results['untagged_files'] else 'Good risk tagging coverage'}
"""

        return report


def demonstrate_simple_tagging():
    """Show how simple risk tagging works."""

    print("RISK TAGGING DEMONSTRATION")
    print("="*50)

    # Example tagged code
    example_code = '''
# Removed process_payment function - TidyLLM does not process payments
'''

    print("Example Tagged Code:")
    print(example_code)

    # Create temp file and scan it
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(example_code)
        temp_path = Path(f.name)

    tagger = RiskTagger()
    result = tagger.tag_file(temp_path)

    print("\nExtracted Risk Information:")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Risk Score: {result['risk_score']}")
    print(f"  Compliance: {', '.join(result['compliance'])}")
    print(f"  Needs Audit: {result['needs_audit']}")

    # Clean up
    temp_path.unlink()

    print("\n" + "="*50)
    print("This simple tagging lets you:")
    print("1. Track compliance requirements automatically")
    print("2. Identify high-risk code for review")
    print("3. Generate audit reports instantly")
    print("4. Score overall system risk")


if __name__ == "__main__":
    demonstrate_simple_tagging()

    # Scan actual v2 directory
    print("\n\nScanning v2 directory for risk tags...")
    tagger = RiskTagger()
    report = tagger.generate_compliance_report(Path("v2"))
    print(report)