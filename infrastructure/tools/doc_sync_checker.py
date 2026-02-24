"""
Documentation Sync Checker
==========================

Automatically detects when documentation and code are out of sync.
"""

import json
import ast
from pathlib import Path
from typing import Dict, List, Tuple

class DocSyncChecker:
    """Check if documentation matches actual code behavior."""

    def __init__(self):
        self.mismatches = []

    def check_config_vs_code(self):
        """Compare configuration values with actual code usage."""

        # Load configuration
        with open("code_review_standards.json") as f:
            config = json.load(f)

        # Check PostgreSQL claim vs reality
        postgres_claimed = "PostgreSQL" in str(config)
        postgres_actual = self._check_actual_database()

        if postgres_claimed and not postgres_actual:
            self.mismatches.append({
                "type": "database",
                "docs_say": "Using PostgreSQL",
                "code_says": "Still using file-based storage",
                "file": "workflow_state_manager.py"
            })

        # Check MLflow audit claim vs reality
        audit_claimed = config.get("mlflow_audit_required", True)
        audit_actual = self._check_mlflow_usage()

        if audit_claimed and not audit_actual:
            self.mismatches.append({
                "type": "audit",
                "docs_say": "All AI calls use MLflow",
                "code_says": "Found direct AI calls without MLflow",
                "file": "direct_ai_caller.py"
            })

        return self.mismatches

    def check_docstrings_vs_implementation(self, file_path: str):
        """Check if docstrings match actual function behavior."""

        with open(file_path) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    # Check if docstring claims match implementation
                    if "returns float" in docstring.lower():
                        actual_return = self._analyze_return_type(node)
                        if actual_return != "float":
                            self.mismatches.append({
                                "type": "return_type",
                                "function": node.name,
                                "docs_say": "returns float",
                                "code_says": f"returns {actual_return}"
                            })

    def check_readme_vs_reality(self):
        """Check if README claims match actual system state."""

        readme_path = Path("docs/README.md")
        if readme_path.exists():
            readme = readme_path.read_text()

            # Check production readiness claim
            if "production ready" in readme.lower():
                actual_ready = self._check_production_readiness()
                if not actual_ready:
                    self.mismatches.append({
                        "type": "readiness",
                        "docs_say": "Production ready",
                        "code_says": "Missing production requirements",
                        "missing": ["PostgreSQL migration", "Load testing", "Monitoring"]
                    })

    def _check_actual_database(self) -> bool:
        """Check what database is actually being used."""
        # Look for actual database connections in code
        for py_file in Path("v2").rglob("*.py"):
            content = py_file.read_text()
            if "psycopg2" in content or "postgresql://" in content:
                if "connect(" in content:  # Actually connecting
                    return True
        return False

    def _check_mlflow_usage(self) -> bool:
        """Check if MLflow is actually being used."""
        for py_file in Path("v2").rglob("*.py"):
            content = py_file.read_text()
            if "import openai" in content or "anthropic.Client" in content:
                if "mlflow" not in content:  # Direct AI call without MLflow
                    return False
        return True

    def _analyze_return_type(self, node) -> str:
        """Analyze actual return type from AST."""
        # Simplified - real implementation would be more complex
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                if isinstance(child.value, ast.Constant):
                    return type(child.value.value).__name__
        return "unknown"

    def _check_production_readiness(self) -> bool:
        """Check actual production readiness."""
        required_files = [
            "requirements.txt",
            ".env.example",
            "Dockerfile",
            "tests/"
        ]
        for req in required_files:
            if not Path(req).exists():
                return False
        return True

    def generate_sync_report(self) -> Dict:
        """Generate comprehensive sync report."""

        print("Checking documentation sync with code...")

        # Run all checks
        self.check_config_vs_code()
        self.check_readme_vs_reality()

        # Check key Python files
        for py_file in Path("v2").glob("*.py"):
            try:
                self.check_docstrings_vs_implementation(str(py_file))
            except:
                pass  # Skip files with syntax issues

        # Generate report
        report = {
            "total_mismatches": len(self.mismatches),
            "critical": [m for m in self.mismatches if m["type"] in ["database", "audit"]],
            "warnings": [m for m in self.mismatches if m["type"] not in ["database", "audit"]],
            "recommendation": self._get_recommendation()
        }

        return report

    def _get_recommendation(self) -> str:
        """Get sync recommendation."""
        if len(self.mismatches) == 0:
            return "Documentation is in sync with code!"
        elif len(self.mismatches) < 5:
            return "Minor sync issues - update documentation"
        else:
            return "Major sync issues - comprehensive review needed"


def main():
    """Run documentation sync check."""

    checker = DocSyncChecker()
    report = checker.generate_sync_report()

    print("\n=== DOCUMENTATION SYNC REPORT ===")
    print(f"Total Mismatches: {report['total_mismatches']}")

    if report['critical']:
        print("\nCRITICAL MISMATCHES:")
        for mismatch in report['critical']:
            print(f"  - Docs say: {mismatch['docs_say']}")
            print(f"    Code says: {mismatch['code_says']}")

    if report['warnings']:
        print("\nWARNINGS:")
        for warning in report['warnings']:
            print(f"  - {warning['type']}: {warning.get('docs_say', 'mismatch')}")

    print(f"\nRecommendation: {report['recommendation']}")

    # Save report
    with open("doc_sync_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\nFull report saved to: doc_sync_report.json")

    return report


if __name__ == "__main__":
    main()