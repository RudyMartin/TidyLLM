#!/usr/bin/env python3
"""
Generate README files for all workflows in the registry
=========================================================

Automatically creates comprehensive README.md files for each workflow
based on the template and actual workflow configuration data.

Usage:
    python generate_workflow_readmes.py                    # Generate all
    python generate_workflow_readmes.py --workflow mvr     # Generate specific
    python generate_workflow_readmes.py --force           # Overwrite existing
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from workflow_registry_system import WorkflowRegistrySystem
except ImportError:
    print("Error: Could not import WorkflowRegistrySystem")
    sys.exit(1)


class WorkflowREADMEGenerator:
    """Generate comprehensive README files for workflows."""

    def __init__(self):
        """Initialize the README generator."""
        self.registry = WorkflowRegistrySystem()
        self.template_path = Path("workflow_registry/WORKFLOW_README_TEMPLATE.md")
        self.workflows_dir = Path("workflow_registry/workflows")

        # Load template
        if self.template_path.exists():
            with open(self.template_path, 'r', encoding='utf-8') as f:
                self.template_content = f.read()
        else:
            print(f"Warning: Template not found at {self.template_path}")
            self.template_content = self._get_minimal_template()

    def _get_minimal_template(self) -> str:
        """Get minimal README template if main template missing."""
        return """# {WORKFLOW_NAME} Workflow

## Overview
**Workflow ID**: `{WORKFLOW_ID}`
**Type**: `{WORKFLOW_TYPE}`
**Status**: `{STATUS}`

{WORKFLOW_DESCRIPTION}

## Quick Start

### CLI Usage
```bash
python workflow_cli.py info {WORKFLOW_ID}
streamlit run unified_boss_portal.py
```

### Python API Usage
```python
from workflow_registry_system import WorkflowRegistrySystem
registry = WorkflowRegistrySystem()
workflow = registry.get_workflow("{WORKFLOW_ID}")
```

## Configuration
{CRITERIA_SETTINGS}

## Template Fields
{TEMPLATE_FIELDS_TABLE}

---
**Generated**: {LAST_UPDATED}
"""

    def get_workflow_data(self, workflow_id: str) -> Dict[str, Any]:
        """Extract workflow data for README generation."""
        workflow_dir = self.workflows_dir / workflow_id

        # Default data
        data = {
            'WORKFLOW_ID': workflow_id,
            'WORKFLOW_NAME': workflow_id.replace('_', ' ').title(),
            'WORKFLOW_TYPE': 'Unknown',
            'PRIORITY_LEVEL': 'normal',
            'STATUS': 'active',
            'WORKFLOW_DESCRIPTION': f'Workflow for {workflow_id} processing',
            'PROCESSING_STRATEGY': 'single_template',
            'ESTIMATED_DURATION': 1.0,
            'LAST_UPDATED': datetime.now().strftime('%Y-%m-%d'),
            'WORKFLOW_VERSION': '1.0'
        }

        # Load from criteria file
        criteria_file = workflow_dir / "criteria" / f"{workflow_id}_criteria.json"
        if criteria_file.exists():
            try:
                with open(criteria_file, 'r', encoding='utf-8') as f:
                    criteria = json.load(f)

                data['WORKFLOW_DESCRIPTION'] = criteria.get('description', data['WORKFLOW_DESCRIPTION'])
                data['WORKFLOW_TYPE'] = criteria.get('type', data['WORKFLOW_TYPE'])

                # Build criteria settings
                scoring_criteria = criteria.get('scoring_criteria', {})
                criteria_table = self._build_criteria_table(scoring_criteria)
                data['CRITERIA_SETTINGS'] = criteria_table

            except Exception as e:
                print(f"Warning: Could not load criteria for {workflow_id}: {e}")
                data['CRITERIA_SETTINGS'] = "No criteria configuration found."

        # Load from fields file
        fields_file = workflow_dir / "resources" / "fields.json"
        if fields_file.exists():
            try:
                with open(fields_file, 'r', encoding='utf-8') as f:
                    fields = json.load(f)

                # Build template fields table
                fields_used = fields.get('fields_used_in_template', [])
                fields_table = self._build_fields_table(fields_used)
                data['TEMPLATE_FIELDS_TABLE'] = fields_table

            except Exception as e:
                print(f"Warning: Could not load fields for {workflow_id}: {e}")
                data['TEMPLATE_FIELDS_TABLE'] = "No field information available."

        # Get workflow registry data
        try:
            workflows = self.registry.get_registered_workflows()
            workflow_info = next(
                (w for w in workflows if w['workflow_id'] == workflow_id),
                None
            )

            if workflow_info:
                data['PRIORITY_LEVEL'] = workflow_info.get('priority_level', data['PRIORITY_LEVEL'])
                data['STATUS'] = workflow_info.get('status', data['STATUS'])
                data['PROCESSING_STRATEGY'] = workflow_info.get('processing_strategy', data['PROCESSING_STRATEGY'])

        except Exception as e:
            print(f"Warning: Could not load registry data for {workflow_id}: {e}")

        # Add additional computed fields
        data.update(self._get_additional_fields(workflow_id, workflow_dir))

        return data

    def _build_criteria_table(self, scoring_criteria: Dict[str, Any]) -> str:
        """Build criteria settings table."""
        if not scoring_criteria:
            return "No scoring criteria defined."

        table = "| Criterion | Weight | Description |\n"
        table += "|-----------|--------|-------------|\n"

        for criterion, config in scoring_criteria.items():
            weight = config.get('weight', 0.0)
            description = config.get('description', 'No description')
            table += f"| {criterion.title()} | {weight:.2f} | {description} |\n"

        return table

    def _build_fields_table(self, fields_used: List[str]) -> str:
        """Build template fields table."""
        if not fields_used:
            return "No template fields defined."

        table = "| Field | Type | Description |\n"
        table += "|-------|------|-------------|\n"

        # Common field types and descriptions
        field_info = {
            'workflow_name': ('string', 'Name of the workflow'),
            'generated_date': ('date', 'Report generation timestamp'),
            'overall_score': ('number', 'Total calculated score'),
            'verdict': ('string', 'Final assessment result'),
            'model_id': ('string', 'Model identifier'),
            'compliance_score': ('number', 'Compliance assessment score'),
            'coverage_score': ('number', 'Coverage assessment score'),
            'clarity_score': ('number', 'Clarity assessment score'),
            'challenge_score': ('number', 'Challenge assessment score'),
            'missing_sections': ('array', 'List of missing sections'),
            'incomplete_sections': ('array', 'List of incomplete sections'),
            'immediate_actions': ('array', 'Priority action items')
        }

        for field in sorted(fields_used):
            field_type, description = field_info.get(field, ('string', 'Template field'))
            table += f"| `{field}` | {field_type} | {description} |\n"

        return table

    def _get_additional_fields(self, workflow_id: str, workflow_dir: Path) -> Dict[str, Any]:
        """Get additional fields for README generation."""
        additional = {}

        # Document types based on workflow type
        workflow_type = workflow_id.lower()
        if 'mvr' in workflow_type:
            additional['SUPPORTED_DOCUMENT_TYPES'] = "- PDF: Model validation reports\n- DOCX: Draft validation documents\n- XLSX: Model data and checklists"
            additional['REQUIRED_FIELDS_LIST'] = "- Model identifier\n- Validation sections\n- Risk classification\n- Validation dates"
            additional['OPTIONAL_FIELDS_LIST'] = "- Model owner\n- Business justification\n- Regulatory requirements"
            additional['OUTPUT_FIELDS_TABLE'] = "| Field | Description |\n|-------|-------------|\n| overall_score | Weighted assessment score |\n| verdict | PASS/FAIL/REVIEW recommendation |\n| missing_sections | Required sections not found |"
            additional['QUALITY_METRICS'] = "- Processing Time: < 30 seconds\n- Accuracy: 95%+ section identification\n- Coverage: 100% standard sections"
            additional['DOCUMENT_CRITERIA_JSON'] = json.dumps({
                "keywords": ["model validation", "mvr", "compliance"],
                "patterns": ["Model ID:", "Validation Date:", "Risk Rating:"],
                "min_pages": 5
            }, indent=2)
        else:
            # Generic defaults
            additional['SUPPORTED_DOCUMENT_TYPES'] = "- PDF: Primary document format\n- DOCX: Word documents\n- XLSX: Spreadsheet data"
            additional['REQUIRED_FIELDS_LIST'] = "- Document identifier\n- Content sections\n- Processing date"
            additional['OPTIONAL_FIELDS_LIST'] = "- Metadata\n- Tags\n- Custom fields"
            additional['OUTPUT_FIELDS_TABLE'] = "| Field | Description |\n|-------|-------------|\n| result | Processing result |\n| status | Execution status |"
            additional['QUALITY_METRICS'] = "- Processing Time: Variable\n- Accuracy: Workflow dependent\n- Coverage: Based on criteria"
            additional['DOCUMENT_CRITERIA_JSON'] = json.dumps({
                "keywords": [workflow_id],
                "patterns": ["Document:", "Date:"],
                "min_pages": 1
            }, indent=2)

        return additional

    def generate_readme(self, workflow_id: str, force: bool = False) -> bool:
        """Generate README for a specific workflow."""
        workflow_dir = self.workflows_dir / workflow_id
        readme_path = workflow_dir / "README.md"

        # Check if workflow directory exists
        if not workflow_dir.exists():
            print(f"Warning: Workflow directory not found: {workflow_dir}")
            return False

        # Check if README already exists
        if readme_path.exists() and not force:
            print(f"README already exists for {workflow_id}. Use --force to overwrite.")
            return False

        # Get workflow data
        try:
            data = self.get_workflow_data(workflow_id)
        except Exception as e:
            print(f"Error getting data for {workflow_id}: {e}")
            return False

        # Generate README content
        try:
            readme_content = self.template_content

            # Replace all placeholders
            for key, value in data.items():
                placeholder = f"{{{key}}}"
                readme_content = readme_content.replace(placeholder, str(value))

            # Write README file
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)

            print(f"[OK] Generated README for {workflow_id}")
            return True

        except Exception as e:
            print(f"Error generating README for {workflow_id}: {e}")
            return False

    def generate_all_readmes(self, force: bool = False) -> int:
        """Generate README files for all workflows."""
        if not self.workflows_dir.exists():
            print(f"Error: Workflows directory not found: {self.workflows_dir}")
            return 0

        workflow_dirs = [d for d in self.workflows_dir.iterdir() if d.is_dir()]

        if not workflow_dirs:
            print("No workflow directories found.")
            return 0

        generated_count = 0

        for workflow_dir in workflow_dirs:
            workflow_id = workflow_dir.name
            if self.generate_readme(workflow_id, force):
                generated_count += 1

        print(f"\n[SUCCESS] Generated {generated_count} README files")
        return generated_count

    def list_workflows(self) -> List[str]:
        """List all available workflows."""
        if not self.workflows_dir.exists():
            return []

        return [d.name for d in self.workflows_dir.iterdir() if d.is_dir()]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate README files for workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--workflow', '-w',
        help='Generate README for specific workflow'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Overwrite existing README files'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available workflows'
    )

    args = parser.parse_args()

    try:
        generator = WorkflowREADMEGenerator()
    except Exception as e:
        print(f"Error initializing generator: {e}")
        sys.exit(1)

    if args.list:
        workflows = generator.list_workflows()
        if workflows:
            print("Available workflows:")
            for workflow in workflows:
                print(f"  - {workflow}")
        else:
            print("No workflows found.")
        return

    if args.workflow:
        # Generate for specific workflow
        success = generator.generate_readme(args.workflow, args.force)
        if not success:
            sys.exit(1)
    else:
        # Generate for all workflows
        count = generator.generate_all_readmes(args.force)
        if count == 0:
            sys.exit(1)


if __name__ == '__main__':
    main()