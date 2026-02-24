#!/usr/bin/env python3
"""
Template Field Extractor
========================

Intelligent extraction of {template_fields} from:
1. criteria.json files - scoring and evaluation requirements
2. markdown prompts - embedded variables in templates
3. project_template_fields - defined in resources JSON

Uses json_scrubber for safe JSON parsing and validation.
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, asdict

# Import the json_scrubber utility
try:
    from tidyllm.utils.json_scrubber import safe_load_json_with_scrubbing, convert_json_to_python_values
    JSON_SCRUBBER_AVAILABLE = True
except ImportError:
    JSON_SCRUBBER_AVAILABLE = False

@dataclass
class TemplateField:
    """Represents a discovered template field"""
    name: str
    source: str  # 'criteria', 'markdown', 'resources'
    description: str
    data_type: str  # 'string', 'number', 'array', 'object'
    required: bool = True
    default_value: Any = None
    validation_rules: List[str] = None

    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = []

@dataclass
class FieldExtractionResult:
    """Results of template field extraction"""
    project_path: str
    fields: List[TemplateField]
    sources_scanned: Dict[str, bool]
    errors: List[str]
    total_fields: int

class TemplateFieldExtractor:
    """
    Extracts template fields from project structure:
    - criteria/*.json files
    - templates/*.md files
    - resources/*.json files with project_template_fields
    """

    def __init__(self, project_path: str):
        """Initialize extractor for a specific project"""
        self.project_path = Path(project_path)
        self.fields = {}  # field_name -> TemplateField
        self.errors = []

        # Regex pattern for extracting {field_name} from markdown
        self.field_pattern = re.compile(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}')

    def extract_all_fields(self) -> FieldExtractionResult:
        """Extract fields from all sources in the project"""
        sources = {
            'criteria': False,
            'templates': False,
            'resources': False
        }

        # Extract from criteria JSON files
        try:
            criteria_fields = self._extract_from_criteria()
            sources['criteria'] = len(criteria_fields) > 0
            for field in criteria_fields:
                self.fields[field.name] = field
        except Exception as e:
            self.errors.append(f"Criteria extraction failed: {e}")

        # Extract from markdown templates
        try:
            template_fields = self._extract_from_templates()
            sources['templates'] = len(template_fields) > 0
            for field in template_fields:
                if field.name not in self.fields:
                    self.fields[field.name] = field
                else:
                    # Merge information from multiple sources
                    self._merge_field_info(self.fields[field.name], field)
        except Exception as e:
            self.errors.append(f"Template extraction failed: {e}")

        # Extract from resources JSON
        try:
            resource_fields = self._extract_from_resources()
            sources['resources'] = len(resource_fields) > 0
            for field in resource_fields:
                if field.name not in self.fields:
                    self.fields[field.name] = field
                else:
                    self._merge_field_info(self.fields[field.name], field)
        except Exception as e:
            self.errors.append(f"Resource extraction failed: {e}")

        return FieldExtractionResult(
            project_path=str(self.project_path),
            fields=list(self.fields.values()),
            sources_scanned=sources,
            errors=self.errors,
            total_fields=len(self.fields)
        )

    def _extract_from_criteria(self) -> List[TemplateField]:
        """Extract fields from criteria/*.json files"""
        fields = []
        criteria_dir = self.project_path / "criteria"

        if not criteria_dir.exists():
            return fields

        for json_file in criteria_dir.glob("*.json"):
            try:
                # Use json_scrubber for safe loading
                if JSON_SCRUBBER_AVAILABLE:
                    result = safe_load_json_with_scrubbing(str(json_file))
                    if not result['success']:
                        self.errors.append(f"Failed to load {json_file}: {result['error']}")
                        continue
                    data = result['data']
                else:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                # Extract fields from criteria structure
                criteria_fields = self._parse_criteria_fields(data, json_file.name)
                fields.extend(criteria_fields)

            except Exception as e:
                self.errors.append(f"Error processing {json_file}: {e}")

        return fields

    def _extract_from_templates(self) -> List[TemplateField]:
        """Extract {field_name} patterns from templates/*.md files"""
        fields = []
        templates_dir = self.project_path / "templates"

        if not templates_dir.exists():
            return fields

        discovered_fields = set()

        for md_file in templates_dir.glob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Find all {field_name} patterns
                matches = self.field_pattern.findall(content)
                discovered_fields.update(matches)

            except Exception as e:
                self.errors.append(f"Error reading {md_file}: {e}")

        # Create TemplateField objects for discovered fields
        for field_name in discovered_fields:
            field = TemplateField(
                name=field_name,
                source='markdown',
                description=f"Template variable found in markdown files",
                data_type=self._infer_data_type(field_name),
                required=True
            )
            fields.append(field)

        return fields

    def _extract_from_resources(self) -> List[TemplateField]:
        """Extract from resources/*.json with project_template_fields"""
        fields = []
        resources_dir = self.project_path / "resources"

        if not resources_dir.exists():
            return fields

        for json_file in resources_dir.glob("*.json"):
            try:
                # Use json_scrubber for safe loading
                if JSON_SCRUBBER_AVAILABLE:
                    result = safe_load_json_with_scrubbing(str(json_file))
                    if not result['success']:
                        self.errors.append(f"Failed to load {json_file}: {result['error']}")
                        continue
                    data = result['data']
                else:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                # Look for project_template_fields
                if 'project_template_fields' in data:
                    template_fields_data = data['project_template_fields']
                    resource_fields = self._parse_template_fields_spec(template_fields_data)
                    fields.extend(resource_fields)

            except Exception as e:
                self.errors.append(f"Error processing {json_file}: {e}")

        return fields

    def _parse_criteria_fields(self, data: Dict, source_file: str) -> List[TemplateField]:
        """Parse fields from criteria JSON structure"""
        fields = []

        # Extract document qualifiers as template fields (only if template_field: true)
        if 'document_qualifiers' in data:
            qualifiers = data['document_qualifiers']
            for field_name, field_config in qualifiers.items():
                # Check if this should be a template field
                if isinstance(field_config, dict) and field_config.get('template_field', False):
                    field_value = field_config.get('value')

                    # Determine data type from value
                    data_type = 'string'
                    if isinstance(field_value, int):
                        data_type = 'integer'
                    elif isinstance(field_value, float):
                        data_type = 'number'
                    elif isinstance(field_value, bool):
                        data_type = 'boolean'
                    elif isinstance(field_value, list):
                        data_type = 'array'
                    elif isinstance(field_value, dict):
                        data_type = 'object'

                    field = TemplateField(
                        name=field_name,
                        source='criteria',
                        description=field_config.get('description', f"Document qualifier: {field_name}"),
                        data_type=data_type,
                        required=True,
                        default_value=field_value
                    )
                    fields.append(field)

        # Extract whitepaper fields as template fields (only if template_field: true)
        if 'whitepaper_fields' in data:
            whitepaper_fields = data['whitepaper_fields']
            for field_name, field_config in whitepaper_fields.items():
                # Check if this should be a template field
                if isinstance(field_config, dict) and field_config.get('template_field', False):
                    field_value = field_config.get('value')

                    data_type = 'string'
                    if isinstance(field_value, list):
                        data_type = 'array'
                    elif isinstance(field_value, dict):
                        data_type = 'object'

                    field = TemplateField(
                        name=field_name,
                        source='criteria',
                        description=field_config.get('description', f"Whitepaper field: {field_name}"),
                        data_type=data_type,
                        required=field_name in ['title', 'abstract', 'authors'],  # Core fields required
                        default_value=field_value
                    )
                    fields.append(field)

        # Look for explicit template_fields definitions
        if 'template_fields' in data:
            template_fields_data = data['template_fields']
            if isinstance(template_fields_data, dict):
                for field_name, field_spec in template_fields_data.items():
                    if isinstance(field_spec, dict):
                        # Extract validation rules
                        validation_rules = []
                        if 'range' in field_spec:
                            validation_rules.append(f"range: {field_spec['range']}")
                        if 'enum' in field_spec:
                            validation_rules.append(f"enum: {field_spec['enum']}")
                        if 'pattern' in field_spec:
                            validation_rules.append(f"pattern: {field_spec['pattern']}")
                        if 'max_length' in field_spec:
                            validation_rules.append(f"max_length: {field_spec['max_length']}")

                        field = TemplateField(
                            name=field_name,
                            source='criteria',
                            description=field_spec.get('description', f'Template field from {source_file}'),
                            data_type=field_spec.get('type', 'string'),
                            required=field_spec.get('required', True),
                            default_value=field_spec.get('default'),
                            validation_rules=validation_rules
                        )
                        fields.append(field)

        # Look for common criteria patterns (legacy support)
        if 'scoring_rubric' in data:
            # Scoring rubric implies these common fields
            common_fields = [
                ('quality_threshold', 'Minimum quality score required'),
                ('scoring_criteria', 'Criteria for evaluation'),
                ('weight_scheme', 'Weighting scheme for scoring')
            ]

            for field_name, desc in common_fields:
                field = TemplateField(
                    name=field_name,
                    source='criteria',
                    description=f"{desc} (from {source_file})",
                    data_type='number' if 'threshold' in field_name else 'object',
                    required=True
                )
                fields.append(field)

        return fields

    def _parse_template_fields_spec(self, spec_data: Dict) -> List[TemplateField]:
        """Parse project_template_fields specification"""
        fields = []

        for field_name, field_spec in spec_data.items():
            if isinstance(field_spec, dict):
                field = TemplateField(
                    name=field_name,
                    source='resources',
                    description=field_spec.get('description', f'Resource-defined field: {field_name}'),
                    data_type=field_spec.get('type', 'string'),
                    required=field_spec.get('required', True),
                    default_value=field_spec.get('default'),
                    validation_rules=field_spec.get('validation', [])
                )
                fields.append(field)
            else:
                # Simple field definition
                field = TemplateField(
                    name=field_name,
                    source='resources',
                    description=f'Resource-defined field: {field_name}',
                    data_type=self._infer_data_type_from_value(field_spec),
                    required=True,
                    default_value=field_spec
                )
                fields.append(field)

        return fields

    def _infer_data_type(self, field_name: str) -> str:
        """Infer data type from field name patterns"""
        name_lower = field_name.lower()

        if any(word in name_lower for word in ['count', 'number', 'score', 'threshold', 'limit']):
            return 'number'
        elif any(word in name_lower for word in ['files', 'items', 'list', 'array']):
            return 'array'
        elif any(word in name_lower for word in ['config', 'settings', 'params', 'criteria']):
            return 'object'
        elif any(word in name_lower for word in ['enabled', 'required', 'valid']):
            return 'boolean'
        else:
            return 'string'

    def _infer_data_type_from_value(self, value: Any) -> str:
        """Infer data type from actual value"""
        if isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, int):
            return 'number'
        elif isinstance(value, float):
            return 'number'
        elif isinstance(value, list):
            return 'array'
        elif isinstance(value, dict):
            return 'object'
        else:
            return 'string'

    def _merge_field_info(self, existing: TemplateField, new: TemplateField):
        """Merge information from multiple sources"""
        # Prefer more detailed descriptions
        if len(new.description) > len(existing.description):
            existing.description = new.description

        # Merge validation rules
        if new.validation_rules:
            existing.validation_rules.extend(new.validation_rules)
            existing.validation_rules = list(set(existing.validation_rules))  # Remove duplicates

        # Use default value if not set
        if existing.default_value is None and new.default_value is not None:
            existing.default_value = new.default_value

    def generate_field_definitions_json(self) -> str:
        """Generate JSON schema for extracted fields with proper boolean formatting"""
        result = self.extract_all_fields()

        schema = {
            "project": str(self.project_path.name),
            "extraction_summary": {
                "total_fields": result.total_fields,
                "sources_found": result.sources_scanned,
                "errors": result.errors
            },
            "project_template_fields": {}
        }

        for field in result.fields:
            schema["project_template_fields"][field.name] = {
                "description": field.description,
                "type": field.data_type,
                "source": field.source,
                "required": field.required,
                "default": field.default_value,
                "validation": field.validation_rules
            }

        # Apply Python value conversions if available, then format with proper JSON structure
        if JSON_SCRUBBER_AVAILABLE:
            schema = convert_json_to_python_values(schema)

        return json.dumps(schema, indent=2, ensure_ascii=True)

# =============================================================================
# CLI Interface and Testing
# =============================================================================

def extract_fields_for_project(project_path: str) -> FieldExtractionResult:
    """Extract template fields for a specific project"""
    extractor = TemplateFieldExtractor(project_path)
    return extractor.extract_all_fields()

def main():
    """CLI entry point for field extraction"""
    import argparse

    parser = argparse.ArgumentParser(description="Extract template fields from workflow projects")
    parser.add_argument("project_path", help="Path to workflow project directory")
    parser.add_argument("--output", "-o", help="Output JSON file for field definitions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    extractor = TemplateFieldExtractor(args.project_path)
    result = extractor.extract_all_fields()

    print(f"Template Field Extraction Results")
    print("=" * 50)
    print(f"Project: {result.project_path}")
    print(f"Total fields found: {result.total_fields}")
    print(f"Sources scanned: {result.sources_scanned}")

    if result.errors:
        print(f"\nErrors encountered:")
        for error in result.errors:
            print(f"  - {error}")

    if args.verbose:
        print(f"\nExtracted fields:")
        for field in result.fields:
            print(f"  {field.name} ({field.data_type}) - {field.description}")

    if args.output:
        schema_json = extractor.generate_field_definitions_json()
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(schema_json)
        print(f"\nField definitions saved to: {args.output}")

if __name__ == "__main__":
    main()