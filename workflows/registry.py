"""
Workflow Registry System - Criterion & Markdown Template Management
===================================================================

@risk: HIGH
@compliance: SR-11-7, SOX-404
@audit: REQUIRED

Central registry for all workflows with:
- Criterion files (JSON) defining validation rules and scoring
- Markdown templates for prompts and documentation
- Admin-configurable through onboarding interface
- Integration with existing Polars processors

This unifies v1 registered workflows with v2 clean architecture.

Risk Note: HIGH - Controls all workflow execution and regulatory compliance validation
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import shutil

@dataclass
class WorkflowCriteria:
    """Criteria definition for a workflow."""
    id: str
    name: str
    description: str
    
    # Scoring criteria
    scoring_rubric: Dict[str, float] = field(default_factory=dict)
    weight_scheme: Dict[str, float] = field(default_factory=dict)
    threshold_values: Dict[str, float] = field(default_factory=dict)
    
    # Validation rules
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)
    
    # Processing configuration
    processing_strategy: str = "single_template"
    priority_level: str = "normal"
    estimated_duration_hours: float = 1.0
    
    # Compliance requirements
    regulatory_standards: List[str] = field(default_factory=list)
    compliance_checks: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    minimum_score: float = 0.7
    confidence_threshold: float = 0.8
    review_required: bool = False

@dataclass
class WorkflowTemplate:
    """Markdown template definition for a workflow."""
    id: str
    name: str
    template_path: str
    template_content: str = ""
    
    # Template metadata
    version: str = "1.0.0"
    author: str = "system"
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Template configuration
    placeholders: List[str] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)
    output_format: str = "json"
    
    # Usage tracking
    usage_count: int = 0
    last_used: Optional[str] = None
    success_rate: float = 1.0

@dataclass
class RegisteredWorkflow:
    """Complete workflow registration with criteria and template."""
    workflow_id: str
    workflow_name: str
    workflow_type: str  # mvr, qa, compliance, financial, etc.
    
    # Core components
    criteria: WorkflowCriteria
    template: WorkflowTemplate
    
    # Workflow metadata
    status: str = "active"  # active, draft, deprecated
    version: str = "1.0.0"
    created_by: str = "system"
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Integration points
    polars_processor: str = "mvr_polars_processor"  # or mvr_clean_processor
    flow_encoding: str = ""
    template_names: List[str] = field(default_factory=list)
    
    # Admin fields (favorite fields for reviewers)
    favorite_fields: Dict[str, List[str]] = field(default_factory=dict)
    custom_validators: List[str] = field(default_factory=list)
    
    # Performance tracking
    total_executions: int = 0
    average_duration: float = 0.0
    success_rate: float = 1.0
    last_execution: Optional[str] = None

class WorkflowRegistrySystem:
    """
    Central workflow registry system for managing all workflows.
    
    Features:
    - Auto-discovery of existing workflows
    - Dynamic criterion generation from templates
    - Admin interface integration
    - Backward compatibility with v1 workflows
    """
    
    def __init__(self, base_path: str = None):
        """Initialize the workflow registry system."""
        if base_path is None:
            base_path = Path(__file__).parent
        
        self.base_path = Path(base_path)
        self.registry_path = self.base_path / "workflow_registry"
        self.workflows_path = self.registry_path / "workflows"  # New individual workflow folders
        self.criteria_path = self.registry_path / "criteria"      # Legacy support
        self.templates_path = self.registry_path / "templates"     # Legacy support
        
        # Create directories if they don't exist
        self.registry_path.mkdir(exist_ok=True)
        self.workflows_path.mkdir(exist_ok=True)
        self.criteria_path.mkdir(exist_ok=True)
        self.templates_path.mkdir(exist_ok=True)
        
        # Registry storage
        self.workflows: Dict[str, RegisteredWorkflow] = {}
        
        # Load existing workflows
        self._load_registry()
        self._discover_workflows()
    
    def _load_registry(self):
        """Load existing workflow registry from disk."""
        registry_file = self.registry_path / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
                for workflow_id, workflow_data in registry_data.items():
                    # Reconstruct workflow objects
                    criteria = WorkflowCriteria(**workflow_data['criteria'])
                    template = WorkflowTemplate(**workflow_data['template'])
                    workflow = RegisteredWorkflow(
                        workflow_id=workflow_id,
                        workflow_name=workflow_data['workflow_name'],
                        workflow_type=workflow_data['workflow_type'],
                        criteria=criteria,
                        template=template,
                        **{k: v for k, v in workflow_data.items() 
                           if k not in ['workflow_id', 'workflow_name', 'workflow_type', 'criteria', 'template']}
                    )
                    self.workflows[workflow_id] = workflow
    
    def _discover_workflows(self):
        """Auto-discover workflows from multiple sources."""
        # 1. Discover from individual workflow folders (new structure)
        for workflow_folder in self.workflows_path.iterdir():
            if workflow_folder.is_dir():
                workflow_id = workflow_folder.name
                if workflow_id not in self.workflows:
                    workflow = self._create_workflow_from_folder(workflow_id, workflow_folder)
                    if workflow:
                        self.workflows[workflow_id] = workflow
        
        # 2. Check v2/prompts directory (legacy support)
        prompts_path = self.base_path / "prompts"
        if prompts_path.exists():
            # Load flow mappings
            flow_mappings_file = prompts_path / "flow_mappings.json"
            if flow_mappings_file.exists():
                with open(flow_mappings_file, 'r') as f:
                    flow_mappings = json.load(f)
                    
                    for flow_key, flow_data in flow_mappings.items():
                        workflow_id = flow_key.strip('[]').replace(' ', '_').lower()
                        
                        # Skip if already registered
                        if workflow_id in self.workflows:
                            continue
                        
                        # Create workflow from flow mapping
                        self._create_workflow_from_flow(workflow_id, flow_key, flow_data)
    
    def _create_workflow_from_flow(self, workflow_id: str, flow_key: str, flow_data: Dict[str, Any]):
        """Create a workflow from flow mapping data."""
        # Extract workflow type from flow encoding
        workflow_type = flow_data.get('flow_encoding', '').split('#')[0].strip('@')
        
        # Create criteria
        criteria = WorkflowCriteria(
            id=f"{workflow_id}_criteria",
            name=f"{flow_key} Criteria",
            description=flow_data.get('description', ''),
            validation_rules=flow_data.get('validation_rules', []),
            processing_strategy=flow_data.get('processing_strategy', 'single_template'),
            priority_level=flow_data.get('priority_level', 'normal'),
            scoring_rubric=self._generate_default_rubric(workflow_type),
            weight_scheme=self._generate_default_weights(workflow_type)
        )
        
        # Create template
        template_names = flow_data.get('template_names', [])
        template_path = ""
        if template_names:
            # Check if template exists in the workflow_registry/templates directory
            expected_template = self.templates_path / f"{template_names[0]}.md"
            if expected_template.exists():
                template_path = f"workflow_registry/templates/{template_names[0]}.md"
            else:
                # Create from existing process_mvr_template.md if available
                process_mvr_template = self.templates_path / "process_mvr_template.md"
                if process_mvr_template.exists() and template_names[0] == "mvr_analysis":
                    expected_template.write_text(process_mvr_template.read_text())
                    template_path = f"workflow_registry/templates/{template_names[0]}.md"
        
        template = WorkflowTemplate(
            id=f"{workflow_id}_template",
            name=f"{flow_key} Template",
            template_path=template_path,
            placeholders=['{document_content}', '{validation_rules}']
        )
        
        # Create registered workflow
        workflow = RegisteredWorkflow(
            workflow_id=workflow_id,
            workflow_name=flow_key.strip('[]'),
            workflow_type=workflow_type or 'general',
            criteria=criteria,
            template=template,
            flow_encoding=flow_data.get('flow_encoding', ''),
            template_names=template_names
        )
        
        self.workflows[workflow_id] = workflow
    
    def _create_workflow_from_folder(self, workflow_id: str, workflow_folder: Path) -> Optional['RegisteredWorkflow']:
        """Create a workflow from individual workflow folder structure."""
        try:
            # Get workflow folder components
            criteria_folder = workflow_folder / "criteria"
            templates_folder = workflow_folder / "templates"
            resources_folder = workflow_folder / "resources"
            
            # Find criteria files
            criteria_files = list(criteria_folder.glob("*.json")) if criteria_folder.exists() else []
            criteria_data = {}
            criteria_path = ""
            
            if criteria_files:
                # Use the first criteria file found
                criteria_file = criteria_files[0]
                criteria_path = str(criteria_file.relative_to(self.base_path))
                with open(criteria_file, 'r') as f:
                    criteria_data = json.load(f)
            
            # Find template files
            template_files = list(templates_folder.glob("*.md")) if templates_folder.exists() else []
            template_path = ""
            template_names = []
            
            if template_files:
                # Use the first template file found
                template_file = template_files[0]
                template_path = str(template_file.relative_to(self.base_path))
                template_names = [template_file.stem for template_file in template_files]
            
            # Extract workflow information
            workflow_name = workflow_id.replace('_', ' ').title()
            workflow_type = criteria_data.get('workflow_type', 'general')
            description = criteria_data.get('description', f'Auto-generated workflow for {workflow_name}')
            
            # Create criteria object
            criteria = WorkflowCriteria(
                id=f"{workflow_id}_criteria",
                name=f"{workflow_name} Criteria",
                description=description,
                validation_rules=criteria_data.get('validation_rules', []),
                processing_strategy=criteria_data.get('processing_strategy', 'single_template'),
                priority_level=criteria_data.get('priority_level', 'normal'),
                scoring_rubric=criteria_data.get('scoring_rubric', {}),
                weight_scheme=criteria_data.get('weight_scheme', {}),
                threshold_values=criteria_data.get('threshold_values', {}),
                required_fields=criteria_data.get('required_fields', []),
                regulatory_standards=criteria_data.get('regulatory_standards', [])
            )
            
            # Create template object
            template = WorkflowTemplate(
                id=f"{workflow_id}_template",
                name=f"{workflow_name} Template",
                template_path=template_path,
                placeholders=criteria_data.get('template_placeholders', ['{document_content}', '{validation_rules}'])
            )
            
            # Create registered workflow
            workflow = RegisteredWorkflow(
                workflow_id=workflow_id,
                workflow_name=workflow_name,
                workflow_type=workflow_type,
                criteria=criteria,
                template=template,
                flow_encoding=criteria_data.get('flow_encoding', f"@{workflow_type}#process!extract@data"),
                template_names=template_names,
                criteria_path=criteria_path,
                template_path=template_path,
                resources_path=str(resources_folder.relative_to(self.base_path)) if resources_folder.exists() else ""
            )
            
            return workflow
            
        except Exception as e:
            print(f"Error creating workflow from folder {workflow_folder}: {e}")
            return None
    
    def get_registered_workflows(self) -> List[Dict[str, Any]]:
        """Get list of all registered workflows with their status and metadata."""
        registered_workflows = []
        
        for workflow_id, workflow in self.workflows.items():
            # Check if workflow has files in new structure
            workflow_folder = self.workflows_path / workflow_id
            has_new_structure = workflow_folder.exists() and (
                (workflow_folder / "criteria").exists() or 
                (workflow_folder / "templates").exists()
            )
            
            # Check legacy structure
            has_legacy_criteria = (self.criteria_path / f"{workflow_id}_criteria.json").exists()
            has_legacy_templates = any((self.templates_path / f"{tpl}.md").exists() for tpl in workflow.template_names)
            
            workflow_info = {
                "workflow_id": workflow_id,
                "workflow_name": workflow.workflow_name,
                "workflow_type": workflow.workflow_type,
                "description": workflow.criteria.description,
                "priority_level": workflow.criteria.priority_level,
                "processing_strategy": workflow.criteria.processing_strategy,
                "template_count": len(workflow.template_names),
                "criteria_file": getattr(workflow, 'criteria_path', ''),
                "template_files": workflow.template_names,
                "has_new_structure": has_new_structure,
                "has_legacy_structure": has_legacy_criteria or has_legacy_templates,
                "status": "active" if (has_new_structure or has_legacy_criteria) else "configured",
                "flow_encoding": workflow.flow_encoding,
                "total_executions": getattr(workflow.criteria, 'total_executions', 0),
                "success_rate": getattr(workflow.criteria, 'success_rate', 1.0),
                "last_execution": getattr(workflow.criteria, 'last_execution', None),
                "resources_available": bool(getattr(workflow, 'resources_path', False))
            }
            
            registered_workflows.append(workflow_info)
        
        # Sort by priority level and name
        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        registered_workflows.sort(key=lambda x: (priority_order.get(x["priority_level"], 2), x["workflow_name"]))
        
        return registered_workflows
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get statistics about registered workflows."""
        workflows = self.get_registered_workflows()
        
        stats = {
            "total_workflows": len(workflows),
            "active_workflows": len([w for w in workflows if w["status"] == "active"]),
            "configured_workflows": len([w for w in workflows if w["status"] == "configured"]),
            "new_structure_count": len([w for w in workflows if w["has_new_structure"]]),
            "legacy_structure_count": len([w for w in workflows if w["has_legacy_structure"]]),
            "workflow_types": {},
            "priority_levels": {},
            "processing_strategies": {},
            "total_executions": sum(w["total_executions"] for w in workflows),
            "average_success_rate": sum(w["success_rate"] for w in workflows) / len(workflows) if workflows else 0
        }
        
        # Count by categories
        for workflow in workflows:
            # Workflow types
            wf_type = workflow["workflow_type"]
            stats["workflow_types"][wf_type] = stats["workflow_types"].get(wf_type, 0) + 1
            
            # Priority levels
            priority = workflow["priority_level"]
            stats["priority_levels"][priority] = stats["priority_levels"].get(priority, 0) + 1
            
            # Processing strategies
            strategy = workflow["processing_strategy"]
            stats["processing_strategies"][strategy] = stats["processing_strategies"].get(strategy, 0) + 1
        
        return stats
    
    def _generate_default_rubric(self, workflow_type: str) -> Dict[str, float]:
        """Generate default scoring rubric based on workflow type."""
        rubrics = {
            'mvr': {
                'document_completeness': 1.0,
                'compliance_score': 1.0,
                'risk_assessment': 1.0,
                'data_quality': 1.0
            },
            'financial': {
                'accuracy': 1.0,
                'completeness': 1.0,
                'risk_metrics': 1.0,
                'compliance': 1.0
            },
            'compliance': {
                'regulatory_adherence': 1.0,
                'documentation': 1.0,
                'risk_mitigation': 1.0,
                'audit_trail': 1.0
            },
            'quality': {
                'accuracy': 1.0,
                'consistency': 1.0,
                'completeness': 1.0,
                'standards_compliance': 1.0
            }
        }
        return rubrics.get(workflow_type, {
            'general_quality': 1.0,
            'completeness': 1.0,
            'accuracy': 1.0,
            'compliance': 1.0
        })
    
    def _generate_default_weights(self, workflow_type: str) -> Dict[str, float]:
        """Generate default weight scheme based on workflow type."""
        weights = {
            'mvr': {
                'document_completeness': 0.25,
                'compliance_score': 0.35,
                'risk_assessment': 0.30,
                'data_quality': 0.10
            },
            'financial': {
                'accuracy': 0.40,
                'completeness': 0.20,
                'risk_metrics': 0.25,
                'compliance': 0.15
            },
            'compliance': {
                'regulatory_adherence': 0.40,
                'documentation': 0.20,
                'risk_mitigation': 0.25,
                'audit_trail': 0.15
            },
            'quality': {
                'accuracy': 0.30,
                'consistency': 0.25,
                'completeness': 0.25,
                'standards_compliance': 0.20
            }
        }
        return weights.get(workflow_type, {
            'general_quality': 0.25,
            'completeness': 0.25,
            'accuracy': 0.25,
            'compliance': 0.25
        })
    
    def register_workflow(self, workflow: RegisteredWorkflow) -> bool:
        """Register a new workflow in the system."""
        try:
            # Save criteria file
            criteria_file = self.criteria_path / f"{workflow.workflow_id}_criteria.json"
            with open(criteria_file, 'w') as f:
                json.dump(asdict(workflow.criteria), f, indent=2)
            
            # Save template file if content exists
            if workflow.template.template_content:
                template_file = self.templates_path / f"{workflow.workflow_id}_template.md"
                with open(template_file, 'w') as f:
                    f.write(workflow.template.template_content)
                workflow.template.template_path = str(template_file)
            
            # Add to registry
            self.workflows[workflow.workflow_id] = workflow
            
            # Save registry
            self.save_registry()
            
            return True
            
        except Exception as e:
            print(f"Error registering workflow: {e}")
            return False
    
    def get_workflow(self, workflow_id: str) -> Optional[RegisteredWorkflow]:
        """Get a registered workflow by ID."""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self, workflow_type: Optional[str] = None, 
                      status: Optional[str] = None) -> List[RegisteredWorkflow]:
        """List all workflows, optionally filtered by type or status."""
        workflows = list(self.workflows.values())
        
        if workflow_type:
            workflows = [w for w in workflows if w.workflow_type == workflow_type]
        
        if status:
            workflows = [w for w in workflows if w.status == status]
        
        return workflows
    
    def update_workflow_criteria(self, workflow_id: str, criteria: WorkflowCriteria) -> bool:
        """Update the criteria for a workflow."""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].criteria = criteria
            
            # Save updated criteria file
            criteria_file = self.criteria_path / f"{workflow_id}_criteria.json"
            with open(criteria_file, 'w') as f:
                json.dump(asdict(criteria), f, indent=2)
            
            self.save_registry()
            return True
        return False
    
    def update_workflow_template(self, workflow_id: str, template: WorkflowTemplate) -> bool:
        """Update the template for a workflow."""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].template = template
            
            # Save updated template file
            if template.template_content:
                template_file = self.templates_path / f"{workflow_id}_template.md"
                with open(template_file, 'w') as f:
                    f.write(template.template_content)
            
            self.save_registry()
            return True
        return False
    
    def add_favorite_fields(self, workflow_id: str, reviewer: str, fields: List[str]) -> bool:
        """Add favorite fields for a specific reviewer."""
        if workflow_id in self.workflows:
            if reviewer not in self.workflows[workflow_id].favorite_fields:
                self.workflows[workflow_id].favorite_fields[reviewer] = []
            
            self.workflows[workflow_id].favorite_fields[reviewer].extend(fields)
            self.workflows[workflow_id].favorite_fields[reviewer] = list(set(
                self.workflows[workflow_id].favorite_fields[reviewer]
            ))
            
            self.save_registry()
            return True
        return False
    
    def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get performance metrics for a workflow."""
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            return {
                'total_executions': workflow.total_executions,
                'average_duration': workflow.average_duration,
                'success_rate': workflow.success_rate,
                'last_execution': workflow.last_execution,
                'status': workflow.status
            }
        return {}
    
    def export_workflow_config(self, workflow_id: str, export_path: Optional[str] = None) -> str:
        """Export a workflow configuration for backup or sharing."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        export_data = {
            'workflow': asdict(workflow),
            'exported_at': datetime.now().isoformat(),
            'version': workflow.version
        }
        
        if export_path:
            export_file = Path(export_path)
        else:
            export_file = self.registry_path / f"{workflow_id}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(export_file)
    
    def import_workflow_config(self, import_path: str) -> bool:
        """Import a workflow configuration from file."""
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            workflow_data = import_data['workflow']
            
            # Reconstruct workflow objects
            criteria = WorkflowCriteria(**workflow_data['criteria'])
            template = WorkflowTemplate(**workflow_data['template'])
            
            workflow = RegisteredWorkflow(
                workflow_id=workflow_data['workflow_id'],
                workflow_name=workflow_data['workflow_name'],
                workflow_type=workflow_data['workflow_type'],
                criteria=criteria,
                template=template,
                **{k: v for k, v in workflow_data.items() 
                   if k not in ['workflow_id', 'workflow_name', 'workflow_type', 'criteria', 'template']}
            )
            
            return self.register_workflow(workflow)
            
        except Exception as e:
            print(f"Error importing workflow: {e}")
            return False
    
    def save_registry(self):
        """Save the current registry to disk."""
        registry_file = self.registry_path / "registry.json"
        
        registry_data = {}
        for workflow_id, workflow in self.workflows.items():
            registry_data[workflow_id] = asdict(workflow)
        
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def generate_admin_config(self) -> Dict[str, Any]:
        """Generate configuration for admin interface."""
        return {
            'workflows': [
                {
                    'id': w.workflow_id,
                    'name': w.workflow_name,
                    'type': w.workflow_type,
                    'status': w.status,
                    'criteria_file': f"{w.workflow_id}_criteria.json",
                    'template_file': w.template.template_path,
                    'favorite_fields': w.favorite_fields,
                    'metrics': self.get_workflow_metrics(w.workflow_id)
                }
                for w in self.workflows.values()
            ],
            'workflow_types': list(set(w.workflow_type for w in self.workflows.values())),
            'total_workflows': len(self.workflows),
            'active_workflows': len([w for w in self.workflows.values() if w.status == 'active'])
        }
    
    def get_all_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get all workflows in dictionary format for admin interface."""
        return {
            workflow_id: {
                'name': workflow.workflow_name,
                'type': workflow.workflow_type,
                'status': workflow.status,
                'criteria_file': f"{workflow_id}_criteria.json" if workflow.criteria else None,
                'template_file': workflow.template.template_path if workflow.template else None,
                'favorite_fields': workflow.favorite_fields,
                'metrics': self.get_workflow_metrics(workflow_id)
            }
            for workflow_id, workflow in self.workflows.items()
        }
    
    def update_workflow_status(self, workflow_id: str, status: str) -> bool:
        """Update workflow status (active/inactive)."""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].status = status
            self.save_registry()
            return True
        return False
    
    def load_criteria(self, workflow_id: str) -> Dict[str, Any]:
        """Load criteria for a specific workflow."""
        criteria_file = self.criteria_path / f"{workflow_id}_criteria.json"
        if criteria_file.exists():
            with open(criteria_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_criteria(self, workflow_id: str, criteria: Dict[str, Any]) -> bool:
        """Save criteria for a specific workflow."""
        try:
            self.criteria_path.mkdir(parents=True, exist_ok=True)
            criteria_file = self.criteria_path / f"{workflow_id}_criteria.json"
            with open(criteria_file, 'w') as f:
                json.dump(criteria, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving criteria for {workflow_id}: {e}")
            return False
    
    def load_template(self, workflow_id: str) -> str:
        """Load template content for a specific workflow."""
        if workflow_id in self.workflows:
            template_path = self.workflows[workflow_id].template.template_path
            # Remove duplicate "templates/" if present
            if template_path.startswith('templates/'):
                template_path = template_path[10:]  # Remove "templates/"
            template_file = self.base_path / "prompts" / "templates" / template_path
            if template_file.exists():
                with open(template_file, 'r', encoding='utf-8') as f:
                    return f.read()
        return ""
    
    def save_template(self, workflow_id: str, content: str) -> bool:
        """Save template content for a specific workflow."""
        try:
            if workflow_id in self.workflows:
                template_path = self.workflows[workflow_id].template.template_path
                template_file = self.base_path / "prompts" / "templates" / template_path
                template_file.parent.mkdir(parents=True, exist_ok=True)
                with open(template_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
        except Exception as e:
            print(f"Error saving template for {workflow_id}: {e}")
        return False


def main():
    """Example usage and initialization."""
    print("Initializing Workflow Registry System...")
    
    registry = WorkflowRegistrySystem()
    
    # List discovered workflows
    print(f"\nDiscovered {len(registry.workflows)} workflows:")
    for workflow in registry.list_workflows():
        print(f"  - {workflow.workflow_name} ({workflow.workflow_type}): {workflow.status}")
        print(f"    Criteria: {workflow.criteria.id}")
        print(f"    Template: {workflow.template.template_path}")
    
    # Generate admin config
    admin_config = registry.generate_admin_config()
    
    # Save admin config for onboarding interface
    admin_config_file = Path(__file__).parent / "workflow_registry" / "admin_config.json"
    with open(admin_config_file, 'w') as f:
        json.dump(admin_config, f, indent=2)
    
    print(f"\nAdmin configuration saved to: {admin_config_file}")
    print(f"Total workflows: {admin_config['total_workflows']}")
    print(f"Active workflows: {admin_config['active_workflows']}")
    
    return registry


if __name__ == "__main__":
    registry = main()