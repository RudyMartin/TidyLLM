#!/usr/bin/env python3
"""
Bracket Commands Registry - Programmatic access to available FLOW bracket commands
================================================================================

This module provides programmatic access to the complete registry of FLOW bracket commands.
Use this for API endpoints, CLI help systems, and integration with external tools.

Usage:
    from tidyllm.flow.examples.bracket_registry import BracketRegistry
    
    registry = BracketRegistry()
    commands = registry.get_all_commands()
    details = registry.get_command_details("[Process MVR]")
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class CommandCategory(Enum):
    """Categories for organizing bracket commands."""
    QA_COMPLIANCE = "qa_compliance"
    DOCUMENT_ANALYSIS = "document_analysis" 
    ADVANCED_ANALYSIS = "advanced_analysis"
    SYSTEM_OPERATIONS = "system_operations"

class ProcessingStrategy(Enum):
    """Processing strategies for bracket commands."""
    SINGLE_TEMPLATE = "single_template"
    MULTI_PERSPECTIVE = "multi_perspective"
    HYBRID_ANALYSIS = "hybrid_analysis"

class Priority(Enum):
    """Priority levels for bracket command processing."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"

@dataclass
class BracketCommand:
    """Represents a bracket command with all its metadata."""
    command: str
    purpose: str
    templates: List[str]
    processing_strategy: ProcessingStrategy
    priority: Priority
    category: CommandCategory
    flow_encoding: str
    validation_rules: List[str]
    examples: List[str]
    related_commands: List[str]

class BracketRegistry:
    """
    Registry of all available FLOW bracket commands with metadata and usage information.
    """
    
    def __init__(self):
        self.commands = self._initialize_registry()
    
    def _initialize_registry(self) -> Dict[str, BracketCommand]:
        """Initialize the complete bracket commands registry."""
        return {
            # QA & Compliance Commands
            "[Process MVR]": BracketCommand(
                command="[Process MVR]",
                purpose="Process Model Validation Report through complete compliance workflow",
                templates=["mvr_analysis", "qa_control"],
                processing_strategy=ProcessingStrategy.MULTI_PERSPECTIVE,
                priority=Priority.HIGH,
                category=CommandCategory.QA_COMPLIANCE,
                flow_encoding="@mvr#process!extract@compliance_data",
                validation_rules=["mvr_document_type", "compliance_standards"],
                examples=["[Process MVR] /path/to/mvr_document.pdf"],
                related_commands=["[Check MVS Compliance]", "[Quality Check]"]
            ),
            
            "[Check MVS Compliance]": BracketCommand(
                command="[Check MVS Compliance]",
                purpose="Validate document against MVS 5.4.3 requirements",
                templates=["compliance_review", "qa_control"],
                processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
                priority=Priority.HIGH,
                category=CommandCategory.QA_COMPLIANCE,
                flow_encoding="@compliance#check!validate@mvs_requirements",
                validation_rules=["MVS_5.4.3", "severity_threshold_medium"],
                examples=["[Check MVS Compliance] /path/to/document.pdf"],
                related_commands=["[Process MVR]", "[Compliance Check]"]
            ),
            
            "[Quality Check]": BracketCommand(
                command="[Quality Check]",
                purpose="Quality assurance review and validation workflow",
                templates=["qa_control"],
                processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
                priority=Priority.NORMAL,
                category=CommandCategory.QA_COMPLIANCE,
                flow_encoding="@quality#check!validate@standards",
                validation_rules=["quality_standards"],
                examples=["[Quality Check] /path/to/document.pdf"],
                related_commands=["[Process MVR]", "[Peer Review]"]
            ),
            
            "[Compliance Check]": BracketCommand(
                command="[Compliance Check]",
                purpose="Regulatory compliance analysis and risk assessment",
                templates=["compliance_review", "qa_control"],
                processing_strategy=ProcessingStrategy.MULTI_PERSPECTIVE,
                priority=Priority.HIGH,
                category=CommandCategory.QA_COMPLIANCE,
                flow_encoding="@compliance#check!validate@regulations",
                validation_rules=["regulatory_requirements"],
                examples=["[Compliance Check] /path/to/regulatory_document.pdf"],
                related_commands=["[Check MVS Compliance]", "[Contract Review]"]
            ),
            
            # Document Analysis Commands
            "[Financial Analysis]": BracketCommand(
                command="[Financial Analysis]",
                purpose="Comprehensive financial document analysis and risk assessment",
                templates=["financial_analysis", "qa_control"],
                processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
                priority=Priority.NORMAL,
                category=CommandCategory.DOCUMENT_ANALYSIS,
                flow_encoding="@financial#analysis!assess@risk_metrics",
                validation_rules=["financial_document_type"],
                examples=["[Financial Analysis] /path/to/financial_report.pdf"],
                related_commands=["[Cost Analysis]", "[Hybrid Analysis]"]
            ),
            
            "[Contract Review]": BracketCommand(
                command="[Contract Review]",
                purpose="Legal contract review with compliance validation",
                templates=["contract_analysis", "compliance_review"],
                processing_strategy=ProcessingStrategy.MULTI_PERSPECTIVE,
                priority=Priority.HIGH,
                category=CommandCategory.DOCUMENT_ANALYSIS,
                flow_encoding="@contract#review!validate@legal_terms",
                validation_rules=["legal_document_type", "contract_complexity"],
                examples=["[Contract Review] /path/to/contract.pdf"],
                related_commands=["[Compliance Check]", "[Peer Review]"]
            ),
            
            "[Data Extraction]": BracketCommand(
                command="[Data Extraction]",
                purpose="Structured data extraction and processing workflow",
                templates=["data_extraction"],
                processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
                priority=Priority.NORMAL,
                category=CommandCategory.DOCUMENT_ANALYSIS,
                flow_encoding="@data#extraction!extract@structured_data",
                validation_rules=["data_structure_validation"],
                examples=["[Data Extraction] /path/to/data_document.pdf"],
                related_commands=["[Document Section View]", "[Quality Check]"]
            ),
            
            "[Document Section View]": BracketCommand(
                command="[Document Section View]",
                purpose="Structured document section analysis for interactive browsing",
                templates=["document_section_view"],
                processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
                priority=Priority.NORMAL,
                category=CommandCategory.DOCUMENT_ANALYSIS,
                flow_encoding="@document#section!analyze@structure",
                validation_rules=["document_structure"],
                examples=["[Document Section View] /path/to/document.pdf"],
                related_commands=["[Data Extraction]", "[Hybrid Analysis]"]
            ),
            
            # Advanced Analysis Commands
            "[Peer Review]": BracketCommand(
                command="[Peer Review]",
                purpose="Expert peer review and professional validation",
                templates=["peer_review", "qa_control"],
                processing_strategy=ProcessingStrategy.MULTI_PERSPECTIVE,
                priority=Priority.CRITICAL,
                category=CommandCategory.ADVANCED_ANALYSIS,
                flow_encoding="@peer#review!validate@expert_opinion",
                validation_rules=["expert_review_required"],
                examples=["[Peer Review] /path/to/research_paper.pdf"],
                related_commands=["[Quality Check]", "[Hybrid Analysis]"]
            ),
            
            "[Hybrid Analysis]": BracketCommand(
                command="[Hybrid Analysis]",
                purpose="Multi-dimensional document analysis combining multiple analytical perspectives",
                templates=["hybrid_analysis", "qa_control"],
                processing_strategy=ProcessingStrategy.HYBRID_ANALYSIS,
                priority=Priority.HIGH,
                category=CommandCategory.ADVANCED_ANALYSIS,
                flow_encoding="@hybrid#analysis!synthesize@multi_framework",
                validation_rules=["multi_framework_applicable"],
                examples=["[Hybrid Analysis] /path/to/complex_document.pdf"],
                related_commands=["[Peer Review]", "[Financial Analysis]", "[Contract Review]"]
            ),
            
            # System Operations Commands
            "[Performance Test]": BracketCommand(
                command="[Performance Test]",
                purpose="Run comprehensive performance benchmark operations",
                templates=["qa_control", "data_extraction"],
                processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
                priority=Priority.NORMAL,
                category=CommandCategory.SYSTEM_OPERATIONS,
                flow_encoding="@performance#test!benchmark@operations",
                validation_rules=["performance_metrics"],
                examples=["[Performance Test] /path/to/test_document.pdf"],
                related_commands=["[Integration Test]", "[Error Analysis]"]
            ),
            
            "[Integration Test]": BracketCommand(
                command="[Integration Test]",
                purpose="Test integration between components and external systems",
                templates=["qa_control"],
                processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
                priority=Priority.NORMAL,
                category=CommandCategory.SYSTEM_OPERATIONS,
                flow_encoding="@integration#test!validate@components",
                validation_rules=["integration_validation"],
                examples=["[Integration Test] /path/to/integration_test.pdf"],
                related_commands=["[Performance Test]", "[Quality Check]"]
            ),
            
            "[Cost Analysis]": BracketCommand(
                command="[Cost Analysis]",
                purpose="Analyze cost patterns and optimization opportunities",
                templates=["financial_analysis"],
                processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
                priority=Priority.NORMAL,
                category=CommandCategory.SYSTEM_OPERATIONS,
                flow_encoding="@cost#analysis!track@operations",
                validation_rules=["cost_tracking"],
                examples=["[Cost Analysis] /path/to/cost_report.pdf"],
                related_commands=["[Financial Analysis]", "[Performance Test]"]
            ),
            
            "[Error Analysis]": BracketCommand(
                command="[Error Analysis]",
                purpose="Analyze error patterns and failure modes",
                templates=["qa_control"],
                processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
                priority=Priority.HIGH,
                category=CommandCategory.SYSTEM_OPERATIONS,
                flow_encoding="@error#analysis!identify@failures",
                validation_rules=["error_patterns"],
                examples=["[Error Analysis] /path/to/error_log.pdf"],
                related_commands=["[Performance Test]", "[Quality Check]"]
            )
        }
    
    def get_all_commands(self) -> List[str]:
        """Get list of all available bracket commands."""
        return list(self.commands.keys())
    
    def get_command_details(self, command: str) -> Optional[BracketCommand]:
        """Get detailed information about a specific bracket command."""
        return self.commands.get(command)
    
    def get_commands_by_category(self, category: CommandCategory) -> List[BracketCommand]:
        """Get all commands in a specific category."""
        return [cmd for cmd in self.commands.values() if cmd.category == category]
    
    def get_commands_by_priority(self, priority: Priority) -> List[BracketCommand]:
        """Get all commands with a specific priority level."""
        return [cmd for cmd in self.commands.values() if cmd.priority == priority]
    
    def search_commands(self, query: str) -> List[BracketCommand]:
        """Search commands by purpose, templates, or validation rules."""
        query_lower = query.lower()
        results = []
        
        for cmd in self.commands.values():
            if (query_lower in cmd.purpose.lower() or
                query_lower in cmd.command.lower() or
                any(query_lower in template for template in cmd.templates) or
                any(query_lower in rule for rule in cmd.validation_rules)):
                results.append(cmd)
        
        return results
    
    def get_template_usage(self) -> Dict[str, List[str]]:
        """Get which commands use each template."""
        template_usage = {}
        
        for cmd in self.commands.values():
            for template in cmd.templates:
                if template not in template_usage:
                    template_usage[template] = []
                template_usage[template].append(cmd.command)
        
        return template_usage
    
    def validate_command(self, command: str) -> bool:
        """Check if a bracket command is valid and registered."""
        return command in self.commands
    
    def get_command_help(self, command: str) -> str:
        """Get formatted help text for a specific command."""
        cmd_details = self.get_command_details(command)
        if not cmd_details:
            return f"Command '{command}' not found in registry."
        
        help_text = f"""
{cmd_details.command}
{'-' * len(cmd_details.command)}

Purpose: {cmd_details.purpose}
Templates: {', '.join(cmd_details.templates)}
Processing: {cmd_details.processing_strategy.value}
Priority: {cmd_details.priority.value}
Category: {cmd_details.category.value}

Examples:
{chr(10).join(f"  {example}" for example in cmd_details.examples)}

Related Commands: {', '.join(cmd_details.related_commands)}
        """
        return help_text.strip()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the bracket commands registry."""
        return {
            "total_commands": len(self.commands),
            "categories": {
                category.value: len(self.get_commands_by_category(category))
                for category in CommandCategory
            },
            "priorities": {
                priority.value: len(self.get_commands_by_priority(priority))
                for priority in Priority
            },
            "processing_strategies": {
                strategy.value: len([cmd for cmd in self.commands.values() 
                                   if cmd.processing_strategy == strategy])
                for strategy in ProcessingStrategy
            },
            "most_used_templates": self._get_most_used_templates()
        }
    
    def _get_most_used_templates(self) -> Dict[str, int]:
        """Get template usage statistics."""
        template_counts = {}
        
        for cmd in self.commands.values():
            for template in cmd.templates:
                template_counts[template] = template_counts.get(template, 0) + 1
        
        return dict(sorted(template_counts.items(), key=lambda x: x[1], reverse=True))

# Convenience functions for common operations
def get_all_bracket_commands() -> List[str]:
    """Get list of all available bracket commands."""
    registry = BracketRegistry()
    return registry.get_all_commands()

def get_bracket_command_help(command: str) -> str:
    """Get help text for a specific bracket command."""
    registry = BracketRegistry()
    return registry.get_command_help(command)

def validate_bracket_command(command: str) -> bool:
    """Validate if a bracket command exists in the registry."""
    registry = BracketRegistry()
    return registry.validate_command(command)

def search_bracket_commands(query: str) -> List[str]:
    """Search bracket commands by query."""
    registry = BracketRegistry()
    results = registry.search_commands(query)
    return [cmd.command for cmd in results]

# Main execution for CLI usage
if __name__ == "__main__":
    import sys
    
    registry = BracketRegistry()
    
    if len(sys.argv) == 1:
        print("Available Bracket Commands:")
        print("=" * 50)
        
        for category in CommandCategory:
            commands = registry.get_commands_by_category(category)
            if commands:
                print(f"\n{category.value.upper().replace('_', ' ')}:")
                for cmd in commands:
                    priority_indicator = "🔴" if cmd.priority == Priority.CRITICAL else "🟡" if cmd.priority == Priority.HIGH else "🟢"
                    print(f"  {priority_indicator} {cmd.command}")
        
        print(f"\nTotal Commands: {len(registry.get_all_commands())}")
        print("\nUse 'python bracket_registry.py [Command Name]' for details")
        
    elif len(sys.argv) == 2:
        command = sys.argv[1]
        if not command.startswith('['):
            command = f"[{command}]"
        
        print(registry.get_command_help(command))
    
    else:
        print("Usage: python bracket_registry.py [command_name]")
        sys.exit(1)