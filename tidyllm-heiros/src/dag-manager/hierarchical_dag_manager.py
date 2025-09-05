#!/usr/bin/env python3
"""
TidyLLM-HeirOS: Hierarchical DAG Flow Manager
============================================

A compliance-focused, hierarchical workflow management system inspired by:
- ROS Behavior Trees for modular robotics control
- Elysia's decision tree architecture with transparency
- Corporate compliance requirements with audit trails

Core Principles:
1. Hierarchical decomposition (like ROS behavior trees)
2. Decision transparency (like Elysia)  
3. SPARSE agreement for documented decisions
4. Compliance audit trails
5. Dynamic flow generation for uncertain processes
"""

import os
import sys
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid
from abc import ABC, abstractmethod

class NodeType(Enum):
    """Types of nodes in the hierarchical DAG"""
    SEQUENCE = "sequence"          # Execute children in order
    SELECTOR = "selector"          # Execute first successful child
    PARALLEL = "parallel"          # Execute children simultaneously  
    CONDITION = "condition"        # Boolean evaluation node
    ACTION = "action"             # Executable task node
    SPARSE_DECISION = "sparse"    # Pre-documented decision
    DYNAMIC_FLOW = "dynamic"      # AI-generated flow

class NodeStatus(Enum):
    """Execution status of nodes"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"
    SKIPPED = "skipped"

class ComplianceLevel(Enum):
    """Compliance transparency levels for corporate users"""
    FULL_TRANSPARENCY = "full"      # Complete audit trail
    SUMMARY_ONLY = "summary"        # Key decisions only
    MINIMAL = "minimal"             # Basic logging
    REGULATORY = "regulatory"       # Compliance-focused

@dataclass
class DecisionAudit:
    """Audit record for compliance tracking"""
    decision_id: str
    node_id: str
    timestamp: datetime
    decision_type: str  # sparse_agreement, dynamic_ai, manual_override
    reasoning: str
    confidence_score: float
    compliance_notes: str = ""
    reviewer_id: Optional[str] = None
    approval_status: str = "pending"

@dataclass
class SparseAgreement:
    """Pre-documented decision with compliance metadata"""
    agreement_id: str
    title: str
    description: str
    conditions: List[str]
    approved_actions: List[str]
    compliance_framework: List[str]  # regulations/standards
    approval_date: datetime
    expiry_date: Optional[datetime]
    risk_assessment: Dict[str, Any]
    stakeholder_approvals: List[str]

class HierarchicalNode(ABC):
    """Base class for all DAG nodes with hierarchical behavior"""
    
    def __init__(self, 
                 node_id: str,
                 node_type: NodeType,
                 name: str,
                 description: str = "",
                 compliance_level: ComplianceLevel = ComplianceLevel.FULL_TRANSPARENCY):
        self.node_id = node_id
        self.node_type = node_type
        self.name = name
        self.description = description
        self.compliance_level = compliance_level
        
        # Hierarchical structure
        self.parent: Optional['HierarchicalNode'] = None
        self.children: List['HierarchicalNode'] = []
        
        # Execution state
        self.status = NodeStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        
        # Compliance tracking
        self.audit_trail: List[DecisionAudit] = []
        self.metadata: Dict[str, Any] = {}
        
        # Context awareness (inspired by Elysia)
        self.global_context: Dict[str, Any] = {}
        self.local_context: Dict[str, Any] = {}
    
    def add_child(self, child: 'HierarchicalNode') -> 'HierarchicalNode':
        """Add child node with parent relationship"""
        child.parent = self
        self.children.append(child)
        return self
    
    def get_hierarchy_path(self) -> List[str]:
        """Get full hierarchical path from root"""
        if self.parent is None:
            return [self.node_id]
        return self.parent.get_hierarchy_path() + [self.node_id]
    
    def create_audit_record(self, 
                          decision_type: str, 
                          reasoning: str,
                          confidence_score: float,
                          compliance_notes: str = "") -> DecisionAudit:
        """Create compliance audit record"""
        audit = DecisionAudit(
            decision_id=str(uuid.uuid4()),
            node_id=self.node_id,
            timestamp=datetime.now(),
            decision_type=decision_type,
            reasoning=reasoning,
            confidence_score=confidence_score,
            compliance_notes=compliance_notes
        )
        self.audit_trail.append(audit)
        return audit
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node with context"""
        pass
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Generate compliance summary for corporate review"""
        return {
            'node_id': self.node_id,
            'name': self.name,
            'hierarchy_path': ' -> '.join(self.get_hierarchy_path()),
            'status': self.status.value,
            'execution_time': self._get_execution_time(),
            'decisions_made': len(self.audit_trail),
            'compliance_level': self.compliance_level.value,
            'risk_factors': self._assess_risk_factors()
        }
    
    def _get_execution_time(self) -> Optional[float]:
        """Calculate execution time in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def _assess_risk_factors(self) -> List[str]:
        """Assess risk factors for compliance"""
        risks = []
        if self.status == NodeStatus.FAILURE:
            risks.append("execution_failure")
        if len(self.audit_trail) == 0:
            risks.append("no_audit_trail")
        if any(audit.confidence_score < 0.7 for audit in self.audit_trail):
            risks.append("low_confidence_decisions")
        return risks

class SequenceNode(HierarchicalNode):
    """Execute children in sequence (like ROS Sequence behavior)"""
    
    def __init__(self, node_id: str, name: str, **kwargs):
        super().__init__(node_id, NodeType.SEQUENCE, name, **kwargs)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all children in sequence"""
        self.status = NodeStatus.RUNNING
        self.start_time = datetime.now()
        
        self.create_audit_record(
            "sequence_execution",
            f"Starting sequence execution of {len(self.children)} children",
            1.0,
            f"Sequence node with {len(self.children)} children"
        )
        
        results = []
        for child in self.children:
            try:
                child_context = {**context, **self.local_context}
                result = child.execute(child_context)
                
                if child.status == NodeStatus.FAILURE:
                    self.status = NodeStatus.FAILURE
                    self.error_message = f"Child {child.node_id} failed: {child.error_message}"
                    break
                
                results.append(result)
                
            except Exception as e:
                self.status = NodeStatus.FAILURE
                self.error_message = f"Exception in child {child.node_id}: {str(e)}"
                break
        
        if self.status == NodeStatus.RUNNING:
            self.status = NodeStatus.SUCCESS
        
        self.end_time = datetime.now()
        
        return {
            'node_id': self.node_id,
            'status': self.status.value,
            'results': results,
            'execution_time': self._get_execution_time()
        }

class SelectorNode(HierarchicalNode):
    """Execute first successful child (like ROS Selector behavior)"""
    
    def __init__(self, node_id: str, name: str, **kwargs):
        super().__init__(node_id, NodeType.SELECTOR, name, **kwargs)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute children until one succeeds"""
        self.status = NodeStatus.RUNNING
        self.start_time = datetime.now()
        
        self.create_audit_record(
            "selector_execution",
            f"Starting selector execution with {len(self.children)} options",
            1.0,
            "Selector node - will execute first successful child"
        )
        
        for child in self.children:
            try:
                child_context = {**context, **self.local_context}
                result = child.execute(child_context)
                
                if child.status == NodeStatus.SUCCESS:
                    self.status = NodeStatus.SUCCESS
                    self.end_time = datetime.now()
                    
                    self.create_audit_record(
                        "selector_success",
                        f"Child {child.node_id} succeeded, stopping selector",
                        0.95,
                        f"Successful execution via child: {child.name}"
                    )
                    
                    return {
                        'node_id': self.node_id,
                        'status': self.status.value,
                        'successful_child': child.node_id,
                        'result': result,
                        'execution_time': self._get_execution_time()
                    }
                    
            except Exception as e:
                # Continue to next child on exception
                continue
        
        # All children failed
        self.status = NodeStatus.FAILURE
        self.error_message = "All selector children failed"
        self.end_time = datetime.now()
        
        return {
            'node_id': self.node_id,
            'status': self.status.value,
            'error': self.error_message,
            'execution_time': self._get_execution_time()
        }

class SparseDecisionNode(HierarchicalNode):
    """Pre-documented decision with SPARSE agreement"""
    
    def __init__(self, 
                 node_id: str, 
                 name: str, 
                 sparse_agreement: SparseAgreement,
                 **kwargs):
        super().__init__(node_id, NodeType.SPARSE_DECISION, name, **kwargs)
        self.sparse_agreement = sparse_agreement
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pre-documented decision"""
        self.status = NodeStatus.RUNNING
        self.start_time = datetime.now()
        
        # Check if SPARSE agreement is still valid
        if (self.sparse_agreement.expiry_date and 
            datetime.now() > self.sparse_agreement.expiry_date):
            
            self.status = NodeStatus.BLOCKED
            self.error_message = "SPARSE agreement expired"
            
            self.create_audit_record(
                "sparse_expired",
                "SPARSE agreement has expired, manual review required",
                0.0,
                f"Agreement {self.sparse_agreement.agreement_id} expired on {self.sparse_agreement.expiry_date}"
            )
            
            return {
                'node_id': self.node_id,
                'status': self.status.value,
                'error': self.error_message,
                'requires_review': True
            }
        
        # Check conditions
        conditions_met = self._check_conditions(context)
        
        if not conditions_met['all_met']:
            self.status = NodeStatus.BLOCKED
            self.error_message = f"SPARSE conditions not met: {conditions_met['failed_conditions']}"
            
            self.create_audit_record(
                "sparse_conditions_failed",
                f"Conditions not met: {conditions_met['failed_conditions']}",
                0.3,
                "SPARSE agreement conditions evaluation failed"
            )
            
            return {
                'node_id': self.node_id,
                'status': self.status.value,
                'error': self.error_message,
                'failed_conditions': conditions_met['failed_conditions']
            }
        
        # Execute approved actions
        self.status = NodeStatus.SUCCESS
        self.end_time = datetime.now()
        
        self.create_audit_record(
            "sparse_executed",
            f"SPARSE agreement {self.sparse_agreement.agreement_id} executed successfully",
            1.0,
            f"Pre-approved decision executed under agreement: {self.sparse_agreement.title}"
        )
        
        return {
            'node_id': self.node_id,
            'status': self.status.value,
            'agreement_id': self.sparse_agreement.agreement_id,
            'actions_executed': self.sparse_agreement.approved_actions,
            'compliance_framework': self.sparse_agreement.compliance_framework,
            'execution_time': self._get_execution_time()
        }
    
    def _check_conditions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if SPARSE agreement conditions are met"""
        failed_conditions = []
        
        for condition in self.sparse_agreement.conditions:
            # Simple condition checking - in practice would be more sophisticated
            if condition not in context or not context[condition]:
                failed_conditions.append(condition)
        
        return {
            'all_met': len(failed_conditions) == 0,
            'failed_conditions': failed_conditions,
            'total_conditions': len(self.sparse_agreement.conditions)
        }

class DynamicFlowNode(HierarchicalNode):
    """AI-generated flow for uncertain processes (inspired by Elysia)"""
    
    def __init__(self, 
                 node_id: str, 
                 name: str,
                 ai_orchestrator: Optional[Callable] = None,
                 **kwargs):
        super().__init__(node_id, NodeType.DYNAMIC_FLOW, name, **kwargs)
        self.ai_orchestrator = ai_orchestrator or self._default_ai_orchestrator
        self.generated_flow: Optional[List[HierarchicalNode]] = None
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate and execute dynamic flow"""
        self.status = NodeStatus.RUNNING
        self.start_time = datetime.now()
        
        self.create_audit_record(
            "dynamic_flow_generation",
            "Starting AI-powered dynamic flow generation",
            0.8,  # Lower confidence due to AI generation
            "Dynamic flow requires AI orchestration for uncertain process"
        )
        
        try:
            # Generate flow using AI orchestrator
            self.generated_flow = self.ai_orchestrator(context, self.global_context)
            
            if not self.generated_flow:
                self.status = NodeStatus.FAILURE
                self.error_message = "AI orchestrator failed to generate flow"
                return {
                    'node_id': self.node_id,
                    'status': self.status.value,
                    'error': self.error_message
                }
            
            # Execute generated flow
            results = []
            for generated_node in self.generated_flow:
                result = generated_node.execute(context)
                results.append(result)
                
                if generated_node.status == NodeStatus.FAILURE:
                    self.status = NodeStatus.FAILURE
                    self.error_message = f"Generated node {generated_node.node_id} failed"
                    break
            
            if self.status == NodeStatus.RUNNING:
                self.status = NodeStatus.SUCCESS
            
            self.end_time = datetime.now()
            
            self.create_audit_record(
                "dynamic_flow_completed",
                f"Dynamic flow completed with {len(self.generated_flow)} generated nodes",
                0.85,
                f"AI-generated workflow executed successfully"
            )
            
            return {
                'node_id': self.node_id,
                'status': self.status.value,
                'generated_nodes': len(self.generated_flow),
                'results': results,
                'execution_time': self._get_execution_time()
            }
            
        except Exception as e:
            self.status = NodeStatus.FAILURE
            self.error_message = f"Dynamic flow generation failed: {str(e)}"
            
            self.create_audit_record(
                "dynamic_flow_error",
                f"AI orchestration failed: {str(e)}",
                0.0,
                "Dynamic flow generation encountered critical error"
            )
            
            return {
                'node_id': self.node_id,
                'status': self.status.value,
                'error': self.error_message
            }
    
    def _default_ai_orchestrator(self, 
                                context: Dict[str, Any], 
                                global_context: Dict[str, Any]) -> List[HierarchicalNode]:
        """Default AI orchestrator - in practice would use LLM"""
        # Mock AI-generated flow
        return [
            ActionNode(
                f"ai_generated_{uuid.uuid4().hex[:8]}", 
                "AI Generated Action",
                action=lambda ctx: {"ai_decision": "mock_action_executed"}
            )
        ]

class ActionNode(HierarchicalNode):
    """Executable action node"""
    
    def __init__(self, 
                 node_id: str, 
                 name: str,
                 action: Callable[[Dict[str, Any]], Dict[str, Any]],
                 **kwargs):
        super().__init__(node_id, NodeType.ACTION, name, **kwargs)
        self.action = action
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action"""
        self.status = NodeStatus.RUNNING
        self.start_time = datetime.now()
        
        try:
            result = self.action({**context, **self.local_context})
            self.status = NodeStatus.SUCCESS
            
            self.create_audit_record(
                "action_executed",
                f"Action {self.name} executed successfully",
                0.95,
                f"Direct action execution completed"
            )
            
        except Exception as e:
            self.status = NodeStatus.FAILURE
            self.error_message = str(e)
            result = {'error': str(e)}
            
            self.create_audit_record(
                "action_failed",
                f"Action {self.name} failed: {str(e)}",
                0.0,
                f"Action execution encountered error"
            )
        
        self.end_time = datetime.now()
        
        return {
            'node_id': self.node_id,
            'status': self.status.value,
            'result': result,
            'execution_time': self._get_execution_time()
        }

class HierarchicalDAGManager:
    """Main DAG manager with compliance and transparency features"""
    
    def __init__(self, 
                 name: str,
                 compliance_level: ComplianceLevel = ComplianceLevel.FULL_TRANSPARENCY):
        self.name = name
        self.compliance_level = compliance_level
        self.root_nodes: List[HierarchicalNode] = []
        self.sparse_agreements: Dict[str, SparseAgreement] = {}
        self.global_context: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
    def add_root_node(self, node: HierarchicalNode) -> 'HierarchicalDAGManager':
        """Add root-level node to DAG"""
        node.global_context = self.global_context
        self.root_nodes.append(node)
        return self
    
    def register_sparse_agreement(self, agreement: SparseAgreement) -> 'HierarchicalDAGManager':
        """Register pre-documented decision"""
        self.sparse_agreements[agreement.agreement_id] = agreement
        return self
    
    def execute_dag(self, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the entire DAG with compliance tracking"""
        if initial_context:
            self.global_context.update(initial_context)
        
        execution_start = datetime.now()
        execution_id = str(uuid.uuid4())
        
        results = []
        overall_status = NodeStatus.SUCCESS
        
        for root_node in self.root_nodes:
            try:
                result = root_node.execute(self.global_context)
                results.append(result)
                
                if root_node.status == NodeStatus.FAILURE:
                    overall_status = NodeStatus.FAILURE
                    
            except Exception as e:
                overall_status = NodeStatus.FAILURE
                results.append({
                    'node_id': root_node.node_id,
                    'status': NodeStatus.FAILURE.value,
                    'error': str(e)
                })
        
        execution_end = datetime.now()
        
        execution_record = {
            'execution_id': execution_id,
            'start_time': execution_start,
            'end_time': execution_end,
            'duration_seconds': (execution_end - execution_start).total_seconds(),
            'overall_status': overall_status.value,
            'nodes_executed': len(results),
            'results': results,
            'compliance_summary': self.generate_compliance_report()
        }
        
        self.execution_history.append(execution_record)
        
        return execution_record
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report for corporate users"""
        all_nodes = self._collect_all_nodes()
        
        total_decisions = sum(len(node.audit_trail) for node in all_nodes)
        total_nodes = len(all_nodes)
        
        risk_summary = {}
        decision_types = {}
        
        for node in all_nodes:
            # Collect risk factors
            for risk in node._assess_risk_factors():
                risk_summary[risk] = risk_summary.get(risk, 0) + 1
            
            # Collect decision types
            for audit in node.audit_trail:
                decision_types[audit.decision_type] = decision_types.get(audit.decision_type, 0) + 1
        
        return {
            'compliance_level': self.compliance_level.value,
            'total_nodes': total_nodes,
            'total_decisions': total_decisions,
            'sparse_agreements_used': len([a for a in self.sparse_agreements.values()]),
            'risk_factors': risk_summary,
            'decision_types': decision_types,
            'nodes_summary': [node.get_compliance_summary() for node in all_nodes],
            'audit_completeness': total_decisions / total_nodes if total_nodes > 0 else 0
        }
    
    def _collect_all_nodes(self) -> List[HierarchicalNode]:
        """Collect all nodes in the DAG for analysis"""
        all_nodes = []
        
        def collect_recursive(node: HierarchicalNode):
            all_nodes.append(node)
            for child in node.children:
                collect_recursive(child)
        
        for root in self.root_nodes:
            collect_recursive(root)
        
        return all_nodes
    
    def visualize_hierarchy(self) -> str:
        """Generate text-based hierarchy visualization"""
        visualization = f"DAG: {self.name}\n"
        visualization += "=" * (len(self.name) + 5) + "\n\n"
        
        for root in self.root_nodes:
            visualization += self._visualize_node(root, 0)
        
        return visualization
    
    def _visualize_node(self, node: HierarchicalNode, depth: int) -> str:
        """Recursively visualize node hierarchy"""
        indent = "  " * depth
        status_symbol = {
            NodeStatus.PENDING: "[-]",
            NodeStatus.RUNNING: "[~]",
            NodeStatus.SUCCESS: "[+]",
            NodeStatus.FAILURE: "[!]", 
            NodeStatus.BLOCKED: "[X]",
            NodeStatus.SKIPPED: "[>]"
        }.get(node.status, "[?]")
        
        viz = f"{indent}{status_symbol} [{node.node_type.value}] {node.name} ({node.node_id})\n"
        
        for child in node.children:
            viz += self._visualize_node(child, depth + 1)
        
        return viz