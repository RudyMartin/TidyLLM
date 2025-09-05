"""
Control Signal Nodes (S) - TidyLLM-HeirOS Electrical System
===========================================================

Control signals manage flow and decisions in the system
These are the signal (S) bus components that orchestrate the process
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from flow_schemas import ElectricalNode, NodePolarity, FlowType

class ControlSignalType(Enum):
    """Types of control signals"""
    ROUTER = "router"
    SCHEDULER = "scheduler"
    VALIDATOR = "validator"
    SEQUENCER = "sequencer"
    CONDITION_GATE = "condition_gate"
    FLOW_MONITOR = "flow_monitor"

class RouterControlSignal(ElectricalNode):
    """Router control signal (S) - manages routing decisions"""
    
    def __init__(self, node_id: str, name: str, routing_logic: Callable):
        super().__init__(node_id, name, NodePolarity.CONTROL_SIGNAL)
        self.routing_logic = routing_logic
        self.routing_history = []
        self.current_route = None
        
        # Router control pins
        self.add_pin("DATA_MONITOR_+", FlowType.INPUT, "input", "3.3V", 50)
        self.add_pin("ROUTE_OUT_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("STATUS_S", FlowType.CONTROL, "output", "3.3V", 5)
        self.add_pin("FAULT_GND", FlowType.OUTPUT, "output", "0V", 5)
    
    def process_electrical_flow(self, input_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Generate routing control signals"""
        
        data_monitor = input_flows.get('DATA_MONITOR_+', {})
        
        try:
            self.current_state = "ANALYZING_ROUTE"
            
            # Analyze data for routing decision
            routing_context = {
                'data_characteristics': data_monitor.get('data', {}),
                'data_quality': data_monitor.get('quality', 1.0),
                'timestamp': datetime.now(),
                'history': self.routing_history[-5:]  # Last 5 routing decisions
            }
            
            # Execute routing logic
            routing_decision = self.routing_logic(routing_context)
            
            # Record routing decision
            route_record = {
                'route_id': routing_decision.get('route_id', 'default'),
                'confidence': routing_decision.get('confidence', 1.0),
                'reasoning': routing_decision.get('reasoning', 'automatic'),
                'timestamp': datetime.now()
            }
            
            self.routing_history.append(route_record)
            self.current_route = route_record['route_id']
            self.current_state = "ROUTE_DETERMINED"
            
            return {
                'ROUTE_OUT_S': {
                    'voltage': '3.3V',
                    'control_signal': 'routing_instruction',
                    'route_id': route_record['route_id'],
                    'confidence': route_record['confidence'],
                    'parameters': routing_decision.get('parameters', {})
                },
                'STATUS_S': {
                    'voltage': '3.3V',
                    'status': 'routing_active',
                    'current_route': self.current_route,
                    'decisions_made': len(self.routing_history)
                },
                'FAULT_GND': {'voltage': '0V', 'error': None}
            }
            
        except Exception as e:
            self.current_state = "FAULT_ROUTING_ERROR"
            return {
                'ROUTE_OUT_S': {'voltage': '0V', 'control_signal': 'routing_failed'},
                'STATUS_S': {'voltage': '0V', 'status': 'routing_error'},
                'FAULT_GND': {
                    'voltage': '3.3V',
                    'error': str(e),
                    'fault_type': 'routing_logic_error'
                }
            }

class SchedulerControlSignal(ElectricalNode):
    """Scheduler control signal (S) - manages timing and sequencing"""
    
    def __init__(self, node_id: str, name: str, scheduling_policy: str = "fifo"):
        super().__init__(node_id, name, NodePolarity.CONTROL_SIGNAL)
        self.scheduling_policy = scheduling_policy  # fifo, priority, round_robin
        self.task_queue = []
        self.active_tasks = []
        self.completed_tasks = 0
        
        # Scheduler control pins
        self.add_pin("TASK_QUEUE_+", FlowType.INPUT, "input", "3.3V", 20)
        self.add_pin("SCHEDULE_OUT_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("TIMING_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("QUEUE_STATUS_S", FlowType.CONTROL, "output", "3.3V", 5)
    
    def process_electrical_flow(self, input_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Generate scheduling control signals"""
        
        task_input = input_flows.get('TASK_QUEUE_+', {})
        
        try:
            self.current_state = "SCHEDULING"
            
            # Add new tasks to queue
            if task_input:
                new_task = {
                    'task_id': f"task_{len(self.task_queue) + 1}",
                    'data': task_input.get('data', {}),
                    'priority': task_input.get('priority', 1),
                    'queued_at': datetime.now(),
                    'status': 'queued'
                }
                self.task_queue.append(new_task)
            
            # Schedule next task based on policy
            scheduled_task = self._schedule_next_task()
            
            current_time = datetime.now()
            
            return {
                'SCHEDULE_OUT_S': {
                    'voltage': '3.3V',
                    'control_signal': 'task_schedule',
                    'scheduled_task': scheduled_task,
                    'policy': self.scheduling_policy,
                    'schedule_time': current_time
                },
                'TIMING_S': {
                    'voltage': '3.3V',
                    'control_signal': 'timing_control',
                    'current_time': current_time,
                    'next_schedule': current_time,  # Simplified
                    'active_tasks': len(self.active_tasks)
                },
                'QUEUE_STATUS_S': {
                    'voltage': '3.3V',
                    'queue_length': len(self.task_queue),
                    'active_tasks': len(self.active_tasks),
                    'completed_tasks': self.completed_tasks,
                    'policy': self.scheduling_policy
                }
            }
            
        except Exception as e:
            self.current_state = "FAULT_SCHEDULING_ERROR"
            return {
                'SCHEDULE_OUT_S': {'voltage': '0V', 'control_signal': 'schedule_failed'},
                'TIMING_S': {'voltage': '0V', 'control_signal': 'timing_failed'},
                'QUEUE_STATUS_S': {'voltage': '0V', 'error': str(e)}
            }
    
    def _schedule_next_task(self) -> Optional[Dict]:
        """Schedule next task based on policy"""
        if not self.task_queue:
            return None
        
        if self.scheduling_policy == "fifo":
            return self.task_queue.pop(0)
        elif self.scheduling_policy == "priority":
            # Sort by priority (higher number = higher priority)
            self.task_queue.sort(key=lambda x: x['priority'], reverse=True)
            return self.task_queue.pop(0)
        else:  # round_robin or default
            return self.task_queue.pop(0)

class ValidatorControlSignal(ElectricalNode):
    """Validator control signal (S) - manages data validation"""
    
    def __init__(self, node_id: str, name: str, validation_rules: List[Dict]):
        super().__init__(node_id, name, NodePolarity.CONTROL_SIGNAL)
        self.validation_rules = validation_rules
        self.validation_count = 0
        self.pass_rate = 1.0
        
        # Validator control pins
        self.add_pin("DATA_IN_+", FlowType.INPUT, "input", "5V", 100)
        self.add_pin("VALIDATION_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("APPROVAL_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("REJECT_GND", FlowType.OUTPUT, "output", "0V", 5)
    
    def process_electrical_flow(self, input_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation control signals"""
        
        data_input = input_flows.get('DATA_IN_+', {})
        
        if not data_input:
            return {
                'VALIDATION_S': {'voltage': '0V', 'validation_result': 'no_data'},
                'APPROVAL_S': {'voltage': '0V', 'approved': False},
                'REJECT_GND': {'voltage': '3.3V', 'reason': 'no_input_data'}
            }
        
        try:
            self.current_state = "VALIDATING"
            
            # Extract data for validation
            data_payload = data_input.get('data', {})
            data_quality = data_input.get('quality', 1.0)
            
            # Run validation rules
            validation_results = []
            all_passed = True
            
            for rule in self.validation_rules:
                rule_result = self._apply_validation_rule(rule, data_payload, data_quality)
                validation_results.append(rule_result)
                if not rule_result['passed']:
                    all_passed = False
            
            self.validation_count += 1
            
            # Update pass rate
            if all_passed:
                self.pass_rate = (self.pass_rate * (self.validation_count - 1) + 1.0) / self.validation_count
            else:
                self.pass_rate = (self.pass_rate * (self.validation_count - 1) + 0.0) / self.validation_count
            
            self.current_state = "VALIDATION_COMPLETE"
            
            if all_passed:
                return {
                    'VALIDATION_S': {
                        'voltage': '3.3V',
                        'control_signal': 'validation_complete',
                        'validation_result': 'passed',
                        'rules_checked': len(self.validation_rules),
                        'pass_rate': self.pass_rate
                    },
                    'APPROVAL_S': {
                        'voltage': '3.3V',
                        'control_signal': 'data_approved',
                        'approved': True,
                        'confidence': min([r['confidence'] for r in validation_results])
                    },
                    'REJECT_GND': {'voltage': '0V', 'reason': None}
                }
            else:
                failed_rules = [r for r in validation_results if not r['passed']]
                return {
                    'VALIDATION_S': {
                        'voltage': '3.3V',
                        'control_signal': 'validation_complete',
                        'validation_result': 'failed',
                        'failed_rules': len(failed_rules),
                        'pass_rate': self.pass_rate
                    },
                    'APPROVAL_S': {'voltage': '0V', 'approved': False},
                    'REJECT_GND': {
                        'voltage': '3.3V',
                        'reason': 'validation_failed',
                        'failed_rules': [r['rule_name'] for r in failed_rules]
                    }
                }
            
        except Exception as e:
            self.current_state = "FAULT_VALIDATION_ERROR"
            return {
                'VALIDATION_S': {'voltage': '0V', 'validation_result': 'error'},
                'APPROVAL_S': {'voltage': '0V', 'approved': False},
                'REJECT_GND': {
                    'voltage': '3.3V',
                    'reason': 'validation_error',
                    'error': str(e)
                }
            }
    
    def _apply_validation_rule(self, rule: Dict, data: Dict, quality: float) -> Dict:
        """Apply a single validation rule"""
        rule_name = rule.get('name', 'unknown_rule')
        rule_type = rule.get('type', 'existence')
        
        try:
            if rule_type == 'existence':
                required_field = rule.get('field', '')
                passed = required_field in data
                confidence = 1.0 if passed else 0.0
            
            elif rule_type == 'quality_threshold':
                threshold = rule.get('threshold', 0.8)
                passed = quality >= threshold
                confidence = quality if passed else quality / threshold
            
            elif rule_type == 'format':
                field = rule.get('field', '')
                pattern = rule.get('pattern', '')
                # Simplified format check
                passed = field in data and isinstance(data[field], str)
                confidence = 1.0 if passed else 0.0
            
            else:
                passed = True  # Unknown rule type passes by default
                confidence = 0.5
            
            return {
                'rule_name': rule_name,
                'rule_type': rule_type,
                'passed': passed,
                'confidence': confidence
            }
            
        except Exception as e:
            return {
                'rule_name': rule_name,
                'rule_type': rule_type,
                'passed': False,
                'confidence': 0.0,
                'error': str(e)
            }

class ConditionGateSignal(ElectricalNode):
    """Condition gate control signal (S) - conditional flow control"""
    
    def __init__(self, node_id: str, name: str, condition_function: Callable):
        super().__init__(node_id, name, NodePolarity.CONTROL_SIGNAL)
        self.condition_function = condition_function
        self.gate_state = "closed"
        self.evaluations = 0
        
        # Condition gate pins
        self.add_pin("CONDITION_IN_+", FlowType.INPUT, "input", "3.3V", 20)
        self.add_pin("GATE_CONTROL_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("PASS_THROUGH_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("BLOCK_GND", FlowType.OUTPUT, "output", "0V", 5)
    
    def process_electrical_flow(self, input_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate condition and control gate state"""
        
        condition_input = input_flows.get('CONDITION_IN_+', {})
        
        try:
            self.current_state = "EVALUATING_CONDITION"
            
            # Evaluate condition
            condition_context = {
                'input_data': condition_input.get('data', {}),
                'input_quality': condition_input.get('quality', 1.0),
                'gate_history': getattr(self, 'evaluation_history', []),
                'current_state': self.gate_state
            }
            
            condition_result = self.condition_function(condition_context)
            should_open = condition_result.get('gate_open', False)
            confidence = condition_result.get('confidence', 1.0)
            
            # Update gate state
            self.gate_state = "open" if should_open else "closed"
            self.evaluations += 1
            
            self.current_state = f"GATE_{self.gate_state.upper()}"
            
            if should_open:
                return {
                    'GATE_CONTROL_S': {
                        'voltage': '3.3V',
                        'control_signal': 'gate_open',
                        'gate_state': 'open',
                        'confidence': confidence,
                        'evaluations': self.evaluations
                    },
                    'PASS_THROUGH_S': {
                        'voltage': '3.3V',
                        'control_signal': 'allow_passage',
                        'condition_met': True,
                        'pass_data': condition_input.get('data', {})
                    },
                    'BLOCK_GND': {'voltage': '0V', 'blocked': False}
                }
            else:
                return {
                    'GATE_CONTROL_S': {
                        'voltage': '3.3V',
                        'control_signal': 'gate_closed',
                        'gate_state': 'closed',
                        'confidence': confidence,
                        'evaluations': self.evaluations
                    },
                    'PASS_THROUGH_S': {'voltage': '0V', 'control_signal': 'block_passage'},
                    'BLOCK_GND': {
                        'voltage': '3.3V',
                        'blocked': True,
                        'reason': condition_result.get('reason', 'condition_not_met')
                    }
                }
            
        except Exception as e:
            self.current_state = "FAULT_CONDITION_ERROR"
            return {
                'GATE_CONTROL_S': {'voltage': '0V', 'control_signal': 'gate_error'},
                'PASS_THROUGH_S': {'voltage': '0V', 'control_signal': 'error'},
                'BLOCK_GND': {
                    'voltage': '3.3V',
                    'blocked': True,
                    'reason': 'condition_evaluation_error',
                    'error': str(e)
                }
            }

# Control Signal Factory

class ControlSignalFactory:
    """Factory for creating control signal nodes"""
    
    @staticmethod
    def create_router(node_id: str, name: str, routing_logic: Callable) -> RouterControlSignal:
        """Create router control signal"""
        return RouterControlSignal(node_id, name, routing_logic)
    
    @staticmethod
    def create_scheduler(node_id: str, name: str, policy: str = "fifo") -> SchedulerControlSignal:
        """Create scheduler control signal"""
        return SchedulerControlSignal(node_id, name, policy)
    
    @staticmethod
    def create_validator(node_id: str, name: str, rules: List[Dict]) -> ValidatorControlSignal:
        """Create validator control signal"""
        return ValidatorControlSignal(node_id, name, rules)
    
    @staticmethod
    def create_condition_gate(node_id: str, name: str, condition_func: Callable) -> ConditionGateSignal:
        """Create condition gate control signal"""
        return ConditionGateSignal(node_id, name, condition_func)

# Demo Functions

def demo_control_signals():
    """Demonstrate control signal nodes"""
    print("Control Signal Nodes Demo (S)")
    print("=" * 40)
    
    # Router demo
    def sample_routing_logic(context):
        data_size = len(str(context.get('data_characteristics', {})))
        return {
            'route_id': 'route_A' if data_size < 100 else 'route_B',
            'confidence': 0.9,
            'reasoning': f'Based on data size: {data_size}'
        }
    
    print("\n[S] Router Control Signal:")
    router = ControlSignalFactory.create_router(
        "router_ctrl", "Data Router", sample_routing_logic
    )
    
    # Scheduler demo
    print("\n[S] Scheduler Control Signal:")
    scheduler = ControlSignalFactory.create_scheduler(
        "sched_ctrl", "Task Scheduler", "priority"
    )
    
    # Validator demo
    validation_rules = [
        {'name': 'data_exists', 'type': 'existence', 'field': 'content'},
        {'name': 'quality_check', 'type': 'quality_threshold', 'threshold': 0.8}
    ]
    
    print("\n[S] Validator Control Signal:")
    validator = ControlSignalFactory.create_validator(
        "valid_ctrl", "Data Validator", validation_rules
    )
    
    # Condition gate demo
    def sample_condition(context):
        quality = context.get('input_quality', 0.0)
        return {
            'gate_open': quality > 0.7,
            'confidence': quality,
            'reason': f'Quality threshold check: {quality}'
        }
    
    print("\n[S] Condition Gate Control Signal:")
    gate = ControlSignalFactory.create_condition_gate(
        "gate_ctrl", "Quality Gate", sample_condition
    )
    
    print("\nControl signals created successfully!")
    print("All control signals ready to orchestrate the process (S)")

if __name__ == "__main__":
    demo_control_signals()