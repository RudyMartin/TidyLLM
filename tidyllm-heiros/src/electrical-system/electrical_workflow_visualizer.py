"""
Electrical Workflow Visualizer for TidyLLM-HeirOS
Engineer-friendly circuit diagram visualization of workflow execution
Power (+), Ground (-), Signal (s) separation with real-time monitoring
"""

import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json

class CircuitSymbol(Enum):
    """ASCII circuit symbols for engineer-friendly visualization"""
    # Power symbols
    VCC_RAIL = "━━━┳━━━"
    POWER_SOURCE = "[PWR]"
    VOLTAGE_REG = "[REG]"
    
    # Ground symbols  
    GROUND_PLANE = "━━━┻━━━"
    CHASSIS_GND = "[GND]"
    EARTH_GND = "[⏚]"
    
    # Signal symbols
    SIGNAL_LINE = "────○────"
    SIGNAL_BUS = "════○════"
    SIGNAL_DIFF = "────○○────"
    
    # Component symbols
    RESISTOR = "──[R]──"
    CAPACITOR = "──||──"
    INDUCTOR = "──◊◊──"
    AMPLIFIER = "─▷─"
    INVERTER = "─▷○─"
    SWITCH = "─/○─"
    
    # Logic symbols
    AND_GATE = "[&]"
    OR_GATE = "[≥1]"
    NOT_GATE = "[1]"
    BUFFER = "[>]"
    
    # Connection symbols
    JUNCTION = "●"
    CROSSOVER = "┼"
    TEE = "┬"
    CORNER = "┐"

@dataclass
class CircuitNode:
    """Visual node for circuit diagram"""
    node_id: str
    symbol: str
    position: tuple[int, int]
    connections: List[str] = field(default_factory=list)
    voltage: float = 0.0
    current: float = 0.0
    power: float = 0.0
    status: str = "inactive"
    label: str = ""

@dataclass
class CircuitConnection:
    """Visual connection between nodes"""
    from_node: str
    to_node: str
    connection_type: str  # power, ground, signal
    signal_name: str = ""
    impedance: float = 50.0
    current_value: Any = None
    
class ElectricalWorkflowVisualizer:
    """Engineer-friendly workflow visualization with circuit diagrams"""
    
    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.nodes: Dict[str, CircuitNode] = {}
        self.connections: List[CircuitConnection] = []
        self.power_rails: Dict[str, Dict] = {}
        self.ground_planes: Dict[str, Dict] = {}
        self.signal_traces: Dict[str, Dict] = {}
        
        # Display settings
        self.width = 120
        self.height = 40
        self.grid: List[List[str]] = []
        self.legend_shown = False
        
        self._init_grid()
        
    def _init_grid(self):
        """Initialize display grid"""
        self.grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
    def add_power_rail(self, rail_id: str, voltage: str, position: int, capacity: int = 1000):
        """Add power rail to visualization"""
        self.power_rails[rail_id] = {
            "voltage": voltage,
            "position": position,
            "capacity": capacity,
            "current_load": 0,
            "status": "active"
        }
        
        # Draw power rail across top
        rail_symbol = f"-- {voltage} Rail ({capacity}mA) --"
        self._draw_horizontal_line(1, position, rail_symbol, "=")
        
    def add_ground_plane(self, ground_id: str, position: int):
        """Add ground plane to visualization"""
        self.ground_planes[ground_id] = {
            "position": position,
            "connections": 0,
            "status": "ready"
        }
        
        # Draw ground plane across bottom
        ground_symbol = "-- Ground Plane --"
        self._draw_horizontal_line(self.height - 2, position, ground_symbol, "=")
        
    def add_component(self, node_id: str, symbol: str, position: tuple[int, int], 
                     label: str = "", component_type: str = "logic"):
        """Add component to circuit diagram"""
        x, y = position
        
        node = CircuitNode(
            node_id=node_id,
            symbol=symbol,
            position=position,
            label=label,
            status="inactive"
        )
        
        self.nodes[node_id] = node
        
        # Draw component on grid
        self._place_component(x, y, symbol, label)
        
    def add_signal_trace(self, trace_id: str, from_pos: tuple[int, int], 
                        to_pos: tuple[int, int], signal_type: str = "digital"):
        """Add signal trace between components"""
        self.signal_traces[trace_id] = {
            "from": from_pos,
            "to": to_pos,
            "type": signal_type,
            "value": None,
            "active": False
        }
        
        # Draw signal trace
        self._draw_trace(from_pos, to_pos, signal_type)
        
    def update_component_status(self, node_id: str, status: str, 
                              voltage: float = 0.0, current: float = 0.0):
        """Update component status and electrical values"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.status = status
            node.voltage = voltage
            node.current = current
            node.power = voltage * current / 1000  # mW
            
    def update_signal_value(self, trace_id: str, value: Any, active: bool = True):
        """Update signal trace value"""
        if trace_id in self.signal_traces:
            self.signal_traces[trace_id]["value"] = value
            self.signal_traces[trace_id]["active"] = active
            
    def _place_component(self, x: int, y: int, symbol: str, label: str):
        """Place component symbol on grid"""
        # Ensure position is within bounds
        if 0 <= y < self.height and 0 <= x < self.width - len(symbol):
            # Place main symbol
            for i, char in enumerate(symbol):
                if x + i < self.width:
                    self.grid[y][x + i] = char
                    
            # Place label below if space
            if label and y + 1 < self.height:
                label_start = max(0, x + len(symbol)//2 - len(label)//2)
                for i, char in enumerate(label[:self.width - label_start]):
                    if label_start + i < self.width:
                        self.grid[y + 1][label_start + i] = char
                        
    def _draw_horizontal_line(self, y: int, start_x: int, text: str, char: str):
        """Draw horizontal line with text"""
        if 0 <= y < self.height:
            # Draw line
            for x in range(start_x, min(self.width, start_x + len(text) + 10)):
                self.grid[y][x] = char
                
            # Place text
            text_pos = start_x + 2
            for i, c in enumerate(text):
                if text_pos + i < self.width:
                    self.grid[y][text_pos + i] = c
                    
    def _draw_trace(self, from_pos: tuple[int, int], to_pos: tuple[int, int], 
                   trace_type: str):
        """Draw signal trace between positions"""
        x1, y1 = from_pos
        x2, y2 = to_pos
        
        # Choose trace character based on type
        trace_chars = {
            "digital": "-",
            "analog": "~",
            "power": "=", 
            "ground": ".",
            "differential": "="
        }
        
        char = trace_chars.get(trace_type, "-")
        
        # Simple horizontal/vertical routing
        if y1 == y2:  # Horizontal
            start_x, end_x = (min(x1, x2), max(x1, x2))
            for x in range(start_x, end_x + 1):
                if 0 <= y1 < self.height and 0 <= x < self.width:
                    self.grid[y1][x] = char
        elif x1 == x2:  # Vertical
            start_y, end_y = (min(y1, y2), max(y1, y2))
            for y in range(start_y, end_y + 1):
                if 0 <= y < self.height and 0 <= x1 < self.width:
                    self.grid[y][x1] = "|" if trace_type == "digital" else "|"
        else:  # L-shaped routing
            # Horizontal first, then vertical
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if 0 <= y1 < self.height and 0 <= x < self.width:
                    self.grid[y1][x] = char
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if 0 <= y < self.height and 0 <= x2 < self.width:
                    self.grid[y][x2] = "|" if trace_type == "digital" else "|"
            # Add corner
            if 0 <= y1 < self.height and 0 <= x2 < self.width:
                self.grid[y1][x2] = "+" if y2 > y1 else "+"
                
    def render_circuit_diagram(self) -> str:
        """Render complete circuit diagram"""
        output = []
        
        # Title
        title = f"[== {self.workflow_name} - Circuit Diagram ==]"
        output.append(title)
        output.append("")
        
        # Legend
        if not self.legend_shown:
            output.append("Legend:")
            output.append("  Power: ===  Signal: ---  Ground: ...")
            output.append("  Active: [*]  Inactive: [ ]  Error: [!]")
            output.append("")
            
        # Circuit diagram
        for row in self.grid:
            output.append("".join(row))
            
        output.append("")
        
        # Status panel
        output.append("══ System Status ══")
        
        # Power rail status
        for rail_id, rail in self.power_rails.items():
            load_percent = (rail["current_load"] / rail["capacity"]) * 100
            status_char = "*" if rail["status"] == "active" else " "
            output.append(f"[{status_char}] {rail['voltage']} Rail: {rail['current_load']}/{rail['capacity']}mA ({load_percent:.1f}%)")
            
        # Component status
        active_components = [n for n in self.nodes.values() if n.status == "active"]
        if active_components:
            output.append(f"Active Components: {len(active_components)}")
            for node in active_components:
                power_str = f"{node.power:.1f}mW" if node.power > 0 else "N/A"
                output.append(f"  {node.label or node.node_id}: {node.voltage:.1f}V, {node.current:.1f}mA, {power_str}")
                
        # Signal activity
        active_signals = {k: v for k, v in self.signal_traces.items() if v["active"]}
        if active_signals:
            output.append(f"Active Signals: {len(active_signals)}")
            for trace_id, trace in active_signals.items():
                value_str = str(trace["value"])[:20] if trace["value"] else "None"
                output.append(f"  {trace_id}: {value_str}")
                
        return "\n".join(output)
        
    def render_timing_diagram(self, execution_log: List[Dict]) -> str:
        """Render timing diagram of execution"""
        output = []
        
        output.append(f"[== {self.workflow_name} - Timing Diagram ==]")
        output.append("")
        
        if not execution_log:
            output.append("No execution data available")
            return "\n".join(output)
            
        # Time axis
        max_time = max(log.get("timestamp", 0) for log in execution_log)
        scale = 50 / max_time if max_time > 0 else 1
        
        output.append("Time: 0" + " " * 45 + f"{max_time:.2f}s")
        output.append("      " + "-" * 50)
        
        # Component timeline
        components = set()
        for log in execution_log:
            if "component" in log:
                components.add(log["component"])
                
        for comp in sorted(components):
            line = f"{comp:8s}: "
            timeline = [" "] * 50
            
            for log in execution_log:
                if log.get("component") == comp:
                    time_pos = int(log.get("timestamp", 0) * scale)
                    if 0 <= time_pos < 50:
                        if log.get("status") == "start":
                            timeline[time_pos] = "["
                        elif log.get("status") == "active":
                            timeline[time_pos] = "="
                        elif log.get("status") == "complete":
                            timeline[time_pos] = "]"
                        elif log.get("status") == "error":
                            timeline[time_pos] = "!"
                            
            line += "".join(timeline)
            output.append(line)
            
        output.append("")
        output.append("Legend: [ Start, = Active, ] Complete, ! Error")
        
        return "\n".join(output)
        
    def render_power_analysis(self) -> str:
        """Render power consumption analysis"""
        output = []
        
        output.append(f"[== {self.workflow_name} - Power Analysis ==]")
        output.append("")
        
        total_power = sum(node.power for node in self.nodes.values())
        total_current = sum(node.current for node in self.nodes.values())
        
        output.append(f"Total System Power: {total_power:.2f}mW")
        output.append(f"Total Current Draw: {total_current:.1f}mA") 
        output.append("")
        
        # Power breakdown by component
        output.append("Component Power Breakdown:")
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.power, reverse=True)
        
        for node in sorted_nodes:
            if node.power > 0:
                percent = (node.power / total_power * 100) if total_power > 0 else 0
                bar_length = int(percent / 2)  # Scale to 50 chars max
                bar = "=" * bar_length + "." * (25 - bar_length)
                output.append(f"{node.label or node.node_id:15s}: {node.power:6.2f}mW [{bar}] {percent:5.1f}%")
                
        # Power efficiency metrics
        output.append("")
        output.append("Power Efficiency Metrics:")
        
        if total_power > 0:
            # Calculate efficiency metrics
            active_nodes = [n for n in self.nodes.values() if n.status == "active"]
            efficiency = len(active_nodes) / len(self.nodes) * 100 if self.nodes else 0
            
            output.append(f"Component Utilization: {len(active_nodes)}/{len(self.nodes)} ({efficiency:.1f}%)")
            
            avg_power = total_power / len(self.nodes) if self.nodes else 0
            output.append(f"Average Power per Component: {avg_power:.2f}mW")
            
            # Power density (power per unit area - simplified)
            power_density = total_power / (self.width * self.height) * 10000  # scaled
            output.append(f"Power Density: {power_density:.3f}mW/unit^2")
            
        return "\n".join(output)

class WorkflowElectricalMonitor:
    """Real-time monitoring of electrical workflow execution"""
    
    def __init__(self, visualizer: ElectricalWorkflowVisualizer):
        self.visualizer = visualizer
        self.execution_log: List[Dict] = []
        self.start_time: float = 0
        
    def start_monitoring(self):
        """Start execution monitoring"""
        self.start_time = time.time()
        self.execution_log = []
        
    def log_component_start(self, component_id: str, voltage: float = 5.0, current: float = 100.0):
        """Log component start"""
        timestamp = time.time() - self.start_time
        self.execution_log.append({
            "timestamp": timestamp,
            "component": component_id,
            "status": "start",
            "voltage": voltage,
            "current": current
        })
        
        self.visualizer.update_component_status(component_id, "active", voltage, current)
        
    def log_component_complete(self, component_id: str, result: Any = None):
        """Log component completion"""
        timestamp = time.time() - self.start_time
        self.execution_log.append({
            "timestamp": timestamp,
            "component": component_id,
            "status": "complete",
            "result": str(result)[:50] if result else None
        })
        
        self.visualizer.update_component_status(component_id, "complete", 0, 0)
        
    def log_signal_update(self, trace_id: str, value: Any):
        """Log signal update"""
        timestamp = time.time() - self.start_time
        self.execution_log.append({
            "timestamp": timestamp,
            "signal": trace_id,
            "value": str(value)[:50] if value else None
        })
        
        self.visualizer.update_signal_value(trace_id, value, True)
        
    def log_error(self, component_id: str, error: str):
        """Log component error"""
        timestamp = time.time() - self.start_time
        self.execution_log.append({
            "timestamp": timestamp,
            "component": component_id,
            "status": "error",
            "error": error
        })
        
        self.visualizer.update_component_status(component_id, "error", 0, 0)
        
    def generate_real_time_display(self) -> str:
        """Generate real-time display of system status"""
        output = []
        
        # System overview
        current_time = time.time() - self.start_time
        output.append(f"[== REAL-TIME ELECTRICAL MONITOR ==]")
        output.append(f"Runtime: {current_time:.2f}s")
        output.append("")
        
        # Recent activity (last 5 events)
        recent_events = self.execution_log[-5:] if len(self.execution_log) > 5 else self.execution_log
        
        if recent_events:
            output.append("Recent Activity:")
            for event in recent_events:
                timestamp = f"{event['timestamp']:6.2f}s"
                if "component" in event:
                    status_icon = {"start": "[+]", "active": "[*]", "complete": "[OK]", "error": "[!]"}
                    icon = status_icon.get(event["status"], "[ ]")
                    output.append(f"  {timestamp} {icon} {event['component']}: {event['status']}")
                elif "signal" in event:
                    output.append(f"  {timestamp} [~] Signal {event['signal']}: {event.get('value', 'N/A')}")
        else:
            output.append("No recent activity")
            
        output.append("")
        
        # Live circuit diagram
        circuit_diagram = self.visualizer.render_circuit_diagram()
        output.append(circuit_diagram)
        
        return "\n".join(output)

# Demo and Examples

async def demo_electrical_workflow_visualizer():
    """Demonstrate electrical workflow visualization"""
    print("=" * 80)
    print("Electrical Workflow Visualizer Demo")
    print("Engineer-friendly circuit diagrams for workflow execution")
    print("=" * 80)
    
    # Create visualizer for MVR workflow
    viz = ElectricalWorkflowVisualizer("MVR Document Processing")
    monitor = WorkflowElectricalMonitor(viz)
    
    # Set up power infrastructure
    viz.add_power_rail("main_5v", "5V", 10, 2000)
    viz.add_power_rail("logic_3v3", "3.3V", 15, 500)
    viz.add_ground_plane("main_gnd", 35)
    
    # Add components
    viz.add_component("doc_input", "[INPUT]", (5, 20), "Document Input", "io")
    viz.add_component("classifier", "[CLASS]", (25, 20), "ML Classifier", "logic")
    viz.add_component("validator", "[VALID]", (45, 20), "Validator", "logic")
    viz.add_component("analyzer", "[ANALYZE]", (65, 20), "Analyzer", "processing")
    viz.add_component("output", "[OUTPUT]", (85, 20), "Report Output", "io")
    
    # Add signal traces
    viz.add_signal_trace("doc_data", (15, 20), (25, 20), "digital")
    viz.add_signal_trace("class_result", (35, 20), (45, 20), "digital") 
    viz.add_signal_trace("valid_result", (55, 20), (65, 20), "digital")
    viz.add_signal_trace("analysis", (75, 20), (85, 20), "digital")
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate workflow execution
    components = ["doc_input", "classifier", "validator", "analyzer", "output"]
    voltages = [3.3, 5.0, 3.3, 5.0, 3.3]
    currents = [50, 200, 100, 300, 75]
    
    print("\n" + "=" * 50)
    print("Starting workflow execution...")
    print("=" * 50)
    
    for i, (comp, voltage, current) in enumerate(zip(components, voltages, currents)):
        print(f"\n[Step {i+1}] Activating {comp}...")
        
        # Log component start
        monitor.log_component_start(comp, voltage, current)
        
        # Show real-time display
        print(monitor.generate_real_time_display())
        
        # Simulate processing time
        await asyncio.sleep(1)
        
        # Log completion with result
        result = f"Processed by {comp}"
        monitor.log_component_complete(comp, result)
        
        # Update signal traces
        if i < len(components) - 1:
            trace_name = f"signal_{i}"
            monitor.log_signal_update(trace_name, result)
    
    print("\n" + "=" * 50)
    print("Workflow execution completed!")
    print("=" * 50)
    
    # Final reports
    print("\n" + viz.render_timing_diagram(monitor.execution_log))
    print("\n" + viz.render_power_analysis())

if __name__ == "__main__":
    print("Electrical Workflow Visualizer")
    print("Circuit diagrams for engineer-friendly workflow monitoring")
    print("Power (+), Ground (-), Signal (s) separation visualization\n")
    
    # Run demo
    asyncio.run(demo_electrical_workflow_visualizer())