"""
Simple ASCII-safe demo of electrical workflow visualization
"""

from electrical_workflow_visualizer import ElectricalWorkflowVisualizer, WorkflowElectricalMonitor
import asyncio
import time

async def simple_demo():
    """Simple demo with ASCII-only output"""
    print("Electrical Workflow Visualizer - Simple Demo")
    print("=" * 50)
    
    # Create visualizer
    viz = ElectricalWorkflowVisualizer("Simple MVR")
    monitor = WorkflowElectricalMonitor(viz)
    
    # Add basic components (simplified)
    viz.nodes["input"] = type('obj', (object,), {
        'node_id': 'input', 'status': 'inactive', 'voltage': 0.0, 
        'current': 0.0, 'power': 0.0, 'label': 'Input'
    })()
    
    viz.nodes["process"] = type('obj', (object,), {
        'node_id': 'process', 'status': 'inactive', 'voltage': 0.0, 
        'current': 0.0, 'power': 0.0, 'label': 'Processor'
    })()
    
    viz.nodes["output"] = type('obj', (object,), {
        'node_id': 'output', 'status': 'inactive', 'voltage': 0.0, 
        'current': 0.0, 'power': 0.0, 'label': 'Output'
    })()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate execution
    components = ["input", "process", "output"]
    voltages = [3.3, 5.0, 3.3]
    currents = [50, 200, 75]
    
    for comp, voltage, current in zip(components, voltages, currents):
        print(f"\nActivating {comp}...")
        
        # Update component
        monitor.log_component_start(comp, voltage, current)
        
        # Show status
        print(f"  Status: Active at {voltage}V, {current}mA")
        print(f"  Power: {voltage * current / 1000:.2f}mW")
        
        await asyncio.sleep(1)
        
        # Complete
        monitor.log_component_complete(comp, f"Result from {comp}")
    
    print("\nExecution completed!")
    
    # Show final stats
    total_power = sum(node.power for node in viz.nodes.values())
    print(f"Total power consumed: {total_power:.2f}mW")
    
    # Show timing info
    print(f"Total execution time: {time.time() - monitor.start_time:.2f}s")
    print(f"Events logged: {len(monitor.execution_log)}")

if __name__ == "__main__":
    asyncio.run(simple_demo())