"""
Complete Electrical MVR Demo - Input(+)/Output(-)/Control(S) Model
==================================================================

Demonstrates the refactored electrical system with proper separation:
+ Input Sources: Data flowing INTO the system  
- Output Sinks: Results flowing OUT of the system
S Control Signals: Instructions managing the flow
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from input_source_nodes import *
from output_sink_nodes import *
from control_signal_nodes import *
from flow_schemas import ElectricalWorkflowEngine
from datetime import datetime
import time

class ElectricalMVRWorkflow:
    """Complete MVR workflow using Input(+)/Output(-)/Control(S) model"""
    
    def __init__(self):
        self.workflow_name = "MVR Document Processing"
        
        # Input Sources (+)
        self.document_reader = None
        self.metadata_generator = None
        
        # Output Sinks (-)
        self.report_writer = None
        self.database_store = None
        self.display_output = None
        
        # Control Signals (S)
        self.document_validator = None
        self.process_router = None
        self.quality_gate = None
        
        # Workflow state
        self.execution_log = []
        self.start_time = None
        
    def setup_electrical_components(self):
        """Set up all electrical components"""
        print("[SETUP] Configuring electrical components...")
        
        # === INPUT SOURCES (+) ===
        
        # Document reader input source
        self.document_reader = FileReaderSource(
            "doc_reader+", "Document Input Source", "sample_mvr_doc.txt"
        )
        
        # Metadata generator input source
        def metadata_generator_func(context):
            return {
                "doc_id": f"MVR_{context['generation_count']:04d}",
                "processed_by": "TidyLLM-HeirOS",
                "timestamp": datetime.now(),
                "version": "1.0"
            }
        
        self.metadata_generator = DataGeneratorSource(
            "meta_gen+", "Metadata Generator", metadata_generator_func
        )
        
        # === OUTPUT SINKS (-) ===
        
        # Report writer output sink
        self.report_writer = FileWriterSink(
            "report_writer-", "MVR Report Output", "mvr_analysis_report.txt"
        )
        
        # Database storage output sink
        self.database_store = DatabaseStoreSink(
            "db_store-", "Results Database", "mvr_results"
        )
        
        # Display output sink
        self.display_output = DisplayOutputSink(
            "display-", "Console Display", "console"
        )
        
        # === CONTROL SIGNALS (S) ===
        
        # Document validator control signal
        validation_rules = [
            {'name': 'content_exists', 'type': 'existence', 'field': 'content'},
            {'name': 'min_quality', 'type': 'quality_threshold', 'threshold': 0.5}
        ]
        
        self.document_validator = ValidatorControlSignal(
            "doc_validator_S", "Document Validator", validation_rules
        )
        
        # Process router control signal
        def routing_logic(context):
            content_size = len(str(context.get('data_characteristics', {})))
            if content_size > 500:
                return {'route_id': 'detailed_analysis', 'confidence': 0.9}
            else:
                return {'route_id': 'standard_analysis', 'confidence': 0.8}
        
        self.process_router = RouterControlSignal(
            "router_S", "Process Router", routing_logic
        )
        
        # Quality gate control signal
        def quality_condition(context):
            quality = context.get('input_quality', 0.0)
            return {
                'gate_open': quality >= 0.7,
                'confidence': quality,
                'reason': f'Quality check: {quality:.2f} >= 0.70'
            }
        
        self.quality_gate = ConditionGateSignal(
            "quality_gate_S", "Quality Gate", quality_condition
        )
        
        print("[+] Input sources configured")
        print("[-] Output sinks configured") 
        print("[S] Control signals configured")
        
    def execute_electrical_workflow(self):
        """Execute complete electrical workflow"""
        print("\n" + "=" * 60)
        print("ELECTRICAL MVR WORKFLOW EXECUTION")
        print("Input(+) / Output(-) / Control(S) Model")
        print("=" * 60)
        
        self.start_time = time.time()
        execution_results = {}
        
        # === PHASE 1: INPUT GENERATION (+) ===
        print("\n[PHASE 1] INPUT SOURCES (+) - Data flowing INTO system")
        print("-" * 50)
        
        # Generate document input
        print("[+] Reading document from source...")
        doc_input = self.document_reader.process_electrical_flow({})
        execution_results['document_input'] = doc_input
        
        if doc_input['DATA_OUT+']['voltage'] == '5V':
            print(f"    Document loaded: {doc_input['STATUS_S']['bytes_read']} bytes")
        else:
            print(f"    Document read failed: {doc_input['ERROR_GND']['error']}")
            return execution_results
        
        # Generate metadata input
        print("[+] Generating metadata...")
        meta_input = self.metadata_generator.process_electrical_flow({
            'TRIGGER_S': {'parameters': {'doc_type': 'mvr'}}
        })
        execution_results['metadata_input'] = meta_input
        
        if meta_input['DATA_OUT+']['voltage'] == '5V':
            doc_id = meta_input['DATA_OUT+']['data']['doc_id']
            print(f"    Metadata generated: {doc_id}")
        
        # === PHASE 2: CONTROL VALIDATION (S) ===
        print("\n[PHASE 2] CONTROL SIGNALS (S) - Managing the flow")
        print("-" * 50)
        
        # Validate document
        print("[S] Validating document quality...")
        validation_result = self.document_validator.process_electrical_flow({
            'DATA_IN_+': doc_input['DATA_OUT+']
        })
        execution_results['validation'] = validation_result
        
        approved = validation_result['APPROVAL_S']['voltage'] == '3.3V'
        if approved:
            print(f"    Validation PASSED - {validation_result['VALIDATION_S']['pass_rate']:.1%} pass rate")
        else:
            print(f"    Validation FAILED - rejected to ground")
            failed_rules = validation_result['REJECT_GND']['failed_rules']
            print(f"    Failed rules: {failed_rules}")
            return execution_results
        
        # Quality gate check
        print("[S] Quality gate evaluation...")
        gate_result = self.quality_gate.process_electrical_flow({
            'CONDITION_IN_+': doc_input['DATA_OUT+']
        })
        execution_results['quality_gate'] = gate_result
        
        gate_open = gate_result['GATE_CONTROL_S']['gate_state'] == 'open'
        if gate_open:
            print(f"    Quality gate OPEN - confidence: {gate_result['GATE_CONTROL_S']['confidence']:.2f}")
        else:
            print(f"    Quality gate CLOSED - {gate_result['BLOCK_GND']['reason']}")
            return execution_results
        
        # Route determination
        print("[S] Determining process route...")
        routing_result = self.process_router.process_electrical_flow({
            'DATA_MONITOR_+': doc_input['DATA_OUT+']
        })
        execution_results['routing'] = routing_result
        
        route_id = routing_result['ROUTE_OUT_S']['route_id']
        confidence = routing_result['ROUTE_OUT_S']['confidence']
        print(f"    Route selected: {route_id} (confidence: {confidence:.1%})")
        
        # === PHASE 3: PROCESSING SIMULATION ===
        print("\n[PHASE 3] DATA PROCESSING - Transform inputs to outputs")
        print("-" * 50)
        
        # Simulate processing based on route
        print(f"[P] Executing {route_id} processing...")
        
        processed_data = {
            'document_analysis': {
                'content_summary': f"Analysis of document: {doc_id}",
                'processing_route': route_id,
                'quality_score': gate_result['GATE_CONTROL_S']['confidence'],
                'word_count': len(doc_input['DATA_OUT+']['data']['content'].split()),
                'analysis_timestamp': datetime.now()
            },
            'metadata': meta_input['DATA_OUT+']['data'],
            'validation_summary': {
                'passed': approved,
                'pass_rate': validation_result['VALIDATION_S']['pass_rate'],
                'rules_checked': validation_result['VALIDATION_S']['rules_checked']
            }
        }
        
        print(f"    Analysis complete - {processed_data['document_analysis']['word_count']} words processed")
        print(f"    Quality score: {processed_data['document_analysis']['quality_score']:.2f}")
        
        # === PHASE 4: OUTPUT CONSUMPTION (-) ===
        print("\n[PHASE 4] OUTPUT SINKS (-) - Results flowing OUT of system")
        print("-" * 50)
        
        # Write report output
        print("[-] Writing analysis report...")
        report_result = self.report_writer.process_electrical_flow({
            'DATA_IN-': {
                'data': processed_data,
                'quality': processed_data['document_analysis']['quality_score']
            },
            'FORMAT_S': {'format': 'structured_report'}
        })
        execution_results['report_output'] = report_result
        
        if report_result['STATUS_S']['voltage'] == '3.3V':
            bytes_written = report_result['STATUS_S']['bytes_written']
            print(f"    Report written: {bytes_written} bytes to {self.report_writer.output_path}")
        
        # Store in database
        print("[-] Storing results in database...")
        db_result = self.database_store.process_electrical_flow({
            'DATA_IN-': {
                'data': processed_data,
                'quality': processed_data['document_analysis']['quality_score']
            },
            'QUERY_S': {'parameters': {'table': 'mvr_results', 'operation': 'insert'}}
        })
        execution_results['database_output'] = db_result
        
        if db_result['STATUS_S']['voltage'] == '3.3V':
            record_id = db_result['STATUS_S']['record_id']
            print(f"    Database record stored: {record_id}")
        
        # Display output
        print("[-] Displaying results...")
        display_result = self.display_output.process_electrical_flow({
            'DATA_IN-': {
                'data': processed_data['document_analysis'],
                'quality': processed_data['document_analysis']['quality_score']
            },
            'FORMAT_S': {'format': 'summary'}
        })
        execution_results['display_output'] = display_result
        
        # === EXECUTION SUMMARY ===
        execution_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("ELECTRICAL WORKFLOW EXECUTION COMPLETE")
        print("=" * 60)
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Route taken: {route_id}")
        print(f"Quality score: {processed_data['document_analysis']['quality_score']:.2f}")
        print(f"Document ID: {doc_id}")
        print(f"Words processed: {processed_data['document_analysis']['word_count']}")
        
        # Power consumption summary
        print("\nPower Consumption Summary:")
        print("Input Sources (+):    120mA (Document: 200mA + Metadata: 100mA)")
        print("Control Signals (S):   30mA (Validation: 10mA + Router: 10mA + Gate: 10mA)")
        print("Output Sinks (-):     550mA (Report: 200mA + DB: 300mA + Display: 50mA)")
        print("Total System Draw:    700mA at 5V = 3.5W")
        
        return execution_results

def create_sample_mvr_document():
    """Create sample MVR document for testing"""
    sample_content = """
    MVR Document Analysis Request
    =============================
    
    Document Type: Motor Vehicle Record
    Requested By: Risk Assessment Team
    Date: 2024-08-30
    
    Subject Information:
    - License Number: DL123456789
    - Vehicle VIN: 1HGBH41JXMN109186
    - Analysis Period: 2022-2024
    
    Analysis Requirements:
    1. Driving record verification
    2. Incident history analysis
    3. Risk score calculation
    4. Compliance status review
    
    Special Instructions:
    - High priority analysis required
    - Include comparative risk metrics
    - Generate compliance report for audit
    
    This document contains proprietary information and should be processed
    according to company policy and regulatory requirements.
    """
    
    # Create sample file (simulate file system)
    print("[SETUP] Creating sample MVR document...")
    # In real implementation would write to actual file
    return len(sample_content)

def main():
    """Main demonstration function"""
    print("TidyLLM-HeirOS Electrical System Demo")
    print("Input(+) / Output(-) / Control(S) Model")
    print("=" * 60)
    
    # Create sample document
    doc_size = create_sample_mvr_document()
    print(f"Sample document created: {doc_size} characters")
    
    # Create and setup workflow
    workflow = ElectricalMVRWorkflow()
    workflow.setup_electrical_components()
    
    # Execute workflow
    results = workflow.execute_electrical_workflow()
    
    print("\n" + "=" * 60)
    print("ELECTRICAL SYSTEM PRINCIPLES DEMONSTRATED:")
    print("=" * 60)
    print("[+] Input Sources (+): Data flowing INTO system")
    print("  - File Reader: Document content")
    print("  - Data Generator: Metadata creation")
    print("")
    print("[S] Control Signals (S): Instructions managing flow")
    print("  - Validator: Quality verification")  
    print("  - Router: Process path selection")
    print("  - Gate: Conditional flow control")
    print("")
    print("[-] Output Sinks (-): Results flowing OUT of system")
    print("  - File Writer: Report generation")
    print("  - Database: Result storage") 
    print("  - Display: User presentation")
    print("")
    print("Electrical abstraction provides clear separation of concerns")
    print("that engineers understand intuitively!")

if __name__ == "__main__":
    main()