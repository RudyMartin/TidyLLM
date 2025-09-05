"""
Output Sink Nodes (-) - TidyLLM-HeirOS Electrical System
=======================================================

Output sinks receive results flowing OUT of the system
These are the negative (-) rail components that consume workflow outputs
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from flow_schemas import ElectricalNode, NodePolarity, FlowType

class OutputSinkType(Enum):
    """Types of output sinks"""
    FILE_WRITER = "file_writer"
    DATABASE_STORE = "database_store"
    API_SENDER = "api_sender"
    DISPLAY_OUTPUT = "display_output"
    EMAIL_SENDER = "email_sender"
    NOTIFICATION_SINK = "notification_sink"

class FileWriterSink(ElectricalNode):
    """File writing output sink (-) - consumes results to write to files"""
    
    def __init__(self, node_id: str, name: str, output_path: str):
        super().__init__(node_id, name, NodePolarity.OUTPUT_SINK)
        self.output_path = output_path
        self.write_mode = "text"  # text, binary, append
        self.bytes_written = 0
        
        # Output sink pins - only inputs data
        self.add_pin("DATA_IN-", FlowType.INPUT, "input", "5V", 200)
        self.add_pin("FORMAT_S", FlowType.CONTROL, "input", "3.3V", 10)
        self.add_pin("STATUS_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("WRITE_GND", FlowType.OUTPUT, "output", "0V", 5)
    
    def process_electrical_flow(self, input_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Consume data and write to file sink"""
        
        data_input = input_flows.get('DATA_IN-', {})
        format_control = input_flows.get('FORMAT_S', {})
        
        if not data_input:
            self.current_state = "FAULT_NO_DATA"
            return {
                'STATUS_S': {'voltage': '0V', 'status': 'no_input_data'},
                'WRITE_GND': {'voltage': '3.3V', 'error': 'no_data_to_write'}
            }
        
        try:
            self.current_state = "WRITING"
            
            # Extract data to write
            data_content = data_input.get('data', {})
            write_format = format_control.get('format', 'text')
            
            # Simulate file writing
            if isinstance(data_content, dict):
                content_str = str(data_content)
            else:
                content_str = str(data_content)
            
            # In real implementation, would write to actual file
            # with open(self.output_path, 'w') as file:
            #     file.write(content_str)
            
            self.bytes_written += len(content_str)
            self.current_state = "WRITE_COMPLETE"
            
            return {
                'STATUS_S': {
                    'voltage': '3.3V',
                    'status': 'write_complete',
                    'bytes_written': len(content_str),
                    'total_bytes': self.bytes_written,
                    'output_path': self.output_path
                },
                'WRITE_GND': {'voltage': '0V', 'error': None}
            }
            
        except Exception as e:
            self.current_state = "FAULT_WRITE_ERROR"
            self.fault_conditions.append(str(e))
            
            return {
                'STATUS_S': {'voltage': '0V', 'status': 'write_failed'},
                'WRITE_GND': {
                    'voltage': '3.3V',
                    'error': str(e),
                    'fault_type': 'file_write_error'
                }
            }

class DatabaseStoreSink(ElectricalNode):
    """Database storage output sink (-) - stores results in database"""
    
    def __init__(self, node_id: str, name: str, table_name: str):
        super().__init__(node_id, name, NodePolarity.OUTPUT_SINK)
        self.table_name = table_name
        self.records_stored = 0
        self.connection_status = "disconnected"
        
        # Database sink pins
        self.add_pin("DATA_IN-", FlowType.INPUT, "input", "5V", 300)
        self.add_pin("QUERY_S", FlowType.CONTROL, "input", "3.3V", 10)
        self.add_pin("STATUS_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("DB_GND", FlowType.OUTPUT, "output", "0V", 5)
    
    def process_electrical_flow(self, input_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Store data in database sink"""
        
        data_input = input_flows.get('DATA_IN-', {})
        query_control = input_flows.get('QUERY_S', {})
        
        if not data_input:
            return {
                'STATUS_S': {'voltage': '0V', 'status': 'no_data'},
                'DB_GND': {'voltage': '3.3V', 'error': 'no_data_to_store'}
            }
        
        try:
            self.current_state = "STORING"
            
            # Extract data for storage
            record_data = data_input.get('data', {})
            query_params = query_control.get('parameters', {})
            
            # Simulate database storage
            storage_record = {
                'table': self.table_name,
                'data': record_data,
                'query_params': query_params,
                'stored_at': datetime.now(),
                'record_id': f"rec_{self.records_stored + 1}"
            }
            
            self.records_stored += 1
            self.connection_status = "connected"
            self.current_state = "STORE_COMPLETE"
            
            return {
                'STATUS_S': {
                    'voltage': '3.3V',
                    'status': 'store_complete',
                    'records_stored': self.records_stored,
                    'table_name': self.table_name,
                    'record_id': storage_record['record_id']
                },
                'DB_GND': {'voltage': '0V', 'error': None}
            }
            
        except Exception as e:
            self.current_state = "FAULT_STORE_ERROR"
            return {
                'STATUS_S': {'voltage': '0V', 'status': 'store_failed'},
                'DB_GND': {
                    'voltage': '3.3V',
                    'error': str(e),
                    'fault_type': 'database_store_error'
                }
            }

class APISenderSink(ElectricalNode):
    """API sender output sink (-) - sends results to external APIs"""
    
    def __init__(self, node_id: str, name: str, endpoint_url: str):
        super().__init__(node_id, name, NodePolarity.OUTPUT_SINK)
        self.endpoint_url = endpoint_url
        self.requests_sent = 0
        self.success_rate = 1.0
        
        # API sender pins
        self.add_pin("DATA_IN-", FlowType.INPUT, "input", "5V", 200)
        self.add_pin("METHOD_S", FlowType.CONTROL, "input", "3.3V", 10)
        self.add_pin("STATUS_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("NETWORK_GND", FlowType.OUTPUT, "output", "0V", 5)
    
    def process_electrical_flow(self, input_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Send data to API endpoint sink"""
        
        data_input = input_flows.get('DATA_IN-', {})
        method_control = input_flows.get('METHOD_S', {})
        
        if not data_input:
            return {
                'STATUS_S': {'voltage': '0V', 'status': 'no_data'},
                'NETWORK_GND': {'voltage': '3.3V', 'error': 'no_data_to_send'}
            }
        
        try:
            self.current_state = "SENDING"
            
            # Extract data for API request
            request_data = data_input.get('data', {})
            http_method = method_control.get('method', 'POST')
            
            # Simulate API request
            api_request = {
                'url': self.endpoint_url,
                'method': http_method,
                'data': request_data,
                'sent_at': datetime.now(),
                'request_id': f"req_{self.requests_sent + 1}"
            }
            
            # Simulate response
            response_status = 200 if self.success_rate > 0.8 else 500
            
            self.requests_sent += 1
            self.current_state = "SEND_COMPLETE"
            
            return {
                'STATUS_S': {
                    'voltage': '3.3V',
                    'status': 'send_complete',
                    'response_status': response_status,
                    'requests_sent': self.requests_sent,
                    'request_id': api_request['request_id']
                },
                'NETWORK_GND': {'voltage': '0V', 'error': None}
            }
            
        except Exception as e:
            self.current_state = "FAULT_SEND_ERROR"
            return {
                'STATUS_S': {'voltage': '0V', 'status': 'send_failed'},
                'NETWORK_GND': {
                    'voltage': '3.3V',
                    'error': str(e),
                    'fault_type': 'api_send_error'
                }
            }

class DisplayOutputSink(ElectricalNode):
    """Display output sink (-) - renders results for user viewing"""
    
    def __init__(self, node_id: str, name: str, display_type: str = "console"):
        super().__init__(node_id, name, NodePolarity.OUTPUT_SINK)
        self.display_type = display_type  # console, gui, web
        self.displays_rendered = 0
        
        # Display sink pins
        self.add_pin("DATA_IN-", FlowType.INPUT, "input", "5V", 100)
        self.add_pin("FORMAT_S", FlowType.CONTROL, "input", "3.3V", 10)
        self.add_pin("RENDER_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("DISPLAY_GND", FlowType.OUTPUT, "output", "0V", 5)
    
    def process_electrical_flow(self, input_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Render data for display sink"""
        
        data_input = input_flows.get('DATA_IN-', {})
        format_control = input_flows.get('FORMAT_S', {})
        
        if not data_input:
            return {
                'RENDER_S': {'voltage': '0V', 'status': 'no_data'},
                'DISPLAY_GND': {'voltage': '3.3V', 'error': 'no_data_to_display'}
            }
        
        try:
            self.current_state = "RENDERING"
            
            # Extract data for display
            display_data = data_input.get('data', {})
            display_format = format_control.get('format', 'text')
            
            # Simulate rendering
            rendered_output = {
                'display_type': self.display_type,
                'format': display_format,
                'content': display_data,
                'rendered_at': datetime.now(),
                'display_id': f"disp_{self.displays_rendered + 1}"
            }
            
            self.displays_rendered += 1
            self.current_state = "DISPLAY_COMPLETE"
            
            # In console mode, actually display
            if self.display_type == "console":
                print(f"\n[DISPLAY OUTPUT] {self.name}:")
                print(f"Format: {display_format}")
                print(f"Content: {display_data}")
                print("-" * 40)
            
            return {
                'RENDER_S': {
                    'voltage': '3.3V',
                    'status': 'render_complete',
                    'displays_rendered': self.displays_rendered,
                    'display_id': rendered_output['display_id'],
                    'format_used': display_format
                },
                'DISPLAY_GND': {'voltage': '0V', 'error': None}
            }
            
        except Exception as e:
            self.current_state = "FAULT_RENDER_ERROR"
            return {
                'RENDER_S': {'voltage': '0V', 'status': 'render_failed'},
                'DISPLAY_GND': {
                    'voltage': '3.3V',
                    'error': str(e),
                    'fault_type': 'display_render_error'
                }
            }

class NotificationSink(ElectricalNode):
    """Notification output sink (-) - sends notifications and alerts"""
    
    def __init__(self, node_id: str, name: str, notification_type: str = "email"):
        super().__init__(node_id, name, NodePolarity.OUTPUT_SINK)
        self.notification_type = notification_type  # email, sms, slack, webhook
        self.notifications_sent = 0
        
        # Notification sink pins
        self.add_pin("MESSAGE_IN-", FlowType.INPUT, "input", "5V", 50)
        self.add_pin("URGENCY_S", FlowType.CONTROL, "input", "3.3V", 10)
        self.add_pin("DELIVERY_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("NOTIFY_GND", FlowType.OUTPUT, "output", "0V", 5)
    
    def process_electrical_flow(self, input_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification to configured sink"""
        
        message_input = input_flows.get('MESSAGE_IN-', {})
        urgency_control = input_flows.get('URGENCY_S', {})
        
        if not message_input:
            return {
                'DELIVERY_S': {'voltage': '0V', 'status': 'no_message'},
                'NOTIFY_GND': {'voltage': '3.3V', 'error': 'no_message_to_send'}
            }
        
        try:
            self.current_state = "NOTIFYING"
            
            # Extract notification data
            message_data = message_input.get('data', {})
            urgency_level = urgency_control.get('urgency', 'normal')
            
            # Create notification
            notification = {
                'type': self.notification_type,
                'message': message_data,
                'urgency': urgency_level,
                'sent_at': datetime.now(),
                'notification_id': f"notify_{self.notifications_sent + 1}"
            }
            
            self.notifications_sent += 1
            self.current_state = "NOTIFY_COMPLETE"
            
            return {
                'DELIVERY_S': {
                    'voltage': '3.3V',
                    'status': 'notification_sent',
                    'delivery_method': self.notification_type,
                    'urgency_level': urgency_level,
                    'notifications_sent': self.notifications_sent,
                    'notification_id': notification['notification_id']
                },
                'NOTIFY_GND': {'voltage': '0V', 'error': None}
            }
            
        except Exception as e:
            self.current_state = "FAULT_NOTIFY_ERROR"
            return {
                'DELIVERY_S': {'voltage': '0V', 'status': 'notify_failed'},
                'NOTIFY_GND': {
                    'voltage': '3.3V',
                    'error': str(e),
                    'fault_type': 'notification_error'
                }
            }

# Output Sink Factory

class OutputSinkFactory:
    """Factory for creating output sink nodes"""
    
    @staticmethod
    def create_file_writer(node_id: str, name: str, output_path: str) -> FileWriterSink:
        """Create file writer output sink"""
        return FileWriterSink(node_id, name, output_path)
    
    @staticmethod
    def create_database_store(node_id: str, name: str, table_name: str) -> DatabaseStoreSink:
        """Create database storage output sink"""
        return DatabaseStoreSink(node_id, name, table_name)
    
    @staticmethod
    def create_api_sender(node_id: str, name: str, endpoint_url: str) -> APISenderSink:
        """Create API sender output sink"""
        return APISenderSink(node_id, name, endpoint_url)
    
    @staticmethod
    def create_display_output(node_id: str, name: str, display_type: str = "console") -> DisplayOutputSink:
        """Create display output sink"""
        return DisplayOutputSink(node_id, name, display_type)
    
    @staticmethod
    def create_notification(node_id: str, name: str, notification_type: str = "email") -> NotificationSink:
        """Create notification output sink"""
        return NotificationSink(node_id, name, notification_type)

# Demo Functions

def demo_output_sinks():
    """Demonstrate output sink nodes"""
    print("Output Sink Nodes Demo (-)")
    print("=" * 40)
    
    # File writer demo
    print("\n[-] File Writer Sink:")
    file_writer = OutputSinkFactory.create_file_writer(
        "file_out", "Report Writer", "output_report.txt"
    )
    
    # Database store demo
    print("\n[-] Database Store Sink:")
    db_store = OutputSinkFactory.create_database_store(
        "db_out", "Results Database", "analysis_results"
    )
    
    # API sender demo
    print("\n[-] API Sender Sink:")
    api_sender = OutputSinkFactory.create_api_sender(
        "api_out", "External API", "https://api.client.com/results"
    )
    
    # Display output demo
    print("\n[-] Display Output Sink:")
    display = OutputSinkFactory.create_display_output(
        "disp_out", "Console Display", "console"
    )
    
    # Test display with sample data
    sample_data = {
        'DATA_IN-': {
            'data': {'result': 'Sample processing complete', 'score': 0.95},
            'quality': 1.0
        },
        'FORMAT_S': {'format': 'structured'}
    }
    
    result = display.process_electrical_flow(sample_data)
    
    print("\nOutput sinks created successfully!")
    print("All sinks ready to consume results from the system (-)")

if __name__ == "__main__":
    demo_output_sinks()