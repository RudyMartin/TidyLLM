"""
Input Source Nodes (+) - TidyLLM-HeirOS Electrical System
========================================================

Input sources provide data flowing INTO the system
These are the positive (+) rail components that feed the workflow
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from flow_schemas import ElectricalNode, NodePolarity, FlowType
import os

class InputSourceType(Enum):
    """Types of input sources"""
    FILE_READER = "file_reader"
    DATA_GENERATOR = "data_generator" 
    SENSOR_INPUT = "sensor_input"
    USER_INPUT = "user_input"
    API_FETCH = "api_fetch"
    DATABASE_QUERY = "database_query"

class FileReaderSource(ElectricalNode):
    """File reading input source (+) - provides file data to system"""
    
    def __init__(self, node_id: str, name: str, file_path: str):
        super().__init__(node_id, name, NodePolarity.INPUT_SOURCE)
        self.file_path = file_path
        self.read_mode = "text"  # text, binary, stream
        
        # Input source pins - only outputs data
        self.add_pin("DATA_OUT+", FlowType.OUTPUT, "output", "5V", 200)
        self.add_pin("STATUS_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("ERROR_GND", FlowType.OUTPUT, "output", "0V", 5)
    
    def process_electrical_flow(self, input_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Generate input data from file source"""
        
        try:
            self.current_state = "READING"
            
            # Simulate file reading (in demo, create content)
            if not os.path.exists(self.file_path):
                content = f"Sample MVR document content for demonstration.\nFile: {self.file_path}\nGenerated at: {datetime.now()}\nThis is simulated content for the electrical system demo."
            else:
                with open(self.file_path, 'r') as file:
                    content = file.read()
            
            # Calculate data characteristics
            data_size = len(content)
            data_quality = 1.0 if data_size > 0 else 0.0
            
            self.current_state = "ACTIVE"
            
            return {
                'DATA_OUT+': {
                    'voltage': '5V',
                    'data': {
                        'content': content,
                        'file_path': self.file_path,
                        'size_bytes': data_size,
                        'read_time': datetime.now()
                    },
                    'quality': data_quality,
                    'source_type': 'file_input'
                },
                'STATUS_S': {
                    'voltage': '3.3V',
                    'status': 'reading_complete',
                    'bytes_read': data_size
                },
                'ERROR_GND': {'voltage': '0V', 'error': None}
            }
            
        except Exception as e:
            self.current_state = "FAULT_READ_ERROR"
            self.fault_conditions.append(str(e))
            
            return {
                'DATA_OUT+': {'voltage': '0V', 'data': None},
                'STATUS_S': {'voltage': '0V', 'status': 'read_failed'},
                'ERROR_GND': {
                    'voltage': '3.3V', 
                    'error': str(e),
                    'fault_type': 'file_read_error'
                }
            }

class DataGeneratorSource(ElectricalNode):
    """Data generator input source (+) - creates synthetic data"""
    
    def __init__(self, node_id: str, name: str, generator_function: Callable):
        super().__init__(node_id, name, NodePolarity.INPUT_SOURCE)
        self.generator_function = generator_function
        self.generation_count = 0
        
        # Input source pins
        self.add_pin("TRIGGER_S", FlowType.CONTROL, "input", "3.3V", 10)
        self.add_pin("DATA_OUT+", FlowType.OUTPUT, "output", "5V", 100)
        self.add_pin("STATUS_S", FlowType.CONTROL, "output", "3.3V", 10)
    
    def process_electrical_flow(self, input_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Generate input data using generator function"""
        
        # Check for trigger signal
        trigger = input_flows.get('TRIGGER_S', {})
        
        try:
            self.current_state = "GENERATING"
            
            # Generate data
            generated_data = self.generator_function({
                'generation_count': self.generation_count,
                'trigger_params': trigger.get('parameters', {})
            })
            
            self.generation_count += 1
            self.current_state = "ACTIVE"
            
            return {
                'DATA_OUT+': {
                    'voltage': '5V',
                    'data': generated_data,
                    'quality': 1.0,
                    'source_type': 'generated_input',
                    'generation_id': self.generation_count
                },
                'STATUS_S': {
                    'voltage': '3.3V',
                    'status': 'generation_complete',
                    'generation_count': self.generation_count
                }
            }
            
        except Exception as e:
            self.current_state = "FAULT_GENERATION_ERROR"
            return {
                'DATA_OUT+': {'voltage': '0V', 'data': None},
                'STATUS_S': {
                    'voltage': '0V', 
                    'status': 'generation_failed',
                    'error': str(e)
                }
            }

class APIFetchSource(ElectricalNode):
    """API fetch input source (+) - retrieves data from external APIs"""
    
    def __init__(self, node_id: str, name: str, api_url: str):
        super().__init__(node_id, name, NodePolarity.INPUT_SOURCE)
        self.api_url = api_url
        self.fetch_count = 0
        self.cache = {}
        
        # API source pins
        self.add_pin("REQUEST_S", FlowType.CONTROL, "input", "3.3V", 10)
        self.add_pin("DATA_OUT+", FlowType.OUTPUT, "output", "5V", 300)
        self.add_pin("STATUS_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("NETWORK_GND", FlowType.OUTPUT, "output", "0V", 5)
    
    def process_electrical_flow(self, input_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch input data from API"""
        
        request_signal = input_flows.get('REQUEST_S', {})
        request_params = request_signal.get('parameters', {})
        
        try:
            self.current_state = "FETCHING"
            
            # Simulate API fetch (in real implementation would use requests library)
            api_response = {
                'url': self.api_url,
                'params': request_params,
                'response_data': f"API response data {self.fetch_count}",
                'status_code': 200,
                'fetch_time': datetime.now()
            }
            
            self.fetch_count += 1
            self.current_state = "ACTIVE"
            
            # Calculate data quality based on response
            data_quality = 1.0 if api_response['status_code'] == 200 else 0.5
            
            return {
                'DATA_OUT+': {
                    'voltage': '5V',
                    'data': api_response,
                    'quality': data_quality,
                    'source_type': 'api_input',
                    'fetch_id': self.fetch_count
                },
                'STATUS_S': {
                    'voltage': '3.3V',
                    'status': 'fetch_complete',
                    'status_code': api_response['status_code']
                },
                'NETWORK_GND': {'voltage': '0V', 'error': None}
            }
            
        except Exception as e:
            self.current_state = "FAULT_FETCH_ERROR"
            return {
                'DATA_OUT+': {'voltage': '0V', 'data': None},
                'STATUS_S': {'voltage': '0V', 'status': 'fetch_failed'},
                'NETWORK_GND': {
                    'voltage': '3.3V',
                    'error': str(e),
                    'fault_type': 'api_fetch_error'
                }
            }

class UserInputSource(ElectricalNode):
    """User input source (+) - handles interactive user data input"""
    
    def __init__(self, node_id: str, name: str, input_schema: Dict):
        super().__init__(node_id, name, NodePolarity.INPUT_SOURCE)
        self.input_schema = input_schema
        self.input_buffer = []
        
        # User input pins
        self.add_pin("USER_DATA+", FlowType.OUTPUT, "output", "5V", 50)
        self.add_pin("VALIDATION_S", FlowType.CONTROL, "output", "3.3V", 10)
        self.add_pin("INPUT_READY_S", FlowType.CONTROL, "output", "3.3V", 5)
    
    def process_electrical_flow(self, input_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input data"""
        
        try:
            self.current_state = "WAITING_FOR_INPUT"
            
            # Simulate user input (in real implementation would interface with UI)
            user_data = {
                'schema_version': self.input_schema.get('version', '1.0'),
                'user_input': f"Sample user input for {self.name}",
                'input_timestamp': datetime.now(),
                'validation_status': 'valid'
            }
            
            # Validate against schema
            is_valid = self._validate_input(user_data)
            
            self.current_state = "INPUT_RECEIVED"
            
            return {
                'USER_DATA+': {
                    'voltage': '5V',
                    'data': user_data,
                    'quality': 1.0 if is_valid else 0.5,
                    'source_type': 'user_input',
                    'valid': is_valid
                },
                'VALIDATION_S': {
                    'voltage': '3.3V',
                    'validation_result': 'valid' if is_valid else 'invalid',
                    'schema_matched': is_valid
                },
                'INPUT_READY_S': {
                    'voltage': '3.3V',
                    'ready': True,
                    'buffer_size': len(self.input_buffer)
                }
            }
            
        except Exception as e:
            self.current_state = "FAULT_INPUT_ERROR"
            return {
                'USER_DATA+': {'voltage': '0V', 'data': None},
                'VALIDATION_S': {'voltage': '0V', 'validation_result': 'error'},
                'INPUT_READY_S': {'voltage': '0V', 'ready': False, 'error': str(e)}
            }
    
    def _validate_input(self, user_data: Dict) -> bool:
        """Validate user input against schema"""
        # Simplified validation
        required_fields = self.input_schema.get('required_fields', [])
        return all(field in user_data for field in required_fields)

# Input Source Factory

class InputSourceFactory:
    """Factory for creating input source nodes"""
    
    @staticmethod
    def create_file_reader(node_id: str, name: str, file_path: str) -> FileReaderSource:
        """Create file reader input source"""
        return FileReaderSource(node_id, name, file_path)
    
    @staticmethod  
    def create_data_generator(node_id: str, name: str, generator_func: Callable) -> DataGeneratorSource:
        """Create data generator input source"""
        return DataGeneratorSource(node_id, name, generator_func)
    
    @staticmethod
    def create_api_fetch(node_id: str, name: str, api_url: str) -> APIFetchSource:
        """Create API fetch input source"""
        return APIFetchSource(node_id, name, api_url)
    
    @staticmethod
    def create_user_input(node_id: str, name: str, schema: Dict) -> UserInputSource:
        """Create user input source"""
        return UserInputSource(node_id, name, schema)

# Demo Functions

def demo_input_sources():
    """Demonstrate input source nodes"""
    print("Input Source Nodes Demo (+)")
    print("=" * 40)
    
    # File reader demo
    print("\n[+] File Reader Source:")
    file_reader = InputSourceFactory.create_file_reader(
        "file_in", "Document Reader", "sample.txt"
    )
    
    # Data generator demo  
    def sample_generator(context):
        return {
            "generated_value": f"Sample data {context['generation_count']}",
            "timestamp": datetime.now(),
            "quality": "high"
        }
    
    print("\n[+] Data Generator Source:")
    data_gen = InputSourceFactory.create_data_generator(
        "data_gen", "Sample Generator", sample_generator
    )
    
    # API fetch demo
    print("\n[+] API Fetch Source:")
    api_fetch = InputSourceFactory.create_api_fetch(
        "api_in", "External API", "https://api.example.com/data"
    )
    
    print("\nInput sources created successfully!")
    print("All sources ready to feed data into the system (+)")

if __name__ == "__main__":
    demo_input_sources()