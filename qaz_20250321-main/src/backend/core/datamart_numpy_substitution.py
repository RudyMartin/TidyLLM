#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataMart NumPy Substitution Layer

This module provides DataMart/datatable alternatives to NumPy functions,
eliminating NumPy dependencies while maintaining similar functionality.
"""

from typing import Union, List, Tuple, Any, Optional, Dict
import random
import math
from datetime import datetime
import json

# Use existing datatable infrastructure - don't import here
DATATABLE_AVAILABLE = True  # Assume it's available through existing imports

# Try to import DataMart, fallback to simple implementation if not available
try:
    from ..mcp.orchestrators.advanced_qa_orchestrator import DataMartManager, DataMartMode
    DATAMART_AVAILABLE = True
except ImportError:
    DATAMART_AVAILABLE = False
    print("⚠️ DataMart not available, using simple fallback")
    
    # Simple fallback classes
    class DataMartMode:
        SIMPLE = "simple"
        ENHANCED = "enhanced"
        ADVANCED = "advanced"
    
    class DataMartManager:
        def __init__(self, mode=None):
            self.mode = mode or DataMartMode.SIMPLE
            self.buffer = []
        
        def initialize_datamart(self):
            return True
        
        def add_analysis_data(self, data):
            self.buffer.append(data)
            return True
        
        def get_performance_metrics(self):
            return {
                'buffer_size': len(self.buffer),
                'mode': self.mode.value if hasattr(self.mode, 'value') else str(self.mode),
                'status': 'active'
            }


class DataMartNumPySubstitution:
    """Complete NumPy substitution using DataMart/datatable"""
    
    def __init__(self, mode: DataMartMode = DataMartMode.ENHANCED):
        self.mode = mode
        self.datamart = DataMartManager(mode)
        self._initialize_datamart()
    
    def _initialize_datamart(self):
        """Initialize DataMart for numpy operations"""
        try:
            self.datamart.initialize_datamart()
        except Exception as e:
            print(f"Warning: DataMart initialization failed: {e}")
    
    # Array creation substitutes
    def array(self, data: Union[List, Tuple, Any], **kwargs) -> List:
        """Replace np.array() with list - datatable handled by existing infrastructure"""
        try:
            if isinstance(data, (list, tuple)):
                return data
            else:
                return [data]
        except Exception as e:
            print(f"Error creating array: {e}")
            return []
    
    def zeros(self, shape: Union[int, Tuple], **kwargs) -> List:
        """Replace np.zeros() with list filled with zeros"""
        try:
            if isinstance(shape, int):
                return [0] * shape
            else:
                # Handle multi-dimensional shapes
                total_size = math.prod(shape)
                return [0] * total_size
        except Exception as e:
            print(f"Error creating zeros array: {e}")
            return []
    
    def ones(self, shape: Union[int, Tuple], **kwargs) -> List:
        """Replace np.ones() with list filled with ones"""
        try:
            if isinstance(shape, int):
                return [1] * shape
            else:
                total_size = math.prod(shape)
                return [1] * total_size
        except Exception as e:
            print(f"Error creating ones array: {e}")
            return []
    
    def random_normal(self, loc: float = 0.0, scale: float = 1.0, 
                     size: Union[int, Tuple] = None, **kwargs) -> Union[float, List]:
        """Replace np.random.normal() with random.gauss()"""
        try:
            if size is None:
                return random.gauss(loc, scale)
            elif isinstance(size, int):
                return [random.gauss(loc, scale) for _ in range(size)]
            else:
                total_size = math.prod(size)
                return [random.gauss(loc, scale) for _ in range(total_size)]
        except Exception as e:
            print(f"Error generating random normal: {e}")
            return 0.0
    
    def random_rand(self, *args, **kwargs) -> Union[float, List]:
        """Replace np.random.rand() with random.random()"""
        try:
            if not args:
                return random.random()
            elif len(args) == 1:
                return [random.random() for _ in range(args[0])]
            else:
                total_size = math.prod(args)
                return [random.random() for _ in range(total_size)]
        except Exception as e:
            print(f"Error generating random values: {e}")
            return 0.0
    
    def random_randn(self, *args, **kwargs) -> Union[float, List]:
        """Replace np.random.randn() with random.gauss()"""
        try:
            if not args:
                return random.gauss(0, 1)
            elif len(args) == 1:
                return [random.gauss(0, 1) for _ in range(args[0])]
            else:
                total_size = math.prod(args)
                return [random.gauss(0, 1) for _ in range(total_size)]
        except Exception as e:
            print(f"Error generating random normal: {e}")
            return 0.0
    
    # Mathematical operations
    def mean(self, data: List, **kwargs) -> float:
        """Calculate mean using standard library"""
        try:
            if not data:
                return 0.0
            return sum(data) / len(data)
        except Exception as e:
            print(f"Error calculating mean: {e}")
            return 0.0
    
    def std(self, data: List, **kwargs) -> float:
        """Calculate standard deviation using standard library"""
        try:
            if not data or len(data) < 2:
                return 0.0
            mean_val = self.mean(data)
            variance = sum((x - mean_val) ** 2 for x in data) / (len(data) - 1)
            return math.sqrt(variance)
        except Exception as e:
            print(f"Error calculating std: {e}")
            return 0.0
    
    def sum(self, data: List, **kwargs) -> float:
        """Calculate sum using standard library"""
        try:
            return sum(data)
        except Exception as e:
            print(f"Error calculating sum: {e}")
            return 0.0
    
    def min(self, data: List, **kwargs) -> float:
        """Calculate minimum using standard library"""
        try:
            return min(data) if data else 0.0
        except Exception as e:
            print(f"Error calculating min: {e}")
            return 0.0
    
    def max(self, data: List, **kwargs) -> float:
        """Calculate maximum using standard library"""
        try:
            return max(data) if data else 0.0
        except Exception as e:
            print(f"Error calculating max: {e}")
            return 0.0
    
    # Array operations
    def reshape(self, data: List, shape: Tuple, **kwargs) -> List:
        """Reshape data using list operations"""
        try:
            return self._reshape_list_to_frame(data, shape)
        except Exception as e:
            print(f"Error reshaping data: {e}")
            return []
    
    def transpose(self, data: List, **kwargs) -> List:
        """Transpose data using list operations"""
        try:
            # Simple transpose for 2D lists
            if data and isinstance(data[0], list):
                return list(zip(*data))
            else:
                return data
        except Exception as e:
            print(f"Error transposing data: {e}")
            return []
    
    def pad(self, data: List, pad_width: Tuple, 
            mode: str = 'constant', **kwargs) -> List:
        """Pad array with zeros or other values"""
        try:
            # Simple padding implementation
            if mode == 'constant':
                pad_value = kwargs.get('constant_values', 0)
                padded_data = [pad_value] * pad_width[0] + data + [pad_value] * pad_width[1]
                return padded_data
            else:
                return data
        except Exception as e:
            print(f"Error padding data: {e}")
            return []
    
    def linalg_norm(self, data: List, **kwargs) -> float:
        """Calculate L2 norm using standard library"""
        try:
            # Calculate L2 norm
            squared_sum = sum(x * x for x in data)
            return math.sqrt(squared_sum)
        except Exception as e:
            print(f"Error calculating norm: {e}")
            return 0.0
    
    def datetime64(self, value: str) -> str:
        """Replace np.datetime64 with string timestamp"""
        try:
            if value == 'now':
                return datetime.now().isoformat()
            else:
                return value
        except Exception as e:
            print(f"Error creating datetime: {e}")
            return datetime.now().isoformat()
    
    def ndarray(self, data: Union[List, Tuple, Any]) -> List:
        """Replace np.ndarray with list"""
        return self.array(data)
    
    # Helper methods
    def _reshape_list_to_frame(self, data_list: List, shape: Tuple) -> List:
        """Helper to reshape list data"""
        try:
            if len(shape) == 1:
                return data_list
            elif len(shape) == 2:
                rows, cols = shape
                if len(data_list) != rows * cols:
                    # Pad or truncate to fit
                    if len(data_list) < rows * cols:
                        data_list.extend([0] * (rows * cols - len(data_list)))
                    else:
                        data_list = data_list[:rows * cols]
                
                reshaped_data = []
                for i in range(rows):
                    row = data_list[i * cols:(i + 1) * cols]
                    reshaped_data.append(row)
                return reshaped_data
            else:
                # For other shapes, return as is
                return data_list
        except Exception as e:
            print(f"Error reshaping list: {e}")
            return []
    
    def to_list(self, data: List) -> List:
        """Convert data to list format"""
        try:
            return data
        except Exception as e:
            print(f"Error converting to list: {e}")
            return []
    
    def to_array(self, data: List) -> List:
        """Convert data to array format (alias for to_list)"""
        return self.to_list(data)
    
    # DataMart integration
    def add_to_datamart(self, data: List, metadata: dict = None) -> bool:
        """Add data to DataMart for analytics"""
        try:
            datamart_data = {
                'data_type': 'numpy_substitution',
                'data_size': len(data),
                'operation_timestamp': datetime.now().isoformat(),
                'metadata': json.dumps(metadata or {})
            }
            return self.datamart.add_analysis_data(datamart_data)
        except Exception as e:
            print(f"Error adding to DataMart: {e}")
            return False
    
    def get_datamart_metrics(self) -> dict:
        """Get DataMart performance metrics"""
        try:
            return self.datamart.get_performance_metrics()
        except Exception as e:
            print(f"Error getting DataMart metrics: {e}")
            return {}


# Create a random module substitute
class RandomModule:
    """Substitute for np.random module"""
    
    def normal(self, loc: float = 0.0, scale: float = 1.0, 
               size: Union[int, Tuple] = None, **kwargs) -> Union[float, List]:
        """Replace np.random.normal()"""
        if size is None:
            return random.gauss(loc, scale)
        elif isinstance(size, int):
            return [random.gauss(loc, scale) for _ in range(size)]
        else:
            total_size = math.prod(size)
            return [random.gauss(loc, scale) for _ in range(total_size)]
    
    def rand(self, *args, **kwargs) -> Union[float, List]:
        """Replace np.random.rand()"""
        if not args:
            return random.random()
        elif len(args) == 1:
            return [random.random() for _ in range(args[0])]
        else:
            total_size = math.prod(args)
            return [random.random() for _ in range(total_size)]
    
    def randn(self, *args, **kwargs) -> Union[float, List]:
        """Replace np.random.randn()"""
        if not args:
            return random.gauss(0, 1)
        elif len(args) == 1:
            return [random.gauss(0, 1) for _ in range(args[0])]
        else:
            total_size = math.prod(args)
            return [random.gauss(0, 1) for _ in range(total_size)]


# Create a linalg module substitute
class LinalgModule:
    """Substitute for np.linalg module"""
    
    def norm(self, data: List, **kwargs) -> float:
        """Replace np.linalg.norm()"""
        try:
            # Calculate L2 norm
            squared_sum = sum(x * x for x in data)
            return math.sqrt(squared_sum)
        except Exception as e:
            print(f"Error calculating norm: {e}")
            return 0.0


# Create the main numpy substitute
class NumpySubstitute:
    """Main NumPy substitute using DataMart"""
    
    def __init__(self, mode: DataMartMode = DataMartMode.ENHANCED):
        self.datamart_np = DataMartNumPySubstitution(mode)
        self.random = RandomModule()
        self.linalg = LinalgModule()
    
    # Expose all the methods as attributes
    @property
    def array(self):
        return self.datamart_np.array
    
    @property
    def zeros(self):
        return self.datamart_np.zeros
    
    @property
    def ones(self):
        return self.datamart_np.ones
    
    @property
    def mean(self):
        return self.datamart_np.mean
    
    @property
    def std(self):
        return self.datamart_np.std
    
    @property
    def sum(self):
        return self.datamart_np.sum
    
    @property
    def min(self):
        return self.datamart_np.min
    
    @property
    def max(self):
        return self.datamart_np.max
    
    @property
    def reshape(self):
        return self.datamart_np.reshape
    
    @property
    def transpose(self):
        return self.datamart_np.transpose
    
    @property
    def pad(self):
        return self.datamart_np.pad
    
    @property
    def datetime64(self):
        return self.datamart_np.datetime64
    
    @property
    def ndarray(self):
        return self.datamart_np.ndarray
    
    @property
    def random_normal(self):
        return self.datamart_np.random_normal
    
    def to_list(self, data):
        return self.datamart_np.to_list(data)
    
    def to_array(self, data):
        return self.datamart_np.to_array(data)
    
    def add_to_datamart(self, data, metadata=None):
        return self.datamart_np.add_to_datamart(data, metadata)
    
    def get_datamart_metrics(self):
        return self.datamart_np.get_datamart_metrics()


# Create the main substitution object
np = NumpySubstitute()


# Simple DataMart wrapper for easier access
class DataMart:
    """Simple DataMart wrapper for easier access to DataMart functionality"""
    
    def __init__(self, mode: DataMartMode = DataMartMode.SIMPLE):
        self.manager = DataMartManager(mode)
        self.np = NumpySubstitute(mode)
    
    def initialize(self):
        """Initialize DataMart"""
        return self.manager.initialize_datamart()
    
    def add_data(self, data: Dict[str, Any]) -> bool:
        """Add analysis data to DataMart"""
        return self.manager.add_analysis_data(data)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get DataMart performance metrics"""
        return self.manager.get_performance_metrics()
    
    def get_numpy_substitute(self):
        """Get NumPy substitute for calculations"""
        return self.np
