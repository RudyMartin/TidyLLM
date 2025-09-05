#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YAML Processing Worker

Dedicated worker for YAML file processing with library detection and fallback mechanisms.
This worker handles YAML files without requiring specific YAML libraries to be installed.
"""

import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class YAMLMode:
    """YAML processing modes"""
    SIMPLE = "simple"      # Basic YAML parsing
    ENHANCED = "enhanced"  # Advanced features
    ADVANCED = "advanced"  # Full YAML capabilities


class YAMLProcessingWorker:
    """Dedicated worker for YAML file processing with library fallback"""
    
    def __init__(self, mode=YAMLMode.SIMPLE):
        self.mode = mode
        self.worker_id = f"yaml_worker_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.yaml_libraries = self._detect_yaml_libraries()
        self.processing_stats = {
            'files_processed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'library_used': None
        }
        
        logger.info(f"YAML Processing Worker initialized in {mode} mode")
        logger.info(f"Available YAML libraries: {list(self.yaml_libraries.keys())}")
    
    def _detect_yaml_libraries(self):
        """Detect available YAML libraries with fallback order"""
        libraries = {}
        
        # Try PyYAML first (most common)
        try:
            import yaml
            libraries['pyyaml'] = yaml
            logger.info("✅ PyYAML detected")
        except ImportError:
            logger.warning("⚠️ PyYAML not available")
        
        # Try ruamel-yaml (more advanced)
        try:
            import ruamel.yaml
            libraries['ruamel'] = ruamel.yaml
            logger.info("✅ ruamel-yaml detected")
        except ImportError:
            logger.warning("⚠️ ruamel-yaml not available")
        
        # Try ruamel_yaml (alternative import)
        try:
            import ruamel_yaml
            libraries['ruamel_alt'] = ruamel_yaml
            logger.info("✅ ruamel_yaml (alt) detected")
        except ImportError:
            logger.warning("⚠️ ruamel_yaml (alt) not available")
        
        if not libraries:
            logger.error("❌ No YAML libraries available - using fallback parser")
        
        return libraries
    
    def process_document(self, file_path):
        """Process YAML file with library fallback"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"YAML file not found: {file_path}")
            
            # Try each available library
            for lib_name, lib in self.yaml_libraries.items():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result = lib.safe_load(f)
                    
                    self.processing_stats['successful_parses'] += 1
                    self.processing_stats['library_used'] = lib_name
                    self.processing_stats['files_processed'] += 1
                    
                    logger.info(f"✅ Successfully parsed {file_path} using {lib_name}")
                    return {
                        'success': True,
                        'data': result,
                        'library_used': lib_name,
                        'file_path': file_path,
                        'processing_time': datetime.now().isoformat(),
                        'worker_id': self.worker_id
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse with {lib_name}: {e}")
                    continue
            
            # If no library worked, try fallback parser
            return self._fallback_yaml_parser(file_path)
            
        except Exception as e:
            self.processing_stats['failed_parses'] += 1
            self.processing_stats['files_processed'] += 1
            
            logger.error(f"❌ Failed to process YAML file {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path,
                'processing_time': datetime.now().isoformat(),
                'worker_id': self.worker_id
            }
    
    def _fallback_yaml_parser(self, file_path):
        """Simple fallback YAML parser for basic YAML files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Very basic YAML-like parsing for simple cases
            result = self._parse_simple_yaml(content)
            
            self.processing_stats['successful_parses'] += 1
            self.processing_stats['library_used'] = 'fallback'
            self.processing_stats['files_processed'] += 1
            
            logger.info(f"✅ Successfully parsed {file_path} using fallback parser")
            return {
                'success': True,
                'data': result,
                'library_used': 'fallback',
                'file_path': file_path,
                'processing_time': datetime.now().isoformat(),
                'worker_id': self.worker_id,
                'warning': 'Using fallback parser - limited YAML features'
            }
            
        except Exception as e:
            self.processing_stats['failed_parses'] += 1
            self.processing_stats['files_processed'] += 1
            
            logger.error(f"❌ Fallback parser failed for {file_path}: {e}")
            return {
                'success': False,
                'error': f"All YAML parsers failed: {str(e)}",
                'file_path': file_path,
                'processing_time': datetime.now().isoformat(),
                'worker_id': self.worker_id
            }
    
    def _parse_simple_yaml(self, content):
        """Simple YAML parser for basic key-value pairs"""
        result = {}
        current_key = None
        current_value = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Check for key-value pairs
            if ':' in line and not line.startswith(' '):
                # Save previous key-value pair
                if current_key and current_value:
                    result[current_key] = '\n'.join(current_value).strip()
                
                # Start new key-value pair
                parts = line.split(':', 1)
                current_key = parts[0].strip()
                current_value = [parts[1].strip()] if len(parts) > 1 else []
            else:
                # Continuation of current value
                if current_key:
                    current_value.append(line)
        
        # Save last key-value pair
        if current_key and current_value:
            result[current_key] = '\n'.join(current_value).strip()
        
        return result
    
    def get_processing_stats(self):
        """Get processing statistics"""
        return {
            'worker_id': self.worker_id,
            'mode': self.mode,
            'available_libraries': list(self.yaml_libraries.keys()),
            'stats': self.processing_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_yaml_file(self, file_path):
        """Validate YAML file without processing"""
        try:
            if not os.path.exists(file_path):
                return {
                    'valid': False,
                    'error': 'File not found',
                    'file_path': file_path
                }
            
            # Check file extension
            if not file_path.endswith(('.yaml', '.yml')):
                return {
                    'valid': False,
                    'error': 'Not a YAML file',
                    'file_path': file_path
                }
            
            # Check if file is readable
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                
                return {
                    'valid': True,
                    'file_path': file_path,
                    'file_size': os.path.getsize(file_path),
                    'first_line': first_line.strip()[:50] + '...' if len(first_line) > 50 else first_line.strip()
                }
                
            except Exception as e:
                return {
                    'valid': False,
                    'error': f'File not readable: {str(e)}',
                    'file_path': file_path
                }
                
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'file_path': file_path
            }
    
    def get_required_dependencies(self):
        """Get list of required dependencies for this worker"""
        return ['PyYAML', 'ruamel-yaml']
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        return {
            'pyyaml': 'pyyaml' in self.yaml_libraries,
            'ruamel_yaml': 'ruamel' in self.yaml_libraries or 'ruamel_alt' in self.yaml_libraries,
            'any_yaml_library': len(self.yaml_libraries) > 0
        }


# Convenience function for easy usage
def load_yaml_file(file_path):
    """Convenience function to load YAML file"""
    worker = YAMLProcessingWorker()
    result = worker.process_document(file_path)
    
    if result['success']:
        return result['data']
    else:
        raise ValueError(f"Failed to load YAML file: {result['error']}")


if __name__ == "__main__":
    # Test the worker
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        worker = YAMLProcessingWorker()
        result = worker.process_document(file_path)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python yaml_processing_worker.py <yaml_file_path>")
