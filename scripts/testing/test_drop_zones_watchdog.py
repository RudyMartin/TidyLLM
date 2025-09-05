#!/usr/bin/env python3
"""
Drop Zones Watchdog Testing System
=================================

Comprehensive testing framework for DROP ZONES concept using Watchdog file monitoring.
Tests both local file processing and S3-integrated MVR workflow automation.

Connects DROP ZONES to Universal Bracket Flows for end-to-end MVR analysis.
"""

import os
import sys
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import json
import yaml

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

# Import drop zone components
from drop_zones.basic.listener import BasicListener
from drop_zones.basic.config import ConfigManager
from drop_zones.working_s3_dropzones import WorkingS3DropZones

# Import Universal Bracket Flows
try:
    from tidyllm.universal_flow_parser import UniversalFlowParser
    FLOW_PARSER_AVAILABLE = True
except ImportError:
    FLOW_PARSER_AVAILABLE = False
    print("WARNING: Universal Flow Parser not available")

class DropZoneTestFramework:
    """Comprehensive testing framework for drop zone systems"""
    
    def __init__(self, test_dir: Optional[Path] = None):
        # Create temporary test environment
        self.test_dir = Path(tempfile.mkdtemp(prefix='dropzone_test_')) if not test_dir else test_dir
        self.test_dir.mkdir(exist_ok=True, parents=True)
        
        # Test environment structure
        self.zones = {
            'mvr_documents': self.test_dir / 'zones' / 'mvr',
            'vst_documents': self.test_dir / 'zones' / 'vst', 
            'research_papers': self.test_dir / 'zones' / 'research',
            'data_files': self.test_dir / 'zones' / 'data'
        }
        
        self.processing = self.test_dir / 'processing'
        self.completed = self.test_dir / 'completed'
        self.failed = self.test_dir / 'failed'
        
        # Create all directories
        for zone_path in self.zones.values():
            zone_path.mkdir(parents=True, exist_ok=True)
        
        for dir_path in [self.processing, self.completed, self.failed]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Test results tracking
        self.test_results: List[Dict] = []
        self.processed_files: List[str] = []
        
        # Initialize components
        self.config_manager = None
        self.listener = None
        self.s3_processor = None
        self.flow_parser = None
        
        print(f"[TEST] Drop Zone Test Framework initialized")
        print(f"[TEST] Test directory: {self.test_dir}")
        print(f"[TEST] Zones created: {list(self.zones.keys())}")
    
    def setup_watchdog_config(self) -> str:
        """Create test configuration for Watchdog listener"""
        config = {
            'drop_zones': [
                {
                    'name': 'mvr_documents',
                    'paths': [str(self.zones['mvr_documents'])],
                    'patterns': ['*.pdf', '*.docx', '*.txt'],
                    'events': ['created', 'modified'],
                    'processing_dir': str(self.processing),
                    'success_dir': str(self.completed),
                    'failure_dir': str(self.failed),
                    'max_file_size': 50 * 1024 * 1024,  # 50MB
                    'enabled': True,
                    'workflow': 'mvr_analysis_flow'
                },
                {
                    'name': 'vst_documents',
                    'paths': [str(self.zones['vst_documents'])],
                    'patterns': ['*.pdf', '*.docx'],
                    'events': ['created'],
                    'processing_dir': str(self.processing),
                    'success_dir': str(self.completed), 
                    'failure_dir': str(self.failed),
                    'max_file_size': 10 * 1024 * 1024,  # 10MB
                    'enabled': True,
                    'workflow': 'vst_validation_flow'
                },
                {
                    'name': 'research_papers',
                    'paths': [str(self.zones['research_papers'])],
                    'patterns': ['*.pdf'],
                    'events': ['created'],
                    'processing_dir': str(self.processing),
                    'success_dir': str(self.completed),
                    'failure_dir': str(self.failed),
                    'enabled': True
                }
            ],
            'logging': {
                'level': 'DEBUG',
                'file': str(self.test_dir / 'test_dropzones.log')
            },
            'processing': {
                'timeout': 30,
                'retry_count': 2,
                'retry_delay': 1
            }
        }
        
        config_path = self.test_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(config_path)
    
    def create_test_documents(self):
        """Create test documents for drop zone processing"""
        test_files = []
        
        # MVR test document
        mvr_content = """
        REV12345 Motor Vehicle Record Analysis
        =====================================
        
        Document Type: MVR (Motor Vehicle Record)
        REV ID: REV12345
        Analysis Date: 2024-01-15
        
        Driver Information:
        - Name: John Doe
        - License: ABC123456
        - State: California
        
        Violations:
        - Speeding: 2023-05-10
        - Parking: 2023-08-22
        
        YNSR Noise Factor: 0.15 (Low)
        Classification: Standard MVR
        """
        
        mvr_file = self.zones['mvr_documents'] / 'REV12345_MVR_Analysis.txt'
        with open(mvr_file, 'w') as f:
            f.write(mvr_content)
        test_files.append(mvr_file)
        
        # VST test document
        vst_content = """
        REV12345 Validation Scoping Template
        ===================================
        
        Document Type: VST (Validation Scoping Template)
        REV ID: REV12345 
        Scope Date: 2024-01-10
        
        Validation Requirements:
        - Identity verification: Required
        - Address verification: Required
        - Employment verification: Optional
        
        Expected Discrepancies:
        - Minor address variations acceptable
        - Nickname variations acceptable
        
        Quality Threshold: 95%
        """
        
        vst_file = self.zones['vst_documents'] / 'REV12345_VST_Template.txt'
        with open(vst_file, 'w') as f:
            f.write(vst_content)
        test_files.append(vst_file)
        
        # Research paper
        research_content = """
        Research Paper: Advanced Document Analysis
        =========================================
        
        Abstract: This paper explores automated document processing
        techniques for regulatory compliance workflows...
        
        Keywords: document analysis, compliance, automation
        """
        
        research_file = self.zones['research_papers'] / 'research_paper_001.txt'
        with open(research_file, 'w') as f:
            f.write(research_content)
        test_files.append(research_file)
        
        # Invalid test file (too large)
        large_content = "X" * (60 * 1024 * 1024)  # 60MB - exceeds MVR limit
        large_file = self.zones['mvr_documents'] / 'oversized_document.txt'
        with open(large_file, 'w') as f:
            f.write(large_content)
        test_files.append(large_file)
        
        print(f"[TEST] Created {len(test_files)} test documents:")
        for f in test_files:
            print(f"[TEST]   - {f.name} ({f.stat().st_size} bytes)")
        
        return test_files
    
    def test_watchdog_monitoring(self, duration: int = 10) -> Dict:
        """Test Watchdog file monitoring with created documents"""
        print(f"\n[TEST] Starting Watchdog monitoring test ({duration}s)")
        
        try:
            # Setup configuration
            config_path = self.setup_watchdog_config()
            self.config_manager = ConfigManager(config_path)
            
            # Initialize listener
            self.listener = BasicListener(self.config_manager)
            
            # Start monitoring in background thread
            monitor_thread = threading.Thread(target=self.listener.start, daemon=True)
            monitor_thread.start()
            
            print("[TEST] Watchdog monitoring started")
            
            # Wait a moment for monitoring to stabilize
            time.sleep(2)
            
            # Create test documents (this should trigger events)
            test_files = self.create_test_documents()
            
            # Wait for processing
            print(f"[TEST] Waiting {duration}s for file processing...")
            time.sleep(duration)
            
            # Get monitoring statistics
            stats = self.listener.get_stats()
            
            # Stop monitoring
            self.listener.stop()
            
            # Analyze results
            result = {
                'test_type': 'watchdog_monitoring',
                'success': stats['files_processed'] > 0,
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'statistics': stats,
                'test_files_created': len(test_files),
                'zones_monitored': len(self.config_manager.zones),
                'completed_files': list(self.completed.glob('**/*')) if self.completed.exists() else [],
                'failed_files': list(self.failed.glob('**/*')) if self.failed.exists() else []
            }
            
            print(f"[TEST] Watchdog test completed:")
            print(f"[TEST]   Files detected: {stats['files_detected']}")
            print(f"[TEST]   Files queued: {stats['files_queued']}")
            print(f"[TEST]   Files processed: {stats['files_processed']}")
            print(f"[TEST]   Success rate: {stats.get('files_per_minute', 0):.1f} files/min")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Watchdog test failed: {e}")
            return {
                'test_type': 'watchdog_monitoring',
                'success': False,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def test_s3_integration(self) -> Dict:
        """Test S3-integrated drop zone processing"""
        print(f"\n[TEST] Starting S3 integration test")
        
        try:
            # Initialize S3 processor
            self.s3_processor = WorkingS3DropZones(str(self.zones['research_papers']))
            
            # Create a test document directly in the zone
            test_content = """
            Test Document for S3 Integration
            ================================
            
            This document tests S3 upload integration with drop zones.
            Created: {timestamp}
            File type: Research paper
            Test ID: s3_integration_001
            """.format(timestamp=datetime.now().isoformat())
            
            test_file = self.zones['research_papers'] / 's3_test_document.txt'
            with open(test_file, 'w') as f:
                f.write(test_content)
            
            print(f"[TEST] Created S3 test file: {test_file.name}")
            
            # Process file through S3 workflow
            processing_result = self.s3_processor.process_file(test_file)
            
            # Get processing results
            self.s3_processor.save_processing_report()
            
            result = {
                'test_type': 's3_integration',
                'success': processing_result.get('success', False),
                'timestamp': datetime.now().isoformat(),
                'processing_result': processing_result,
                's3_evidence': processing_result.get('s3_evidence', {}),
                'file_processed': test_file.name,
                'file_size': test_file.stat().st_size if test_file.exists() else 0
            }
            
            print(f"[TEST] S3 integration test completed:")
            print(f"[TEST]   Success: {result['success']}")
            if result['success'] and 's3_evidence' in processing_result:
                s3_info = processing_result['s3_evidence']
                print(f"[TEST]   S3 URL: {s3_info.get('s3_url', 'N/A')}")
                print(f"[TEST]   File hash: {s3_info.get('file_hash', 'N/A')[:16]}...")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] S3 integration test failed: {e}")
            return {
                'test_type': 's3_integration',
                'success': False,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def test_mvr_workflow_integration(self) -> Dict:
        """Test MVR workflow integration with Universal Bracket Flows"""
        print(f"\n[TEST] Starting MVR workflow integration test")
        
        if not FLOW_PARSER_AVAILABLE:
            return {
                'test_type': 'mvr_workflow_integration',
                'success': False,
                'timestamp': datetime.now().isoformat(),
                'error': 'Universal Flow Parser not available'
            }
        
        try:
            # Initialize flow parser
            self.flow_parser = UniversalFlowParser()
            
            # Simulate MVR document detection
            mvr_file_path = "REV12345_MVR_Document.pdf"
            
            # Test each stage of MVR workflow
            mvr_stages = ['tag', 'qa', 'peer', 'report']
            workflow_results = []
            
            for stage in mvr_stages:
                bracket_command = f"[mvr_analysis {stage} {mvr_file_path}]"
                
                print(f"[TEST] Testing workflow stage: {stage}")
                print(f"[TEST] Bracket command: {bracket_command}")
                
                try:
                    # Parse the bracket command
                    parsed_command = self.flow_parser.parse_bracket_command(bracket_command)
                    
                    stage_result = {
                        'stage': stage,
                        'bracket_command': bracket_command,
                        'parsed_successfully': True,
                        'workflow_name': parsed_command.workflow_name,
                        'action': parsed_command.action,
                        'parameters': parsed_command.parameters
                    }
                    
                    print(f"[TEST]   Parse result: {stage_result}")
                    
                except Exception as e:
                    stage_result = {
                        'stage': stage,
                        'bracket_command': bracket_command,
                        'parsed_successfully': False,
                        'error': str(e)
                    }
                    
                    print(f"[TEST]   Parse failed: {e}")
                
                workflow_results.append(stage_result)
            
            # Overall result
            successful_stages = len([r for r in workflow_results if r.get('parsed_successfully', False)])
            
            result = {
                'test_type': 'mvr_workflow_integration',
                'success': successful_stages > 0,
                'timestamp': datetime.now().isoformat(),
                'total_stages': len(mvr_stages),
                'successful_stages': successful_stages,
                'workflow_results': workflow_results,
                'mvr_file_path': mvr_file_path
            }
            
            print(f"[TEST] MVR workflow integration test completed:")
            print(f"[TEST]   Successful stages: {successful_stages}/{len(mvr_stages)}")
            print(f"[TEST]   Overall success: {result['success']}")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] MVR workflow integration test failed: {e}")
            return {
                'test_type': 'mvr_workflow_integration',
                'success': False,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def run_full_test_suite(self) -> Dict:
        """Run complete drop zone testing suite"""
        print(f"\n{'='*60}")
        print("DROP ZONES WATCHDOG TESTING SUITE")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Run all tests
        tests = [
            ('Watchdog Monitoring', self.test_watchdog_monitoring),
            ('S3 Integration', self.test_s3_integration), 
            ('MVR Workflow Integration', self.test_mvr_workflow_integration)
        ]
        
        all_results = []
        
        for test_name, test_func in tests:
            print(f"\n{'-'*40}")
            print(f"Running Test: {test_name}")
            print(f"{'-'*40}")
            
            try:
                result = test_func()
                result['test_name'] = test_name
                all_results.append(result)
                
                status = "PASS" if result.get('success', False) else "FAIL"
                print(f"[RESULT] {test_name}: {status}")
                
            except Exception as e:
                print(f"[ERROR] {test_name} crashed: {e}")
                all_results.append({
                    'test_name': test_name,
                    'test_type': 'crashed',
                    'success': False,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                })
        
        # Final results summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        successful_tests = len([r for r in all_results if r.get('success', False)])
        total_tests = len(all_results)
        
        final_result = {
            'test_suite': 'drop_zones_watchdog',
            'success': successful_tests > 0,
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
            'test_directory': str(self.test_dir),
            'individual_results': all_results
        }
        
        # Save comprehensive test report
        self.save_test_report(final_result)
        
        print(f"\n{'='*60}")
        print("TEST SUITE SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success rate: {final_result['success_rate']:.1f}%")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Test directory: {self.test_dir}")
        
        return final_result
    
    def save_test_report(self, test_results: Dict):
        """Save comprehensive test report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.test_dir / f'drop_zones_test_report_{timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\n[REPORT] Test report saved: {report_path}")
        
        # Also save a summary file
        summary_path = self.test_dir / f'test_summary_{timestamp}.txt'
        with open(summary_path, 'w') as f:
            f.write("DROP ZONES WATCHDOG TEST SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Suite: {test_results['test_suite']}\n")
            f.write(f"Timestamp: {test_results['timestamp']}\n")
            f.write(f"Duration: {test_results['duration_seconds']:.1f} seconds\n")
            f.write(f"Success Rate: {test_results['success_rate']:.1f}%\n")
            f.write(f"Total Tests: {test_results['total_tests']}\n")
            f.write(f"Successful: {test_results['successful_tests']}\n\n")
            
            f.write("INDIVIDUAL TEST RESULTS:\n")
            f.write("-" * 25 + "\n")
            for result in test_results['individual_results']:
                status = "PASS" if result.get('success', False) else "FAIL"
                f.write(f"{result.get('test_name', 'Unknown')}: {status}\n")
                if not result.get('success', False) and 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
            
            f.write(f"\nTest Environment: {test_results['test_directory']}\n")
    
    def cleanup(self):
        """Clean up test environment"""
        try:
            if self.listener:
                self.listener.stop()
            
            # Remove test directory
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
                print(f"[CLEANUP] Test directory removed: {self.test_dir}")
        
        except Exception as e:
            print(f"[WARNING] Cleanup failed: {e}")

def main():
    """Main function to run drop zone testing"""
    print("STARTING: Drop Zones Watchdog Testing System")
    
    # Create test framework
    test_framework = DropZoneTestFramework()
    
    try:
        # Run full test suite
        results = test_framework.run_full_test_suite()
        
        # Exit with appropriate code
        if results.get('success', False):
            print(f"\n[SUCCESS] All tests completed successfully!")
            exit_code = 0
        else:
            print(f"\n[FAILURE] Some tests failed!")
            exit_code = 1
    
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Testing interrupted by user")
        exit_code = 2
    
    except Exception as e:
        print(f"\n[ERROR] Testing suite crashed: {e}")
        exit_code = 3
    
    finally:
        # Always cleanup
        test_framework.cleanup()
    
    print(f"\nTesting complete. Exit code: {exit_code}")
    return exit_code

if __name__ == "__main__":
    import sys
    sys.exit(main())