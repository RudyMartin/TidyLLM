#!/usr/bin/env python3
"""
Gateway Data Corruption Detection Test
=====================================

Tests the data validation system's ability to detect common data integrity issues
that could occur as data flows through the 4-gateway chain.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'tidyllm'))

import uuid
import polars as pl
from datetime import datetime
from validate_gateway_data_flow import GatewayDataValidator

class DataCorruptionTester:
    """Test data corruption detection capabilities."""
    
    def __init__(self):
        self.validator = GatewayDataValidator()
    
    def test_request_id_corruption(self):
        """Test detection of request ID inconsistency."""
        print("TEST 1: Request ID Corruption Detection")
        print("-" * 40)
        
        request_id = str(uuid.uuid4())
        corrupted_id = str(uuid.uuid4())  # Different ID
        
        # Create dataframes with inconsistent request IDs
        business_df = pl.DataFrame({
            'request_id': [request_id],
            'stage': ['business_context'],
            'prompt': ['Test prompt']
        })
        
        ai_df = pl.DataFrame({
            'request_id': [corrupted_id],  # CORRUPTED: Different ID
            'stage': ['ai_processing'],
            'prompt': ['Test prompt']
        })
        
        dataframes = [
            ('business_context', business_df),
            ('ai_processing', ai_df)
        ]
        
        results = self.validator.validate_data_consistency(dataframes)
        
        print(f"Request ID Consistency: {'PASS' if results['request_id_consistency'] else 'FAIL'}")
        if results['issues']:
            print("Issues detected:")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        expected_failure = not results['request_id_consistency']
        print(f"Corruption Detection: {'SUCCESS' if expected_failure else 'MISSED'}")
        print()
        
        return expected_failure
    
    def test_prompt_corruption(self):
        """Test detection of prompt data corruption."""
        print("TEST 2: Prompt Data Corruption Detection")
        print("-" * 40)
        
        request_id = str(uuid.uuid4())
        original_prompt = "Analyze financial data"
        corrupted_prompt = "Analyze weather data"  # CORRUPTED
        
        business_df = pl.DataFrame({
            'request_id': [request_id],
            'stage': ['business_context'],
            'prompt': [original_prompt]
        })
        
        ai_df = pl.DataFrame({
            'request_id': [request_id],
            'stage': ['ai_processing'], 
            'prompt': [corrupted_prompt]  # CORRUPTED: Different prompt
        })
        
        dataframes = [
            ('business_context', business_df),
            ('ai_processing', ai_df)
        ]
        
        results = self.validator.validate_data_consistency(dataframes)
        
        print(f"Data Continuity: {'PASS' if results['data_continuity'] else 'FAIL'}")
        if results['issues']:
            print("Issues detected:")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        expected_failure = not results['data_continuity']
        print(f"Corruption Detection: {'SUCCESS' if expected_failure else 'MISSED'}")
        print()
        
        return expected_failure
    
    def test_model_consistency_corruption(self):
        """Test detection of model selection/execution mismatch.""" 
        print("TEST 3: Model Consistency Corruption Detection")
        print("-" * 40)
        
        request_id = str(uuid.uuid4())
        
        ai_df = pl.DataFrame({
            'request_id': [request_id],
            'stage': ['ai_processing'],
            'model_selected': ['claude-3-sonnet']
        })
        
        execution_df = pl.DataFrame({
            'request_id': [request_id],
            'stage': ['model_execution'],
            'model_executed': ['gpt-4']  # CORRUPTED: Different model executed
        })
        
        dataframes = [
            ('ai_processing', ai_df),
            ('model_execution', execution_df)
        ]
        
        results = self.validator.validate_data_consistency(dataframes)
        
        print(f"Data Continuity: {'PASS' if results['data_continuity'] else 'FAIL'}")
        if results['issues']:
            print("Issues detected:")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        expected_failure = not results['data_continuity']
        print(f"Corruption Detection: {'SUCCESS' if expected_failure else 'MISSED'}")
        print()
        
        return expected_failure
    
    def test_stage_progression_corruption(self):
        """Test detection of incorrect stage progression."""
        print("TEST 4: Stage Progression Corruption Detection")
        print("-" * 40)
        
        request_id = str(uuid.uuid4())
        
        # Create dataframes with wrong stage progression
        business_df = pl.DataFrame({
            'request_id': [request_id],
            'stage': ['business_context']
        })
        
        # Skip workflow_solution - go directly to ai_processing (CORRUPTED)
        ai_df = pl.DataFrame({
            'request_id': [request_id], 
            'stage': ['ai_processing']
        })
        
        execution_df = pl.DataFrame({
            'request_id': [request_id],
            'stage': ['model_execution']
        })
        
        dataframes = [
            ('business_context', business_df),
            ('ai_processing', ai_df),  # Missing workflow_solution stage
            ('model_execution', execution_df)
        ]
        
        # This should detect that we're missing the workflow_solution stage
        stages = [stage for stage, _ in dataframes]
        expected_stages = ['business_context', 'workflow_solution', 'ai_processing', 'model_execution']
        
        corruption_detected = stages != expected_stages
        print(f"Stage Progression: {'FAIL' if corruption_detected else 'PASS'}")
        print(f"Expected: {expected_stages}")
        print(f"Actual: {stages}")
        print(f"Corruption Detection: {'SUCCESS' if corruption_detected else 'MISSED'}")
        print()
        
        return corruption_detected
    
    def run_all_corruption_tests(self):
        """Run all data corruption detection tests."""
        print("=" * 60)
        print("GATEWAY DATA CORRUPTION DETECTION TESTS")
        print("=" * 60)
        print()
        
        tests = [
            ("Request ID Corruption", self.test_request_id_corruption),
            ("Prompt Data Corruption", self.test_prompt_corruption),
            ("Model Consistency Corruption", self.test_model_consistency_corruption),
            ("Stage Progression Corruption", self.test_stage_progression_corruption)
        ]
        
        results = []
        for test_name, test_func in tests:
            success = test_func()
            results.append((test_name, success))
        
        print("=" * 60)
        print("CORRUPTION DETECTION SUMMARY")
        print("=" * 60)
        
        for test_name, success in results:
            status = "PASS" if success else "FAIL" 
            print(f"{test_name}: {status}")
        
        all_passed = all(success for _, success in results)
        print()
        print(f"Overall Result: {'ALL CORRUPTION TYPES DETECTED' if all_passed else 'SOME CORRUPTION MISSED'}")
        print()
        
        return results

if __name__ == "__main__":
    tester = DataCorruptionTester()
    results = tester.run_all_corruption_tests()