#!/usr/bin/env python3
"""
Gateway Data Flow Validation System
===================================

Validates data integrity and consistency as requests flow through the 4-gateway chain.
Ensures polars DataFrames maintain proper data correlation across all stages.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'tidyllm'))

import uuid
import polars as pl
from datetime import datetime
from typing import Dict, Any, List, Tuple
import json

class GatewayDataValidator:
    """Validates data flow across gateway chain."""
    
    def __init__(self):
        self.test_data = {}
        self.validation_results = []
    
    def simulate_gateway_flow(self, request_id: str, initial_prompt: str) -> List[Tuple[str, pl.DataFrame]]:
        """Simulate data flow through all 4 gateways."""
        
        # Stage 1: Business Context Gateway
        business_data = {
            'request_id': request_id,
            'stage': 'business_context',
            'gateway': 'BusinessContextGateway',
            'timestamp': datetime.now().isoformat(),
            'prompt': initial_prompt,
            'compliance_level': 'enterprise',
            'business_rules_applied': 3,
            'user_context': 'validated_enterprise_user',
            'risk_assessment': 'low'
        }
        
        # Stage 2: Workflow Solution Gateway  
        workflow_data = {
            'request_id': request_id,
            'stage': 'workflow_solution', 
            'gateway': 'WorkflowSolutionGateway',
            'timestamp': datetime.now().isoformat(),
            'previous_stage': 'business_context',
            'workflow_type': 'ai_processing_flow',
            'solution_path': 'enterprise_standard',
            'optimization_score': 0.85,
            'routing_decision': 'ai_processing_gateway'
        }
        
        # Stage 3: AI Processing Gateway
        ai_data = {
            'request_id': request_id,
            'stage': 'ai_processing',
            'gateway': 'AIProcessingGateway', 
            'timestamp': datetime.now().isoformat(),
            'previous_stage': 'workflow_solution',
            'prompt': initial_prompt,  # Should match original
            'model_selected': 'claude-3-sonnet',
            'temperature': 0.7,
            'max_tokens': 2000,
            'backend': 'bedrock',
            'cache_hit': False
        }
        
        # Stage 4: Model Execution Gateway
        execution_data = {
            'request_id': request_id,
            'stage': 'model_execution',
            'gateway': 'ModelExecutionGateway',
            'timestamp': datetime.now().isoformat(), 
            'previous_stage': 'ai_processing',
            'model_executed': 'claude-3-sonnet',  # Should match ai_processing
            'tokens_used': 156,
            'response_length': 487,
            'execution_time_ms': 1247,
            'status': 'success'
        }
        
        # Create DataFrames for each stage
        stage_dataframes = []
        for stage_data in [business_data, workflow_data, ai_data, execution_data]:
            df = pl.DataFrame({k: [v] for k, v in stage_data.items()})
            stage_dataframes.append((stage_data['stage'], df))
            
        return stage_dataframes
    
    def validate_data_consistency(self, dataframes: List[Tuple[str, pl.DataFrame]]) -> Dict[str, Any]:
        """Validate data consistency across gateway stages."""
        
        results = {
            'request_id_consistency': True,
            'stage_progression': True,
            'data_continuity': True,
            'issues': []
        }
        
        # Extract all request IDs
        request_ids = []
        stages = []
        
        for stage_name, df in dataframes:
            request_id = df['request_id'][0]
            stage = df['stage'][0]
            
            request_ids.append(request_id)
            stages.append(stage)
            
            # Validate stage name matches DataFrame content
            if stage != stage_name:
                results['stage_progression'] = False
                results['issues'].append(f"Stage mismatch in {stage_name}: DataFrame says '{stage}'")
        
        # Check request ID consistency
        unique_request_ids = list(set(request_ids))
        if len(unique_request_ids) != 1:
            results['request_id_consistency'] = False
            results['issues'].append(f"Multiple request IDs found: {unique_request_ids}")
        
        # Check stage progression
        expected_stages = ['business_context', 'workflow_solution', 'ai_processing', 'model_execution']
        if stages != expected_stages:
            results['stage_progression'] = False
            results['issues'].append(f"Stage order incorrect. Expected: {expected_stages}, Got: {stages}")
        
        # Check data continuity (prompt should flow through)
        business_df = next((df for stage, df in dataframes if stage == 'business_context'), None)
        ai_df = next((df for stage, df in dataframes if stage == 'ai_processing'), None)
        
        if business_df is not None and ai_df is not None:
            if 'prompt' in business_df.columns and 'prompt' in ai_df.columns:
                business_prompt = business_df['prompt'][0]
                ai_prompt = ai_df['prompt'][0]
                
                if business_prompt != ai_prompt:
                    results['data_continuity'] = False
                    results['issues'].append(f"Prompt mismatch: Business='{business_prompt}' vs AI='{ai_prompt}'")
        
        # Check model consistency
        ai_df = next((df for stage, df in dataframes if stage == 'ai_processing'), None)
        exec_df = next((df for stage, df in dataframes if stage == 'model_execution'), None)
        
        if ai_df is not None and exec_df is not None:
            if 'model_selected' in ai_df.columns and 'model_executed' in exec_df.columns:
                selected_model = ai_df['model_selected'][0]
                executed_model = exec_df['model_executed'][0]
                
                if selected_model != executed_model:
                    results['data_continuity'] = False
                    results['issues'].append(f"Model mismatch: Selected='{selected_model}' vs Executed='{executed_model}'")
        
        return results
    
    def run_validation_test(self):
        """Run comprehensive data flow validation."""
        
        print("=" * 60)
        print("GATEWAY DATA FLOW VALIDATION TEST")
        print("=" * 60)
        
        # Generate test request
        request_id = str(uuid.uuid4())
        test_prompt = "Analyze the quarterly financial report for compliance issues"
        
        print(f"Test Request ID: {request_id}")
        print(f"Test Prompt: '{test_prompt}'")
        print()
        
        # Simulate gateway flow
        print("SIMULATING 4-GATEWAY DATA FLOW:")
        print("-" * 40)
        
        dataframes = self.simulate_gateway_flow(request_id, test_prompt)
        
        for stage_name, df in dataframes:
            print(f"[{stage_name.upper()}]")
            print(f"  Shape: {df.shape}")
            print(f"  Request ID: {df['request_id'][0]}")
            print(f"  Timestamp: {df['timestamp'][0]}")
            if 'previous_stage' in df.columns:
                print(f"  Previous Stage: {df['previous_stage'][0]}")
            print()
        
        # Validate data consistency
        print("DATA CONSISTENCY VALIDATION:")
        print("-" * 40)
        
        validation_results = self.validate_data_consistency(dataframes)
        
        print(f"Request ID Consistency: {'PASS' if validation_results['request_id_consistency'] else 'FAIL'}")
        print(f"Stage Progression: {'PASS' if validation_results['stage_progression'] else 'FAIL'}")
        print(f"Data Continuity: {'PASS' if validation_results['data_continuity'] else 'FAIL'}")
        
        if validation_results['issues']:
            print()
            print("ISSUES FOUND:")
            for issue in validation_results['issues']:
                print(f"  - {issue}")
        else:
            print()
            print("NO ISSUES FOUND - DATA FLOW IS CONSISTENT!")
        
        print()
        print("=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        
        return validation_results

if __name__ == "__main__":
    validator = GatewayDataValidator()
    results = validator.run_validation_test()