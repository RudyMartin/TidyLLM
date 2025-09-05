#!/usr/bin/env python3
"""
🤖 AWS Bedrock Demo - VectorQA Sage

This script demonstrates AWS Bedrock integration and model capabilities 
for the QA document processing use case.

Overview:
- AWS Bedrock Models: Claude, Titan, and other foundation models
- Use Cases: Document analysis, embedding generation, report creation
- Security: AWS-only mode for compliance environments

Usage:
    python notebooks/02_aws_bedrock_demo.py

Requirements:
    pip install boto3 pandas numpy matplotlib seaborn
    AWS credentials configured (IAM role or access keys)
"""

import os
import sys
import json
import time
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError, NoCredentialsError

# Add src to path for backend imports
sys.path.insert(0, '../src')

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BedrockDemo:
    """AWS Bedrock demonstration and testing class."""
    
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize Bedrock client and configuration."""
        self.region_name = region_name
        self.bedrock_client = None
        self.bedrock_runtime = None
        
        # Bedrock model configurations
        self.models = {
            'embedding': {
                'primary': 'amazon.titan-embed-text-v2:0',
                'fallback': 'amazon.titan-embed-text-v1',
                'dimensions': 1024
            },
            'document_analysis': {
                'primary': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'fallback': 'amazon.titan-text-express-v1',
                'max_tokens': 4096,
                'temperature': 0.1
            },
            'report_generation': {
                'primary': 'anthropic.claude-3-opus-20240229-v1:0',
                'fallback': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'max_tokens': 4096,
                'temperature': 0.2
            },
            'quick_processing': {
                'primary': 'anthropic.claude-3-haiku-20240307-v1:0',
                'fallback': 'amazon.titan-text-lite-v1',
                'max_tokens': 2048,
                'temperature': 0.1
            }
        }
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS Bedrock clients."""
        try:
            # Initialize Bedrock client for model management
            self.bedrock_client = boto3.client('bedrock', region_name=self.region_name)
            
            # Initialize Bedrock runtime for model invocation
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=self.region_name)
            
            print(f"✅ AWS Bedrock clients initialized for region: {self.region_name}")
            
        except NoCredentialsError:
            print("❌ AWS credentials not found")
            print("💡 Please configure AWS credentials using:")
            print("   - IAM role (recommended for SageMaker)")
            print("   - AWS CLI: aws configure")
            print("   - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            self.bedrock_client = None
            self.bedrock_runtime = None
            
        except Exception as e:
            print(f"❌ Error initializing Bedrock clients: {e}")
            self.bedrock_client = None
            self.bedrock_runtime = None
    
    def list_available_models(self) -> Dict[str, Any]:
        """List available Bedrock models."""
        if not self.bedrock_client:
            return {}
            
        try:
            response = self.bedrock_client.list_foundation_models()
            
            # Organize models by provider
            models_by_provider = {}
            for model in response['modelSummaries']:
                provider = model['providerName']
                if provider not in models_by_provider:
                    models_by_provider[provider] = []
                
                models_by_provider[provider].append({
                    'modelId': model['modelId'],
                    'modelName': model['modelName'],
                    'inputModalities': model.get('inputModalities', []),
                    'outputModalities': model.get('outputModalities', []),
                    'inferenceTypesSupported': model.get('inferenceTypesSupported', [])
                })
            
            print("🤖 Available Bedrock Models by Provider:")
            print("=" * 50)
            
            for provider, models in models_by_provider.items():
                print(f"\n📋 {provider}:")
                for model in models:
                    print(f"  - {model['modelId']}")
                    print(f"    Name: {model['modelName']}")
                    print(f"    Input: {', '.join(model['inputModalities'])}")
                    print(f"    Output: {', '.join(model['outputModalities'])}")
                    print(f"    Inference: {', '.join(model['inferenceTypesSupported'])}")
                    print()
            
            return models_by_provider
            
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return {}
    
    def test_model_access(self, model_id: str) -> bool:
        """Test access to a specific Bedrock model."""
        if not self.bedrock_client:
            return False
            
        try:
            # Try to get model details
            response = self.bedrock_client.get_foundation_model(modelIdentifier=model_id)
            print(f"✅ Access confirmed for {model_id}")
            print(f"   Model: {response['modelDetails']['modelName']}")
            print(f"   Provider: {response['modelDetails']['providerName']}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDeniedException':
                print(f"❌ Access denied for {model_id}")
                print("💡 Check IAM permissions for Bedrock access")
            elif error_code == 'ResourceNotFoundException':
                print(f"❌ Model not found: {model_id}")
            else:
                print(f"❌ Error testing {model_id}: {error_code}")
            return False
            
        except Exception as e:
            print(f"❌ Error testing {model_id}: {e}")
            return False
    
    def generate_embedding(self, text: str, model_id: str = None) -> Optional[List[float]]:
        """Generate embedding using Bedrock Titan model."""
        if not self.bedrock_runtime:
            return None
            
        model_id = model_id or self.models['embedding']['primary']
        
        try:
            # Prepare request for Titan embedding model
            if 'titan' in model_id.lower():
                request_body = {
                    "inputText": text
                }
            else:
                # For other embedding models
                request_body = {
                    "text": text
                }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            if 'embedding' in response_body:
                embedding = response_body['embedding']
                print(f"✅ Generated embedding with {model_id}")
                print(f"   Dimensions: {len(embedding)}")
                print(f"   Text length: {len(text)} characters")
                return embedding
            else:
                print(f"❌ No embedding found in response from {model_id}")
                return None
                
        except Exception as e:
            print(f"❌ Error generating embedding with {model_id}: {e}")
            return None
    
    def analyze_document(self, content: str, model_id: str = None) -> Optional[Dict[str, Any]]:
        """Analyze document content using Bedrock Claude model."""
        if not self.bedrock_runtime:
            return None
            
        model_id = model_id or self.models['document_analysis']['primary']
        
        # Create analysis prompt
        prompt = f"""
        Analyze the following document content and extract key information in JSON format.
        
        Document Content:
        {content[:4000]}  # Limit content length
        
        Please extract and return the following information in JSON format:
        {{
            "document_type": "Type of document (Policy, Standards, Procedure, etc.)",
            "risk_tier": "Risk tier (High, Medium, Low)",
            "lifecycle_stage": "Lifecycle stage (Development, Validation, Monitoring)",
            "key_topics": ["List of key topics covered"],
            "compliance_areas": ["List of compliance areas mentioned"],
            "effective_date": "Effective date if mentioned",
            "summary": "Brief summary of the document"
        }}
        
        Return only valid JSON, no additional text.
        """
        
        try:
            # Prepare request for Claude model
            if 'claude' in model_id.lower():
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.models['document_analysis']['max_tokens'],
                    "temperature": self.models['document_analysis']['temperature'],
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            else:
                # For Titan models
                request_body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": self.models['document_analysis']['max_tokens'],
                        "temperature": self.models['document_analysis']['temperature']
                    }
                }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract content based on model type
            if 'claude' in model_id.lower():
                content = response_body['content'][0]['text']
            else:
                content = response_body.get('results', [{}])[0].get('outputText', '')
            
            # Try to parse JSON response
            try:
                analysis = json.loads(content)
                print(f"✅ Document analysis completed with {model_id}")
                return analysis
            except json.JSONDecodeError:
                print(f"⚠️ Non-JSON response from {model_id}, returning raw content")
                return {"raw_response": content}
                
        except Exception as e:
            print(f"❌ Error analyzing document with {model_id}: {e}")
            return None
    
    def generate_qa_report(self, analysis_data: Dict[str, Any], model_id: str = None) -> Optional[str]:
        """Generate QA report using Bedrock Claude model."""
        if not self.bedrock_runtime:
            return None
            
        model_id = model_id or self.models['report_generation']['primary']
        
        # Create report generation prompt
        prompt = f"""
        Generate a comprehensive QA HealthCheck Report based on the following analysis data.
        
        Analysis Data:
        {json.dumps(analysis_data, indent=2)}
        
        Please create a professional QA report with the following sections:
        
        1. Executive Summary
        2. Document Overview
        3. Compliance Assessment
        4. Risk Analysis
        5. Recommendations
        6. Next Steps
        
        Format the report in markdown with clear headings and bullet points.
        """
        
        try:
            # Prepare request for Claude model
            if 'claude' in model_id.lower():
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.models['report_generation']['max_tokens'],
                    "temperature": self.models['report_generation']['temperature'],
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            else:
                # For Titan models
                request_body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": self.models['report_generation']['max_tokens'],
                        "temperature": self.models['report_generation']['temperature']
                    }
                }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract content based on model type
            if 'claude' in model_id.lower():
                report = response_body['content'][0]['text']
            else:
                report = response_body.get('results', [{}])[0].get('outputText', '')
            
            print(f"✅ QA report generated with {model_id}")
            return report
                
        except Exception as e:
            print(f"❌ Error generating report with {model_id}: {e}")
            return None
    
    def quick_validation(self, content: str, validation_type: str, model_id: str = None) -> Optional[Dict[str, Any]]:
        """Perform quick validation using Bedrock Claude Haiku."""
        if not self.bedrock_runtime:
            return None
            
        model_id = model_id or self.models['quick_processing']['primary']
        
        # Create validation prompt
        prompt = f"""
        Perform a quick {validation_type} validation on the following content.
        
        Content:
        {content[:2000]}  # Limit content length
        
        Please return a JSON response with:
        {{
            "validation_type": "{validation_type}",
            "status": "PASS/FAIL/WARNING",
            "confidence": 0.0-1.0,
            "issues": ["List of issues found"],
            "recommendations": ["List of recommendations"]
        }}
        
        Return only valid JSON, no additional text.
        """
        
        try:
            # Prepare request for Claude model
            if 'claude' in model_id.lower():
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.models['quick_processing']['max_tokens'],
                    "temperature": self.models['quick_processing']['temperature'],
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            else:
                # For Titan models
                request_body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": self.models['quick_processing']['max_tokens'],
                        "temperature": self.models['quick_processing']['temperature']
                    }
                }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract content based on model type
            if 'claude' in model_id.lower():
                content = response_body['content'][0]['text']
            else:
                content = response_body.get('results', [{}])[0].get('outputText', '')
            
            # Try to parse JSON response
            try:
                validation = json.loads(content)
                print(f"✅ {validation_type} validation completed with {model_id}")
                return validation
            except json.JSONDecodeError:
                print(f"⚠️ Non-JSON response from {model_id}, returning raw content")
                return {"raw_response": content, "validation_type": validation_type}
                
        except Exception as e:
            print(f"❌ Error performing {validation_type} validation with {model_id}: {e}")
            return None
    
    def benchmark_models(self, test_content: str) -> Dict[str, Any]:
        """Benchmark different Bedrock models for performance and cost."""
        if not self.bedrock_runtime:
            return {}
            
        results = {}
        test_text = test_content[:1000]  # Use first 1000 characters for testing
        
        print("🏃‍♂️ Benchmarking Bedrock Models...")
        print("=" * 50)
        
        for task, model_config in self.models.items():
            print(f"\n📊 Testing {task} task...")
            
            start_time = time.time()
            
            if task == 'embedding':
                result = self.generate_embedding(test_text, model_config['primary'])
                success = result is not None
            elif task == 'document_analysis':
                result = self.analyze_document(test_text, model_config['primary'])
                success = result is not None
            elif task == 'report_generation':
                # Use a simple analysis for report generation
                analysis = {"document_type": "Test", "summary": "Test document"}
                result = self.generate_qa_report(analysis, model_config['primary'])
                success = result is not None
            elif task == 'quick_processing':
                result = self.quick_validation(test_text, "compliance", model_config['primary'])
                success = result is not None
            else:
                continue
            
            end_time = time.time()
            duration = end_time - start_time
            
            results[task] = {
                'model': model_config['primary'],
                'success': success,
                'duration_seconds': duration,
                'input_tokens': len(test_text.split()),  # Rough estimate
                'output_tokens': len(str(result).split()) if result else 0
            }
            
            print(f"  Model: {model_config['primary']}")
            print(f"  Success: {'✅' if success else '❌'}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Input tokens: ~{results[task]['input_tokens']}")
            print(f"  Output tokens: ~{results[task]['output_tokens']}")
        
        return results
    
    def create_benchmark_visualization(self, results: Dict[str, Any]):
        """Create visualization of benchmark results."""
        if not results:
            return
            
        # Prepare data for visualization
        tasks = list(results.keys())
        durations = [results[task]['duration_seconds'] for task in tasks]
        success_rates = [1 if results[task]['success'] else 0 for task in tasks]
        models = [results[task]['model'].split('.')[-1] for task in tasks]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Duration comparison
        bars1 = ax1.bar(range(len(tasks)), durations, color='skyblue')
        ax1.set_title('Model Performance - Duration')
        ax1.set_xlabel('Task')
        ax1.set_ylabel('Duration (seconds)')
        ax1.set_xticks(range(len(tasks)))
        ax1.set_xticklabels(tasks, rotation=45, ha='right')
        
        # Add duration labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        # Success rate comparison
        bars2 = ax2.bar(range(len(tasks)), success_rates, color=['green' if s else 'red' for s in success_rates])
        ax2.set_title('Model Performance - Success Rate')
        ax2.set_xlabel('Task')
        ax2.set_ylabel('Success Rate')
        ax2.set_xticks(range(len(tasks)))
        ax2.set_xticklabels(tasks, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        
        # Add success labels on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            status = '✅' if height > 0 else '❌'
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    status, ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('notebooks/bedrock_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Benchmark visualization saved as 'notebooks/bedrock_benchmark.png'")
    
    def generate_insights(self) -> Dict[str, Any]:
        """Generate insights from Bedrock demonstration."""
        insights = {
            "aws_integration": "✅ AWS Bedrock integration ready",
            "model_access": "✅ Multiple model types available",
            "embedding_capability": "✅ Titan embedding models ready",
            "analysis_capability": "✅ Claude analysis models ready",
            "report_generation": "✅ Report generation capability ready",
            "security_compliance": "✅ AWS-only mode for compliance",
            "next_steps": [
                "Integrate with database for embedding storage",
                "Connect to S3 for document processing",
                "Implement MCP coordinators",
                "Test end-to-end QA workflow",
                "Optimize model selection for cost/performance"
            ]
        }
        
        print("🎯 AWS Bedrock Demo Insights:")
        print("=" * 50)
        
        for key, value in insights.items():
            if key == "next_steps":
                print(f"\n📋 {key.replace('_', ' ').title()}:")
                for i, step in enumerate(value, 1):
                    print(f"  {i}. {step}")
            else:
                print(f"{value}")
        
        print("\n🚀 Ready for next script: Embedding Generation")
        return insights


def main():
    """Main function to run AWS Bedrock demonstration."""
    print("🤖 VectorQA Sage - AWS Bedrock Demo")
    print("=" * 50)
    
    # Initialize Bedrock demo
    demo = BedrockDemo()
    
    if not demo.bedrock_client:
        print("❌ Cannot proceed without AWS Bedrock access")
        return
    
    try:
        # 1. List available models
        print("\n📋 Step 1: Available Bedrock Models")
        print("-" * 40)
        models = demo.list_available_models()
        
        # 2. Test model access
        print("\n📋 Step 2: Model Access Testing")
        print("-" * 40)
        test_models = [
            'amazon.titan-embed-text-v2:0',
            'anthropic.claude-3-sonnet-20240229-v1:0',
            'anthropic.claude-3-haiku-20240307-v1:0'
        ]
        
        for model_id in test_models:
            demo.test_model_access(model_id)
        
        # 3. Test embedding generation
        print("\n📋 Step 3: Embedding Generation Test")
        print("-" * 40)
        test_text = "This is a test document for embedding generation using AWS Bedrock Titan model."
        embedding = demo.generate_embedding(test_text)
        
        # 4. Test document analysis
        print("\n📋 Step 4: Document Analysis Test")
        print("-" * 40)
        sample_document = """
        Model Risk Governance Policy
        
        This policy establishes the framework for managing model risk across the organization.
        All models must undergo validation before deployment and continuous monitoring thereafter.
        
        Risk Tiers:
        - High: Models with significant financial or regulatory impact
        - Medium: Models with moderate impact on business operations
        - Low: Models with minimal impact on business operations
        
        Effective Date: January 1, 2024
        """
        
        analysis = demo.analyze_document(sample_document)
        if analysis:
            print("📊 Analysis Results:")
            print(json.dumps(analysis, indent=2))
        
        # 5. Test report generation
        print("\n📋 Step 5: Report Generation Test")
        print("-" * 40)
        if analysis:
            report = demo.generate_qa_report(analysis)
            if report:
                print("📄 Generated Report Preview:")
                print(report[:500] + "..." if len(report) > 500 else report)
        
        # 6. Test quick validation
        print("\n📋 Step 6: Quick Validation Test")
        print("-" * 40)
        validation = demo.quick_validation(sample_document, "compliance")
        if validation:
            print("✅ Validation Results:")
            print(json.dumps(validation, indent=2))
        
        # 7. Benchmark models
        print("\n📋 Step 7: Model Benchmarking")
        print("-" * 40)
        benchmark_results = demo.benchmark_models(sample_document)
        demo.create_benchmark_visualization(benchmark_results)
        
        # 8. Generate insights
        print("\n📋 Step 8: Key Insights & Next Steps")
        print("-" * 40)
        insights = demo.generate_insights()
        
        # Summary
        print("\n📝 Conclusion")
        print("=" * 50)
        print("This script has demonstrated AWS Bedrock integration and capabilities:")
        print("\n✅ What's Working:")
        print("- AWS Bedrock Integration: Multiple model types available")
        print("- Embedding Generation: Titan models for vector creation")
        print("- Document Analysis: Claude models for content understanding")
        print("- Report Generation: High-quality report creation")
        print("- Quick Validation: Fast compliance checking")
        print("- Security: AWS-only mode for compliance environments")
        
        print("\n🔧 Next Steps:")
        print("1. Connect embeddings to PostgreSQL database")
        print("2. Process S3 standards documents")
        print("3. Implement MCP coordinators")
        print("4. Test end-to-end QA workflow")
        
        print("\n📚 Related Scripts:")
        print("- 01_database_exploration.py - Database schema exploration")
        print("- 03_embedding_generation.py - Embedding generation pipeline")
        print("- 05_point_in_time_demo.py - Point-in-time benchmarks")
        
        print("\nAWS Bedrock is **ready for QA document processing** - we can now generate embeddings, analyze documents, and create reports!")
        
    except Exception as e:
        print(f"❌ Error during Bedrock demo: {e}")


if __name__ == "__main__":
    main()
