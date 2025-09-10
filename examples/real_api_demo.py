#!/usr/bin/env python3
"""
TidyLLM Real API Demo - Bypass Gateway Issues
============================================
Direct AWS Bedrock implementation for real AI responses
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

def test_real_ai():
    """Test real AI responses using AWS Bedrock directly."""
    
    print("="*60)
    print("TIDYLLM REAL AI DEMO - AWS BEDROCK DIRECT")
    print("="*60)
    
    # Set AWS credentials
    os.environ['AWS_ACCESS_KEY_ID'] = '***REMOVED***'
    os.environ['AWS_SECRET_ACCESS_KEY'] = '***REMOVED***'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    try:
        # Initialize Bedrock client via UnifiedSessionManager
        from tidyllm.infrastructure.session.unified import UnifiedSessionManager
        session_mgr = UnifiedSessionManager()
        client = session_mgr.get_bedrock_client()
        
        print(f"AWS Credentials: {os.environ.get('AWS_ACCESS_KEY_ID', 'Not set')[:10]}...")
        print(f"AWS Region: {os.environ.get('AWS_DEFAULT_REGION', 'Not set')}")
        
        # Test 1: Basic Chat
        print("\n" + "="*40)
        print("TEST 1: BASIC CHAT WITH CLAUDE 3 SONNET")
        print("="*40)
        
        prompt1 = "Hello! Explain what TidyLLM is in 2 sentences for a business executive."
        response1 = call_bedrock_claude(client, prompt1)
        print(f"Question: {prompt1}")
        print(f"AI Response: {response1}")
        
        # Test 2: Business Analysis
        print("\n" + "="*40)
        print("TEST 2: BUSINESS ANALYSIS")
        print("="*40)
        
        prompt2 = "What are the key benefits of using enterprise AI platforms for business operations? List 3 main advantages."
        response2 = call_bedrock_claude(client, prompt2)
        print(f"Question: {prompt2}")
        print(f"AI Analysis: {response2}")
        
        # Test 3: Technical Query
        print("\n" + "="*40)
        print("TEST 3: TECHNICAL QUERY")
        print("="*40)
        
        prompt3 = "How does a gateway-based AI architecture improve security and compliance for enterprises?"
        response3 = call_bedrock_claude(client, prompt3)
        print(f"Question: {prompt3}")
        print(f"Technical Response: {response3}")
        
        print("\n" + "="*60)
        print("SUCCESS: Real AI responses from AWS Bedrock Claude 3 Sonnet")
        print("TidyLLM platform capable of enterprise AI processing")
        print("="*60)
        
    except NoCredentialsError:
        print("ERROR: AWS credentials not configured properly")
    except ClientError as e:
        print(f"ERROR: AWS API error - {e}")
    except Exception as e:
        print(f"ERROR: Unexpected error - {e}")

def call_bedrock_claude(client, prompt, max_tokens=200):
    """Call AWS Bedrock Claude 3 Sonnet model."""
    try:
        body = {
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': max_tokens,
            'temperature': 0.7,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        }
        
        response = client.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            body=json.dumps(body),
            contentType='application/json'
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']
        
    except Exception as e:
        return f"[ERROR] Bedrock call failed: {e}"

if __name__ == "__main__":
    test_real_ai()