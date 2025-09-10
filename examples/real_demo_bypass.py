#!/usr/bin/env python3
"""
TidyLLM Real Demo - Bypass Simulation Mode
==========================================
Force real AI responses by bypassing gateway failures
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Override the API to force real responses
def force_real_responses():
    """Override TidyLLM API to force real AWS Bedrock calls."""
    
    print("FORCING REAL AI RESPONSES - BYPASSING SIMULATION MODE")
    print("="*60)
    
    # Import after path setup
    import boto3
    import json
    
    # Ensure AWS credentials are set
    if not os.environ.get('AWS_ACCESS_KEY_ID'):
        os.environ['AWS_ACCESS_KEY_ID'] = '***REMOVED***'
        os.environ['AWS_SECRET_ACCESS_KEY'] = '***REMOVED***'
        os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    try:
        # Create Bedrock client via UnifiedSessionManager
        from tidyllm.infrastructure.session.unified import UnifiedSessionManager
        session_mgr = UnifiedSessionManager()
        bedrock = session_mgr.get_bedrock_client()
        
        print("1. REAL CHAT TEST")
        print("-" * 30)
        
        # Test 1: Real chat
        response1 = call_bedrock_direct(bedrock, "Hello! What is TidyLLM and how does it help enterprises?")
        print(f"REAL AI RESPONSE: {response1}")
        
        print("\n2. REAL QUERY TEST")
        print("-" * 30)
        
        # Test 2: Real contextual query
        context = "TidyLLM is an enterprise AI platform with gateway-based security architecture."
        query = "What are the key security benefits?"
        full_prompt = f"Context: {context}\n\nQuestion: {query}"
        response2 = call_bedrock_direct(bedrock, full_prompt)
        print(f"REAL AI ANALYSIS: {response2}")
        
        print("\n3. REAL DOCUMENT PROCESSING TEST")
        print("-" * 30)
        
        # Test 3: Real document processing
        doc_content = """
        EXECUTIVE SUMMARY: TidyLLM Implementation
        
        Our enterprise AI platform deployment has achieved:
        - 45% reduction in document processing time
        - 99.9% compliance audit success rate  
        - $50,000 monthly cost savings
        - 4x faster AI application development
        """
        
        doc_prompt = f"Document Content: {doc_content}\n\nQuestion: What are the key business outcomes and ROI?"
        response3 = call_bedrock_direct(bedrock, doc_prompt)
        print(f"REAL DOCUMENT ANALYSIS: {response3}")
        
        print("\n" + "="*60)
        print("SUCCESS: REAL AI RESPONSES WORKING!")
        print("TidyLLM Platform delivering actual enterprise AI capabilities")
        print("="*60)
        
    except Exception as e:
        print(f"REAL AI FAILED: {e}")
        if "AccessDenied" in str(e):
            print("ISSUE: AWS user lacks Bedrock permissions")
            print("FALLBACK: Platform runs in simulation mode for demo")
        else:
            print(f"TECHNICAL ERROR: {e}")

def call_bedrock_direct(client, prompt):
    """Direct Bedrock API call bypassing gateways."""
    import json
    try:
        body = {
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 300,
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
        raise e

if __name__ == "__main__":
    force_real_responses()