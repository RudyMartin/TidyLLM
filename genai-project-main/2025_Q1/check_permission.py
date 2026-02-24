import boto3
import json

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime')

def generate_embedding(text):
    """Generate embedding for the given text using Bedrock."""
    try:
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text})
        )
        embedding = json.loads(response['body'].read())['embedding']
        return embedding
    except Exception as e:
        print(f"Error from genration due to  {e}")

#Then call to test the data
value = generate_embedding("This is a test value")
print (value)
