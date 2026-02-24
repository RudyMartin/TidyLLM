import boto3
import json

# Replace with your actual secret ARN
SECRET_ARN = "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-bedrock-creds"

# 1. Create a Secrets Manager client (using your *current* login context)
secrets_client = boto3.client("secretsmanager", region_name="us-east-1")

# 2. Retrieve the secret
response = secrets_client.get_secret_value(SecretId=SECRET_ARN)
secret_dict = json.loads(response["SecretString"])

# Extract credentials
aws_access_key_id = secret_dict["aws_access_key_id"]
aws_secret_access_key = secret_dict["aws_secret_access_key"]
region = secret_dict.get("region", "us-east-1")

# 3. Create a Bedrock client with those credentials
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# 4. Test: list models
models = bedrock_client.list_foundation_models()
print(models)
