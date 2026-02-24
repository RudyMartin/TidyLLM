#pip install psycopg2-binary boto3

import boto3
import psycopg2
from botocore.exceptions import NoCredentialsError

# Set your AWS region
region_name = 'your-region'

# Secret ARN and Database details
secret_arn = 'arn:aws:secretsmanager:your-region:your-account-id:secret:your-secret-id'
endpoint = 'your-db-instance-name.region.rds.amazonaws.com'
database_name = 'your-database-name'

# Initialize a Secrets Manager client
client = boto3.client('secretsmanager', region_name=region_name)

def get_db_credentials(secret_arn):
    try:
        # Retrieve the secret
        get_secret_value_response = client.get_secret_value(SecretId=secret_arn)
        secret = get_secret_value_response['SecretString']
        return secret
    except NoCredentialsError:
        print("Error: No valid AWS credentials found.")
        return None

# Get the credentials
credentials = get_db_credentials(secret_arn)
if credentials:
    # Use credentials (username and password)
    import json
    credentials = json.loads(credentials)
    username = credentials['username']
    password = credentials['password']

    # Connect to the PostgreSQL database
    try:
        connection = psycopg2.connect(
            dbname=database_name,
            user=username,
            password=password,
            host=endpoint,
            port=5432  # Default PostgreSQL port
        )
        print("Successfully connected to the database!")
    except Exception as e:
        print("Error while connecting to PostgreSQL:", e)
