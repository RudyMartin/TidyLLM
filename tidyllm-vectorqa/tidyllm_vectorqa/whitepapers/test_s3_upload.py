"""
Quick S3 upload test script
"""

def test_s3_upload():
    try:
        import boto3
        
        # Test credentials
        s3_client = boto3.client('s3')
        response = s3_client.list_buckets()
        print(f"SUCCESS: Connected to S3 - Found {len(response['Buckets'])} buckets")
        
        # Show available buckets
        for bucket in response['Buckets']:
            print(f"  - {bucket['Name']}")
        
        # Test actual upload
        from paper_repository import get_paper_repository
        from backend_config import get_backend_config
        
        backend_config = get_backend_config()
        repo = get_paper_repository(backend_config)
        
        # Try upload to your bucket
        bucket_name = input("Enter your S3 bucket name: ")
        result = repo.sync_to_s3(bucket_name, 'papers/')
        
        if result["success"]:
            print(f"SUCCESS: {result['message']}")
            print(f"Uploaded: {result['uploaded_count']} papers")
        else:
            print(f"FAILED: {result['message']}")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_s3_upload()