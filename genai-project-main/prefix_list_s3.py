import boto3
from configuration import s3_client

def list_folders_s3(bucket_name, prefix=""):
    """
    Lists all subfolders within a given prefix in an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        prefix (str, optional): Folder path to filter (default: "" for the root level).

    Returns:
        List[str]: List of subfolder names (prefixes).
    """
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter="/")

        if "CommonPrefixes" in response:
            return [folder["Prefix"] for folder in response["CommonPrefixes"]]
        else:
            print(f"⚠️ No folders found in {bucket_name}/{prefix}")
            return []

    except Exception as e:
        print(f"❌ ERROR: Access Denied or Missing Permissions for {bucket_name}/{prefix} → {e}")
        return []


import boto3
from configuration import s3_client

def list_json_files_s3(bucket_name, prefix=""):
    """
    Lists all JSON files in an S3 bucket under a specific prefix.

    Args:
        bucket_name (str): Name of the S3 bucket.
        prefix (str, optional): Folder path to filter (default: "" for all files).

    Returns:
        List[str]: List of JSON file paths in S3.
    """
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if "Contents" in response:
            return [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".json")]
        else:
            print(f"⚠️ No JSON files found in {bucket_name}/{prefix}")
            return []

    except Exception as e:
        print(f"❌ ERROR: Access Denied or Missing Permissions for {bucket_name}/{prefix} → {e}")
        return []
