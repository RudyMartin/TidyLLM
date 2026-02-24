import os

def list_json_files(folder_path):
    """
    Lists all JSON files in a given folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        List[str]: List of JSON file paths.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' not found.")

    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]

import boto3

def list_json_files_s3(bucket_name, prefix=""):
    """
    Recursively lists all JSON files in an S3 bucket, including subdirectories.

    Args:
        bucket_name (str): Name of the S3 bucket.
        prefix (str, optional): Folder path within the bucket to filter (default: "").

    Returns:
        List[str]: List of S3 object keys (JSON file paths).
    """
    s3_client = boto3.client("s3")
    json_files = []
    continuation_token = None  # Handles paginated results

    while True:
        # Fetch the list of files, handling pagination
        list_params = {"Bucket": bucket_name, "Prefix": prefix}
        if continuation_token:
            list_params["ContinuationToken"] = continuation_token
        
        response = s3_client.list_objects_v2(**list_params)

        # Check if any files exist
        if "Contents" in response:
            json_files.extend([obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".json")])

        # Check for more pages of results
        continuation_token = response.get("NextContinuationToken")
        if not continuation_token:
            break  # Exit if no more files

    return json_files


def list_folders_s3(bucket_name, prefix=""):
    """
    Lists all folder-like prefixes in an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        prefix (str, optional): Starting path for folders (default: "").

    Returns:
        List[str]: List of folder names (prefixes).
    """
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter="/")

    # Extract folder names from the response
    if "CommonPrefixes" in response:
        return [folder["Prefix"] for folder in response["CommonPrefixes"]]
    else:
        return []

