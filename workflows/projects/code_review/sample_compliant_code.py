#!/usr/bin/env python3
"""
Sample compliant code for code review testing
This code follows all architectural requirements
"""

import polars as pl
import boto3
from pathlib import Path
from typing import Dict, Any

class DocumentProcessingAdapter:
    """
    Document processing adapter following hexagonal architecture
    Uses S3-first approach with proper app folder cleanup
    """

    def __init__(self, s3_bucket: str):
        self.s3_client = boto3.client('s3')
        self.bucket = s3_bucket

    def process_document(self, s3_key: str) -> Dict[str, Any]:
        """
        Process document directly from S3 without local storage
        Returns results that can be stored back to S3
        """
        try:
            # Read directly from S3
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            content = response['Body'].read()

            # Use polars for data processing (not pandas)
            if s3_key.endswith('.csv'):
                df = pl.read_csv(content)
                processed_data = df.select([
                    pl.col("*").fill_null(""),
                    pl.col("amount").cast(pl.Float64).alias("processed_amount")
                ])
            else:
                processed_data = {"raw_content": content.decode('utf-8')}

            # Return structured result for S3 storage
            return {
                'success': True,
                'source_key': s3_key,
                'processed_data': processed_data.to_dict() if hasattr(processed_data, 'to_dict') else processed_data,
                'processing_timestamp': pl.datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'source_key': s3_key
            }

    def cleanup_temp_resources(self):
        """
        Cleanup any temporary resources (following app cleanup requirements)
        """
        # No local files to cleanup since we use S3-first approach
        pass