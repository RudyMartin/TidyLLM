#!/usr/bin/env python3
"""
S3 File Processor for MVR Review System
Processes files when they land in S3 bucket with Review ID matching and validation
"""

import boto3
import json
import re
import logging
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3FileProcessor:
    """Process files landing in S3 bucket with MVR Review validation"""
    
    def __init__(self, config_path: str = None):
        """Initialize the S3 file processor"""
        self.s3_client = boto3.client('s3')
        self.sqs_client = boto3.client('sqs')
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Review ID patterns from existing MCP code
        self.review_id_patterns = [
            r'Review ID[:\s]*([A-Z]{3}\d{5})',
            r'Review[:\s]*([A-Z]{3}\d{5})',
            r'([A-Z]{3}\d{5})',
            r'Review ID[:\s]*(\d{5})',
        ]
        
        # File classification patterns
        self.file_classification = self._load_file_classification()
        
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '../../dev_configs/file_classification.yaml')
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if YAML loading fails"""
        return {
            's3_buckets': {
                'landing': os.environ.get('S3_LANDING_BUCKET', 'mvr-landing-bucket'),
                'catalog': os.environ.get('S3_CATALOG_BUCKET', 'mvr-catalog-bucket'),
                'quarantine': os.environ.get('S3_QUARANTINE_BUCKET', 'mvr-quarantine-bucket')
            },
            'sqs_queue': os.environ.get('SQS_NOTIFICATION_QUEUE', 'mvr-file-notifications'),
            'validation': {
                'max_file_size_mb': 100,
                'allowed_extensions': ['.pdf', '.docx', '.txt', '.md', '.yaml', '.yml', '.json', '.csv', '.xlsx', '.py', '.r', '.sql', '.ipynb'],
                'required_fields': ['review_id', 'model_type', 'risk_tier'],
                'virus_scan': True,
                'content_validation': True
            },
            'processing': {
                'batch_size': 10,
                'timeout_seconds': 300,
                'retry_attempts': 3
            }
        }
    
    def _load_file_classification(self) -> Dict:
        """Load file classification rules"""
        return {
            'unclassified': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.yaml', '.yml', '.json', '.csv', '.xlsx', '.py', '.r', '.sql', '.ipynb', '.html', '.xml', '.log'],
                'category': "Unclassified (U)",
                'priority': "Low"
            },
            'whitepaper': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt'],
                'category': "Whitepaper (W)",
                'priority': "Medium"
            },
            'validation_scoping_template': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.yaml', '.yml', '.json'],
                'category': "Validation Scoping Template (VST)",
                'priority': "High"
            },
            'annual_model_review': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.xlsx', '.xls'],
                'category': "Annual Model Review (AMR)",
                'priority': "High"
            },
            'model_documentation_assessment': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.yaml', '.yml', '.json'],
                'category': "Model Documentation Assessment (MDA)",
                'priority': "High"
            },
            'model_development_document': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.yaml', '.yml', '.json', '.py', '.r', '.sql', '.ipynb'],
                'category': "Model Development Document (MDD)",
                'priority': "High"
            },
            'model_development_plan': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.yaml', '.yml', '.json', '.xlsx', '.xls'],
                'category': "Model Development Plan (MDP)",
                'priority': "High"
            },
            'sdc_validation_report': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.xlsx', '.xls'],
                'category': "SDC Validation Report (SVR)",
                'priority': "High"
            },
            'qa_template': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.yaml', '.yml', '.json', '.xlsx', '.xls'],
                'category': "QA Template (QAT)",
                'priority': "Medium"
            },
            'implementation_testing_document': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.yaml', '.yml', '.json', '.xlsx', '.xls'],
                'category': "Implementation Testing Document (ITD)",
                'priority': "High"
            },
            'model_implementation_plan': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.yaml', '.yml', '.json', '.xlsx', '.xls'],
                'category': "Model Implementation Plan (MIP)",
                'priority': "High"
            },
            'standards_procedures': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.yaml', '.yml', '.json', '.xlsx', '.xls'],
                'category': "Standards/Procedures (SP)",
                'priority': "Medium"
            },
            'model_performance_monitoring_plan': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.yaml', '.yml', '.json', '.xlsx', '.xls'],
                'category': "Model Performance Monitoring (MPM) Plan",
                'priority': "High"
            },
            'post_development_recalibrated_model_plan': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.yaml', '.yml', '.json', '.xlsx', '.xls'],
                'category': "Post-Development Recalibrated Model (PDRM) Plan",
                'priority': "High"
            },
            'modeling_area_prioritization_plan': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.yaml', '.yml', '.json', '.xlsx', '.xls'],
                'category': "Modeling Area Prioritization Plan (MAPP)",
                'priority': "High"
            }
        }
    
    def extract_review_id(self, content: str) -> Optional[str]:
        """Extract Review ID from file content using existing MCP patterns"""
        for pattern in self.review_id_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                review_id = match.group(1)
                if not review_id.startswith('REV'):
                    review_id = f"REV{review_id.zfill(5)}"
                return review_id
        return None
    
    def classify_file(self, filename: str, content: str) -> Dict:
        """Classify file based on extension and content"""
        file_ext = Path(filename).suffix.lower()
        
        # First try content-based classification
        content_classification = self._classify_by_content(content)
        if content_classification:
            return content_classification
        
        # Fall back to extension-based classification
        for doc_type, config in self.file_classification.items():
            if file_ext in config['extensions']:
                return {
                    'type': doc_type,
                    'category': config['category'],
                    'priority': config['priority'],
                    'classification_method': 'extension_based'
                }
        
        return {
            'type': 'unclassified',
            'category': 'Unclassified (U)',
            'priority': 'Low',
            'classification_method': 'unknown'
        }
    
    def _classify_by_content(self, content: str) -> Optional[Dict]:
        """Classify file based on content keywords"""
        content_lower = content[:1000].lower()  # Analyze first 1000 characters
        
        # Content-based classification keywords
        classification_keywords = {
            'whitepaper': ['abstract', 'introduction', 'methodology', 'conclusion', 'references', 'bibliography', 'research', 'study', 'analysis', 'paper'],
            'validation_scoping_template': ['validation scope', 'stakeholders', 'timeline', 'resources', 'validation framework', 'scoping document', 'validation plan'],
            'annual_model_review': ['annual review', 'model performance', 'risk assessment', 'review period', 'annual assessment', 'comprehensive review', 'yearly evaluation'],
            'model_documentation_assessment': ['documentation assessment', 'documentation quality', 'missing elements', 'improvement areas', 'documentation score', 'completeness assessment'],
            'model_development_document': ['model architecture', 'development methodology', 'data sources', 'implementation details', 'model specification', 'development process'],
            'model_development_plan': ['development plan', 'development timeline', 'resource planning', 'methodology framework', 'success criteria', 'development roadmap', 'strategic plan'],
            'sdc_validation_report': ['validation report', 'test results', 'defects found', 'validation phase', 'software development cycle', 'validation conclusion'],
            'qa_template': ['qa template', 'quality assurance', 'checklist', 'assessment criteria', 'quality metrics', 'qa framework', 'quality checklist'],
            'implementation_testing_document': ['implementation testing', 'test plan', 'test cases', 'deployment validation', 'testing documentation', 'test results'],
            'model_implementation_plan': ['implementation plan', 'implementation timeline', 'deployment strategy', 'resource requirements', 'risk mitigation', 'implementation roadmap'],
            'standards_procedures': ['standards', 'procedures', 'governance framework', 'organizational standards', 'procedure steps', 'compliance requirements'],
            'model_performance_monitoring_plan': ['performance monitoring', 'monitoring framework', 'performance metrics', 'alert thresholds', 'escalation procedures', 'production monitoring', 'model performance'],
            'post_development_recalibrated_model_plan': ['recalibration', 'model updates', 'post development', 'recalibrated model', 'update procedures', 'deployment strategy', 'model maintenance'],
            'modeling_area_prioritization_plan': ['modeling area', 'prioritization', 'resource allocation', 'strategic planning', 'area management', 'prioritization criteria', 'timeline framework', 'success metrics']
        }
        
        best_match = None
        highest_score = 0
        
        for doc_type, keywords in classification_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > highest_score and score >= 2:  # Require at least 2 keyword matches
                highest_score = score
                best_match = doc_type
        
        if best_match:
            config = self.file_classification[best_match]
            return {
                'type': best_match,
                'category': config['category'],
                'priority': config['priority'],
                'classification_method': 'content_analysis'
            }
        
        return None
    
    def validate_file(self, bucket: str, key: str, content: str) -> Dict:
        """Validate file before processing"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_size_mb': 0,
            'file_extension': Path(key).suffix.lower(),
            'review_id': None,
            'classification': None
        }
        
        try:
            # Get file size
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            file_size_bytes = response['ContentLength']
            file_size_mb = file_size_bytes / (1024 * 1024)
            validation_result['file_size_mb'] = file_size_mb
            
            # Check file size
            max_size = self.config['validation']['max_file_size_mb']
            if file_size_mb > max_size:
                validation_result['valid'] = False
                validation_result['errors'].append(f"File size {file_size_mb:.2f}MB exceeds maximum {max_size}MB")
            
            # Check file extension
            allowed_extensions = self.config['validation']['allowed_extensions']
            if validation_result['file_extension'] not in allowed_extensions:
                validation_result['valid'] = False
                validation_result['errors'].append(f"File extension {validation_result['file_extension']} not allowed")
            
            # Extract Review ID
            review_id = self.extract_review_id(content)
            validation_result['review_id'] = review_id
            
            if not review_id:
                validation_result['warnings'].append("No Review ID found in file content")
            
            # Classify file
            classification = self.classify_file(key, content)
            validation_result['classification'] = classification
            
            # Content validation (basic checks)
            if self.config['validation']['content_validation']:
                if len(content.strip()) == 0:
                    validation_result['valid'] = False
                    validation_result['errors'].append("File content is empty")
                
                # Check for suspicious content patterns
                suspicious_patterns = [
                    r'<script[^>]*>',
                    r'javascript:',
                    r'vbscript:',
                    r'data:text/html'
                ]
                
                for pattern in suspicious_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        validation_result['valid'] = False
                        validation_result['errors'].append(f"Suspicious content pattern detected: {pattern}")
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def process_s3_event(self, event: Dict) -> Dict:
        """Process S3 event when file lands in bucket"""
        logger.info(f"Processing S3 event: {json.dumps(event, default=str)}")
        
        results = {
            'processed_files': [],
            'errors': [],
            'review_id_groups': {},
            'summary': {
                'total_files': 0,
                'valid_files': 0,
                'invalid_files': 0,
                'moved_to_catalog': 0,
                'moved_to_quarantine': 0
            }
        }
        
        try:
            # Process each S3 record in the event
            for record in event.get('Records', []):
                if record.get('eventSource') == 'aws:s3':
                    s3_info = record['s3']
                    bucket = s3_info['bucket']['name']
                    key = s3_info['object']['key']
                    
                    results['summary']['total_files'] += 1
                    
                    # Skip if not in landing bucket
                    if bucket != self.config['s3_buckets']['landing']:
                        logger.info(f"Skipping file {key} - not in landing bucket")
                        continue
                    
                    # Download and process file
                    file_result = self._process_single_file(bucket, key)
                    results['processed_files'].append(file_result)
                    
                    if file_result['valid']:
                        results['summary']['valid_files'] += 1
                        
                        # Group by Review ID
                        review_id = file_result['review_id']
                        if review_id:
                            if review_id not in results['review_id_groups']:
                                results['review_id_groups'][review_id] = []
                            results['review_id_groups'][review_id].append({
                                'filename': key,
                                'size_mb': file_result['file_size_mb'],
                                'category': file_result['classification']['category'],
                                'priority': file_result['classification']['priority']
                            })
                        
                        # Move to catalog
                        if self._move_to_catalog(bucket, key, file_result):
                            results['summary']['moved_to_catalog'] += 1
                        else:
                            results['summary']['moved_to_quarantine'] += 1
                    else:
                        results['summary']['invalid_files'] += 1
                        results['summary']['moved_to_quarantine'] += 1
                        
                        # Move to quarantine
                        self._move_to_quarantine(bucket, key, file_result)
            
            # Send notification
            self._send_notification(results)
            
        except Exception as e:
            error_msg = f"Error processing S3 event: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        logger.info(f"Processing complete: {json.dumps(results['summary'], default=str)}")
        return results
    
    def _process_single_file(self, bucket: str, key: str) -> Dict:
        """Process a single file from S3"""
        try:
            # Download file content
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8', errors='ignore')
            
            # Validate file
            validation_result = self.validate_file(bucket, key, content)
            
            return {
                'bucket': bucket,
                'key': key,
                'valid': validation_result['valid'],
                'errors': validation_result['errors'],
                'warnings': validation_result['warnings'],
                'file_size_mb': validation_result['file_size_mb'],
                'file_extension': validation_result['file_extension'],
                'review_id': validation_result['review_id'],
                'classification': validation_result['classification'],
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing file {key}: {str(e)}")
            return {
                'bucket': bucket,
                'key': key,
                'valid': False,
                'errors': [f"Processing error: {str(e)}"],
                'warnings': [],
                'file_size_mb': 0,
                'file_extension': Path(key).suffix.lower(),
                'review_id': None,
                'classification': None,
                'processed_at': datetime.now().isoformat()
            }
    
    def _move_to_catalog(self, source_bucket: str, key: str, file_result: Dict) -> bool:
        """Move valid file to catalog bucket"""
        try:
            catalog_bucket = self.config['s3_buckets']['catalog']
            
            # Create new key with Review ID prefix if available
            review_id = file_result['review_id']
            if review_id:
                new_key = f"{review_id}/{key}"
            else:
                new_key = f"unclassified/{key}"
            
            # Copy to catalog bucket
            self.s3_client.copy_object(
                Bucket=catalog_bucket,
                CopySource={'Bucket': source_bucket, 'Key': key},
                Key=new_key,
                Metadata={
                    'review_id': review_id or 'unknown',
                    'category': file_result['classification']['category'],
                    'priority': file_result['classification']['priority'],
                    'processed_at': file_result['processed_at']
                }
            )
            
            # Delete from landing bucket
            self.s3_client.delete_object(Bucket=source_bucket, Key=key)
            
            logger.info(f"Moved {key} to catalog as {new_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving {key} to catalog: {str(e)}")
            return False
    
    def _move_to_quarantine(self, source_bucket: str, key: str, file_result: Dict) -> bool:
        """Move invalid file to quarantine bucket"""
        try:
            quarantine_bucket = self.config['s3_buckets']['quarantine']
            
            # Create quarantine key with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_key = f"quarantine/{timestamp}/{key}"
            
            # Copy to quarantine bucket
            self.s3_client.copy_object(
                Bucket=quarantine_bucket,
                CopySource={'Bucket': source_bucket, 'Key': key},
                Key=quarantine_key,
                Metadata={
                    'original_bucket': source_bucket,
                    'original_key': key,
                    'errors': json.dumps(file_result['errors']),
                    'quarantined_at': file_result['processed_at']
                }
            )
            
            # Delete from landing bucket
            self.s3_client.delete_object(Bucket=source_bucket, Key=key)
            
            logger.info(f"Moved {key} to quarantine as {quarantine_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving {key} to quarantine: {str(e)}")
            return False
    
    def _send_notification(self, results: Dict) -> None:
        """Send notification about processing results"""
        try:
            queue_url = self.config['sqs_queue']
            
            message = {
                'timestamp': datetime.now().isoformat(),
                'summary': results['summary'],
                'review_id_groups': results['review_id_groups'],
                'errors': results['errors']
            }
            
            self.sqs_client.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(message)
            )
            
            logger.info(f"Sent notification to {queue_url}")
            
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")


def lambda_handler(event, context):
    """AWS Lambda handler function"""
    try:
        processor = S3FileProcessor()
        results = processor.process_s3_event(event)
        
        return {
            'statusCode': 200,
            'body': json.dumps(results, default=str)
        }
        
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


if __name__ == "__main__":
    # Test the processor
    processor = S3FileProcessor()
    
    # Simulate S3 event
    test_event = {
        'Records': [
            {
                'eventSource': 'aws:s3',
                's3': {
                    'bucket': {'name': 'mvr-landing-bucket'},
                    'object': {'key': 'test_document.pdf'}
                }
            }
        ]
    }
    
    results = processor.process_s3_event(test_event)
    print(json.dumps(results, indent=2, default=str))


