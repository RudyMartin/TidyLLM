"""
TidyLLM S3-Integrated Universal Flow Parser
===========================================

Extends UniversalFlowParser with S3 trigger support:
- S3 file drops trigger bracket workflows
- Bracket commands as filenames: [mvr_analysis].trigger
- Drop zone monitoring from YAML workflow definitions
- Lambda-compatible S3 event processing
- Maintains S3-first architecture principles

Works across: CLI, API, UI, Chat, AND S3 Events
"""

import json
import boto3
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
from botocore.exceptions import ClientError, NoCredentialsError

from .universal_flow_parser import UniversalFlowParser, FlowParserInterface, BracketCommand, FlowExecution, FlowExecutionStatus

logger = logging.getLogger("s3_flow_parser")

@dataclass 
class S3Event:
    """Represents an S3 event that could trigger a workflow."""
    bucket_name: str
    object_key: str
    event_name: str  # s3:ObjectCreated:Put, etc.
    event_time: str
    object_size: int = 0
    etag: str = ""
    
    @classmethod
    def from_lambda_record(cls, record: Dict[str, Any]) -> 'S3Event':
        """Create S3Event from Lambda event record."""
        s3_data = record['s3']
        return cls(
            bucket_name=s3_data['bucket']['name'],
            object_key=s3_data['object']['key'],
            event_name=record['eventName'],
            event_time=record['eventTime'],
            object_size=s3_data['object'].get('size', 0),
            etag=s3_data['object'].get('eTag', '')
        )

@dataclass
class S3TriggerRule:
    """Rule for matching S3 events to workflow triggers."""
    workflow_name: str
    trigger_patterns: List[str]  # File patterns that trigger this workflow
    drop_zones: List[str]        # Drop zones that trigger this workflow
    action: Optional[str] = None # Specific action to execute
    conditions: Dict[str, Any] = field(default_factory=dict)
    
class S3FlowParser(UniversalFlowParser):
    """
    S3-integrated flow parser that handles bracket triggers from S3 events.
    
    Supports:
    - Bracket command files: [mvr_analysis].trigger
    - Drop zone monitoring: files in mvr_tag/ → trigger mvr_analysis
    - S3 event processing: Lambda-compatible event handling
    - S3-first processing: maintain cloud-native architecture
    """
    
    def __init__(self, workflows_dir: str = "workflows", s3_client: Optional[boto3.client] = None):
        """Initialize S3 flow parser."""
        super().__init__(workflows_dir)
        
        # Initialize S3 client
        self.s3_client = s3_client
        if self.s3_client is None:
            try:
                # Use UnifiedSessionManager for S3 client
                from tidyllm.infrastructure.session.unified import UnifiedSessionManager
                session_mgr = UnifiedSessionManager()
                self.s3_client = session_mgr.get_s3_client()
                logger.info("Initialized S3 client via UnifiedSessionManager")
            except (NoCredentialsError, Exception) as e:
                logger.warning(f"Could not initialize S3 client via UnifiedSessionManager: {e}")
                # NO FALLBACK - UnifiedSessionManager is required
                logger.error(f"Could not initialize S3 client via UnifiedSessionManager: {e}")
                raise RuntimeError("S3FlowParser: UnifiedSessionManager is required for S3 access")
        
        # S3 trigger configuration
        self.trigger_rules: List[S3TriggerRule] = []
        self.trigger_bucket_config: Dict[str, Any] = {}
        
        # Load S3 trigger rules from workflows
        self._load_s3_trigger_rules()
    
    def _load_s3_trigger_rules(self):
        """Load S3 trigger rules from workflow YAML definitions."""
        self.trigger_rules = []
        
        for workflow_name, workflow_def in self.workflows.items():
            # Skip duplicate workflows (same workflow with different keys)
            if workflow_name != workflow_def.get('workflow_name', workflow_name):
                continue
            
            trigger_rule = S3TriggerRule(
                workflow_name=workflow_name,
                trigger_patterns=[],
                drop_zones=[]
            )
            
            # Extract drop zones from stages
            stages = workflow_def.get('stages', {})
            for stage_name, stage_def in stages.items():
                drop_zone = stage_def.get('drop_zone', '')
                if drop_zone:
                    trigger_rule.drop_zones.append(drop_zone.rstrip('/'))
            
            # Add bracket trigger pattern
            trigger_rule.trigger_patterns = [
                f"[{workflow_name}].trigger",
                f"[{workflow_name} *].trigger",  # With actions
                f"triggers/[{workflow_name}]*"   # In triggers folder
            ]
            
            # Extract S3 configuration if present
            s3_config = workflow_def.get('s3_triggers', {})
            if s3_config:
                trigger_rule.conditions = s3_config
            
            self.trigger_rules.append(trigger_rule)
            logger.info(f"Loaded S3 triggers for {workflow_name}: {len(trigger_rule.drop_zones)} drop zones")
    
    def match_s3_event_to_workflow(self, s3_event: S3Event) -> Optional[Tuple[str, str, List[str]]]:
        """
        Match S3 event to workflow trigger.
        
        Returns:
            Tuple of (workflow_name, action, parameters) or None if no match
        """
        object_key = s3_event.object_key
        
        # Check bracket trigger files first
        bracket_match = self._match_bracket_trigger_file(object_key)
        if bracket_match:
            return bracket_match
        
        # Check drop zone matches
        drop_zone_match = self._match_drop_zone_trigger(object_key)
        if drop_zone_match:
            return drop_zone_match
        
        return None
    
    def _match_bracket_trigger_file(self, object_key: str) -> Optional[Tuple[str, str, List[str]]]:
        """Match bracket trigger files like [mvr_analysis].trigger"""
        # Extract filename from S3 key
        filename = Path(object_key).name
        
        # Check if it's a bracket trigger file
        bracket_pattern = re.compile(r'\[([^\]]+)\]\.trigger$')
        match = bracket_pattern.match(filename)
        
        if match:
            bracket_content = match.group(1)
            try:
                command = BracketCommand.parse(f"[{bracket_content}]")
                
                # Check if workflow exists
                if command.workflow_name in self.workflows:
                    return (command.workflow_name, command.action or 'start', command.parameters)
            except Exception as e:
                logger.error(f"Failed to parse bracket trigger {filename}: {e}")
        
        return None
    
    def _match_drop_zone_trigger(self, object_key: str) -> Optional[Tuple[str, str, List[str]]]:
        """Match drop zone patterns from workflow definitions."""
        object_path = Path(object_key)
        
        for rule in self.trigger_rules:
            for drop_zone in rule.drop_zones:
                # Check if object is in this drop zone
                if str(object_path.parent) == drop_zone or str(object_path).startswith(f"{drop_zone}/"):
                    # Determine action based on drop zone
                    action = self._get_action_for_drop_zone(rule.workflow_name, drop_zone)
                    return (rule.workflow_name, action, [])
        
        return None
    
    def _get_action_for_drop_zone(self, workflow_name: str, drop_zone: str) -> str:
        """Get the appropriate action for a drop zone."""
        workflow_def = self.workflows[workflow_name]
        stages = workflow_def.get('stages', {})
        
        # Find stage that uses this drop zone
        for stage_name, stage_def in stages.items():
            if stage_def.get('drop_zone', '').rstrip('/') == drop_zone:
                return stage_name
        
        return 'start'  # Default action
    
    async def process_s3_event(self, s3_event: S3Event, context: Optional[Dict[str, Any]] = None) -> Optional[FlowExecution]:
        """
        Process S3 event and trigger appropriate workflow.
        
        Args:
            s3_event: S3 event object
            context: Optional context (Lambda context, etc.)
            
        Returns:
            FlowExecution if workflow was triggered, None otherwise
        """
        try:
            # Match event to workflow
            workflow_match = self.match_s3_event_to_workflow(s3_event)
            
            if not workflow_match:
                logger.info(f"No workflow match for S3 object: {s3_event.object_key}")
                return None
            
            workflow_name, action, parameters = workflow_match
            
            # Create bracket command
            if action == 'start':
                bracket_command = f"[{workflow_name}]"
            else:
                bracket_command = f"[{workflow_name} {action}]"
            
            # Add S3 context
            s3_context = {
                "trigger_type": "s3_event",
                "bucket_name": s3_event.bucket_name,
                "object_key": s3_event.object_key,
                "event_name": s3_event.event_name,
                "event_time": s3_event.event_time,
                "object_size": s3_event.object_size
            }
            
            if context:
                s3_context.update(context)
            
            # Execute workflow
            logger.info(f"Triggering workflow from S3: {bracket_command}")
            execution = await self.execute_bracket_command(bracket_command, s3_context)
            
            # Store S3 trigger info in execution
            execution.metadata.update(s3_context)
            
            return execution
            
        except Exception as e:
            logger.error(f"Failed to process S3 event: {e}")
            return None
    
    async def process_lambda_event(self, lambda_event: Dict[str, Any], lambda_context: Any = None) -> List[FlowExecution]:
        """
        Process Lambda event containing S3 records.
        
        Args:
            lambda_event: Lambda event with S3 records
            lambda_context: Lambda context object
            
        Returns:
            List of FlowExecution objects for triggered workflows
        """
        executions = []
        
        # Extract S3 records from Lambda event
        records = lambda_event.get('Records', [])
        
        for record in records:
            try:
                # Parse S3 event from Lambda record
                s3_event = S3Event.from_lambda_record(record)
                
                # Add Lambda context
                context = {
                    "lambda_request_id": getattr(lambda_context, 'aws_request_id', ''),
                    "lambda_function_name": getattr(lambda_context, 'function_name', ''),
                    "lambda_remaining_time": getattr(lambda_context, 'get_remaining_time_in_millis', lambda: 0)()
                }
                
                # Process S3 event
                execution = await self.process_s3_event(s3_event, context)
                
                if execution:
                    executions.append(execution)
                    
            except Exception as e:
                logger.error(f"Failed to process Lambda record: {e}")
        
        return executions
    
    def create_trigger_file(self, bucket: str, workflow_name: str, action: str = None, **metadata) -> str:
        """
        Create a trigger file in S3 to initiate workflow.
        
        Args:
            bucket: S3 bucket name
            workflow_name: Name of workflow to trigger
            action: Optional specific action
            **metadata: Additional metadata to store in trigger file
            
        Returns:
            S3 key of created trigger file
        """
        if not self.s3_client:
            raise RuntimeError("S3 client not available")
        
        # Create trigger filename
        if action:
            trigger_key = f"triggers/[{workflow_name} {action}].trigger"
        else:
            trigger_key = f"triggers/[{workflow_name}].trigger"
        
        # Create trigger metadata
        trigger_metadata = {
            "workflow_name": workflow_name,
            "action": action or "start",
            "created_at": datetime.now().isoformat(),
            "created_by": "s3_flow_parser"
        }
        trigger_metadata.update(metadata)
        
        # Upload trigger file
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=trigger_key,
                Body=json.dumps(trigger_metadata, indent=2),
                ContentType='application/json',
                Metadata={
                    'workflow-name': workflow_name,
                    'action': action or 'start',
                    'created-by': 'tidyllm-flow-parser'
                }
            )
            
            logger.info(f"Created trigger file: s3://{bucket}/{trigger_key}")
            return trigger_key
            
        except Exception as e:
            logger.error(f"Failed to create trigger file: {e}")
            raise
    
    def setup_s3_bucket_triggers(self, bucket: str, lambda_arn: str) -> Dict[str, Any]:
        """
        Setup S3 bucket notification configuration for workflow triggers.
        
        Args:
            bucket: S3 bucket name
            lambda_arn: Lambda function ARN to trigger
            
        Returns:
            Configuration summary
        """
        if not self.s3_client:
            raise RuntimeError("S3 client not available")
        
        try:
            # Create notification configuration
            notification_config = {
                'LambdaConfigurations': [
                    {
                        'Id': 'tidyllm-workflow-triggers',
                        'LambdaFunctionArn': lambda_arn,
                        'Events': ['s3:ObjectCreated:*'],
                        'Filter': {
                            'Key': {
                                'FilterRules': [
                                    {'Name': 'prefix', 'Value': 'triggers/'},
                                    {'Name': 'suffix', 'Value': '.trigger'}
                                ]
                            }
                        }
                    }
                ]
            }
            
            # Add drop zone triggers
            for rule in self.trigger_rules:
                for drop_zone in rule.drop_zones:
                    notification_config['LambdaConfigurations'].append({
                        'Id': f'tidyllm-dropzone-{rule.workflow_name}-{drop_zone.replace("/", "-")}',
                        'LambdaFunctionArn': lambda_arn,
                        'Events': ['s3:ObjectCreated:*'],
                        'Filter': {
                            'Key': {
                                'FilterRules': [
                                    {'Name': 'prefix', 'Value': f'{drop_zone}/'}
                                ]
                            }
                        }
                    })
            
            # Apply configuration
            self.s3_client.put_bucket_notification_configuration(
                Bucket=bucket,
                NotificationConfiguration=notification_config
            )
            
            logger.info(f"Configured S3 bucket {bucket} for workflow triggers")
            
            return {
                "bucket": bucket,
                "lambda_arn": lambda_arn,
                "trigger_rules": len(self.trigger_rules),
                "notification_configs": len(notification_config['LambdaConfigurations'])
            }
            
        except Exception as e:
            logger.error(f"Failed to setup S3 bucket triggers: {e}")
            raise

class S3FlowParserInterface(FlowParserInterface):
    """
    S3-integrated interface adapter for different access points.
    Extends FlowParserInterface with S3 trigger capabilities.
    """
    
    def __init__(self, workflows_dir: str = "workflows", s3_client: Optional[boto3.client] = None):
        # Initialize with S3FlowParser instead of UniversalFlowParser
        self.parser = S3FlowParser(workflows_dir, s3_client)
    
    # S3 Interface Methods
    async def s3_process_event(self, s3_event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process S3 event from API/Lambda call."""
        try:
            s3_event = S3Event(
                bucket_name=s3_event_data['bucket_name'],
                object_key=s3_event_data['object_key'],
                event_name=s3_event_data.get('event_name', 's3:ObjectCreated:Put'),
                event_time=s3_event_data.get('event_time', datetime.now().isoformat()),
                object_size=s3_event_data.get('object_size', 0)
            )
            
            execution = await self.parser.process_s3_event(s3_event)
            
            if execution:
                return {
                    "triggered": True,
                    "execution_id": execution.execution_id,
                    "workflow_name": execution.workflow_name,
                    "status": execution.status.value,
                    "trigger_type": "s3_event"
                }
            else:
                return {
                    "triggered": False,
                    "message": "No workflow matched S3 event"
                }
                
        except Exception as e:
            return {
                "triggered": False,
                "error": str(e)
            }
    
    async def s3_process_lambda_event(self, lambda_event: Dict[str, Any], lambda_context: Any = None) -> Dict[str, Any]:
        """Process Lambda event containing S3 records."""
        try:
            executions = await self.parser.process_lambda_event(lambda_event, lambda_context)
            
            return {
                "processed": True,
                "executions_triggered": len(executions),
                "executions": [
                    {
                        "execution_id": ex.execution_id,
                        "workflow_name": ex.workflow_name,
                        "status": ex.status.value
                    }
                    for ex in executions
                ]
            }
            
        except Exception as e:
            return {
                "processed": False,
                "error": str(e)
            }
    
    def s3_create_trigger(self, bucket: str, workflow_name: str, action: str = None, **metadata) -> Dict[str, Any]:
        """Create S3 trigger file."""
        try:
            trigger_key = self.parser.create_trigger_file(bucket, workflow_name, action, **metadata)
            
            return {
                "created": True,
                "bucket": bucket,
                "trigger_key": trigger_key,
                "workflow_name": workflow_name,
                "action": action or "start"
            }
            
        except Exception as e:
            return {
                "created": False,
                "error": str(e)
            }
    
    def s3_setup_bucket(self, bucket: str, lambda_arn: str) -> Dict[str, Any]:
        """Setup S3 bucket for workflow triggers.""" 
        try:
            result = self.parser.setup_s3_bucket_triggers(bucket, lambda_arn)
            result["configured"] = True
            return result
            
        except Exception as e:
            return {
                "configured": False,
                "error": str(e)
            }
    
    def s3_list_trigger_rules(self) -> List[Dict[str, Any]]:
        """List all S3 trigger rules."""
        return [
            {
                "workflow_name": rule.workflow_name,
                "trigger_patterns": rule.trigger_patterns,
                "drop_zones": rule.drop_zones,
                "action": rule.action
            }
            for rule in self.parser.trigger_rules
        ]

# Global S3 parser instance
_global_s3_parser = None

def get_s3_flow_parser(workflows_dir: str = "workflows", s3_client: Optional[boto3.client] = None) -> S3FlowParserInterface:
    """Get global S3 flow parser interface."""
    global _global_s3_parser
    if _global_s3_parser is None:
        _global_s3_parser = S3FlowParserInterface(workflows_dir, s3_client)
    return _global_s3_parser

# Lambda handler function
async def lambda_handler(event, context):
    """
    AWS Lambda handler for S3-triggered workflows.
    
    Usage:
        Deploy this function and configure S3 bucket notifications
        to trigger this Lambda when files are uploaded.
    """
    parser = get_s3_flow_parser()
    
    try:
        result = await parser.s3_process_lambda_event(event, context)
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Lambda handler error: {e}")
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'processed': False,
                'error': str(e)
            })
        }

# Convenience functions
def create_s3_trigger(bucket: str, workflow_name: str, action: str = None, **metadata) -> Dict[str, Any]:
    """Convenience function to create S3 trigger."""
    parser = get_s3_flow_parser()
    return parser.s3_create_trigger(bucket, workflow_name, action, **metadata)

def setup_s3_bucket_triggers(bucket: str, lambda_arn: str) -> Dict[str, Any]:
    """Convenience function to setup S3 bucket triggers."""
    parser = get_s3_flow_parser()
    return parser.s3_setup_bucket(bucket, lambda_arn)

if __name__ == "__main__":
    # Test the S3 parser
    import asyncio
    
    async def test_s3_parser():
        parser = S3FlowParserInterface("workflows")
        
        print("S3 Trigger Rules:")
        rules = parser.s3_list_trigger_rules()
        for rule in rules:
            print(f"  - {rule['workflow_name']}: {rule['drop_zones']}")
        
        print("\nTesting S3 trigger file matching:")
        test_files = [
            "[mvr_analysis].trigger",
            "triggers/[robots3 create_domain_rag].trigger", 
            "mvr_tag/document.pdf",
            "02_processed/data.csv"
        ]
        
        for test_file in test_files:
            s3_event = S3Event(
                bucket_name="test-bucket",
                object_key=test_file,
                event_name="s3:ObjectCreated:Put",
                event_time=datetime.now().isoformat()
            )
            
            match = parser.parser.match_s3_event_to_workflow(s3_event)
            if match:
                workflow, action, params = match
                print(f"  {test_file} → [{workflow} {action}]")
            else:
                print(f"  {test_file} → No match")
        
        # Test creating a trigger file (mock)
        print(f"\nTesting trigger creation:")
        print("create_s3_trigger('my-bucket', 'mvr_analysis', 'start')")
        print("→ Would create: triggers/[mvr_analysis start].trigger")
    
    asyncio.run(test_s3_parser())