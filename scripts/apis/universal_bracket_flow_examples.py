"""
Universal Bracket Integration Examples
======================================

Shows how the same bracket commands work across ALL interfaces:
- CLI: tidyllm "[mvr_analysis]" 
- API: POST /flow {"command": "[mvr_analysis]"}
- UI: Text input box with bracket detection
- Chat: Natural bracket parsing in conversation
- S3: Drop [mvr_analysis].trigger file

Single YAML definition → Universal bracket access
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List

# Import the S3-integrated flow parser
from tidyllm.s3_flow_parser import get_s3_flow_parser, S3Event

class UniversalBracketExamples:
    """Examples showing bracket commands across all interfaces."""
    
    def __init__(self):
        self.parser = get_s3_flow_parser("workflows")
    
    # =========================================================================
    # CLI INTERFACE EXAMPLES
    # =========================================================================
    
    async def example_cli_interface(self):
        """CLI interface examples."""
        print("=" * 60)
        print("CLI INTERFACE - Bracket Commands")
        print("=" * 60)
        
        cli_commands = [
            "[mvr_analysis]",                    # Start full workflow
            "[mvr_analysis mvr_tag]",           # Run specific stage
            "[robots3 create_domain_rag]",      # Domain RAG creation
            "[compliance_check start]"         # Explicit start action
        ]
        
        for cmd in cli_commands:
            print(f"\n$ tidyllm '{cmd}'")
            
            # Execute via CLI interface
            result = await self.parser.cli_execute(cmd, user_id="cli_user")
            
            print(f"✅ Execution ID: {result['execution_id']}")
            print(f"📋 Status: {result['status']}")
            print(f"⚙️  Workflow: {result['workflow_name']}")
            
            if result.get('errors'):
                print(f"❌ Errors: {result['errors']}")
    
    # =========================================================================
    # API INTERFACE EXAMPLES  
    # =========================================================================
    
    async def example_api_interface(self):
        """API interface examples."""
        print("=" * 60)
        print("API INTERFACE - REST Endpoints")
        print("=" * 60)
        
        api_requests = [
            {
                "endpoint": "POST /flow/execute",
                "payload": {
                    "command": "[mvr_analysis]",
                    "context": {"user_id": "api_user", "session_id": "sess123"}
                }
            },
            {
                "endpoint": "POST /flow/execute",
                "payload": {
                    "command": "[robots3 embed]",
                    "context": {"source": "api", "priority": "high"}
                }
            }
        ]
        
        for request in api_requests:
            print(f"\n{request['endpoint']}")
            print(f"Payload: {json.dumps(request['payload'], indent=2)}")
            
            # Execute via API interface
            result = await self.parser.api_execute(request['payload'])
            
            print(f"Response:")
            print(f"  execution_id: {result['execution_id']}")
            print(f"  status: {result['status']}")
            print(f"  started_at: {result['started_at']}")
    
    # =========================================================================
    # UI INTERFACE EXAMPLES
    # =========================================================================
    
    def example_ui_interface(self):
        """UI interface examples."""
        print("=" * 60)
        print("UI INTERFACE - Streamlit/Web Interface")
        print("=" * 60)
        
        ui_inputs = [
            "Please run [mvr_analysis] on the uploaded documents",
            "I need to [robots3 create_domain_rag] for the technical docs",
            "Can you [compliance_check] and [audit_trail generate] the results?",
            "Start [mvr_analysis mvr_qa] stage specifically"
        ]
        
        for user_input in ui_inputs:
            print(f"\nUser Input: '{user_input}'")
            
            # Detect brackets in UI input
            detected = self.parser.ui_detect_and_execute(user_input)
            
            print(f"Detected Brackets: {len(detected)} commands")
            for detection in detected:
                if detection['valid']:
                    print(f"  ✅ {detection['bracket']} → {detection['workflow_name']}")
                    if detection.get('action'):
                        print(f"     Action: {detection['action']}")
                    if detection.get('parameters'):
                        print(f"     Parameters: {detection['parameters']}")
                else:
                    print(f"  ❌ {detection['bracket']} → {detection.get('error', 'Invalid')}")
    
    # =========================================================================
    # CHAT INTERFACE EXAMPLES
    # =========================================================================
    
    async def example_chat_interface(self):
        """Chat interface examples."""
        print("=" * 60)
        print("CHAT INTERFACE - Conversational Bracket Commands")
        print("=" * 60)
        
        chat_messages = [
            "Hi! Can you help me analyze these MVR documents? [mvr_analysis]",
            "I uploaded some robotics papers, please [robots3 ingest] them",
            "Let's [compliance_check] the results and then [generate_report]",
            "Just checking - no brackets in this message"
        ]
        
        for message in chat_messages:
            print(f"\nUser: {message}")
            
            # Process chat message
            result = await self.parser.chat_process_message(
                message, 
                context={"user_id": "chat_user", "channel": "general"}
            )
            
            if result['has_brackets']:
                print(f"🤖 Assistant: Found {len(result['brackets_found'])} workflow commands!")
                for execution in result['executions']:
                    print(f"     ⚡ Executing {execution['bracket']}")
                    print(f"     📋 Status: {execution['status']}")
            else:
                print(f"🤖 Assistant: No workflow commands detected.")
    
    # =========================================================================
    # S3 INTERFACE EXAMPLES
    # =========================================================================
    
    async def example_s3_interface(self):
        """S3 interface examples."""
        print("=" * 60)
        print("S3 INTERFACE - File-Based Triggers")
        print("=" * 60)
        
        # S3 trigger file examples
        s3_examples = [
            {
                "type": "Bracket Trigger File",
                "bucket": "workflows-bucket",
                "key": "triggers/[mvr_analysis].trigger",
                "description": "Drop this file to trigger MVR analysis"
            },
            {
                "type": "Drop Zone File",
                "bucket": "processing-bucket", 
                "key": "mvr_tag/document.pdf",
                "description": "Document in MVR tag drop zone"
            },
            {
                "type": "Specific Action Trigger",
                "bucket": "workflows-bucket",
                "key": "triggers/[robots3 embed].trigger", 
                "description": "Trigger specific embedding stage"
            }
        ]
        
        for example in s3_examples:
            print(f"\n{example['type']}:")
            print(f"  S3 Location: s3://{example['bucket']}/{example['key']}")
            print(f"  Description: {example['description']}")
            
            # Simulate S3 event
            s3_event = S3Event(
                bucket_name=example['bucket'],
                object_key=example['key'],
                event_name="s3:ObjectCreated:Put",
                event_time=datetime.now().isoformat(),
                object_size=1024
            )
            
            # Process S3 event
            execution = await self.parser.parser.process_s3_event(s3_event)
            
            if execution:
                print(f"  ✅ Triggered: {execution.workflow_name}")
                print(f"  🆔 Execution ID: {execution.execution_id}")
                print(f"  📊 Status: {execution.status.value}")
            else:
                print(f"  ❌ No workflow triggered")
    
    # =========================================================================
    # INTEGRATION EXAMPLES
    # =========================================================================
    
    async def example_cross_interface_workflow(self):
        """Example showing same workflow across different interfaces."""
        print("=" * 60) 
        print("CROSS-INTERFACE WORKFLOW - Same Bracket, Different Access")
        print("=" * 60)
        
        workflow_name = "mvr_analysis"
        bracket_command = f"[{workflow_name}]"
        
        print(f"Workflow: {workflow_name}")
        print(f"Bracket Command: {bracket_command}")
        print()
        
        interfaces = [
            {
                "name": "CLI",
                "command": f"tidyllm '{bracket_command}'",
                "method": lambda: self.parser.cli_execute(bracket_command)
            },
            {
                "name": "API", 
                "command": f"POST /flow/execute {{'command': '{bracket_command}'}}",
                "method": lambda: self.parser.api_execute({"command": bracket_command})
            },
            {
                "name": "Chat",
                "command": f"User message: 'Please run {bracket_command}'",
                "method": lambda: self.parser.chat_process_message(f"Please run {bracket_command}")
            }
        ]
        
        results = []
        for interface in interfaces:
            print(f"{interface['name']} Interface:")
            print(f"  Command: {interface['command']}")
            
            try:
                result = await interface['method']()
                execution_id = result.get('execution_id') or result.get('executions', [{}])[0].get('execution_id', 'N/A')
                print(f"  ✅ Execution ID: {execution_id}")
                results.append({"interface": interface['name'], "success": True, "execution_id": execution_id})
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({"interface": interface['name'], "success": False, "error": str(e)})
            print()
        
        # Summary
        print("Cross-Interface Summary:")
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"  {status} {result['interface']}: {result.get('execution_id', result.get('error', 'Failed'))}")
    
    # =========================================================================
    # WORKFLOW DISCOVERY EXAMPLES
    # =========================================================================
    
    def example_workflow_discovery(self):
        """Examples showing workflow discovery and help."""
        print("=" * 60)
        print("WORKFLOW DISCOVERY - Available Commands")  
        print("=" * 60)
        
        # List available workflows
        workflows = self.parser.list_workflows()
        
        print(f"Available Workflows: {len(workflows)}")
        for wf in workflows:
            print(f"  📋 [{wf['workflow_name']}]")
            print(f"     Description: {wf['description']}")
            print(f"     Stages: {wf['stages']}")
            print(f"     Version: {wf['version']}")
            print()
        
        # Show help for specific workflow
        if workflows:
            example_workflow = workflows[0]['workflow_name']
            print(f"Help for [{example_workflow}]:")
            help_info = self.parser.get_workflow_help(example_workflow)
            
            print(f"  Description: {help_info['description']}")
            print(f"  Available Actions: {help_info['available_actions']}")
            print(f"  Example Commands:")
            for cmd in help_info['example_commands']:
                print(f"    - {cmd}")
    
    # =========================================================================
    # S3 SETUP EXAMPLES
    # =========================================================================
    
    def example_s3_setup(self):
        """Examples showing S3 trigger setup.""" 
        print("=" * 60)
        print("S3 TRIGGER SETUP - Bucket Configuration")
        print("=" * 60)
        
        # List trigger rules
        rules = self.parser.s3_list_trigger_rules()
        
        print("S3 Trigger Rules:")
        for rule in rules:
            print(f"  📋 {rule['workflow_name']}:")
            print(f"     Trigger Patterns: {rule['trigger_patterns']}")
            print(f"     Drop Zones: {rule['drop_zones']}")
            print()
        
        # Example bucket setup (mock)
        bucket_name = "tidyllm-workflows"
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:tidyllm-flow-processor"
        
        print(f"Setting up S3 bucket '{bucket_name}' for workflow triggers:")
        print(f"Lambda ARN: {lambda_arn}")
        print()
        print("Configuration would create notifications for:")
        print("  - triggers/*.trigger files → Lambda function")
        for rule in rules:
            for drop_zone in rule['drop_zones']:
                print(f"  - {drop_zone}/* files → {rule['workflow_name']} workflow")


# =========================================================================
# MAIN DEMONSTRATION
# =========================================================================

async def run_all_examples():
    """Run all interface examples."""
    examples = UniversalBracketExamples()
    
    print("🔥 TIDYLLM UNIVERSAL BRACKET SYSTEM DEMONSTRATION 🔥")
    print("Same YAML workflow → Same bracket syntax → All interfaces")
    print()
    
    # Run interface examples
    await examples.example_cli_interface()
    await examples.example_api_interface()
    examples.example_ui_interface() 
    await examples.example_chat_interface()
    await examples.example_s3_interface()
    
    # Show integration and discovery
    await examples.example_cross_interface_workflow()
    examples.example_workflow_discovery()
    examples.example_s3_setup()
    
    print("=" * 60)
    print("🎉 UNIVERSAL BRACKET SYSTEM COMPLETE!")
    print("=" * 60)
    print("KEY BENEFITS:")
    print("✅ Single YAML definition → Universal access")
    print("✅ Same bracket syntax across CLI/API/UI/Chat/S3")
    print("✅ S3-first processing with cloud triggers")
    print("✅ TidyLLM stack compliant (tlm, tidyllm-sentence, polars)")
    print("✅ Easy workflow discovery and help")
    print("✅ Enterprise controls and audit trails")
    print()
    print("USAGE EXAMPLES:")
    print("  CLI: tidyllm '[mvr_analysis]'")
    print("  API: POST /flow {'command': '[mvr_analysis]'}")
    print("  UI:  Text input: 'Please run [mvr_analysis]'")
    print("  Chat: 'Hi! Can you [mvr_analysis] these docs?'")
    print("  S3:   Drop file: triggers/[mvr_analysis].trigger")

if __name__ == "__main__":
    asyncio.run(run_all_examples())