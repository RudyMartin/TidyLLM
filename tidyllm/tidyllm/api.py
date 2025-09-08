"""
TidyLLM Basic API - Simple Functions for Common Tasks
====================================================

Provides simple, beginner-friendly functions that work out of the box.
No complex gateway setup required for basic usage.

Basic Usage:
    import tidyllm
    
    # Simple chat
    response = tidyllm.chat("Hello, how are you?")
    
    # Process document
    result = tidyllm.process_document("document.pdf")
    
    # Query with context
    answer = tidyllm.query("What is machine learning?")
"""

import os
import logging
import yaml
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class TidyLLMSimpleAPI:
    """Simple API wrapper for TidyLLM basic functionality."""
    
    def __init__(self):
        self._initialized = False
        self._default_model = "anthropic/claude-3-sonnet-20240229"
        self._api_key = None
        self._setup_credentials()
    
    def _setup_credentials(self):
        """Setup API credentials from settings.yaml and environment."""
        self.config = self._load_config()
        
        # Setup AWS credentials from config
        if self.config.get('aws', {}).get('access_key_id'):
            aws_config = self.config['aws']
            os.environ['AWS_ACCESS_KEY_ID'] = aws_config['access_key_id']
            os.environ['AWS_SECRET_ACCESS_KEY'] = aws_config['secret_access_key'] 
            os.environ['AWS_DEFAULT_REGION'] = aws_config['default_region']
            
        # Setup API keys from config
        api_keys = self.config.get('api_keys', {})
        if api_keys.get('anthropic'):
            os.environ['ANTHROPIC_API_KEY'] = api_keys['anthropic']
        if api_keys.get('openai'):
            os.environ['OPENAI_API_KEY'] = api_keys['openai']
        
        # Try to get API key from various sources
        self._api_key = (
            os.getenv('ANTHROPIC_API_KEY') or 
            os.getenv('OPENAI_API_KEY') or
            os.getenv('AWS_ACCESS_KEY_ID')  # For Bedrock
        )
        
        if self._api_key:
            self._initialized = True
            logger.info("TidyLLM API credentials found")
        else:
            logger.warning("No API credentials found. Functions will run in simulation mode.")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from settings.yaml."""
        # Look for settings.yaml in multiple locations
        config_paths = [
            Path.cwd() / "settings.yaml",
            Path(__file__).parent.parent.parent / "settings.yaml",  # Root of repo
            Path.home() / ".tidyllm" / "settings.yaml"
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f) or {}
                    logger.info(f"Loaded config from: {config_path}")
                    return config
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")
        
        logger.info("No config file found, using defaults")
        return {}
    
    def chat(self, message: str, model: Optional[str] = None, **kwargs) -> str:
        """
        Simple chat function - send a message and get a response.
        
        Args:
            message: The message to send
            model: Optional model override 
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Response string from the AI model
            
        Example:
            >>> import tidyllm
            >>> response = tidyllm.chat("Hello, how are you?")
            >>> print(response)
        """
        if not self._initialized:
            return self._simulate_chat(message)
        
        try:
            # Try to use actual AI service
            return self._real_chat(message, model, **kwargs)
        except Exception as e:
            logger.warning(f"Real chat failed: {e}, falling back to simulation")
            return self._simulate_chat(message)
    
    def _real_chat(self, message: str, model: Optional[str], **kwargs) -> str:
        """Attempt real chat with AI service - Routes through gateway chain for audit compliance."""
        
        # AUDIT COMPLIANCE: All AI requests must go through gateway chain
        # User sees simple API, but everything is logged and controlled
        try:
            return self._route_through_gateways(message, model, **kwargs)
        except Exception as e:
            logger.error(f"Gateway chain failed: {e}")
            # If gateways fail, fall back to simulation (audit log this failure)
            logger.warning("AUDIT WARNING: Gateway chain failed, using simulation mode")
            return self._simulate_chat(message)
    
    def _route_through_gateways(self, message: str, model: Optional[str], **kwargs) -> str:
        """Route request through the audit compliance gateway chain."""
        
        try:
            # Import gateway system using absolute imports
            from tidyllm.gateways import init_gateways, AIRequest, WorkflowRequest  
            from tidyllm.gateways.corporate_llm_gateway import LLMRequest
            
            # Initialize gateway registry (cached globally for performance)
            if not hasattr(self, '_gateway_registry'):
                self._gateway_registry = init_gateways({
                    "corporate_llm": {
                        "budget_limit_daily_usd": self.config.get('cost_optimization', {}).get('limits', {}).get('daily_limit', 10.0),
                        "compliance_mode": True,
                        "audit_enabled": True
                    },
                    "ai_processing": {
                        "backend": "bedrock",  # Primary backend
                        "fallback_enabled": True,
                        "cache_enabled": True
                    },
                    "workflow_optimizer": {
                        "optimization_level": "basic",  # For simple chat, don't over-optimize
                        "audit_mode": True
                    }
                })
            
            # Step 1: Route through Corporate LLM Gateway (audit + compliance)
            corporate_gateway = self._gateway_registry.get("corporate_llm")
            if corporate_gateway:
                llm_request = LLMRequest(
                    prompt=message,
                    model=model,
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', 1000),
                    audit_reason="basic_api_chat",  # Required for compliance
                    user_id=kwargs.get('user_id', 'api_user'),
                    session_id=kwargs.get('session_id', f"api_session_{int(time.time())}")
                )
                
                # This automatically routes through the full chain:
                # Corporate LLM -> AI Processing -> Workflow Optimizer (if needed)
                response = corporate_gateway.process_llm_request(llm_request)
                
                if response.success:
                    return response.content
                else:
                    logger.error(f"Corporate gateway failed: {response.error}")
                    raise Exception(f"Gateway processing failed: {response.error}")
            
            else:
                raise Exception("Corporate LLM Gateway not available")
                
        except ImportError as e:
            logger.error(f"Gateway system not available: {e}")
            # Fall back to direct implementation but still try to log for audit
            return self._direct_fallback_with_audit_log(message, model, **kwargs)
        except Exception as e:
            logger.error(f"Gateway routing failed: {e}")
            raise
    
    def _direct_fallback_with_audit_log(self, message: str, model: Optional[str], **kwargs) -> str:
        """Direct fallback when gateways unavailable, but still log for audit."""
        
        # Log the direct access for audit purposes
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'direct_fallback',
            'message': message[:100] + '...' if len(message) > 100 else message,
            'model': model,
            'reason': 'gateway_system_unavailable',
            'user_id': kwargs.get('user_id', 'api_user')
        }
        
        # Try to log to audit file
        try:
            audit_file = Path("tidyllm_audit.log")
            with open(audit_file, 'a') as f:
                f.write(f"{json.dumps(audit_entry)}\n")
        except Exception as e:
            logger.warning(f"Could not write audit log: {e}")
        
        # Check settings for enabled services
        api_keys = self.config.get('api_keys', {})
        
        # Try AWS Bedrock (primary integration) - but log this direct access
        if os.getenv('AWS_ACCESS_KEY_ID'):
            try:
                logger.warning("AUDIT: Using direct AWS Bedrock access (gateway bypass)")
                return self._bedrock_chat(message, model, **kwargs)
            except Exception as e:
                logger.error(f"AWS Bedrock error: {e}")
        
        # Try Anthropic (if enabled) - but log this direct access  
        if api_keys.get('anthropic_enabled', True) and os.getenv('ANTHROPIC_API_KEY'):
            try:
                logger.warning("AUDIT: Using direct Anthropic access (gateway bypass)")
                import anthropic
                client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                
                response = client.messages.create(
                    model=model or "claude-3-sonnet-20240229",
                    max_tokens=kwargs.get('max_tokens', 1000),
                    temperature=kwargs.get('temperature', 0.7),
                    messages=[{"role": "user", "content": message}]
                )
                return response.content[0].text
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
        
        # Try OpenAI (if enabled) - but log this direct access
        if api_keys.get('openai_enabled', True) and os.getenv('OPENAI_API_KEY'):
            try:
                logger.warning("AUDIT: Using direct OpenAI access (gateway bypass)")
                import openai
                client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                
                response = client.chat.completions.create(
                    model=model or "gpt-3.5-turbo",
                    messages=[{"role": "user", "content": message}],
                    max_tokens=kwargs.get('max_tokens', 1000),
                    temperature=kwargs.get('temperature', 0.7)
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
        
        # If all else fails, simulate but log this too
        logger.warning("AUDIT: All AI services failed, using simulation mode")
        return self._simulate_chat(message)
    
    def _bedrock_chat(self, message: str, model: Optional[str], **kwargs) -> str:
        """Chat using AWS Bedrock - ROUTED THROUGH GATEWAY FOR AUDIT COMPLIANCE."""
        try:
            # AUDIT COMPLIANCE: Route through gateway system instead of direct boto3
            from tidyllm.gateways.gateway_registry import get_global_registry
            
            registry = get_global_registry()
            ai_gateway = registry.get('ai_processing')
            
            if ai_gateway:
                logger.info("AUDIT: Routing Bedrock request through AI Processing Gateway")
                from tidyllm.gateways.ai_processing_gateway import AIRequest
                
                request = AIRequest(
                    prompt=message,
                    model=model or "claude-3-sonnet",
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', 1000),
                    metadata={'source': 'api_bedrock_fallback', 'user_id': kwargs.get('user_id', 'api_user')}
                )
                
                response = ai_gateway.process_ai_request(request)
                if response.status.value == 'SUCCESS':
                    return response.data
                else:
                    raise Exception(f"Gateway processing failed: {response.status.value}")
            
            # Fallback to direct access only if gateway unavailable (still log for audit)
            logger.warning("AUDIT: Gateway unavailable - using direct Bedrock access")
            import boto3
            import json
            
            # Create Bedrock client  
            bedrock = boto3.client(
                'bedrock-runtime',
                region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            )
            
            # Use Claude model on Bedrock by default
            bedrock_model = model or "anthropic.claude-3-sonnet-20240229-v1:0"
            
            # Prepare request for Claude format
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": kwargs.get('max_tokens', 1000),
                "temperature": kwargs.get('temperature', 0.7),
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            }
            
            # Call Bedrock
            response = bedrock.invoke_model(
                modelId=bedrock_model,
                body=json.dumps(body),
                contentType="application/json"
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
            
        except ImportError:
            logger.warning("boto3 library not available for Bedrock")
            raise
        except Exception as e:
            logger.error(f"Bedrock API error: {e}")
            raise
    
    def _simulate_chat(self, message: str) -> str:
        """Simulate chat response when no API is available."""
        simulated_responses = {
            "hello": "Hello! I'm TidyLLM running in simulation mode. To use real AI models, please set up API credentials.",
            "how are you": "I'm doing well, thank you! This is a simulated response from TidyLLM.",
            "what is machine learning": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
            "help": "TidyLLM Basic API - Available functions: chat(), query(), process_document(). Set ANTHROPIC_API_KEY or OPENAI_API_KEY for real AI responses."
        }
        
        message_lower = message.lower()
        for key, response in simulated_responses.items():
            if key in message_lower:
                return f"[SIMULATION] {response}"
        
        return f"[SIMULATION] Thank you for your message: '{message}'. This is a simulated response. Set up API credentials for real AI responses."
    
    def query(self, question: str, context: Optional[str] = None, model: Optional[str] = None, **kwargs) -> str:
        """
        Query with optional context - useful for Q&A with documents.
        
        Args:
            question: The question to ask
            context: Optional context/document content  
            model: Optional model override
            **kwargs: Additional parameters
            
        Returns:
            Answer string from the AI model
            
        Example:
            >>> import tidyllm
            >>> answer = tidyllm.query("What is the main topic?", context="This document is about Python programming...")
            >>> print(answer)
        """
        if context:
            message = f"Context: {context}\n\nQuestion: {question}"
        else:
            message = question
        
        return self.chat(message, model, **kwargs)
    
    def process_document(self, file_path: Union[str, Path], question: Optional[str] = None, **kwargs) -> str:
        """
        Process a document and optionally ask a question about it.
        
        Args:
            file_path: Path to document file
            question: Optional question to ask about the document
            **kwargs: Additional parameters
            
        Returns:
            Document analysis or answer to question
            
        Example:
            >>> import tidyllm
            >>> result = tidyllm.process_document("document.pdf", "What is the main topic?")
            >>> print(result)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return f"[ERROR] File not found: {file_path}"
        
        # Try to extract text from document
        try:
            content = self._extract_document_content(file_path)
            
            if question:
                return self.query(question, context=content, **kwargs)
            else:
                return self.query("Please summarize this document.", context=content, **kwargs)
                
        except Exception as e:
            return f"[ERROR] Failed to process document: {e}"
    
    def _extract_document_content(self, file_path: Path) -> str:
        """Extract text content from various document formats."""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.txt':
            return file_path.read_text(encoding='utf-8')
        
        elif file_ext == '.pdf':
            try:
                import PyPDF2
                content = ""
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
                return content
            except ImportError:
                return f"[ERROR] PyPDF2 not available. Install with: pip install PyPDF2"
            except Exception as e:
                return f"[ERROR] PDF extraction failed: {e}"
        
        elif file_ext in ['.doc', '.docx']:
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                return f"[ERROR] python-docx not available. Install with: pip install python-docx"
            except Exception as e:
                return f"[ERROR] Word document extraction failed: {e}"
        
        else:
            return f"[ERROR] Unsupported file format: {file_ext}. Supported: .txt, .pdf, .doc, .docx"
    
    def list_models(self) -> list:
        """List available models."""
        return [
            "anthropic/claude-3-sonnet-20240229",
            "anthropic/claude-3-haiku-20240307", 
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "openai/gpt-4-turbo"
        ]
    
    def set_model(self, model: str):
        """Set default model."""
        self._default_model = model
        logger.info(f"Default model set to: {model}")
    
    def status(self) -> Dict[str, Any]:
        """Get API status information."""
        api_keys = self.config.get('api_keys', {})
        
        # Check gateway availability
        gateway_status = self._check_gateway_availability()
        
        return {
            "initialized": self._initialized,
            "default_model": self._default_model,
            "architecture": "gateway_compliant",
            "audit_mode": True,
            "gateway_chain": {
                "corporate_llm": gateway_status.get('corporate_llm', False),
                "ai_processing": gateway_status.get('ai_processing', False), 
                "workflow_optimizer": gateway_status.get('workflow_optimizer', False)
            },
            "primary_service": "aws_bedrock_via_gateways",
            "has_aws_key": bool(os.getenv('AWS_ACCESS_KEY_ID')),
            "has_aws_secret": bool(os.getenv('AWS_SECRET_ACCESS_KEY')),
            "aws_region": os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
            "has_anthropic_key": bool(os.getenv('ANTHROPIC_API_KEY')),
            "has_openai_key": bool(os.getenv('OPENAI_API_KEY')),
            "anthropic_enabled": api_keys.get('anthropic_enabled', True),
            "openai_enabled": api_keys.get('openai_enabled', True),
            "available_models": self.list_models(),
            "audit_logging": True,
            "compliance_mode": True
        }
    
    def _check_gateway_availability(self) -> Dict[str, bool]:
        """Check if gateway components are available."""
        try:
            from tidyllm.gateways import init_gateways
            from tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway
            from tidyllm.gateways.ai_processing_gateway import AIProcessingGateway
            from tidyllm.gateways.workflow_optimizer_gateway import WorkflowOptimizerGateway
            
            return {
                "corporate_llm": True,
                "ai_processing": True,
                "workflow_optimizer": True
            }
        except ImportError as e:
            logger.warning(f"Gateway system not fully available: {e}")
            return {
                "corporate_llm": False,
                "ai_processing": False,
                "workflow_optimizer": False
            }


# Global instance
_api = TidyLLMSimpleAPI()

# Export simple functions at module level
def chat(message: str, model: Optional[str] = None, **kwargs) -> str:
    """Simple chat function."""
    return _api.chat(message, model, **kwargs)

def query(question: str, context: Optional[str] = None, model: Optional[str] = None, **kwargs) -> str:
    """Query with optional context."""
    return _api.query(question, context, model, **kwargs)

def process_document(file_path: Union[str, Path], question: Optional[str] = None, **kwargs) -> str:
    """Process a document."""
    return _api.process_document(file_path, question, **kwargs)

def list_models() -> list:
    """List available models."""
    return _api.list_models()

def set_model(model: str):
    """Set default model."""
    return _api.set_model(model)

def status() -> Dict[str, Any]:
    """Get API status."""
    return _api.status()